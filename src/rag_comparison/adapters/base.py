import asyncio
import fcntl
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from time import perf_counter

from ..config import Settings
from ..models import BackendName, Retrieval, Update, safe_name
from ..provider import Gateway


class Adapter(ABC):
    def __init__(self, name: BackendName, namespace: str, settings: Settings, gateway: Gateway):
        self.name, self.namespace = name, safe_name(namespace)
        self.settings, self.gateway = settings, gateway
        self.path = settings.data_dir.resolve() / namespace / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "manifest.json"
        self.state = {"events": {}, "documents": {}, "config": settings.public()}
        self._lock = asyncio.Lock()
        self._lease = None
        # Neo4j groups are isolated by the absolute local storage directory AND namespace.
        self.group_id = "rb_" + hashlib.sha256(str(self.path).encode()).hexdigest()[:28]

    async def open(self):
        self._lease = (self.path / "writer.lock").open("a")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lease.close()
            self._lease = None
            raise RuntimeError(
                f"Namespace {self.namespace}/{self.name} is in use by another process."
            ) from exc
        try:
            if self.manifest_path.exists():
                self.state = json.loads(self.manifest_path.read_text())
                if self.state["config"] != self.settings.public():
                    raise ValueError(
                        "This namespace uses different model settings. Use a fresh namespace."
                    )
            self.state["group_id"] = self.group_id
            await self._open()
            self._save()
        except BaseException:
            await self.close()
            raise
        return self

    def _save(self):
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2))
        temporary.replace(self.manifest_path)

    async def ingest(self, update: Update) -> dict:
        async with self._lock:
            # Probe labels are evaluation data, never ingestion input.
            payload = update.model_dump(mode="json", exclude={"probes"})
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            previous = self.state["events"].get(update.id)
            if previous:
                if previous["digest"] != digest:
                    raise ValueError(
                        "An existing event id cannot be reused with different content."
                    )
                return {"status": "duplicate", "latency_ms": 0, "event_id": update.id}
            started = perf_counter()
            result = await self._ingest(update)
            result.update(
                {
                    "status": "applied",
                    "event_id": update.id,
                    "latency_ms": round((perf_counter() - started) * 1000, 2),
                }
            )
            self.state["events"][update.id] = {"digest": digest, "result": result}
            self._save()
            return result

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        as_of: datetime | None = None,
    ) -> Retrieval:
        async with self._lock:
            started = perf_counter()
            result = await self._retrieve(query, top_k, as_of)
            result.latency_ms = round((perf_counter() - started) * 1000, 2)
            return result

    @abstractmethod
    async def _open(self): ...

    @abstractmethod
    async def _ingest(self, update: Update) -> dict: ...

    @abstractmethod
    async def _retrieve(self, query: str, top_k: int, as_of: datetime | None) -> Retrieval: ...

    async def close(self):
        if self._lease:
            fcntl.flock(self._lease, fcntl.LOCK_UN)
            self._lease.close()
            self._lease = None
