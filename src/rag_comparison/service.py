import asyncio
from contextlib import asynccontextmanager

from .adapters import make_adapter
from .config import Settings
from .models import IngestRequest, RetrieveRequest
from .provider import Gateway


class RetrievalService:
    """Single-process adapter pool shared by HTTP and MCP entrypoints."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.gateway = Gateway(settings)
        self.adapters = {}
        self.lock = asyncio.Lock()

    async def adapter(self, backend, namespace):
        key = (backend, namespace)
        async with self.lock:
            if key not in self.adapters:
                adapter = make_adapter(backend, namespace, self.settings, self.gateway)
                with self.gateway.scope(backend, "agent:setup"):
                    await adapter.open()
                self.adapters[key] = adapter
            return self.adapters[key]

    async def ingest(self, request: IngestRequest):
        adapter = await self.adapter(request.backend, request.namespace)
        with self.gateway.scope(request.backend, "agent:ingest"):
            result = await adapter.ingest(request.update)
        return {"backend": request.backend, "namespace": request.namespace, **result}

    async def retrieve(self, request: RetrieveRequest):
        adapter = await self.adapter(request.backend, request.namespace)
        with self.gateway.scope(request.backend, "agent:retrieve"):
            retrieval = await adapter.retrieve(request.query, request.top_k, request.as_of)
        answer = None
        if request.answer:
            with self.gateway.scope(request.backend, "agent:answer"):
                answer = await self.gateway.answer(retrieval, context_tokens=request.context_tokens)
        return {
            "retrieval": retrieval.model_dump(mode="json"),
            "answer": answer.model_dump() if answer else None,
        }

    async def close(self):
        try:
            await asyncio.gather(
                *(a.close() for a in self.adapters.values()), return_exceptions=True
            )
        finally:
            await self.gateway.close()


@asynccontextmanager
async def service_context(settings=None):
    service = RetrievalService(settings or Settings())
    try:
        yield service
    finally:
        await service.close()
