import hashlib
from datetime import datetime

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

from ..models import Evidence, Retrieval, Update
from .base import Adapter


class LightRAGAdapter(Adapter):
    async def _open(self):
        async def llm(prompt, system_prompt=None, history_messages=None, **kwargs):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages or [])
            messages.append({"role": "user", "content": prompt})
            # LightRAG passes cache/storage internals through kwargs; do not send them to a provider.
            options = {
                k: v for k, v in kwargs.items() if k in {"temperature", "max_tokens", "top_p"}
            }
            return await self.gateway.complete(messages, **options)

        async def embed(texts, **kwargs):
            return await self.gateway.embed(texts)

        self.rag = LightRAG(
            working_dir=str(self.path / "storage"),
            workspace=self.group_id,
            llm_model_func=llm,
            llm_model_name=self.settings.model,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.settings.embedding_dim,
                func=embed,
                max_token_size=8192,
                model_name=self.settings.embedding_model,
            ),
            llm_model_max_async=self.settings.concurrency,
            embedding_func_max_async=self.settings.concurrency,
            default_embedding_timeout=int(self.settings.request_timeout),
            default_llm_timeout=int(self.settings.request_timeout),
            entity_extract_max_gleaning=1,
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            max_parallel_insert=1,
        )
        await self.rag.initialize_storages()
        await initialize_pipeline_status(workspace=self.group_id)

    async def _delete(self, doc_id: str):
        result = await self.rag.adelete_by_doc_id(doc_id)
        if result.status not in {"success", "not_found"}:
            raise RuntimeError(f"LightRAG deletion failed: {result.status}: {result.message}")

    async def _ingest(self, update: Update) -> dict:
        previous = self.state["documents"].get(update.document_id, [])
        deleted = 0
        if update.operation == "delete" or self.name == "lightrag-replace":
            for doc_id in previous:
                await self._delete(doc_id)
                deleted += 1
            previous = []
            # Persist deletion before reinsertion: failures remain inspectable and can be retried.
            self.state["documents"][update.document_id] = []
            self._save()
        if update.operation == "delete":
            return {"documents_deleted": deleted, "documents_inserted": 0}
        doc_id = "doc-" + hashlib.sha256(update.id.encode()).hexdigest()[:32]
        await self.rag.ainsert(
            update.source_text(), ids=doc_id, file_paths=f"{update.document_id}/{update.id}.txt"
        )
        status = await self.rag.doc_status.get_by_id(doc_id)
        status_value = (
            status.get("status") if isinstance(status, dict) else getattr(status, "status", None)
        )
        if getattr(status_value, "value", status_value) != "processed":
            raise RuntimeError(f"LightRAG ingestion did not finish successfully: {status_value}")
        self.state["documents"][update.document_id] = list(dict.fromkeys([*previous, doc_id]))
        return {"documents_deleted": deleted, "documents_inserted": 1, "document_id": doc_id}

    async def _retrieve(self, query: str, top_k: int, as_of: datetime | None) -> Retrieval:
        warnings = []
        search_query = query
        if as_of:
            search_query += f" As of {as_of.isoformat()}."
            warnings.append(
                "LightRAG has no native validity filter. as_of is a query hint; evidence may span dates."
            )
        raw = await self.rag.aquery_data(
            search_query,
            param=QueryParam(
                mode="mix",
                top_k=top_k,
                chunk_top_k=top_k,
                max_total_tokens=12000,
                enable_rerank=False,
            ),
        )
        data = raw.get("data") or {}
        if raw.get("status") != "success" and data:
            raise RuntimeError(f"LightRAG retrieval failed: {raw.get('message')}")
        if raw.get("status") != "success":
            warnings.append(str(raw.get("message", "No retrieval results.")))
        evidence = []
        # Preserve the context chosen by LightRAG rather than imposing a custom reranker.
        for category, text_key, id_key in [
            ("chunks", "content", "chunk_id"),
            ("relationships", "description", "reference_id"),
            ("entities", "description", "entity_name"),
        ]:
            for i, item in enumerate(data.get(category, [])):
                source = item.get("file_path", "")
                evidence.append(
                    Evidence(
                        id=f"{category}:{item.get(id_key) or i}:{i}",
                        text=item.get(text_key, ""),
                        kind=category,
                        source_ids=[source] if source else [],
                        metadata=item,
                    )
                )
        return Retrieval(
            backend=self.name,
            query=query,
            as_of=as_of,
            native_temporal_filter=False,
            evidence=evidence,
            warnings=warnings,
        )

    async def close(self):
        try:
            if hasattr(self, "rag"):
                await self.rag.finalize_storages()
        finally:
            await super().close()
