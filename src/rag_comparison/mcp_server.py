from contextlib import asynccontextmanager
from datetime import datetime

from mcp.server.fastmcp import Context, FastMCP

from .models import BACKENDS, BackendName, IngestRequest, RetrieveRequest, Update
from .service import service_context


def run_mcp():
    @asynccontextmanager
    async def lifespan(server):
        async with service_context() as service:
            yield service

    server = FastMCP("Temporal RAG Lab retrieval", lifespan=lifespan)

    @server.tool()
    async def retrieve(
        query: str,
        ctx: Context,
        backend: BackendName = "graphiti",
        namespace: str = "agent",
        top_k: int = 10,
        as_of: datetime | None = None,
        answer: bool = True,
        context_tokens: int = 4096,
    ) -> dict:
        """Retrieve cited evidence and an optional grounded answer. Graphiti supports valid-time
        (as_of) filtering. LightRAG as_of is a query hint, not a database filter.
        Timestamps require a timezone, for example 2025-01-01T00:00:00Z.
        Use the same namespace you ingested into; benchmark namespaces appear in report.json.
        """
        service = ctx.request_context.lifespan_context
        return await service.retrieve(
            RetrieveRequest(
                query=query,
                backend=backend,
                namespace=namespace,
                top_k=top_k,
                as_of=as_of,
                answer=answer,
                context_tokens=context_tokens,
            )
        )

    @server.tool()
    async def compare_retrieval(
        query: str,
        ctx: Context,
        namespace: str = "agent",
        as_of: datetime | None = None,
        top_k: int = 10,
    ) -> dict:
        """Compare all three real backends for already-ingested knowledge in one namespace.
        Each backend must be populated separately. Returns errors per backend, never fake results.
        """
        from .provider import safe_error

        service = ctx.request_context.lifespan_context
        results = {}
        for backend in BACKENDS:
            try:
                results[backend] = await service.retrieve(
                    RetrieveRequest(
                        query=query,
                        backend=backend,
                        namespace=namespace,
                        as_of=as_of,
                        top_k=top_k,
                    )
                )
            except Exception as exc:
                results[backend] = {"error": safe_error(exc, service.settings)}
        return results

    @server.tool()
    async def ingest_update(
        event_id: str,
        document_id: str,
        text: str,
        effective_at: datetime,
        ctx: Context,
        backend: BackendName = "graphiti",
        namespace: str = "agent",
    ) -> dict:
        """Ingest a dated update. Reusing an identical event id is idempotent; changed content
        requires a new event id. Select each backend separately to populate a comparison.
        """
        service = ctx.request_context.lifespan_context
        return await service.ingest(
            IngestRequest(
                backend=backend,
                namespace=namespace,
                update=Update(
                    id=event_id, document_id=document_id, text=text, effective_at=effective_at
                ),
            )
        )

    server.run(transport="stdio")
