from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rag_comparison.adapters.base import Adapter
from rag_comparison.adapters.graphiti import GraphitiAdapter, temporal_filters
from rag_comparison.adapters.lightrag import LightRAGAdapter
from rag_comparison.config import Settings
from rag_comparison.models import Retrieval, Update


class MemoryAdapter(Adapter):
    async def _open(self):
        self.writes = 0

    async def _ingest(self, update):
        self.writes += 1
        return {"write": self.writes}

    async def _retrieve(self, query, top_k, as_of):
        return Retrieval(backend=self.name, query=query, native_temporal_filter=False)


def config(tmp_path):
    return Settings(data_dir=tmp_path, openrouter_api_key="test-key", _env_file=None)


def event(**overrides):
    return Update(
        id="event1",
        document_id="doc1",
        text="Alice owns Atlas.",
        effective_at=datetime(2025, 1, 1, tzinfo=UTC),
    ).model_copy(update=overrides)


async def test_idempotency_survives_restart_and_rejects_id_collision(tmp_path):
    settings = config(tmp_path)
    adapter = await MemoryAdapter("lightrag-append", "test", settings, None).open()
    assert (await adapter.ingest(event()))["status"] == "applied"
    assert (await adapter.ingest(event()))["status"] == "duplicate"
    assert adapter.writes == 1
    with pytest.raises(ValueError, match="different content"):
        await adapter.ingest(event(text="Bob owns Atlas."))
    await adapter.close()
    reopened = await MemoryAdapter("lightrag-append", "test", settings, None).open()
    assert (await reopened.ingest(event()))["status"] == "duplicate"
    assert reopened.writes == 0
    await reopened.close()


async def test_namespace_lease_and_model_compatibility(tmp_path):
    settings = config(tmp_path)
    first = await MemoryAdapter("graphiti", "test", settings, None).open()
    with pytest.raises(RuntimeError, match="another process"):
        await MemoryAdapter("graphiti", "test", settings, None).open()
    await first.close()
    changed = settings.model_copy(update={"embedding_dim": 100})
    with pytest.raises(ValueError, match="different model"):
        await MemoryAdapter("graphiti", "test", changed, None).open()


@pytest.mark.parametrize(
    "backend,expected_deleted", [("lightrag-append", 0), ("lightrag-replace", 1)]
)
async def test_lightrag_update_policy_and_checked_deletion(tmp_path, backend, expected_deleted):
    adapter = LightRAGAdapter(backend, "test", config(tmp_path), None)
    adapter.state["documents"]["doc1"] = ["old-doc"]
    adapter.rag = SimpleNamespace(
        adelete_by_doc_id=AsyncMock(
            return_value=SimpleNamespace(status="success", message="deletion result")
        ),
        ainsert=AsyncMock(),
        doc_status=SimpleNamespace(get_by_id=AsyncMock(return_value={"status": "processed"})),
    )
    result = await adapter.ingest(event())
    assert adapter.rag.adelete_by_doc_id.await_count == expected_deleted
    assert result["documents_deleted"] == expected_deleted
    assert len(adapter.state["documents"]["doc1"]) == (2 if not expected_deleted else 1)
    if expected_deleted:
        adapter.rag.adelete_by_doc_id.return_value.status = "fail"
        with pytest.raises(RuntimeError, match="deletion failed"):
            await adapter.ingest(event(id="event2"))


async def test_lightrag_does_not_treat_swallowed_ingestion_error_as_success(tmp_path):
    adapter = LightRAGAdapter("lightrag-append", "test", config(tmp_path), None)
    adapter.rag = SimpleNamespace(
        ainsert=AsyncMock(),
        doc_status=SimpleNamespace(get_by_id=AsyncMock(return_value={"status": "failed"})),
    )
    with pytest.raises(RuntimeError, match="did not finish"):
        await adapter.ingest(event())
    assert not adapter.state["events"]


async def test_graphiti_lets_library_create_new_episode_uuid(tmp_path):
    adapter = GraphitiAdapter("graphiti", "test", config(tmp_path), None)
    adapter.graph = SimpleNamespace(
        add_episode=AsyncMock(
            return_value=SimpleNamespace(
                episode=SimpleNamespace(uuid="created-uuid"), nodes=[], edges=[]
            )
        )
    )
    result = await adapter.ingest(event())
    assert "uuid" not in adapter.graph.add_episode.call_args.kwargs
    assert adapter.graph.add_episode.call_args.kwargs["reference_time"] == event().effective_at
    assert result["episode_id"] == "created-uuid"


def test_native_temporal_filter_is_half_open_and_includes_undated_facts():
    from graphiti_core.driver.driver import GraphProvider
    from graphiti_core.search.search_filters import edge_search_filter_query_constructor

    instant = datetime(2025, 2, 1, tzinfo=UTC)
    queries, params = edge_search_filter_query_constructor(
        temporal_filters(instant), GraphProvider.NEO4J
    )
    assert queries == [
        "((e.valid_at <= $valid_at_0) OR (e.valid_at IS NULL))",
        "((e.invalid_at > $invalid_at_0) OR (e.invalid_at IS NULL))",
    ]
    assert params == {"valid_at_0": instant, "invalid_at_0": instant}


async def test_graphiti_embedder_accepts_search_query_list_and_json_fallback(tmp_path, monkeypatch):
    import numpy as np

    import rag_comparison.adapters.graphiti as module

    gateway = SimpleNamespace(
        settings=config(tmp_path),
        client=object(),
        embed=AsyncMock(return_value=np.ones((1, 1536), dtype=np.float32)),
    )
    graph = SimpleNamespace(build_indices_and_constraints=AsyncMock())
    captured = {}

    def construct(**kwargs):
        captured.update(kwargs)
        return graph

    monkeypatch.setattr(module, "Graphiti", construct)
    adapter = GraphitiAdapter("graphiti", "test", gateway.settings, gateway)
    await adapter._open()
    assert len(await captured["embedder"].create(["query"])) == 1536
    assert len(await captured["embedder"].create("query")) == 1536
    assert captured["llm_client"].structured_output_mode == "json_object"
