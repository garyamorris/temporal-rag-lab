import json
from contextlib import nullcontext
from datetime import UTC, datetime

import httpx
import pytest

from rag_comparison.config import Settings
from rag_comparison.harness import run_suite
from rag_comparison.models import Answer, Evidence, Probe, Retrieval, Scenario, Suite, Update
from rag_comparison.provider import Gateway, safe_error, usage_summary
from rag_comparison.report import write_html


def test_missing_prices_are_unknown_not_free():
    result = usage_summary([{"status": 200, "cost_usd": None}])
    assert result["reported_cost_usd"] is None
    assert result["requests_without_reported_cost"] == 1
    partial = usage_summary([{"status": 200, "cost_usd": 0.02}, {"status": 200}])
    assert partial["reported_cost_usd"] == 0.02
    assert partial["requests_without_reported_cost"] == 1


def test_secrets_are_redacted():
    settings = Settings(
        openrouter_api_key="sk-or-v1-test", neo4j_password="privatepw", _env_file=None
    )
    assert (
        safe_error(ValueError("sk-or-v1-test privatepw sk-fallback"), settings)
        == "[REDACTED] [REDACTED] [REDACTED]"
    )
    assert "sk-or-v1-test" not in json.dumps(settings.public())


async def test_openrouter_gateway_routes_chat_embeddings_and_measures_usage():
    settings = Settings(openrouter_api_key="router-test", embedding_dim=2, _env_file=None)
    gateway = Gateway(settings)
    requests = []

    async def handler(request):
        requests.append(request)
        assert request.url.host == "openrouter.ai"
        assert request.headers["Authorization"] == "Bearer router-test"
        body = json.loads(request.content)
        if request.url.path.endswith("embeddings"):
            assert body["dimensions"] == 2
            return httpx.Response(
                200,
                json={
                    "data": [{"index": 0, "embedding": [1, 0], "object": "embedding"}],
                    "model": settings.embedding_model,
                    "object": "list",
                    "usage": {"prompt_tokens": 2, "total_tokens": 2, "cost": 0.001},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "request1",
                "model": settings.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                    "cost": 0.002,
                },
            },
        )

    # Replace the HTTP transport while retaining the real SDK and measurement hooks.
    await gateway.http.aclose()
    gateway.http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [gateway._request], "response": [gateway._response]},
    )
    gateway.client._client = gateway.http
    with gateway.scope("graphiti", "test"):
        assert await gateway.complete([{"role": "user", "content": "secret prompt"}]) == "OK"
        assert (await gateway.embed(["hello"])).shape == (1, 2)
    assert len(requests) == 2
    assert usage_summary(gateway.events)["reported_cost_usd"] == 0.003
    assert all(event["backend"] == "graphiti" for event in gateway.events)
    assert "secret prompt" not in json.dumps(gateway.events)
    assert "router-test" not in json.dumps(gateway.events)
    await gateway.close()


class FakeGateway:
    def __init__(self, settings):
        self.events = []

    def scope(self, *args):
        return nullcontext()

    async def answer(self, retrieval):
        # The harness sends no gold labels into the answer path.
        assert "SECRET_GOLD" not in retrieval.model_dump_json()
        return Answer(answer="Alice", evidence_ids=["e"])

    async def close(self):
        pass


class FakeAdapter:
    def __init__(self, name, namespace, settings, gateway):
        self.name = name
        self.closed = False

    async def open(self):
        if self.name == "graphiti":
            raise RuntimeError("database offline")

    async def ingest(self, update):
        return {"latency_ms": 100}

    async def retrieve(self, query, as_of=None):
        return Retrieval(
            backend=self.name,
            query=query,
            native_temporal_filter=False,
            evidence=[Evidence(id="e", text="Alice", kind="chunk")],
        )

    async def close(self):
        self.closed = True


async def test_harness_persists_failures_coverage_raw_evidence_and_no_gold_leak(tmp_path):
    settings = Settings(runs_dir=tmp_path, _env_file=None)
    suite = Suite(
        name="test",
        description="test",
        scenarios=[
            Scenario(
                id="scenario",
                description="test",
                steps=[
                    Update(
                        id="event",
                        document_id="doc",
                        text="Alice",
                        effective_at=datetime.now(UTC),
                        probes=[
                            Probe(id="probe", query="Owner?", expected_any=["Alice", "SECRET_GOLD"])
                        ],
                    )
                ],
            )
        ],
    )
    report = await run_suite(
        suite,
        settings,
        ["graphiti", "lightrag-append"],
        "test-run",
        adapter_factory=FakeAdapter,
        gateway_factory=FakeGateway,
        progress=lambda _: None,
    )
    assert report["status"] == "partial"
    assert report["summary"]["graphiti"]["answer_accuracy"] is None
    assert report["summary"]["graphiti"]["probes_completed"] == 0
    assert report["summary"]["graphiti"]["probes_planned"] == 1
    assert report["summary"]["lightrag-append"]["answer_accuracy"] == 1
    assert report["probes"][0]["retrieval"]["evidence"][0]["text"] == "Alice"
    saved = json.loads((tmp_path / "test-run/report.json").read_text())
    assert saved["status"] == "partial"
    assert (tmp_path / "test-run/report.html").exists()
    with pytest.raises(FileExistsError):
        await run_suite(
            suite,
            settings,
            ["lightrag-append"],
            "test-run",
            adapter_factory=FakeAdapter,
            gateway_factory=FakeGateway,
        )


def test_report_does_not_inject_untrusted_evidence_into_script(tmp_path):
    path = tmp_path / "report.html"
    write_html({"payload": "</script><script>alert('x')</script>"}, path)
    assert "<script>alert" not in path.read_text()
    assert "__REPORT__" not in path.read_text()


async def test_answer_cannot_cite_evidence_omitted_by_budget():
    from unittest.mock import AsyncMock

    gateway = Gateway(Settings(openrouter_api_key="test-only", _env_file=None))
    gateway.complete = AsyncMock(
        return_value=json.dumps({"answer": "Bob", "evidence_ids": ["hidden"]})
    )
    retrieval = Retrieval(
        backend="graphiti",
        query="Owner?",
        native_temporal_filter=True,
        evidence=[
            Evidence(id="hidden", text="large text " * 5000, kind="fact"),
            Evidence(id="shown", text="Bob owns Atlas.", kind="fact"),
        ],
    )
    try:
        with pytest.raises(ValueError, match="not retrieved"):
            await gateway.answer(retrieval, context_tokens=128)
        body = json.loads(gateway.complete.call_args.args[0][1]["content"])
        assert [item["id"] for item in body["evidence"]] == ["shown"]
        assert retrieval.answer_context.tokens_used <= 128
    finally:
        await gateway.close()


async def test_current_scenario_clock_is_not_misclassified_as_historical(tmp_path):
    instant = datetime(2025, 1, 1, tzinfo=UTC)
    suite = Suite(
        name="clock",
        description="test",
        scenarios=[
            Scenario(
                id="clock",
                description="test",
                metadata={"history_depth": 12, "arrival_order": "late", "seed": 7},
                steps=[
                    Update(
                        id="event",
                        document_id="doc",
                        text="Alice",
                        effective_at=instant,
                        probes=[
                            Probe(
                                id="current",
                                query="Owner?",
                                expected_any=["Alice"],
                                as_of=instant,
                                temporal_role="current",
                                dimension="temporal",
                            )
                        ],
                    )
                ],
            )
        ],
    )
    report = await run_suite(
        suite,
        Settings(runs_dir=tmp_path, _env_file=None),
        ["lightrag-append"],
        "clock",
        adapter_factory=FakeAdapter,
        gateway_factory=FakeGateway,
        progress=lambda _: None,
    )
    summary = report["summary"]["lightrag-append"]
    assert summary["current_scored"] == 1
    assert summary["historical_scored"] == 0
    assert summary["by_dimension"]["temporal"]["accuracy"] == 1
    assert summary["by_condition"][0]["context_tokens"] == 4096
