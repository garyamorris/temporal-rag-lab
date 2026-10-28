from fastapi.testclient import TestClient

from rag_comparison.api import create_app
from rag_comparison.config import Settings


class FakeService:
    def __init__(self, settings):
        self.settings = settings
        self.closed = False

    async def retrieve(self, request):
        if request.namespace == "unavailable":
            raise RuntimeError("upstream sk-or-sensitive failed")
        return {"backend": request.backend, "query": request.query}

    async def ingest(self, request):
        return {"status": "applied", "event_id": request.update.id}

    async def close(self):
        self.closed = True


def test_http_contract_and_validation_and_secret_redaction(tmp_path):
    settings = Settings(runs_dir=tmp_path, openrouter_api_key="sk-or-sensitive", _env_file=None)
    app = create_app(settings, service_factory=FakeService)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["provider"] == "openrouter"
        assert "sk-or-sensitive" not in client.get("/health").text
        assert len(client.get("/api/suite").json()["scenarios"]) == 4
        assert (
            client.post("/api/retrieve", json={"query": "Owner?"}).json()["backend"] == "graphiti"
        )
        assert (
            client.post(
                "/api/retrieve", json={"query": "Owner?", "namespace": "../oops"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/retrieve", json={"query": "Owner?", "as_of": "2025-01-01"}
            ).status_code
            == 422
        )
        failure = client.post("/api/retrieve", json={"query": "Owner?", "namespace": "unavailable"})
        assert failure.status_code == 502
        assert "sk-or-sensitive" not in failure.text
        assert "REDACTED" in failure.text
        assert client.get("/api/runs").json() == []
        assert client.get("/api/runs/missing").status_code == 404
        assert client.post("/api/runs", json={"scenarios": ["typo"]}).status_code == 400
        assert (
            client.post("/api/runs", json={"backends": ["graphiti", "graphiti"]}).status_code == 422
        )
    assert app.state.service.closed


def test_temporal_workloads_and_completed_dataset_discovery(tmp_path):
    import json

    from rag_comparison.api import builtin_suite_path
    from rag_comparison.harness import load_suite

    settings = Settings(runs_dir=tmp_path, _env_file=None)
    directory = tmp_path / "completed"
    directory.mkdir()
    suite = load_suite(builtin_suite_path()).model_dump(mode="json")
    (directory / "report.json").write_text(
        json.dumps(
            {
                "run_id": "completed",
                "started_at": "2026-10-28T20:00:00Z",
                "status": "complete",
                "suite": suite,
                "summary": {},
                "updates": [
                    {
                        "backend": "graphiti",
                        "namespace": "indexed-owner",
                        "scenario": "owner-change",
                        "status": "ok",
                    },
                    {
                        "backend": "graphiti",
                        "namespace": "indexed-owner",
                        "scenario": "owner-change",
                        "status": "ok",
                    },
                ],
            }
        )
    )
    with TestClient(create_app(settings, service_factory=FakeService)) as client:
        pilot = client.get("/api/suite?kind=temporal-pilot").json()
        assert len(pilot["scenarios"]) == 1
        assert pilot["metadata"]["depths"] == [12]
        assert len(client.get("/api/suite?kind=temporal-stress").json()["scenarios"]) == 18
        assert client.get("/api/suite?kind=unknown").status_code == 422
        datasets = client.get("/api/datasets").json()
        assert len(datasets) == 1
        assert datasets[0]["namespace"] == "indexed-owner"
        assert "Project Atlas" in datasets[0]["example_query"]
        saved = json.loads((directory / "report.json").read_text())
        saved["publication"] = {"read_only": True}
        saved["updates"][0].pop("namespace")
        (directory / "report.json").write_text(json.dumps(saved))
        assert client.get("/api/datasets").json() == []
