import json

import pytest

from rag_comparison.publication import export_report, public_snapshot


def sample_report():
    return {
        "status": "complete",
        "errors": [],
        "requests": [{"request_id": "provider-trace", "total_tokens": 20}],
        "updates": [{"namespace": "local-namespace", "result": {"status": "applied"}}],
        "probes": [
            {
                "namespace": "local-namespace",
                "answer": {"answer": "Alice"},
                "scores": {"answer_correct": True},
            }
        ],
        "suite": {"name": "Synthetic source text"},
        "summary": {"reported_cost_usd": 0.01},
        "methodology": {"limitations": []},
    }


def test_publication_omits_operational_metadata_without_rewriting_results():
    original = sample_report()
    published = public_snapshot(original)
    assert published["requests"] == []
    assert "namespace" not in published["updates"][0]
    assert "namespace" not in published["probes"][0]
    assert published["probes"][0]["answer"] == original["probes"][0]["answer"]
    assert published["probes"][0]["scores"] == original["probes"][0]["scores"]
    assert published["summary"] == original["summary"]
    assert published["suite"] == original["suite"]
    assert published["publication"]["read_only"]
    assert original["requests"][0]["request_id"] == "provider-trace"
    assert original["probes"][0]["namespace"] == "local-namespace"
    assert original["methodology"]["limitations"] == []


def test_incomplete_or_error_reports_cannot_be_exported():
    report = sample_report()
    report["status"] = "running"
    with pytest.raises(ValueError, match="completed"):
        public_snapshot(report)
    report["status"] = "complete"
    report["errors"] = [{"error": "provider diagnostic"}]
    with pytest.raises(ValueError, match="error-free"):
        public_snapshot(report)


def test_export_creates_new_read_only_artifacts_and_preserves_source(tmp_path):
    source = tmp_path / "original.json"
    original = json.dumps(sample_report())
    source.write_text(original)
    target = tmp_path / "published"
    assert export_report(source, target) == target / "report.html"
    assert source.read_text() == original
    assert (target / "suite.json").exists()
    published = json.loads((target / "report.json").read_text())
    assert len(published["publication"]["source_report_sha256"]) == 64
    assert "provider-trace" not in (target / "report.json").read_text()
    assert "local-namespace" not in (target / "report.json").read_text()
    with pytest.raises(FileExistsError):
        export_report(source, target)
