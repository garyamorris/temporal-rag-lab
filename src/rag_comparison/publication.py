"""Create shareable snapshots of reviewed benchmark reports."""

import copy
import hashlib
import json
from pathlib import Path

from .report import write_html


def public_snapshot(report: dict) -> dict:
    """Omit operational identifiers, without rewriting scores, answers, or evidence.

    Callers must review document text before exporting. This is not a general PII scrubber.
    """
    if report.get("status") != "complete" or report.get("errors"):
        raise ValueError("Only completed, error-free reports can be published as snapshots.")
    snapshot = copy.deepcopy(report)
    snapshot["requests"] = []
    for row in [*snapshot.get("updates", []), *snapshot.get("probes", [])]:
        row.pop("namespace", None)
    snapshot["publication"] = {
        "read_only": True,
        "omitted": ["per-request usage ledger", "runtime namespace identifiers"],
        "preserved": ["suite", "answers", "evidence", "scores", "aggregate usage", "versions"],
    }
    snapshot["methodology"]["limitations"].insert(
        0,
        "Published snapshot: runtime namespaces and the per-request usage ledger were omitted. "
        "Scores, evidence and aggregate usage are preserved. No live retrieval index is included.",
    )
    return snapshot


def export_report(source: Path, output: Path) -> Path:
    raw = source.read_bytes()
    snapshot = public_snapshot(json.loads(raw))
    snapshot["publication"]["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    (output / "suite.json").write_text(json.dumps(snapshot["suite"], indent=2) + "\n")
    write_html(snapshot, output / "report.html")
    return output / "report.html"
