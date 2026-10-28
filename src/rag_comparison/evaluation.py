import json
import re
from collections import defaultdict
from statistics import mean

from .models import Answer, Probe, Retrieval
from .provider import normalize_evidence, usage_summary


def mentions(text: str, aliases: list[str]) -> bool:
    """Case-insensitive whole-phrase match; aliases are literal, not regular expressions."""
    return any(
        re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.I) for alias in aliases
    )


def score_probe(probe: Probe, retrieval: Retrieval, answer: Answer | None) -> dict:
    context = retrieval.context()
    answer_expected = mentions(answer.answer, probe.expected_any) if answer else None
    answer_stale = mentions(answer.answer, probe.forbidden_any) if answer else None
    correct = (answer_expected and not answer_stale) if answer else None
    model_context = None
    if retrieval.answer_context is not None:
        selected = set(retrieval.answer_context.evidence_ids)
        model_context = "\n".join(
            json.dumps(normalize_evidence(e)) for e in retrieval.evidence if e.id in selected
        )
    expected_retrieved = mentions(context, probe.expected_any)
    expected_shown = (
        mentions(model_context, probe.expected_any) if model_context is not None else None
    )
    if answer is None:
        outcome = "unscored"
    elif correct:
        outcome = "correct"
    elif not expected_retrieved:
        outcome = "retrieval_missing_expected_alias"
    elif expected_shown is False:
        outcome = "context_budget_missing_expected_alias"
    elif answer_stale:
        outcome = "answer_used_obsolete_alias"
    else:
        outcome = "answer_incorrect_despite_expected_alias"
    return {
        "outcome": outcome,
        "expected_in_answer_context": expected_shown,
        "forbidden_mention_in_answer_context": mentions(model_context, probe.forbidden_any)
        if model_context is not None
        else None,
        "answer_correct": (answer_expected and not answer_stale) if answer else None,
        "expected_in_answer": answer_expected,
        "forbidden_in_answer": answer_stale,
        "expected_in_evidence": mentions(context, probe.expected_any),
        # This is exposure, not a claim that an old fact is necessarily wrong in historical context.
        "forbidden_mention_in_evidence": mentions(context, probe.forbidden_any),
        "has_citation": bool(answer.evidence_ids) if answer else None,
    }


def summarize(report: dict) -> dict:
    phases = {}
    for scenario in report["suite"]["scenarios"]:
        seen = set()
        for step in scenario["steps"]:
            phase = (
                "delete"
                if step["operation"] == "delete"
                else ("update" if step["document_id"] in seen else "insert")
            )
            phases[(scenario["id"], step["id"])] = phase
            seen.add(step["document_id"])
    for row in report["updates"]:
        row["phase"] = phases[(row["scenario"], row["step"])]
    grouped = defaultdict(list)
    for row in report["probes"]:
        grouped[row["backend"]].append(row)
    result = {}
    for backend in report["backends"]:
        rows = grouped[backend]
        ok = [r for r in rows if r.get("status") == "ok"]
        answered = [r for r in ok if r["scores"]["answer_correct"] is not None]
        historical = [
            r
            for r in answered
            if (
                r["probe"].get("temporal_role")
                or ("historical" if r["probe"].get("as_of") else "current")
            )
            == "historical"
        ]
        current = [r for r in answered if r not in historical]
        writes = [r for r in report["updates"] if r["backend"] == backend and r["status"] == "ok"]
        events = [e for e in report["requests"] if e["backend"] == backend]
        errors = [r for r in report["errors"] if r["backend"] == backend]

        def accuracy(items):
            return round(mean(r["scores"]["answer_correct"] for r in items), 4) if items else None

        dimensions = defaultdict(list)
        conditions = defaultdict(list)
        scenario_metadata = {s["id"]: s.get("metadata", {}) for s in report["suite"]["scenarios"]}
        for row in answered:
            dimensions[row["probe"].get("dimension", "general")].append(row)
            meta = scenario_metadata[row["scenario"]]
            if all(key in meta for key in ("history_depth", "arrival_order", "seed")):
                conditions[
                    (
                        meta["history_depth"],
                        meta["arrival_order"],
                        row["probe"].get("context_tokens") or 4096,
                    )
                ].append(row)

        def group_metrics(items):
            selected = [
                r for r in items if r["scores"].get("expected_in_answer_context") is not None
            ]
            covered = [r for r in selected if r["scores"]["expected_in_answer_context"]]
            outcomes = defaultdict(int)
            for row in items:
                outcomes[row["scores"].get("outcome", "legacy")] += 1
            return {
                "scored": len(items),
                "correct": sum(r["scores"]["answer_correct"] for r in items),
                "accuracy": accuracy(items),
                "expected_alias_in_answer_context_rate": round(
                    mean(r["scores"]["expected_in_answer_context"] for r in selected), 4
                )
                if selected
                else None,
                "accuracy_given_expected_alias_in_context": accuracy(covered),
                "outcomes": dict(outcomes),
            }

        result[backend] = {
            "by_dimension": {name: group_metrics(items) for name, items in dimensions.items()},
            "by_condition": [
                {
                    "history_depth": key[0],
                    "arrival_order": key[1],
                    "context_tokens": key[2],
                    "seeds": sorted({scenario_metadata[r["scenario"]]["seed"] for r in items}),
                    **group_metrics(items),
                }
                for key, items in sorted(conditions.items())
            ],
            "probes_completed": len(ok),
            "probes_failed": len(rows) - len(ok),
            "probes_planned": report["planned_probes_per_backend"],
            "answer_accuracy": accuracy(answered),
            "answers_scored": len(answered),
            "current_accuracy": accuracy(current),
            "current_scored": len(current),
            "historical_accuracy": accuracy(historical),
            "historical_scored": len(historical),
            "forbidden_evidence_rate": round(
                mean(r["scores"]["forbidden_mention_in_evidence"] for r in ok), 4
            )
            if ok
            else None,
            **{
                f"mean_{phase}_ms": round(
                    mean(r["result"]["latency_ms"] for r in writes if r["phase"] == phase), 2
                )
                if any(r["phase"] == phase for r in writes)
                else None
                for phase in ["insert", "update", "delete"]
            },
            "mean_retrieval_ms": round(mean(r["retrieval"]["latency_ms"] for r in ok), 2)
            if ok
            else None,
            "update_count": len(writes),
            "error_count": len(errors),
            "usage": usage_summary(events),
        }
    return result
