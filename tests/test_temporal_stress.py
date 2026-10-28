import json
from datetime import UTC, datetime, timedelta

import pytest
import tiktoken

from rag_comparison.evaluation import score_probe
from rag_comparison.models import Answer, Evidence, Probe, Retrieval
from rag_comparison.provider import pack_evidence
from rag_comparison.stress import Assignment, owner_at, temporal_stress_suite


def test_oracle_uses_effective_time_and_half_open_boundaries_not_arrival():
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    stream = [
        Assignment(t0, "Alice", 0),
        Assignment(t0 + timedelta(days=14), "Clara", 2),
        Assignment(t0 + timedelta(days=7), "Bob", 1),
    ]
    assert owner_at(stream, t0 + timedelta(days=6)) == "Alice"
    assert owner_at(stream, t0 + timedelta(days=7)) == "Bob"
    assert owner_at(stream, t0 + timedelta(days=14)) == "Clara"
    with pytest.raises(ValueError):
        owner_at(stream, t0 - timedelta(seconds=1))


def test_generated_order_variants_are_paired_deterministic_and_gold_is_not_in_source():
    suite = temporal_stress_suite(depths=(12,), seeds=(7,), budgets=(512, 2048))
    assert (
        suite.model_dump_json()
        == temporal_stress_suite(depths=(12,), seeds=(7,), budgets=(512, 2048)).model_dump_json()
    )
    chronological, late = suite.scenarios
    assert chronological.steps[-1].probes == late.steps[-1].probes
    canonical = {step.id: step.model_dump(exclude={"probes"}) for step in chronological.steps}
    shuffled = {step.id: step.model_dump(exclude={"probes"}) for step in late.steps}
    assert canonical == shuffled
    assert [s.id for s in chronological.steps] != [s.id for s in late.steps]
    assert len(chronological.steps) == 17  # 12 owners, 3 independent audits, 2 distractors
    assert len(chronological.steps[-1].probes) == 8
    assert all(p.dimension == "temporal" for p in late.steps[-1].probes)
    assert len({p.expected_any[0] for p in late.steps[-1].probes}) == 3
    assert "expected_any" not in "\n".join(s.source_text() for s in late.steps)
    assert all("replaces" not in s.text and "until" not in s.text for s in late.steps)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depths": (2,)},
        {"depths": (41,)},
        {"seeds": ()},
        {"orders": ("wrong",)},
        {"budgets": (64,)},
        {"depths": (4, 4)},
    ],
)
def test_invalid_design_factors_are_rejected(kwargs):
    with pytest.raises(ValueError):
        temporal_stress_suite(**kwargs)


def test_context_budget_is_enforced_on_complete_records_and_metadata_is_not_duplicated():
    retrieval = Retrieval(
        backend="lightrag-append",
        query="Owner?",
        native_temporal_filter=False,
        evidence=[
            Evidence(id="oversized", text="oversized " * 5000, kind="chunk"),
            Evidence(
                id="fact",
                text="Bob owns Atlas.",
                kind="relation",
                metadata={
                    "description": "Bob owns Atlas.",
                    "private_internal": "IGNORE THIS",
                    "src_id": "Bob",
                    "tgt_id": "Atlas",
                },
            ),
        ],
    )
    records = pack_evidence(retrieval, 128)
    encoded = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    assert len(tiktoken.get_encoding("cl100k_base").encode(encoded)) <= 128
    assert records[0]["id"] == "fact"
    assert records[0]["source_entity"] == "Bob"
    assert "IGNORE THIS" not in encoded
    assert "metadata" not in encoded
    assert retrieval.answer_context.omitted_evidence_ids == ["oversized"]
    assert retrieval.answer_context.evidence_ids == ["fact"]


def test_diagnostics_separate_retrieval_budget_and_generation_failures():
    probe = Probe(id="p", query="Owner?", expected_any=["Bob"], forbidden_any=["Alice"])
    retrieval = Retrieval(
        backend="graphiti",
        query="Owner?",
        native_temporal_filter=True,
        evidence=[Evidence(id="old", text="Alice owns Atlas.", kind="fact")],
    )
    pack_evidence(retrieval, 128)
    assert (
        score_probe(probe, retrieval, Answer(answer="Alice"))["outcome"]
        == "retrieval_missing_expected_alias"
    )
    retrieval.evidence.append(Evidence(id="new", text="Bob owns Atlas. " * 5000, kind="fact"))
    pack_evidence(retrieval, 128)
    assert (
        score_probe(probe, retrieval, Answer(answer="Alice"))["outcome"]
        == "context_budget_missing_expected_alias"
    )
    retrieval.evidence[-1].text = "Bob owns Atlas."
    pack_evidence(retrieval, 128)
    assert (
        score_probe(probe, retrieval, Answer(answer="Alice"))["outcome"]
        == "answer_used_obsolete_alias"
    )
    assert (
        score_probe(probe, retrieval, Answer(answer="Unclear"))["outcome"]
        == "answer_incorrect_despite_expected_alias"
    )
    assert score_probe(probe, retrieval, Answer(answer="Bob"))["outcome"] == "correct"
