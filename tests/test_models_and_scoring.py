from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rag_comparison.api import builtin_suite_path, select_scenarios
from rag_comparison.evaluation import mentions, score_probe
from rag_comparison.harness import load_suite
from rag_comparison.models import (
    Answer,
    Evidence,
    Probe,
    Retrieval,
    RetrieveRequest,
    Update,
    safe_name,
)


def test_suite_has_current_history_late_arrival_and_deletion():
    suite = load_suite(builtin_suite_path())
    assert len(suite.scenarios) == 4
    assert sum(len(step.probes) for scenario in suite.scenarios for step in scenario.steps) == 12
    late = select_scenarios(suite, ["late-arrival"]).scenarios[0]
    assert late.steps[-1].effective_at < late.steps[-2].effective_at
    assert any(
        step.operation == "delete" for scenario in suite.scenarios for step in scenario.steps
    )
    with pytest.raises(ValueError, match="Unknown scenarios"):
        select_scenarios(suite, ["typo"])


@pytest.mark.parametrize("value", ["../escape", "a/b", "", "a.b", "x" * 81])
def test_names_cannot_escape_storage(value):
    with pytest.raises(ValueError):
        safe_name(value)


def test_timezone_and_empty_update_validation():
    with pytest.raises(ValidationError, match="timezone"):
        Update(id="a", document_id="a", text="x", effective_at="2025-01-01T00:00:00")
    with pytest.raises(ValidationError, match="require text"):
        Update(id="a", document_id="a", effective_at=datetime.now(UTC))
    with pytest.raises(ValidationError):
        RetrieveRequest(query="x", top_k=0)


def test_literal_aliases_are_whole_phrases_not_substrings_or_regex():
    assert mentions("Owner: BOB CHEN.", ["Bob Chen"])
    assert not mentions("Annabelle", ["Ann"])
    assert not mentions("12.50", ["2.5"])
    assert mentions("Use #support-help.", ["support-help"])
    assert mentions("literal a+b", ["a+b"])
    assert not mentions("aaab", ["a+b"])


def test_obsolete_evidence_is_distinct_from_answer_error():
    probe = Probe(id="p", query="Owner?", expected_any=["Bob"], forbidden_any=["Alice"])
    retrieval = Retrieval(
        backend="lightrag-append",
        query="Owner?",
        native_temporal_filter=False,
        evidence=[Evidence(id="e", text="Alice was owner; Bob is owner.", kind="chunk")],
    )
    score = score_probe(probe, retrieval, Answer(answer="Bob", evidence_ids=["e"]))
    assert score["answer_correct"] is True
    assert score["forbidden_mention_in_evidence"] is True
    assert score_probe(probe, retrieval, Answer(answer="Alice"))["answer_correct"] is False
    assert score_probe(probe, retrieval, None)["answer_correct"] is None
