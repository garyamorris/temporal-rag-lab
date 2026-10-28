# Temporal accuracy, measured separately

The initial suite was too small and explicit to discriminate temporal handling. Both
Graphiti and LightRAG append answered all seven ownership/time probes correctly. Mixing
in a launch-date attribute retrieval failure made the overall score answer a broader
question than the user intended. Those original raw results are preserved.

Graphiti's temporal model supplies a mechanism for storing and filtering validity
intervals. It does not guarantee perfect entity extraction, contradiction resolution,
invalidation dates, retrieval ranking, or generated answers. The initial run already
contained overlapping owner intervals despite correct final answers.

## Controlled design

The bundled full design varies:

| Factor | Values |
| --- | --- |
| Successive owners | 4, 12, 32 |
| Name permutation / arrival-shuffle seeds | 7, 23, 41 |
| Arrival order | Chronological; latest arrives before shuffled middle revisions |
| Evidence token budget | 512, 2048 |
| Queries after all revisions arrive | Current state at the scenario clock; early history; middle history; exact change boundary |

Each scenario also includes a distinct project with a similar name, an unchanged security
reviewer, and independent audit documents confirming some ownership facts. Assignments
state the new sole owner and effective date, without naming the old owner or supplying
explicit end dates. The engine or answer model must reconcile the history.

Order variants contain identical source records and gold questions, differing only in
arrival order. Source copies preserve the original effective time. Every backend receives
those same dated records; the harness supplies neither gold edges nor a Graphiti-specific
ontology. The generator is deterministic and exported suites are saved with every run.

The oracle is a plain event-log calculation: for a query instant, take the ownership event
with the latest effective timestamp less than or equal to that instant. It does not use an
LLM. Gold values never enter ingestion or answer prompts. All temporal queries use an
explicit scenario clock; probes marked current remain current-state probes in summaries,
even though that clock is sent through the as_of API.

The full design has 18 scenarios, 396 writes and 144 probes **per backend**. The bounded
pilot uses one 12-owner late-arrival scenario: 17 writes and eight probes per backend.
The full design has several seeds; the pilot is not a statistical replication.

## Evidence control and diagnostics

Each backend retains its native retrieval operation. Before shared answer generation,
evidence is normalized to ids, text, source ids, available validity dates and semantic
entity/relation names. Storage metadata and duplicate text are excluded. Complete records
are greedily packed in retrieval order under a shared cl100k_base token cap. Oversized
records are skipped, and every omitted id is recorded. There is no oracle reranking or
truncation of individual facts. The evidence JSON uses the same compact serialization
for packing and transmission. Question and system-prompt tokens are outside this cap.

This creates a controlled answer-evidence budget while preserving the real architectural
difference between compressed facts and retrieved source chunks. It is not equal storage,
equal ingestion compute, equal retrieval compute, or identical retrieval units. Limited
budgets can reveal useful compression and ranking differences; large-budget conditions
help distinguish those from temporal reasoning itself.

Answer accuracy still uses inspectable literal aliases. Additional diagnostic outcomes:

- Expected alias absent from retrieved evidence: potential retrieval/extraction coverage miss.
- Expected alias retrieved but absent from the packed evidence: context-budget coverage miss.
- Obsolete alias used in the answer: stale-value answer.
- Expected alias available, but answer still wrong: potential generation/disambiguation failure.

Forbidden-alias exposure also includes distractors from other roles and can be high even when date filtering is correct. It is not a semantic stale-fact precision metric.

Alias presence is not proof that a fact entails the answer. An alias can appear in a
negative, unrelated or obsolete statement. Diagnostics therefore suggest where to inspect,
not definitive component blame. The report includes raw evidence and the exact ids shown.

## Scope and interpretation

This tests relational ownership churn, not a complete business workload. It does not yet
cover source-authority ranking, event-time versus knowledge-time reconstruction,
retroactive corrections to past truth, future commitments, or multi-hop temporal joins.
Those need separate oracles and should not be claimed from this dataset.

Paired date/budget probes are correlated. Report counts and multiple seeds; do not treat
all probe answers as independent trials. Predeclare conditions and compare both engines
across them, rather than choosing a complexity level after observing which engine wins.
Increasing complexity can expose Graphiti's intended advantages, but can also expose its
extraction or invalidation weaknesses. A tie or a loss remains a valid experimental result.

A larger representative corpus, repeated runs and a more capable shared extraction-model
ablation would be useful next steps. Published Zep or GraphRAG comparisons should not be
assumed to establish a universal Graphiti-versus-LightRAG ranking.

Primary references: [Graphiti's temporal model and provider requirements](https://github.com/getzep/graphiti),
[Graphiti 0.31.2 search filters](https://github.com/getzep/graphiti/blob/v0.31.2/graphiti_core/search/search_filters.py).
