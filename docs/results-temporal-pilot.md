# Temporal stress pilot: 28 October 2026

The first smoke suite did not discriminate temporal accuracy: Graphiti and LightRAG append
both scored 7/7 on its temporal subset. Its overall score also included attribute retrieval
and deletion/answer-grounding checks. That headline was too broad for judging temporal handling.

The stronger pilot used 12 consecutive ownership changes, with the latest revision arriving
before shuffled middle revisions. It included three independent audit copies, a stable reviewer
role, and a similarly named distinct project. Source records gave new assignments and effective
dates, without supplying explicit end dates or naming the person superseded. Ground truth came
from a deterministic event log. This was dataset seed 7, not a model-sampling seed.

Both native adapters used the same OpenRouter extraction/answer model and embedder. Answer
inputs used complete normalized evidence records, excluding duplicate storage metadata,
packed in native retrieval order under a common evidence-token cap.

| Evidence budget | Graphiti | LightRAG append | Measurement |
| --- | ---: | ---: | --- |
| 512 tokens | 4/4 | 0/4 | Predeclared pilot condition |
| 2,048 tokens | 4/4 | 1/4 | Predeclared pilot condition |
| 8,192 tokens | 4/4 | 2/4 | Posthoc replay of the same saved retrieval; no new ingestion/search |

The predeclared pilot completed 34 writes and 16 probes with zero execution errors. Its
reported OpenRouter cost was $0.087451. The context diagnostic added eight answer calls,
zero execution errors, and $0.007976 reported cost. These amounts exclude other development
and interactive playground requests.

## What the difference means

At 512 tokens, all four LightRAG failures had the expected alias in the raw retrieval but
outside the evidence packed for the answer model. At 2,048 tokens, two failures still had
that budget/coverage problem; one answer chose an obsolete alias despite the expected alias
being available; one was correct.

The posthoc 8,192-token diagnostic checked whether the small evidence caps alone explained
the outcome. Every expected alias was then present in both backends' packed evidence.
LightRAG's current-state and middle-history answers were correct. Its early-history and exact
change-boundary answers selected other owners. Graphiti answered all four correctly.

This is consistent with the expected temporal advantage: when a history contains many
competing versions, validity filtering can deliver a smaller set of date-relevant facts,
reducing both context pressure and the answer model's disambiguation work. More space helped
LightRAG, but did not completely remove the gap in this case.

The result does not establish that Graphiti always extracts correct entities, sets correct
intervals, or ranks all relevant facts. The original small run contained overlapping
intervals despite correct answers. More complexity can expose those weaknesses too.

## Limits

This is one synthetic ownership scenario, one dataset seed, one arrival order, one extraction
model and native retrieval policy per backend. The four date questions, repeated budgets,
and posthoc replay are correlated; they are not independent statistical replications.
The 8,192-token test was chosen after seeing the pilot, and is explicitly diagnostic.

The context-exposure metric is also only a literal alias count. In this workload the unchanged reviewer is a forbidden owner answer, but a legitimate reviewer fact. Its presence can make that exposure metric 100% even when ownership intervals are filtered correctly; it must not be interpreted as asserted-stale-fact precision.

Alias presence is not proof of entailment. The diagnostic tags help identify where to inspect,
but do not definitively assign component blame. Native retrieval units differ, and the LightRAG
adapter places source chunks before relations and entities when packing. A different query mode,
reranker, context packing policy or application-level temporal filter could change the outcome.

The full prebuilt design has 18 depth/seed/order scenarios and is ready to run; it has **not**
been executed as part of this pilot. Neither these results nor the original smoke score should
be presented as a universal ranking.

- [Published pilot snapshot](benchmarks/temporal-stress-pilot-20261028/report.html)
- [Published posthoc context diagnostic](benchmarks/temporal-context-control-8192/report.html)
- [Design and oracle](temporal-benchmark.md)

Reviewed public data is in [the pilot JSON](benchmarks/temporal-stress-pilot-20261028/report.json) and [the context diagnostic JSON](benchmarks/temporal-context-control-8192/report.json). The diagnostic records its original source-report SHA-256, source probe ids and measurement mode. Runtime namespaces and the per-request ledger were omitted from these public snapshots; answers, evidence, scores and aggregate usage are preserved. The original local run files remain unchanged. See the root README for viewing HTML snapshots locally.
