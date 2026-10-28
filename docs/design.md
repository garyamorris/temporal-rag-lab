# Design and interpretation

## Shared interface

The adapters expose `ingest(Update)` and `retrieve(query, top_k, as_of)`. `RetrievalService` pools them for the HTTP and MCP entrypoints. CLI operations use the same service. Model traffic goes through a single OpenRouter gateway with phase and backend attribution carried through async context variables. HTTP response hooks record token usage, model ids, request ids, reported cost and latency; request headers and prompt bodies are not recorded.

The harness runs each backend sequentially for each scenario. Every scenario gets an isolated namespace. A run records its exact suite, model configuration and installed library versions. A backend failure does not invent scores for skipped probes. Coverage is displayed next to scores. Runs do not use precomputed outcomes.

The local manifest maps logical document ids to source ids and event ids to content hashes. It rejects event id collisions. An advisory file lease prevents two processes writing the same namespace simultaneously, while an async lock serializes calls within a process. These are local harness protections, not distributed transactions. Database writes and the local manifest are not one atomic commit; a process crash during ingestion can require a fresh namespace. In particular, deletion and reinsertion are separate LightRAG operations. A partial replacement is surfaced as an error.

## Updates

Graphiti receives the same dated text as LightRAG, with `reference_time=effective_at`. Each change becomes a new episode. Graphiti's own extraction, deduplication and contradiction logic controls fact reconciliation; the harness does not rewrite its facts or supply gold edges. Episode deletion uses its public `remove_episode` operation. Shared-source behavior is tested rather than assumed.

LightRAG append inserts revisions under distinct physical document ids. Replace deletes the ids tracked for the logical document, checks the deletion result, and inserts the new version. Its public deletion operation can rebuild entities and relations supported by remaining documents. Ingestion success is checked in document status because its pipeline can record an extraction failure without raising it to the caller.

Replace means latest **arrival**, deliberately. A production system could keep an external effective-date catalog, select snapshots, or impose event-order rules before calling LightRAG. Those are additional application policies and would make useful extra comparison tracks. This harness leaves that work visible in the late-arrival experiment.

## Temporal retrieval

Graphiti search uses hybrid RRF with an explicit half-open interval:

    (valid_at <= as_of OR valid_at IS NULL)
    AND (invalid_at > as_of OR invalid_at IS NULL)

For a latest-state query, `as_of` defaults to the current UTC instant. Undated facts remain eligible. Metadata includes `created_at` and `expired_at` to inspect what the library stores. This harness does not offer knowledge-time reconstruction: Graphiti can mutate validity intervals during reconciliation, and filtering record timestamps alone would not faithfully reconstruct an earlier belief state.

LightRAG uses `aquery_data(mode="mix")`, combining its graph results with vector-retrieved source chunks. An `as_of` timestamp is added as a query hint, and the shared answer model sees it. The result explicitly says it has no native validity filter. Dated source text can nevertheless support correct historical answers, which are scored normally.

Graphiti returns up to ten facts. LightRAG uses graph and chunk limits of ten plus a 12,000-token retrieval budget. These are library-native retrieval settings and differ in retrieval units. New answer generation normalizes semantic evidence and packs complete records under a shared token cap (4096 by default, or the probe-specific cap), with omitted ids recorded. This bounds answer evidence, while storage, ingestion and retrieval compute remain different. Historical run schema 1 used uncapped raw evidence including metadata; its results are preserved.

## Answers and scoring

The same model creates answers from the structured evidence returned by each backend. It sees only the question, requested date and retrieved evidence. Gold labels never enter ingestion, retrieval or answer prompts. The answer schema contains concise answer text and citation ids. Unknown citation ids cause an error. Citation presence is recorded, but is not a proof of semantic entailment.

Scoring checks whole, case-insensitive literal aliases. A correct answer contains an expected alias and no forbidden alias. Evidence exposure is a separate metric: obsolete values can legitimately appear in historical or corrective text. Both metrics are intentionally inspectable rather than presented as comprehensive semantic evaluation.

Separate initial insertion, subsequent update and deletion latencies include all work until the adapter returns. LightRAG replacement includes deleting and rebuilding affected data. Retrieval latency excludes shared answer generation; token/cost accounting identifies answer calls separately. Costs are values reported by OpenRouter, not price estimates. Provider retries appear as additional requests. Cache behavior is left to the native libraries within a run and isolated between fresh runs.

## Extending the experiment

Useful next experiments include larger real corpora, repeated runs with confidence intervals, controlled overlapping-source churn, alternative context packing and retrieval policies, typed Graphiti ontologies, additional LightRAG query modes, explicit effective-date replacement policy, and independent blinded answer judging. These are not included in the built-in results.
