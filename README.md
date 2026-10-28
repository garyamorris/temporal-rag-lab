# Temporal RAG Lab

[![CI](https://github.com/garyamorris/temporal-rag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/garyamorris/temporal-rag-lab/actions/workflows/ci.yml)

**Compare retrieval systems when the facts change.** An update-focused harness for
[Graphiti](https://github.com/getzep/graphiti) and [LightRAG](https://github.com/HKUDS/LightRAG),
with dated event streams, controlled answer-context budgets, inspectable evidence, a local
dashboard, and HTTP/MCP tools for agents. All model and embedding calls use OpenRouter.

This is an experimental comparison tool. Its synthetic examples help diagnose behavior;
they do not establish a universal ranking or a guarantee of temporal correctness.

## Quick start

Requires [uv](https://docs.astral.sh/uv/), Python 3.12 or 3.13, and macOS or Linux
(Windows: WSL2). Graphiti also needs Docker for the included Neo4j service.

```sh
git clone https://github.com/garyamorris/temporal-rag-lab.git
cd temporal-rag-lab
uv sync --locked

# Validate the bundled smoke suite without credentials, API calls or a database.
uv run raglab run --dry-run

# Put OPENROUTER_API_KEY in your shell or a local, gitignored .env file.
# .env.example lists the supported settings.
docker compose up -d --wait
uv run raglab doctor --live
uv run raglab serve
```

Open [the dashboard](http://127.0.0.1:8765) or [the HTTP API reference](http://127.0.0.1:8765/docs).
The dashboard selects workloads, runs comparisons, filters evidence and queries indexed datasets.
`doctor --live` and benchmark runs incur OpenRouter API charges; dry runs and unit tests do not.

Defaults are `openai/gpt-4.1-mini` and `openai/text-embedding-3-small` (1536 dimensions),
both served through OpenRouter. No direct OpenAI API key is required. See [.env.example](.env.example).
HTTP and Neo4j bind to localhost. The local service is intended for development, not public hosting.

## Comparison tracks

| Track | Update policy | Historical queries |
| --- | --- | --- |
| `graphiti` | Add a dated episode; let Graphiti extract, reconcile and invalidate facts | Native fact filtering on `valid_at` and `invalid_at` |
| `lightrag-append` | Insert each revision as a distinct document | Retain versions and reason over dated text; `as_of` is a query hint |
| `lightrag-replace` | Delete tracked earlier revisions of a logical document, then insert the new one | Superseded sources are removed; surviving text may still describe history |

Append is the main LightRAG comparator. Arrival-order replacement is an explicit policy
ablation; an application can implement other effective-date or snapshot policies.

Graphiti uses its hybrid fact-edge search. LightRAG uses `mode="mix"` with chunks, entities
and relations. These retrieval units differ. Before answer generation, complete normalized
evidence records are packed under a shared token cap, with every selected and omitted id recorded.
The cap is 4096 evidence tokens by default; stress probes override it. Storage metadata is
excluded while semantic entity names and relation endpoints are retained.

## Workloads

- **Smoke:** owner changes, launch-date corrections, late arrivals and shared-source deletion.
  This mixes temporal, retrieval and lifecycle checks; the report scores them separately.
- **Temporal pilot:** 12 successive owners, delayed middle revisions, duplicate audit sources,
  a stable reviewer and a similarly named project. Queries cover current state, early and
  middle history, and an exact transition boundary, at 512/2048 evidence-token caps.
- **Temporal stress:** 18 paired depth/seed/order scenarios, varying 4/12/32 owners,
  three dataset seeds, chronological/late arrivals and two evidence budgets.

```sh
# One small scenario.
uv run raglab run --scenario owner-change

# Bounded temporal pilot against the two primary tracks.
uv run raglab run --suite examples/temporal-pilot.json --backends graphiti,lightrag-append

# Preview the full sweep before spending time or credits.
# It contains 396 writes and 144 probes PER backend.
uv run raglab run --suite examples/temporal-stress.json --dry-run

# Generate a new predeclared design; output must not already exist.
uv run raglab generate-stress --output examples/my-stress.json \
  --depths 4,12,32 --seeds 7,23,41 --orders chronological,late --budgets 512,2048

# Retrieve without generating or scoring answers.
uv run raglab run --scenario late-arrival --no-answer
```

Each run saves `suite.json`, `report.json` and a standalone `report.html` in `runs/<run-id>/`.
Run ids are never overwritten. Every run/scenario/backend has an isolated namespace, so a
fresh run cannot reuse an earlier index or extraction cache. Reports are checkpointed;
errors produce partial coverage and a nonzero CLI exit code.

The full sweep can take substantial time. Generator seeds control dataset names and arrival
order, not model sampling. Paired date/budget probes are correlated. See the
[temporal benchmark design and oracle](docs/temporal-benchmark.md).

## Included findings and evidence

The original smoke test was a **7/7 temporal tie** between Graphiti and LightRAG append.
Its broader headline score also included attribute-retrieval and lifecycle failures.

A harder, single-seed, late-arrival pilot produced:

| Evidence budget | Graphiti | LightRAG append | Condition |
| --- | ---: | ---: | --- |
| 512 tokens | 4/4 | 0/4 | Predeclared pilot |
| 2,048 tokens | 4/4 | 1/4 | Predeclared pilot |
| 8,192 tokens | 4/4 | 2/4 | Posthoc replay of the same saved retrieval |

At smaller caps, most LightRAG misses involved expected values omitted from the answer
context. At 8192 tokens, every expected alias was available, but two answers still selected
other owners. This is consistent with an advantage from narrowing competing dated facts;
it is not a guarantee that Graphiti's extraction or invalidation is always correct.

Read the [pilot findings and limitations](docs/results-temporal-pilot.md) and
[corrected smoke-test interpretation](docs/results-2026-10-28.md).

Reviewed synthetic snapshots include raw evidence, answers, scores, inputs, versions and
aggregate usage. Runtime namespaces and per-request logs have been removed. They contain
no live index and are not used to fabricate fresh benchmark outcomes.

| Snapshot | Data | Standalone viewer |
| --- | --- | --- |
| Smoke | [JSON](docs/benchmarks/comparison-20261028/report.json) | [HTML](docs/benchmarks/comparison-20261028/report.html) |
| Temporal pilot | [JSON](docs/benchmarks/temporal-stress-pilot-20261028/report.json) | [HTML](docs/benchmarks/temporal-stress-pilot-20261028/report.html) |
| Context diagnostic | [JSON](docs/benchmarks/temporal-context-control-8192/report.json) | [HTML](docs/benchmarks/temporal-context-control-8192/report.html) |

GitHub displays HTML as source. View the saved reports locally without a key or database:

```sh
python3 -m http.server 8766 --directory docs/benchmarks
```

Then open [the saved pilot viewer](http://127.0.0.1:8766/temporal-stress-pilot-20261028/report.html).

## Scoring and limits

Answer scoring checks literal, case-insensitive whole-phrase aliases: at least one expected
alias and no forbidden alias. Gold labels are excluded from ingestion and answer prompts.
The shared answer generator returns citation ids, which must refer to evidence actually shown.

Reports separate retrieval/alias coverage, context-budget omissions, obsolete answer values,
and wrong answers despite available expected aliases. Alias presence is not proof of entailment.
Other answer values in context may be legitimate history, future facts or different roles;
that metric does not measure asserted stale facts.

The adapters differ in storage, extraction and retrieval units. Context caps do not equalize
ingestion or retrieval compute. Small synthetic samples, one extraction model and correlated
probes cannot establish statistical significance or a general product ranking. Model outputs
may vary between runs. See [design and interpretation](docs/design.md).

## Agent retrieval

Populate a persistent namespace, then query it:

```sh
uv run raglab ingest examples/ingest.json
uv run raglab query 'Who owns Project Atlas?' --namespace agent
uv run raglab query 'Who owned Project Atlas on January 15, 2025?' \
  --namespace agent --as-of 2025-01-15T00:00:00Z
```

New content needs a new event id and the same logical document id. Identical event ids are
idempotent across restarts; changed content under an existing id is rejected. Ingestion
selects one backend; repeat it for each track to populate a comparison.

```sh
curl -s http://127.0.0.1:8765/api/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"backend":"graphiti","namespace":"agent","query":"Who owns Project Atlas?","context_tokens":2048,"answer":true}'
```

Responses include structured evidence, provenance, available validity intervals, retrieval
latency, selected context ids/tokens, and an optional grounded answer. `as_of` uses valid time;
the harness does not reconstruct old database snapshots or earlier knowledge-time beliefs.

Start an MCP stdio server:

```sh
uv run raglab mcp
```

Tools: `retrieve`, `compare_retrieval`, `ingest_update`. Logs go to stderr. Adapt
[examples/mcp.json](examples/mcp.json) to your absolute checkout path. The process must inherit
`OPENROUTER_API_KEY` or read the project's `.env`. On macOS, an absolute path to the uv binary
may also be needed in desktop MCP clients.

Each process holds an exclusive lease on an opened backend namespace. Close one process
before opening that same namespace in another; different namespaces can run independently.
The legacy `ragbench` command remains available as an alias for `raglab`.

## Exporting reviewed results

```sh
uv run raglab export-report runs/YOUR-RUN/report.json --output exported/YOUR-RUN
```

The output directory must be new. This removes the request ledger and runtime namespace
identifiers, preserving answers, evidence, scores and aggregate usage. It does **not** scrub
private document text; review the source data before publication. Live runs and indexes
stay in gitignored `runs/` and `.ragbench/` directories.

## Development

```sh
uv sync --locked
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
uv build
```

Unit tests need no API key or database. tiktoken can download its public encoding on first
use. CI runs tests, lint, workload validation and package builds on Python 3.12 and 3.13.

Dependencies are locked in `uv.lock`: Graphiti 0.31.2, LightRAG 1.5.7, and Neo4j 5.26.0 in
Compose. LightRAG uses local JSON, NanoVectorDB and NetworkX stores. Graphiti uses
`json_object` structured-output mode for the tested OpenRouter routes, and its product
analytics are disabled by default. The internal Python module and Compose project retain
their original `rag_comparison` / `rag-comparison` names for local compatibility.

`docker compose down` stops Neo4j while preserving its volume. Stop the dashboard with Ctrl-C.
The harness is [MIT licensed](LICENSE); dependencies retain their own licences.

## References

- [Graphiti model, operations and provider requirements](https://github.com/getzep/graphiti)
- [Graphiti 0.31.2 implementation](https://github.com/getzep/graphiti/tree/v0.31.2/graphiti_core)
- [LightRAG overview and operations](https://github.com/HKUDS/LightRAG)
- [LightRAG 1.5.7 package](https://pypi.org/project/lightrag-hku/1.5.7/)
- [OpenRouter embeddings](https://openrouter.ai/docs/api_reference/embeddings)
