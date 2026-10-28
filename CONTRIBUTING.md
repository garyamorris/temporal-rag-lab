# Contributing

Temporal RAG Lab compares retrieval behavior as facts change. Contributions should improve
reproducibility and diagnosis without designing a workload to guarantee a particular winner.

## Local checks

Python 3.12 or 3.13 and uv are required. Unit tests use mocked providers and storage; they
need neither a model key nor a running graph database. tiktoken may download its public
encoding data on the first test run.

```sh
uv sync --locked
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
uv build
```

## Benchmark changes

- Save the exact suite and generator parameters with a run.
- Keep source text, model settings, questions and evidence budgets shared across backends.
- Keep gold answers outside ingestion, retrieval and answer prompts.
- Separate temporal accuracy, retrieval coverage, context-budget effects and generation errors.
- Label posthoc diagnostics and distinguish them from predeclared conditions.
- Report coverage, failures, seeds and limitations alongside scores; paired probes are correlated.
- Preserve original result files when changing evaluation methods.

Runtime indexes, .env files and run directories are gitignored. Only export synthetic or
otherwise reviewed data for publication. The report exporter removes per-request logs and
runtime namespace identifiers; it does not remove private text from ingested documents.

Graphiti, LightRAG and all other dependencies retain their own licences. This repository
contains the comparison harness and synthetic fixtures, not vendored copies of those projects.
