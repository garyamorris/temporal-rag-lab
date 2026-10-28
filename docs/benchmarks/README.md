# Reviewed synthetic benchmark snapshots

These files preserve inputs, answers, retrieved evidence, scores, versions and aggregate
usage from the initial experiments. They are read-only examples, not live retrieval indexes.

Per-request usage ledgers and local namespace identifiers are omitted. Each snapshot's
publication metadata records the SHA-256 of its original, unmodified local report.
The context-control report's source digest refers to that original pilot report, rather
than the redacted public copy.

| Snapshot | Purpose |
| --- | --- |
| comparison-20261028 | Mixed smoke checks; temporal subset was a 7/7 tie between the primary tracks |
| temporal-stress-pilot-20261028 | One 12-owner, late-arrival scenario at 512/2048 evidence tokens |
| temporal-context-control-8192 | Posthoc answer replay from saved retrieval at 8192 tokens |

The full multi-seed stress design is supplied as an input fixture. These snapshots do not
represent a completed multi-seed study or establish a universal product ranking.

To view the standalone HTML reports without credentials:

```sh
python3 -m http.server 8766 --directory docs/benchmarks
```

Open http://127.0.0.1:8766/temporal-stress-pilot-20261028/report.html.
JSON and suite files are beside each report. The root README and results documents explain
how to run fresh experiments and interpret the diagnostic metrics.
