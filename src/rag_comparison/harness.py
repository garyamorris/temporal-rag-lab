import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from pathlib import Path

from .adapters import make_adapter
from .config import Settings
from .evaluation import score_probe, summarize
from .models import BackendName, Suite, safe_name, utc_now
from .provider import Gateway, safe_error


def load_suite(path: Path) -> Suite:
    return Suite.model_validate_json(path.read_text())


def save_report(report: dict, output: Path, gateway: Gateway):
    from .report import write_html

    report["requests"] = gateway.events
    report["summary"] = summarize(report)
    target = output / "report.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(target)
    write_html(report, output / "report.html")


async def run_suite(
    suite: Suite,
    settings: Settings,
    backends: list[BackendName],
    run_id: str,
    *,
    with_answers: bool = True,
    progress: Callable[[str], None] = print,
    adapter_factory=make_adapter,
    gateway_factory=Gateway,
) -> dict:
    safe_name(run_id)
    output = settings.runs_dir / run_id
    # Fresh namespaces prevent cached/warm runs being presented as cold measurements.
    output.mkdir(parents=True, exist_ok=False)
    if not backends or len(backends) != len(set(backends)):
        raise ValueError("Select at least one distinct backend.")
    gateway = gateway_factory(settings)
    report = {
        "schema_version": 2,
        "run_id": run_id,
        "suite": suite.model_dump(mode="json"),
        "started_at": utc_now().isoformat(),
        "status": "running",
        "config": settings.public(),
        "backends": backends,
        "answers_enabled": with_answers,
        "versions": {
            p: importlib.metadata.version(p) for p in ["lightrag-hku", "graphiti-core", "openai"]
        },
        "planned_probes_per_backend": sum(len(s.probes) for c in suite.scenarios for s in c.steps),
        "updates": [],
        "probes": [],
        "errors": [],
        "requests": [],
        "summary": {},
        "methodology": {
            "lightrag_mode": "mix",
            "top_k": 10,
            "answer_context": "Complete normalized evidence records in retrieval order; metadata excluded; shared cl100k_base token cap.",
            "default_answer_context_tokens": 4096,
            "answer_model_shared": True,
            "graphiti_search": "hybrid RRF with explicit validity interval filter",
            "graphiti_structured_output": "json_object (schema in prompt for OpenRouter compatibility)",
            "latency": "Initial inserts, subsequent updates, and deletions are measured separately. Update includes deletion and reconstruction for the replace policy.",
            "scoring": "Literal case-insensitive whole-phrase aliases in generated answers; gold labels never sent to the answer model.",
            "limitations": [
                "Small synthetic scenarios; results depend on extraction model, prompts, backend, and retrieval settings.",
                "LightRAG returns chunks, entities and relations; Graphiti returns facts. A shared evidence token cap bounds answer context; retrieval units still differ.",
                "Historical LightRAG queries use date hints, not native interval filters.",
                "Forbidden-alias exposure can include history, future facts or different roles; it does not establish an asserted stale fact.",
                "Replace policy removes superseded documents and can lose source history.",
                "Latency includes provider/network/database time; no claim of statistical significance.",
                "API cost is the sum reported by OpenRouter; missing costs remain unknown.",
            ],
        },
    }
    (output / "suite.json").write_text(suite.model_dump_json(indent=2))
    save_report(report, output, gateway)
    try:
        for scenario in suite.scenarios:
            for backend in backends:
                namespace = (
                    "run-"
                    + hashlib.sha256(f"{output.resolve()}:{scenario.id}".encode()).hexdigest()[:24]
                )
                progress(f"{scenario.id} / {backend}: initializing")
                adapter = adapter_factory(backend, namespace, settings, gateway)
                try:
                    with gateway.scope(backend, "setup"):
                        await adapter.open()
                    for step in scenario.steps:
                        progress(f"{scenario.id} / {backend}: update {step.id}")
                        with gateway.scope(backend, f"ingest:{scenario.id}:{step.id}"):
                            result = await adapter.ingest(step)
                        report["updates"].append(
                            {
                                "backend": backend,
                                "namespace": namespace,
                                "scenario": scenario.id,
                                "step": step.id,
                                "status": "ok",
                                "result": result,
                            }
                        )
                        save_report(report, output, gateway)
                        for probe in step.probes:
                            progress(f"{scenario.id} / {backend}: probe {probe.id}")
                            row = {
                                "backend": backend,
                                "namespace": namespace,
                                "scenario": scenario.id,
                                "step": step.id,
                                "probe": probe.model_dump(mode="json"),
                            }
                            try:
                                with gateway.scope(backend, f"retrieve:{scenario.id}:{probe.id}"):
                                    retrieval = await adapter.retrieve(
                                        probe.query, as_of=probe.as_of
                                    )
                                answer = None
                                if with_answers:
                                    with gateway.scope(backend, f"answer:{scenario.id}:{probe.id}"):
                                        answer = await gateway.answer(
                                            retrieval,
                                            **(
                                                {"context_tokens": probe.context_tokens}
                                                if probe.context_tokens is not None
                                                else {}
                                            ),
                                        )
                                row.update(
                                    {
                                        "status": "ok",
                                        "retrieval": retrieval.model_dump(mode="json"),
                                        "answer": answer.model_dump() if answer else None,
                                        "scores": score_probe(probe, retrieval, answer),
                                    }
                                )
                            except Exception as exc:
                                message = safe_error(exc, settings)
                                row.update({"status": "error", "error": message})
                                report["errors"].append(
                                    {
                                        "backend": backend,
                                        "scenario": scenario.id,
                                        "probe": probe.id,
                                        "error": message,
                                    }
                                )
                            report["probes"].append(row)
                            save_report(report, output, gateway)
                except Exception as exc:
                    message = safe_error(exc, settings)
                    progress(f"{scenario.id} / {backend}: ERROR {message}")
                    report["errors"].append(
                        {"backend": backend, "scenario": scenario.id, "error": message}
                    )
                finally:
                    try:
                        await adapter.close()
                    except Exception as exc:
                        report["errors"].append(
                            {
                                "backend": backend,
                                "scenario": scenario.id,
                                "error": safe_error(exc, settings),
                            }
                        )
                    save_report(report, output, gateway)
        report["status"] = "partial" if report["errors"] else "complete"
    except BaseException:
        report["status"] = "interrupted"
        raise
    finally:
        report["finished_at"] = utc_now().isoformat()
        save_report(report, output, gateway)
        await gateway.close()
    return report
