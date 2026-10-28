import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .api import builtin_suite_path, select_scenarios
from .config import Settings
from .harness import load_suite, run_suite
from .models import BACKENDS, IngestRequest, RetrieveRequest, utc_now
from .provider import Gateway, safe_error, usage_summary
from .service import service_context

app = typer.Typer(
    no_args_is_help=True, help="Temporal RAG Lab: compare retrieval systems when knowledge changes."
)
console = Console()


def setup_logging():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


def display_summary(report):
    table = Table(title=f"{report['run_id']} · {report['status']}")
    for label in ["Backend", "Coverage", "Current", "Historical", "Mean update", "API cost"]:
        table.add_column(label)

    def percent(value):
        return f"{value:.0%}" if value is not None else "—"

    for backend, summary in report["summary"].items():
        cost = summary["usage"]["reported_cost_usd"]
        latency = summary["mean_update_ms"]
        table.add_row(
            backend,
            f"{summary['probes_completed']}/{summary['probes_planned']}",
            percent(summary["current_accuracy"]),
            percent(summary["historical_accuracy"]),
            f"{latency / 1000:.2f}s" if latency is not None else "—",
            f"${cost:.4f}" if cost is not None else "unknown",
        )
    console.print(table)


@app.command()
def run(
    suite: Path | None = typer.Option(
        None, help="JSON suite; defaults to the bundled update suite."
    ),
    backends: str = typer.Option(",".join(BACKENDS), help="Comma-separated backend names."),
    scenario: list[str] | None = typer.Option(
        None, help="Select a scenario; repeat to select several."
    ),
    run_id: str | None = typer.Option(None),
    no_answer: bool = typer.Option(False, help="Retrieve only; do not score generated answers."),
    dry_run: bool = typer.Option(
        False, help="Validate and show the plan without API or database calls."
    ),
):
    """Replay updates and probes, then save JSON and an interactive HTML report."""
    setup_logging()
    settings = Settings()
    try:
        selected = [b.strip() for b in backends.split(",")]
        if any(b not in BACKENDS for b in selected) or len(set(selected)) != len(selected):
            raise ValueError(f"Select distinct backends from: {', '.join(BACKENDS)}")
        dataset = select_scenarios(load_suite(suite or builtin_suite_path()), scenario)
        if dry_run:
            console.print_json(
                json.dumps(
                    {
                        "suite": dataset.name,
                        "backends": selected,
                        "scenarios": [s.id for s in dataset.scenarios],
                        "updates_per_backend": sum(len(s.steps) for s in dataset.scenarios),
                        "probes_per_backend": sum(
                            len(s.probes) for c in dataset.scenarios for s in c.steps
                        ),
                        "models": settings.public(),
                    }
                )
            )
            return
        settings.require_key()
        name = run_id or utc_now().strftime("run-%Y%m%d-%H%M%S-%f")
        report = asyncio.run(
            run_suite(
                dataset,
                settings,
                selected,
                name,
                with_answers=not no_answer,
                progress=console.print,
            )
        )
        display_summary(report)
        console.print(f"Report: {(settings.runs_dir / name / 'report.html').resolve()}")
        if report["status"] != "complete":
            raise typer.Exit(1)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        console.print(f"[red]{safe_error(exc, settings)}[/red]")
        raise typer.Exit(1) from exc


@app.command("generate-stress")
def generate_stress(
    output: Path = typer.Option(Path("examples/temporal-stress.json")),
    depths: str = typer.Option("4,12,32"),
    seeds: str = typer.Option("7,23,41"),
    orders: str = typer.Option("chronological,late"),
    budgets: str = typer.Option("512,2048"),
):
    """Generate a deterministic factorial workload, without model or database calls."""
    from .stress import temporal_stress_suite

    try:
        suite = temporal_stress_suite(
            depths=tuple(int(x) for x in depths.split(",")),
            seeds=tuple(int(x) for x in seeds.split(",")),
            orders=tuple(orders.split(",")),
            budgets=tuple(int(x) for x in budgets.split(",")),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        # An explicit generator output should never silently overwrite an experiment.
        with output.open("x") as stream:
            stream.write(suite.model_dump_json(indent=2))
        console.print(f"Created {output.resolve()} · {len(suite.scenarios)} scenarios")
        console.print(
            f"Writes/backend: {sum(len(s.steps) for s in suite.scenarios)} · "
            f"probes/backend: {sum(len(s.probes) for c in suite.scenarios for s in c.steps)}"
        )
    except (ValueError, FileExistsError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("export-report")
def export_public_report(
    source: Path = typer.Argument(..., exists=True), output: Path = typer.Option(...)
):
    """Export a reviewed report as a read-only snapshot, omitting request logs and namespaces.

    Document text is preserved; review it before sharing. The output directory must be new.
    """
    from .publication import export_report

    try:
        target = export_report(source, output)
        console.print(f"Published snapshot: {target.resolve()}")
    except (ValueError, FileExistsError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def doctor(
    live: bool = typer.Option(False, help="Also make a tiny paid chat and embedding request."),
):
    """Check configuration, database connectivity, and optionally OpenRouter access."""
    setup_logging()
    settings = Settings()

    async def check():
        from neo4j import AsyncGraphDatabase

        checks = {
            "openrouter_key_present": bool(settings.openrouter_api_key.get_secret_value()),
            "models": settings.public(),
        }
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
            connection_timeout=5,
        )
        try:
            await driver.verify_connectivity()
            checks["neo4j"] = "reachable"
        except Exception as exc:
            checks["neo4j"] = safe_error(exc, settings)
        finally:
            await driver.close()
        if live:
            gateway = Gateway(settings)
            try:
                with gateway.scope("system", "doctor"):
                    await gateway.complete(
                        [{"role": "user", "content": "Reply with OK."}], max_tokens=8
                    )
                    vectors = await gateway.embed(["health check"])
                checks["openrouter"] = "chat and embeddings reachable"
                checks["embedding_shape"] = list(vectors.shape)
                checks["usage"] = usage_summary(gateway.events)
            finally:
                await gateway.close()
        return checks

    try:
        result = asyncio.run(check())
        console.print_json(json.dumps(result))
        if result["neo4j"] != "reachable" or not result["openrouter_key_present"]:
            raise typer.Exit(1)
    except Exception as exc:
        if isinstance(exc, typer.Exit):
            raise
        console.print(f"[red]{safe_error(exc, settings)}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def ingest(file: Path = typer.Argument(..., exists=True)):
    """Apply a JSON IngestRequest to a persistent namespace."""
    setup_logging()
    request = IngestRequest.model_validate_json(file.read_text())

    async def work():
        async with service_context() as service:
            return await service.ingest(request)

    console.print_json(json.dumps(asyncio.run(work())))


@app.command()
def query(
    query: str,
    backend: str = "graphiti",
    namespace: str = "agent",
    as_of: str | None = None,
    top_k: int = 10,
    no_answer: bool = False,
    context_tokens: int = 4096,
):
    """Retrieve evidence and a grounded answer from an ingested namespace."""
    setup_logging()
    request = RetrieveRequest(
        query=query,
        backend=backend,
        namespace=namespace,
        as_of=as_of,
        top_k=top_k,
        answer=not no_answer,
        context_tokens=context_tokens,
    )

    async def work():
        async with service_context() as service:
            return await service.retrieve(request)

    console.print_json(json.dumps(asyncio.run(work())))


@app.command()
def serve(port: int = 8765):
    """Start the local comparison dashboard and retrieval API."""
    import uvicorn

    setup_logging()
    uvicorn.run("rag_comparison.api:create_app", factory=True, host="127.0.0.1", port=port)


@app.command()
def mcp():
    """Expose retrieval, comparison and ingestion tools over MCP stdio."""
    setup_logging()
    from .mcp_server import run_mcp

    run_mcp()


if __name__ == "__main__":
    app()
