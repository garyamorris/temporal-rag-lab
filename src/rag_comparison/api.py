import asyncio
import json
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .harness import load_suite, run_suite
from .models import BACKENDS, BackendName, IngestRequest, RetrieveRequest, safe_name, utc_now
from .provider import safe_error
from .service import RetrievalService

SuiteKind = Literal["smoke", "temporal-pilot", "temporal-stress"]


class RunRequest(BaseModel):
    suite: SuiteKind = "smoke"
    backends: list[BackendName] = Field(default_factory=lambda: list(BACKENDS), min_length=1)
    scenarios: list[str] = Field(default_factory=list)
    with_answers: bool = True

    @field_validator("backends")
    @classmethod
    def distinct(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Backends must be distinct.")
        return value


def builtin_suite_path(kind: SuiteKind = "smoke") -> Path:
    filename = {
        "smoke": "updates.json",
        "temporal-pilot": "temporal-pilot.json",
        "temporal-stress": "temporal-stress.json",
    }[kind]
    return Path(str(files("rag_comparison").joinpath(f"data/{filename}")))


def select_scenarios(suite, ids):
    if ids:
        unknown = set(ids) - {s.id for s in suite.scenarios}
        if unknown:
            raise ValueError(f"Unknown scenarios: {', '.join(sorted(unknown))}")
        suite = suite.model_copy(update={"scenarios": [s for s in suite.scenarios if s.id in ids]})
    return suite


def create_app(settings: Settings | None = None, service_factory=RetrievalService) -> FastAPI:
    settings = settings or Settings()
    jobs = {}

    @asynccontextmanager
    async def lifespan(app):
        service = service_factory(settings)
        app.state.service = service
        try:
            yield
        finally:
            for task in jobs.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*jobs.values(), return_exceptions=True)
            await service.close()

    app = FastAPI(title="Temporal RAG Lab", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def home():
        return HOME

    @app.get("/health")
    async def health():
        return {"status": "ok", "provider": "openrouter", "config": settings.public()}

    @app.get("/api/suite")
    async def suite_info(kind: SuiteKind = "smoke"):
        return load_suite(builtin_suite_path(kind)).model_dump(mode="json")

    @app.post("/api/ingest")
    async def ingest(request: IngestRequest):
        try:
            return await app.state.service.ingest(request)
        except ValueError as exc:
            raise HTTPException(400, safe_error(exc, settings)) from exc
        except Exception as exc:
            raise HTTPException(502, safe_error(exc, settings)) from exc

    @app.post("/api/retrieve")
    async def retrieve(request: RetrieveRequest):
        try:
            return await app.state.service.retrieve(request)
        except ValueError as exc:
            raise HTTPException(400, safe_error(exc, settings)) from exc
        except Exception as exc:
            raise HTTPException(502, safe_error(exc, settings)) from exc

    @app.post("/api/runs", status_code=202)
    async def start_run(request: RunRequest):
        if any(not task.done() for task in jobs.values()):
            raise HTTPException(409, "A comparison is already running in this server.")
        try:
            suite = select_scenarios(
                load_suite(builtin_suite_path(request.suite)), request.scenarios
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        run_id = utc_now().strftime("run-%Y%m%d-%H%M%S-%f")

        async def work():
            return await run_suite(
                suite, settings, request.backends, run_id, with_answers=request.with_answers
            )

        jobs[run_id] = asyncio.create_task(work())
        return {
            "run_id": run_id,
            "status_url": f"/api/runs/{run_id}",
            "report_url": f"/reports/{run_id}",
        }

    @app.get("/api/runs")
    async def list_runs():
        paths = sorted(settings.runs_dir.glob("*/report.json"), reverse=True)
        result = []
        for path in paths:
            report = json.loads(path.read_text())
            result.append({k: report[k] for k in ["run_id", "status", "started_at", "summary"]})
        return sorted(result, key=lambda row: row["started_at"], reverse=True)[:100]

    @app.get("/api/datasets")
    async def datasets():
        available = []
        reports = [json.loads(path.read_text()) for path in settings.runs_dir.glob("*/report.json")]
        for report in sorted(reports, key=lambda item: item["started_at"], reverse=True):
            if report["status"] != "complete" or report.get("publication", {}).get("read_only"):
                continue
            scenarios = {item["id"]: item for item in report["suite"]["scenarios"]}
            seen = set()
            for row in report["updates"]:
                key = (row["backend"], row["namespace"])
                if row["status"] != "ok" or key in seen:
                    continue
                seen.add(key)
                probes = [
                    probe
                    for step in scenarios[row["scenario"]]["steps"]
                    for probe in step["probes"]
                ]
                example = next(
                    (
                        probe
                        for probe in reversed(probes)
                        if probe.get("category") in {"current", "changed-current"}
                    ),
                    probes[-1] if probes else {},
                )
                available.append(
                    {
                        "run_id": report["run_id"],
                        "scenario": row["scenario"],
                        "backend": row["backend"],
                        "namespace": row["namespace"],
                        "example_query": example.get("query", ""),
                        "example_as_of": example.get("as_of"),
                    }
                )
        return available

    def report_path(run_id):
        try:
            safe_name(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        path = settings.runs_dir / run_id / "report.json"
        if not path.is_file():
            task = jobs.get(run_id)
            if task and task.done() and not task.cancelled() and task.exception():
                raise HTTPException(500, safe_error(task.exception(), settings))
            raise HTTPException(404, "Run not found or initializing.")
        return path

    @app.get("/api/runs/{run_id}")
    async def read_run(run_id: str):
        return json.loads(report_path(run_id).read_text())

    @app.get("/reports/{run_id}")
    async def read_report(run_id: str):
        from .report import render_html

        return HTMLResponse(render_html(json.loads(report_path(run_id).read_text())))

    return app


HOME = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Temporal RAG Lab</title><style>
:root{font:16px/1.6 system-ui;color:#e6e9ee;background:#101319;color-scheme:dark}body{max-width:1080px;margin:auto;padding:48px 24px}h1{font-size:48px;letter-spacing:-2px;line-height:1.1}h2{font-size:22px;margin-top:0}p,small,label{color:#b1bbc8}.eyebrow{font:12px ui-monospace,monospace;letter-spacing:2px;color:#8ee1bd}section{padding:26px;background:#191e27;border:1px solid #303641;border-radius:12px;margin:24px 0}button{background:#8ee1bd;color:#101319;font-weight:650;cursor:pointer}button,input,select{font:inherit;border:1px solid #485362;padding:10px 14px;border-radius:7px}input,select{background:#101319;color:#e6e9ee}label{display:block;margin:10px 0}label input[type=checkbox]{margin-right:9px}button:disabled{opacity:.5}a{color:#8ee1bd}.row{display:flex;flex-wrap:wrap;gap:10px}.row input{flex:1}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}#progress{color:#8ee1bd}.run{border-bottom:1px solid #303641;padding:14px 0;display:flex;justify-content:space-between;gap:20px}.muted{color:#8995a7}
</style></head><body><div class="eyebrow">TEMPORAL RAG LAB / RETRIEVAL SYSTEMS</div><h1>When the facts<br><span style="color:#8ee1bd">change.</span></h1><p>Replay dated updates through Graphiti and LightRAG. Compare the answers, inspect stale facts, and see what survives a change.</p><section><h2>Run a comparison</h2><label>Workload <select id="suite" aria-label="Workload"><option value="smoke">Diagnostic smoke suite</option><option value="temporal-pilot">Temporal pilot · 12 owners, late arrivals</option><option value="temporal-stress">Temporal stress · 18 scenarios</option></select></label><p id="plan"></p><p id="models"></p><div class="row"><label><input type="checkbox" name="backend" value="graphiti" checked>Graphiti</label><label><input type="checkbox" name="backend" value="lightrag-append" checked>LightRAG · append</label><label><input type="checkbox" name="backend" value="lightrag-replace" checked>LightRAG · replace</label></div><div id="scenarios"></div><p><small>Runs use your configured OpenRouter key and incur API charges. Choose a small subset for a quick experiment; the plan shows writes and probes per backend.</small></p><button id="start">Run selected scenarios</button><p id="progress" role="status"></p></section><section><h2>Saved comparisons</h2><div id="runs">Loading…</div></section><section><h2>Agent retrieval playground</h2><p>Select an indexed comparison dataset, or use a custom namespace you have ingested. <a href="/docs">API documentation</a> includes ingestion and retrieval.</p><label>Dataset <select id="dataset" aria-label="Dataset"><option value="">Custom namespace</option></select></label><div class="row"><select id="backend" aria-label="Retrieval backend"><option>graphiti</option><option>lightrag-append</option><option>lightrag-replace</option></select><input id="namespace" aria-label="Namespace" placeholder="Namespace" value="agent"></div><p class="row"><input id="query" aria-label="Question" placeholder="Who owns Project Atlas?"><input id="asof" aria-label="As of timestamp" placeholder="As of (optional ISO timestamp)"></p><button id="example">Use example question</button> <button id="retrieve">Retrieve evidence</button><pre id="result"></pre></section><p class="muted">Graphiti: temporal episodes and validity intervals. LightRAG: document extraction with explicit append or replacement policy. Synthetic runs measure behavior; they do not establish a universal winner.</p><script>
const $=id=>document.getElementById(id),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function json(url,options){const r=await fetch(url,options);const data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data}
async function runs(){const data=await json('/api/runs');$('runs').innerHTML=data.map(r=>`<div class="run"><a href="/reports/${encodeURIComponent(r.run_id)}">${esc(r.run_id)}</a><small>${esc(r.status)}</small></div>`).join('')||'<p class="muted">No saved runs yet.</p>'}
json('/health').then(d=>$('models').textContent=d.config.model+' · '+d.config.embedding_model);
async function loadSuite(){const s=await json('/api/suite?kind='+encodeURIComponent($('suite').value));$('scenarios').innerHTML=s.scenarios.map(c=>`<label><input type="checkbox" name="scenario" value="${esc(c.id)}" checked><strong>${esc(c.id)}</strong> — ${esc(c.description)}</label>`).join('');const writes=s.scenarios.reduce((n,c)=>n+c.steps.length,0),probes=s.scenarios.reduce((n,c)=>n+c.steps.reduce((m,s)=>m+s.probes.length,0),0);$('plan').textContent=`${s.scenarios.length} scenarios · ${writes} writes and ${probes} probes per backend. `+(s.metadata?.kind==='smoke'?'Mixed diagnostic checks; not a temporal ranking.':'Controlled temporal workload; full sweeps can take substantial time.')} $('suite').onchange=loadSuite;loadSuite();runs();
$('start').onclick=async()=>{const backends=[...document.querySelectorAll('[name=backend]:checked')].map(e=>e.value),scenarios=[...document.querySelectorAll('[name=scenario]:checked')].map(e=>e.value);if(!backends.length||!scenarios.length){$('progress').textContent='Select at least one backend and scenario.';return}$('start').disabled=true;try{const job=await json('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({backends,scenarios,suite:$('suite').value})});$('progress').textContent='Running '+job.run_id+'…';const poll=setInterval(async()=>{try{const r=await json(job.status_url);$('progress').textContent=`${r.status} · ${r.updates.length} updates · ${r.probes.length} probes`;if(r.status!=='running'){clearInterval(poll);$('start').disabled=false;await runs();location.href=job.report_url}}catch(e){clearInterval(poll);$('start').disabled=false;$('progress').textContent=e.message}},3000)}catch(e){$('progress').textContent=e.message;$('start').disabled=false}};
let datasets=[];
function chooseDataset(){const selected=datasets[Number($('dataset').value)];if($('dataset').value!==''&&selected){$('namespace').value=selected.namespace;$('query').placeholder=selected.example_query;$('asof').value=selected.example_as_of||''}else{$('namespace').value='agent';$('query').placeholder='Question for your custom namespace'}}
function loadDatasets(){const matching=datasets.map((d,i)=>({d,i})).filter(x=>x.d.backend===$('backend').value);$('dataset').innerHTML='<option value="">Custom namespace</option>'+matching.map(({d,i})=>`<option value="${i}">${esc(d.scenario)} · ${esc(d.run_id)}</option>`).join('');if(matching.length)$('dataset').value=String(matching[0].i);chooseDataset()}
json('/api/datasets').then(items=>{datasets=items;loadDatasets()});$('backend').onchange=loadDatasets;$('dataset').onchange=chooseDataset;$('example').onclick=()=>{const selected=datasets[Number($('dataset').value)];if($('dataset').value!==''&&selected){$('query').value=selected.example_query;$('asof').value=selected.example_as_of||''}};
$('retrieve').onclick=async()=>{$('retrieve').disabled=true;$('result').textContent='Retrieving…';try{$('result').textContent=JSON.stringify(await json('/api/retrieve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({backend:$('backend').value,namespace:$('namespace').value,query:$('query').value,as_of:$('asof').value||null})}),null,2)}catch(e){$('result').textContent=e.message}finally{$('retrieve').disabled=false}};
</script></body></html>"""
