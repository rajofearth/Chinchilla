"""Live local dashboard for persistent llama-server evaluations."""
from __future__ import annotations
import asyncio, json, sys, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_runner import run, ROOT, load_matrix

app=FastAPI(title="Mars checkpoint evaluation")
STATIC=Path(__file__).parent/"static"; RESULTS=ROOT/"evals"/"results"; MODELS=ROOT/"evals"/"models.yaml"; CASES=ROOT/"evals"/"cases"/"agent_realistic_v1.json"
runs: dict[str, asyncio.Queue] = {}
class RunRequest(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["base", "grpo-5400", "grpo-5500", "grpo-5600"])

def history():
    out=[]
    for p in sorted(RESULTS.glob("*.json"), key=lambda x:x.stat().st_mtime, reverse=True):
        try:
            data=json.loads(p.read_text()); out.append({"run_id":data.get("run_id",p.stem),"file":p.name,"summary":data.get("summary",{}),"models":[m.get("id") for m in data.get("models",[])],"status":data.get("status","completed")})
        except (OSError,json.JSONDecodeError): continue
    return out
@app.get("/")
def index(): return FileResponse(STATIC/"index.html")
@app.get("/api/health")
def health(): return {"status":"ok","active_runs":len(runs),"historical_runs":len(history())}
@app.get("/api/models")
def models():
    cfg, items=load_matrix(MODELS); return {"models":[{"id":m["id"],"label":m["label"],"lora":m.get("lora")} for m in items]}
@app.get("/api/cases")
def cases(): return json.loads(CASES.read_text())
@app.get("/api/runs")
def list_runs(): return {"runs":history()}
@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    matches=list(RESULTS.glob(f"{run_id}.json"))+list(RESULTS.glob("*.json"))
    for p in matches:
        try:
            data=json.loads(p.read_text())
            if data.get("run_id")==run_id or p.stem==run_id: return data
        except (OSError,json.JSONDecodeError): pass
    raise HTTPException(404,"run not found")
@app.post("/api/runs")
async def start(req: RunRequest):
    cfg, available=load_matrix(MODELS); valid={m["id"] for m in available}
    if not req.models or not set(req.models)<=valid: raise HTTPException(400,"unknown or empty model selection")
    run_id=uuid.uuid4().hex[:12]; q=asyncio.Queue(); runs[run_id]=q
    async def work():
        loop=asyncio.get_running_loop()
        def emit(event): asyncio.run_coroutine_threadsafe(q.put(event), loop)
        try:
            await asyncio.to_thread(run, MODELS, CASES, req.models, RESULTS/f"{run_id}.json", emit, run_id)
            await q.put({"type":"run_finished","run_id":run_id})
        except Exception as exc: await q.put({"type":"error","run_id":run_id,"message":str(exc)})
    asyncio.create_task(work()); return {"run_id":run_id,"events_url":f"/api/runs/{run_id}/events"}
@app.get("/api/runs/{run_id}/events")
async def events(run_id: str):
    q=runs.get(run_id)
    if not q: raise HTTPException(404,"run not active")
    async def stream():
        while True:
            event=await q.get(); yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in {"run_finished","error"}: break
        runs.pop(run_id,None)
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
