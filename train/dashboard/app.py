"""Live local dashboard for persistent llama-server evaluations."""
from __future__ import annotations
import asyncio, json, sys, time, uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_runner import run, ROOT, load_matrix, LlamaServer, OpenAIClient

app=FastAPI(title="Mars checkpoint evaluation")
STATIC=Path(__file__).parent/"static"; RESULTS=ROOT/"evals"/"results"; MODELS=ROOT/"evals"/"models.yaml"; CASES=ROOT/"evals"/"cases"/"agent_realistic_v1.json"; CHATS=ROOT/"evals"/"chats"
runs: dict[str, asyncio.Queue] = {}
chat_streams: dict[str, tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = {}
chat_server: object | None = None
chat_server_model: str | None = None
chat_server_lock = asyncio.Lock()

class ChatCreate(BaseModel):
    model: str = "base"
    title: str = "New conversation"

class ChatMessage(BaseModel):
    content: str = Field(min_length=1, max_length=32000)


def chat_path(chat_id: str) -> Path:
    if not chat_id or not all(c.isalnum() or c in "-_" for c in chat_id):
        raise HTTPException(400, "invalid chat id")
    return CHATS / f"{chat_id}.json"


def read_chat(chat_id: str) -> dict:
    path = chat_path(chat_id)
    try: return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError: raise HTTPException(404, "chat not found")
    except json.JSONDecodeError: raise HTTPException(500, "chat data is invalid")


def save_chat(chat: dict) -> None:
    CHATS.mkdir(parents=True, exist_ok=True)
    path = chat_path(chat["id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(chat, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def chat_summary(chat: dict) -> dict:
    return {"id": chat["id"], "model": chat["model"], "title": chat["title"], "created_at": chat["created_at"], "updated_at": chat["updated_at"], "message_count": len(chat.get("messages", []))}


def list_chats():
    chats = []
    for path in CHATS.glob("*.json"):
        try: chats.append(chat_summary(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, KeyError): continue
    return sorted(chats, key=lambda item: item["updated_at"], reverse=True)


def ensure_chat_server(cfg: dict, model: dict):
    global chat_server, chat_server_model
    if chat_server is not None and chat_server_model == model["id"]:
        return chat_server
    if chat_server is not None:
        chat_server.stop()
    server = LlamaServer(cfg, model, int(cfg["server"].get("port", 18080)))
    server.start()
    chat_server, chat_server_model = server, model["id"]
    return server
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
@app.get("/chat")
def chat_page(): return FileResponse(STATIC/"chat.html", headers={"Cache-Control": "no-store, max-age=0"})
@app.get("/api/health")
def health(): return {"status":"ok","active_runs":len(runs),"historical_runs":len(history())}
@app.get("/api/models")
def models():
    cfg, items=load_matrix(MODELS); return {"models":[{"id":m["id"],"label":m["label"],"lora":m.get("lora")} for m in items]}
@app.get("/api/cases")
def cases(): return json.loads(CASES.read_text())

@app.get("/api/chats")
def chats(): return {"chats": list_chats()}

@app.post("/api/chats")
def create_chat(req: ChatCreate):
    cfg, available = load_matrix(MODELS)
    model = next((m for m in available if m["id"] == req.model), None)
    if not model: raise HTTPException(400, "unknown model")
    now = time.time()
    chat = {"id": uuid.uuid4().hex[:12], "model": model["id"], "title": req.title.strip()[:120] or "New conversation", "created_at": now, "updated_at": now, "messages": []}
    save_chat(chat)
    return chat

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str): return read_chat(chat_id)

@app.post("/api/chats/{chat_id}/messages")
async def send_chat_message(chat_id: str, req: ChatMessage):
    chat = read_chat(chat_id)
    if chat_id in chat_streams: raise HTTPException(409, "a response is already being generated")
    cfg, available = load_matrix(MODELS)
    model = next((m for m in available if m["id"] == chat["model"]), None)
    if not model: raise HTTPException(400, "model is no longer configured")
    user = {"role": "user", "content": req.content.strip()}
    chat["messages"].append(user); chat["updated_at"] = time.time(); save_chat(chat)
    queue, loop = asyncio.Queue(), asyncio.get_running_loop(); chat_streams[chat_id] = (queue, loop)
    async def work():
        nonlocal chat
        def put(event): asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        assistant = {"role": "assistant", "content": "", "reasoning": "", "status": "generating", "started_at": time.time()}
        try:
            already_loaded = chat_server is not None and chat_server_model == model["id"]
            put({"type": "status", "status": "generating" if already_loaded else "loading", "model": model["id"]})
            async with chat_server_lock:
                server = await asyncio.to_thread(ensure_chat_server, cfg, model)
                put({"type": "status", "status": "generating", "model": model["id"]})
                client = OpenAIClient(server.port, server.client_host)
                def chunk(item):
                    if item.get("reasoning"):
                        assistant["reasoning"] += item["reasoning"]; put({"type": "reasoning", "delta": item["reasoning"]})
                    if item.get("text"):
                        assistant["content"] += item["text"]; put({"type": "delta", "delta": item["text"]})
                prompt_messages = [{"role": item["role"], "content": item.get("content", "")} for item in chat["messages"]]
                result = await asyncio.to_thread(client.stream, model["id"], prompt_messages, {"temperature": cfg["server"].get("temperature", 0.2), "max_tokens": cfg["server"].get("max_tokens", 512)}, None, chunk)
            assistant.update({k: result.get(k) for k in ("elapsed_ms", "ttft_ms", "generation_ms", "prompt_tokens", "completion_tokens", "total_tokens", "tok_per_sec", "token_source")})
            assistant["status"] = "completed"; assistant["finished_at"] = time.time(); chat["messages"].append(assistant); chat["updated_at"] = time.time(); save_chat(chat)
            await queue.put({"type": "done", "message": assistant, "chat": chat_summary(chat)})
        except Exception as exc:
            assistant.update({"status": "error", "error": str(exc), "finished_at": time.time()})
            chat["messages"].append(assistant); chat["updated_at"] = time.time(); save_chat(chat)
            await queue.put({"type": "error", "message": str(exc), "message_record": assistant})
        finally:
            await queue.put({"type": "close"})
    asyncio.create_task(work())
    async def stream():
        while True:
            event = await queue.get(); yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] == "close": break
        chat_streams.pop(chat_id, None)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control":"no-cache", "X-Accel-Buffering":"no"})

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
