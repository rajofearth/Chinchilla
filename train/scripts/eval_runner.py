"""Persistent llama-server/OpenAI evaluation runner for Mars checkpoints."""
from __future__ import annotations
import argparse, json, os, re, subprocess, time, uuid
from pathlib import Path
from urllib.request import Request, urlopen
import yaml

ROOT = Path(__file__).resolve().parents[1]

def _path(value: str | None) -> Path | None:
    if not value: return None
    value = os.path.expandvars(value)
    # Permit the same manifest from WSL and Windows. WSL can launch .exe files
    # through /mnt/<drive>, while Windows retains drive-letter paths.
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\\\/]", value):
        value = "/mnt/" + value[0].lower() + value[2:].replace("\\\\", "/").replace("\\", "/")
    p = Path(value)
    return p if p.is_absolute() else ROOT.parent / p

class LlamaServer:
    def __init__(self, cfg: dict, model: dict, port: int):
        self.cfg, self.model, self.port = cfg, model, port
        self.process = None
    def start(self):
        exe = _path(self.cfg["server"]["executable"])
        base = _path(self.cfg["base"]["model"])
        lora = _path(self.model.get("lora"))
        if not base or not base.exists(): raise FileNotFoundError(f"base GGUF missing: {base}")
        if lora and not lora.exists(): raise FileNotFoundError(f"LoRA GGUF missing: {lora}")
        cmd = [exe, "--model", str(base), "--host", self.cfg["server"].get("host", "127.0.0.1"), "--port", str(self.port), "--alias", self.model["id"], "--ctx-size", str(self.cfg["server"].get("ctx_size", 4096)), "--threads", str(self.cfg["server"].get("threads", 8)), "--parallel", "1", "--no-cont-batching", "--jinja", "--metrics"]
        if lora: cmd += ["--lora", str(lora)]
        self.process = subprocess.Popen([str(x) for x in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                with urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status in (200, 204): return
            except Exception: time.sleep(1)
            if self.process.poll() is not None: raise RuntimeError(f"llama-server exited with {self.process.returncode}")
        raise TimeoutError("llama-server readiness timeout")
    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=10)
            except subprocess.TimeoutExpired: self.process.kill()

class OpenAIClient:
    def __init__(self, port: int): self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
    def stream(self, model: str, messages: list[dict], generation: dict, on_delta=None):
        body = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}, **generation}
        req = Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}, method="POST")
        started = time.perf_counter(); first = None; text = ""; events=[]; usage = None; reasoning = ""
        with urlopen(req, timeout=120) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                try: event = json.loads(data)
                except json.JSONDecodeError: continue
                events.append(event)
                if isinstance(event.get("usage"), dict): usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    reasoning_piece = delta.get("reasoning_content") or ""
                    reasoning += reasoning_piece
                    piece = delta.get("content") or ""
                    if (piece or reasoning_piece) and first is None: first = time.perf_counter()
                    text += piece
                    if (piece or reasoning_piece) and on_delta: on_delta(piece or reasoning_piece)
        elapsed = time.perf_counter() - started
        elapsed_ms = round(elapsed * 1000); ttft_ms = round((first-started)*1000) if first else None
        generation_ms = max(elapsed_ms - ttft_ms, 0) if ttft_ms is not None else elapsed_ms
        completion_tokens = usage.get("completion_tokens") if usage else None
        prompt_tokens = usage.get("prompt_tokens") if usage else None
        total_tokens = usage.get("total_tokens") if usage else None
        tok_per_sec = round(completion_tokens / (generation_ms / 1000), 3) if completion_tokens and generation_ms > 0 else None
        return {"text": text, "reasoning": reasoning, "elapsed_ms": elapsed_ms, "ttft_ms": ttft_ms, "generation_ms": generation_ms, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens, "tok_per_sec": tok_per_sec, "token_source": "server_usage" if usage else "missing", "chars": len(text)}

def score(case, text):
    e = case.get("expect", {}); low = text.lower(); result = {"checks": {}, "expected": e}
    if "tool" in e:
        result["checks"]["tool"] = e["tool"] in text
        result["checks"]["args"] = all(re.search(rf"\b{re.escape(a)}\s*=|[\"']{re.escape(a)}[\"']\s*:", text) for a in e.get("args", []))
        result["pass"] = result["checks"]["tool"] and result["checks"]["args"]
        result["explanation"] = "Valid expected tool and arguments." if result["pass"] else f"Expected {e['tool']} with args {e.get('args', [])}; tool={result['checks']['tool']}, args={result['checks']['args']}."
    elif e.get("type") == "json":
        try: json.loads(text.strip().strip("`").replace("json\n", "", 1)); result["pass"] = True; result["explanation"] = "Response parsed as JSON."
        except Exception as exc: result["pass"] = False; result["explanation"] = f"JSON parse failed: {exc}."
    elif e.get("type") == "one_sentence":
        count = len(re.findall(r"[.!?]", text)); result["pass"] = count == 1; result["explanation"] = f"Observed {count} sentence terminators; expected 1."
    elif e.get("type") == "three_items":
        count = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+", text)); result["pass"] = count == 3; result["explanation"] = f"Observed {count} list items; expected 3."
    elif "function" in e:
        result["pass"] = bool(re.search(rf"\bdef\s+{re.escape(e['function'])}\s*\(", text)); result["explanation"] = f"Function {e['function']} {'found' if result['pass'] else 'not found'}."
    elif "contains" in e: result["pass"] = e["contains"].lower() in low; result["explanation"] = f"Expected text {e['contains']!r}."
    elif "contains_any" in e: result["pass"] = any(x.lower() in low for x in e["contains_any"]); result["explanation"] = f"Expected one of {e['contains_any']}."
    else: result["pass"] = False; result["explanation"] = "No scoring rule matched."
    return result

def load_matrix(path: Path):
    cfg = yaml.safe_load(path.read_text())
    base = cfg["base"]
    models=[]
    for item in cfg["models"]:
        m = dict(base); m.update(item); models.append(m)
    return cfg, models

def _totals(responses, total):
    completed=[r for r in responses if r.get("status")=="completed"]; valid_tokens=[r for r in completed if r.get("completion_tokens") is not None]
    elapsed=sum(r.get("elapsed_ms",0) for r in completed); generation=sum(r.get("generation_ms",0) for r in completed)
    completion=sum(r["completion_tokens"] for r in valid_tokens) if valid_tokens else None
    prompt=sum(r["prompt_tokens"] for r in completed if r.get("prompt_tokens") is not None) or None
    return {"passed":sum(bool(r.get("score",{}).get("pass")) for r in completed),"completed":len(completed),"failed":total-len(completed),"total":total,"pass_rate":round(sum(bool(r.get("score",{}).get("pass")) for r in completed)/total,3) if total else 0,"elapsed_ms":elapsed,"generation_ms":generation,"avg_ttft_ms":round(sum(r["ttft_ms"] for r in completed if r.get("ttft_ms") is not None)/max(1,sum(r.get("ttft_ms") is not None for r in completed)),1),"prompt_tokens":prompt,"completion_tokens":completion,"total_tokens":(prompt+completion if prompt is not None and completion is not None else None),"tok_per_sec":round(completion/(generation/1000),3) if completion and generation else None,"token_coverage":len(valid_tokens)}

def run(models_path: Path, cases_path: Path, selected=None, output=None, on_event=None):
    cfg, models = load_matrix(models_path); cases = json.loads(cases_path.read_text())["cases"]
    if selected: models = [m for m in models if m["id"] in selected]
    run_id = uuid.uuid4().hex[:12]; result={"schema_version":1,"run_id":run_id,"models":[],"cases":[]}
    port = int(cfg["server"].get("port",18080))
    for model in models:
        server=LlamaServer(cfg, model, port); model_result={"id":model["id"],"label":model["label"],"responses":[]}
        try:
            if on_event: on_event({"type":"model_start","model":model["id"]})
            server.start(); client=OpenAIClient(port)
            for case in cases:
                if on_event: on_event({"type":"prompt","model":model["id"],"case":case["id"],"prompt":case["messages"][-1]["content"]})
                try:
                    got=client.stream(model["id"],case["messages"],{"temperature":cfg["server"].get("temperature",0.2),"seed":cfg["server"].get("seed",42),"max_tokens":cfg["server"].get("max_tokens",256)},lambda x: on_event({"type":"delta","model":model["id"],"case":case["id"],"text":x}) if on_event else None)
                    got["case_id"]=case["id"]; got["suite"]=case.get("suite"); got["prompt"]=case["messages"][-1]["content"]; got["expect"]=case.get("expect", {}); got["status"]="completed"; got["score"]=score(case,got["text"]); model_result["responses"].append(got)
                    if on_event: on_event({"type":"finished","model":model["id"],"case":case["id"],"response":got["text"],"score":got["score"],"metrics":{k:got.get(k) for k in ("elapsed_ms","ttft_ms","generation_ms","prompt_tokens","completion_tokens","tok_per_sec")}})
                except Exception as exc:
                    error={"case_id":case["id"],"suite":case.get("suite"),"prompt":case["messages"][-1]["content"],"status":"error","error":str(exc)}; model_result["responses"].append(error)
                    if on_event: on_event({"type":"case_error","model":model["id"],"case":case["id"],"error":str(exc)})
        except Exception as exc: model_result["error"]=str(exc)
        finally: server.stop()
        model_result["totals"] = _totals(model_result["responses"], len(cases)); result["models"].append(model_result); port += 1
    result["summary"]={m["id"]:m["totals"] for m in result["models"]}
    if output:
        output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--models",default=str(ROOT/"evals/models.yaml")); ap.add_argument("--cases",default=str(ROOT/"evals/cases/smoke_v1.json")); ap.add_argument("--model",action="append"); ap.add_argument("--output"); args=ap.parse_args()
    print(json.dumps(run(Path(args.models),Path(args.cases),args.model,Path(args.output) if args.output else ROOT/"evals/results"/f"run-{int(time.time())}.json"),indent=2))
