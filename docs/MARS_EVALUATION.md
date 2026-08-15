# Mars-1.0 Evaluation

## Status

The canonical local evaluator is under `train/`. It compares the same Q4 GGUF base model with no adapter and with GRPO checkpoints. Results are produced by a persistent `llama-server` process: one process/model load per matrix row, followed by independent OpenAI-compatible requests for every case.

The current local matrix is:

- `base`
- `grpo-5400`
- `grpo-5500`
- `grpo-5600`

Only `grpo-5600` currently has a converted local GGUF LoRA adapter. The other checkpoint adapters must be converted before their rows can run.

## Run the dashboard

From PowerShell at the repository root:

```powershell
uv run --with fastapi --with uvicorn python -m uvicorn train.dashboard.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

Select the models and click **Run full suite**. The page shows the active prompt, streamed response text, and per-case pass/fail status. The browser receives SSE events while the runner sends stateless requests to the persistent server.

## Run headless

```powershell
uv run --with pyyaml python train/scripts/eval_runner.py `
  --models train/evals/models.yaml `
  --cases train/evals/cases/smoke_v1.json `
  --model base `
  --model grpo-5600 `
  --output train/evals/results/manual-run.json
```

Each model is loaded once. Each test receives a complete `messages` array and no prior assistant turn, so context is cleared logically between cases without restarting the model.

## Artifacts

- Model matrix: `train/evals/models.yaml`
- Versioned cases: `train/evals/cases/smoke_v1.json`
- Runner: `train/scripts/eval_runner.py`
- Dashboard: `train/dashboard/app.py`
- Static UI: `train/dashboard/static/index.html`
- Results: `train/evals/results/`

## Scoring

The smoke suite reports structural checks, not claims of BFCL, IFEval, or HumanEval parity:

- tool name and required-argument checks
- valid JSON output
- exact one-sentence output
- exactly three numbered/list items
- function-definition presence
- exact answer/keyword checks

Coding checks are currently syntax/shape smoke checks. They are not executable HumanEval `pass@1`. Missing or timed-out requests are recorded as errors and excluded from the completed-case numerator while remaining visible in the total denominator.

## Reproducibility

Results should record the base GGUF path, adapter path, llama.cpp build, server flags, seed, temperature, context size, and suite version. Compare base and candidates only within the same runtime and quantization. The Transformers evaluator and GGUF dashboard results are separate protocols and must not be merged into one score table.

## Conversion

The GRPO 5600 adapter was converted with llama.cpp's `convert_lora_to_gguf.py` using the LFM2.5 config/tokenizer metadata and emitted an F16 LoRA adapter. The base is the official `LiquidAI/LFM2.5-8B-A1B-Q4_K_M.gguf` artifact.
