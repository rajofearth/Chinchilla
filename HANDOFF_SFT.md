# Handoff — Mars-1.0 SFT Agent Training (Resume)

**Generated:** 2026-06-30  
**Workspace:** `maheryashrajtest` (Modal via WSL)  
**Config:** `train/configs/sft_agent.yaml`

---

## Current State

SFT Agent (stage 1 of the Mars-1.0 pipeline) is **running**.  
**Active app:** `ap-pcAVXKe8SyU7chr7aESgKz`  
**GPU:** A100-80GB  
**Resumed from:** `/vol/checkpoints/sft/mars_sft_agent/20260627_174008/checkpoint-5400`  
**Current step:** ~5,518 / 15,992 (~35%)  
**Rate:** ~4.3s/it  
**Timeout:** 20h (from `pipeline.yaml`)  
**Status:** Alive, detached run

Previous run (`ap-iZLS1vZGC2JJuuTZIh53sI`) died at step 5,400/15,991 (~33.8%) after 6.9h — reason unknown (logs expired due to 1-day retention on Starter plan).

---

## Config Changes Made

- `train/configs/sft_agent.yaml`: `gpu: A100-80GB`, `batch_size: 8`, `grad_accum: 2`, `resume_from_checkpoint: /vol/checkpoints/sft/mars_sft_agent/20260627_174008/checkpoint-5400`
- `train/configs/pipeline.yaml`: `gpu: A100-80GB`, `timeout: 72000` (20h)

Changelog from A10 baseline:
| Param | Before (A10) | After (A100-80GB) |
|-------|-------------|-------------------|
| gpu | A10G | A100-80GB |
| per_device_train_batch_size | 1 | 8 |
| gradient_accumulation_steps | 16 | 2 |
| resume_from_checkpoint | — | checkpoint-5400 |

Effective batch remains 16 (8 × 2) — training dynamics preserved.

---

## Checkpoint Structure

Checkpoints saved every 200 steps to volume at:
```
/vol/checkpoints/sft/mars_sft_agent/<run_id>/checkpoint-NNNN/
```

Each checkpoint contains: LoRA adapter weights (`adapter_model.safetensors` + `adapter_config.json`), optimizer state, scheduler, trainer state, RNG state, tokenizer. Volume committed after each save via `VolumeCommitCallback`.

Last 5 checkpoints retained (`save_total_limit=5`). Final adapter saved to `<run_id>/final/` on completion.

Checkpoint directories from current runs:
- `20260627_174008/` — Original run (checkpoints 4600-5400)
- `20260629_1818/` — Current run (active)

---

## PEFT Monkey-Patch (PEFT 0.19.1)

`common.py` contains a generic monkey-patch that filters unexpected kwargs from `WeightConverter.__init__`. This is necessary because PEFT 0.19.1 passes new common kwargs to converter constructors that don't expect them, causing `TypeError`. The patch wraps `__init__` to accept `**kwargs` and only pass known params.

If PEFT is upgraded, verify whether this patch can be removed.

---

## Dataset Caching

Pre-tokenized dataset caching is implemented. Tokenized dataset is saved to `/vol/tokenized/` to speed up resume — the trainer can load pre-tokenized data instead of re-tokenizing from scratch on each resume.

**Current state:** `/vol/tokenized/` is empty (not yet populated — first full run will populate it).

---

## Resume Flow

When run dies (timeout/error):
1. Identify latest checkpoint path on volume: `/vol/checkpoints/sft/mars_sft_agent/<run_id>/checkpoint-<last_step>`
2. Edit `sft_agent.yaml` → set `resume_from_checkpoint` to that path
3. Run:
   ```bash
   PATH="$HOME/.local/bin:$PATH" modal run --detach train/modal_app.py --stage sft --config sft_agent.yaml
   ```
4. Trainer loads LoRA weights + optimizer + scheduler from checkpoint, continues

---

## Billing & Budget

| Item | Cost |
|------|------|
| Previous run (6.9h, 5,400 steps) | $18.81 |
| Resume est. (~12.4h, remaining steps) | ~$31 |
| Total SFT agent est. | ~$50 |
| Monthly credit | $30/mo (Starter plan) |
| Status | Over credit; user unconcerned |

Real Modal GPU pricing per hour:
- A10G: $1.10
- A100-80GB: $2.50 ← **in use**
- H100: $3.95

---

## Volume Layout

```
/vol/
├── bases/LiquidAI__LFM2.5-8B-A1B/    # 15.8 GiB
├── checkpoints/sft/mars_sft_agent/
│   ├── 20260627_174008/               # Original run (ckpt 4600-5400)
│   └── 20260629_1818/                 # Current run (active)
├── data/hf-cache/                     # HF cache dir
├── tokenized/                         # Empty (caching not yet populated)
├── exports/                           # Empty
└── evals/                             # Empty
```

---

## What's Next (Pipeline)

After SFT agent completes:

1. **SFT code** — full FT (no LoRA), `sft_code.yaml` has `peft: enabled: false`. Needs A100-80GB.
2. **DPO** — LoRA on A10G (8B MoE fits).
3. **GRPO** — A100-80GB needed for multiple rollouts.
4. **Eval** — lm-eval results to `/vol/evals/`.
5. **Export** — merged HF repo to `/vol/exports/`.

---

## Notes for Next Agent

1. **Current run** is detached — check with `modal app list`. Monitor: `modal app logs ap-pcAVXKe8SyU7chr7aESgKz --tail 20 --timestamps`.
2. **Two Modal profiles** exist: `maheryashrajtest` (WSL, active) and `rajofearth` (Windows Python). Use WSL with `export PATH="$HOME/.local/bin:$PATH"`.
3. **`pyyaml`** installed via `uv tool install modal --reinstall --with pyyaml` — needed because `modal run` imports `modal_app.py` which does `import yaml`.
4. **If run times out:** find latest checkpoint on volume, update `resume_from_checkpoint`, re-run with the PATH prefix.
5. **PEFT patch** is in `common.py` — if upgrading PEFT, test that the patch is no longer needed.
6. **Dataset caching** writes to `/vol/tokenized/` — ensure volume has space (monitor with `modal volume ls mars-train-vol`).

## Suggested Skills
- `modal` — for Modal cloud platform interactions (billing, logs, app management)
