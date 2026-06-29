# Handoff — Mars-1.0 SFT Agent Training

**Date:** 2026-06-30  
**Agent leaving:** Zed coding agent  
**Workspace:** `maheryashrajtest` (Modal via WSL)

---

## Current State

- **Model:** LiquidAI/LFM2.5-8B-A1B (instruct, MoE, ~8B active)
- **Training stage:** SFT Agent (LoRA)
- **Status:** Running, detached — `ap-pcAVXKe8SyU7chr7aESgKz`
- **Step:** ~5,518 / 15,992 (35%) at ~4.3s/it
- **Resumed from:** `/vol/checkpoints/sft/mars_sft_agent/20260627_174008/checkpoint-5400`
- **GPU:** A100-80GB, batch=8, grad_accum=2, effective batch=16
- **Timeout:** 20h (`pipeline.yaml`)

### Volume Layout

```
/vol/
├── bases/LiquidAI__LFM2.5-8B-A1B/    # 15.8 GiB
├── checkpoints/sft/mars_sft_agent/
│   ├── 20260627_174008/               # Original run (ckpt 4600-5400)
│   └── 20260629_1818/                 # Current run (active)
├── data/hf-cache/                     # HF cache
├── tokenized/                         # Empty (caching not yet populated)
├── exports/                           # Empty
└── evals/                             # Empty
```

---

## What Was Accomplished

| Work | Description |
|------|-------------|
| PEFT 0.19.1 monkey-patch | `common.py` filters unexpected kwargs from `WeightConverter.__init__`. Generic patch — wrap `__init__` to accept `**kwargs`. |
| Dataset caching | Pre-tokenized dataset is saved to `/vol/tokenized/` to speed up resume. Not yet populated — first full run will fill it. |
| Config tuning | `sft_agent.yaml` scaled from A10G (batch=1, accum=16) to A100-80GB (batch=8, accum=2). Effective batch=16 preserved. |
| A10 → A100 migration | Changed GPU, increased timeout to 20h, verified MoE 8B LoRA fits on 80GB. |

---

## What's Next (Pipeline Order)

1. **SFT Agent** ← current (LoRA, ~10,474 steps remaining, est. ~12.5h)
2. **SFT Code** — full FT (`peft: enabled: false`), needs A100-80GB
3. **DPO** — LoRA on A10G (8B MoE fits)
4. **GRPO** — A100-80GB for multiple rollouts
5. **Eval** — lm-eval results to `/vol/evals/`
6. **Export** — merged HF repo to `/vol/exports/`

---

## Commands Reference

```bash
# Run SFT agent (detached)
PATH="$HOME/.local/bin:$PATH" modal run --detach train/modal_app.py --stage sft --config sft_agent.yaml

# Check if run is alive
modal app list

# Tail logs from active run
modal app logs ap-pcAVXKe8SyU7chr7aESgKz --tail 20 --timestamps

# List checkpoints on volume
modal volume ls mars-train-vol checkpoints/sft/mars_sft_agent/20260629_1818/

# SSH into running container (if debugging needed)
modal app exec ap-pcAVXKe8SyU7chr7aESgKz -- /bin/bash

# Check volume space
modal volume ls mars-train-vol
```

## Resume (if run dies)

1. Find latest checkpoint:
   ```bash
   modal volume ls mars-train-vol checkpoints/sft/mars_sft_agent/20260629_1818/
   ```
2. Update `resume_from_checkpoint` in `train/configs/sft_agent.yaml`
3. Re-run with the PATH prefix command above

---

## Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| PEFT TypeError | `WeightConverter.__init__()` got unexpected kwarg | Monkey-patch in `common.py` already applied |
| Modal profile mismatch | "App not found" or auth errors | Ensure `maheryashrajtest` profile active in WSL |
| Logs expired | `modal app logs` returns nothing after 24h | Starter plan — check before expiry or upgrade |
| Volume space | Training hangs mid-save | `modal volume ls mars-train-vol` to check |
| PATH missing | `modal: command not found` | Prefix with `PATH="$HOME/.local/bin:$PATH"` |

---

## Key Files

- `train/modal_app.py` — Modal entry point
- `train/configs/sft_agent.yaml` — SFT agent config (batch, accum, resume path)
- `train/configs/pipeline.yaml` — Pipeline defaults (GPU, timeout)
- `train/common.py` — PEFT monkey-patch, shared utilities
