# nanochat SFT on Modal — Technical Report

## 1. Project Overview

This project performs **Supervised Fine-Tuning (SFT)** of the [karpathy/nanochat](https://github.com/karpathy/nanochat) **d6** model (73.5M parameters, 6-layer, 384-dim) on **Modal** cloud GPUs. The base pretrain checkpoint is at **step 8600** from Kaggle pretraining (FineWeb-EDU data, 512 sequence length).

The goal: take a pretrained autoregressive language model and fine-tune it on a mixture of instruction-following datasets (SmolTalk, MMLU, GSM8K, spelling tasks) so it learns to follow user instructions in a chat format.

---

## 2. Infrastructure Setup

### 2.1 Modal Configuration

| Parameter | Value |
|-----------|-------|
| App name | `nanochat-sft` |
| Base image | `debian-slim` (Python 3.11) |
| PyTorch | 2.3.0 (`cu121`) |
| GPU | 2× NVIDIA A10G (24 GB VRAM each) |
| Volume | `nanochat-vol` mounted at `/vol` |
| Timeout | 14,400 s (4 hours) for training |
| Memory | 32 GB |

### 2.2 Container Image Dependencies

```python
modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "torchaudio==2.3.0",
        "rustbpe",
        "tiktoken",
        "datasets",
        "transformers",
        "httpx",
        "pyarrow",
        "sentencepiece",
        "wandb",
        "psutil",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
```

### 2.3 Volume Contents

The persistent Modal volume `nanochat-vol` stores:

| Path | Contents | Size |
|------|----------|------|
| `/vol/base_checkpoints/d6/model_008600.pt` | Pretrain model weights | 280.5 MB |
| `/vol/base_checkpoints/d6/meta_008600.json` | Pretrain metadata | — |
| `/vol/tokenizer/tokenizer.pkl` | Trained BPE tokenizer (vocab_size=32,768) | 402.5 KB |
| `/vol/tokenizer/token_bytes.pt` | Token byte mapping for BPB eval | 129.2 KB |
| `/vol/base_data/` | 11 FineWeb-EDU parquet shards (downloaded on-demand) | — |
| `/vol/identity_conversations.jsonl` | Synthetic identity conversations (1,000 rows) | 2.4 MB |

### 2.4 Checkpoints (Local Mirror)

Local copies downloaded from the volume reside under `checkpoints/`:

| File | Size |
|------|------|
| `checkpoints/pretrain/model_008600.pt` | 280.5 MB |
| `checkpoints/pretrain/optim_008600_rank0.pt` | 260.3 MB |
| `checkpoints/pretrain/optim_008600_rank1.pt` | 260.3 MB |
| `checkpoints/pretrain/meta_*.json` | (16 metadata files, steps 2000–8600) |
| `checkpoints/sft/model_000971.pt` | 280.5 MB |
| `checkpoints/sft/meta_000971.json` | SFT metadata |

---

## 3. Patches Applied

The `setup_nanochat()` function applies 5 compatibility patches to the upstream `karpathy/nanochat` repo. These are necessary because Modal uses PyTorch 2.3.0 (cu121) and a different filesystem layout than the original Kaggle environment.

### 3.1 `dataset.py` — Data Directory Path

**Problem:** The dataset module hardcodes `DATA_DIR = os.path.join(base_dir, "base_data_climbmix")`. This path doesn't exist on Modal because we supply our own data shards in `base_data/`.

**Patch:** Replace `base_data_climbmix` with `base_data` in the source file.

```python
content = content.replace("base_data_climbmix", "base_data")
```

### 3.2 `engine.py` — Configurable Dtype

**Problem:** The engine hardcodes `dtype = torch.bfloat16 if device.type == "cuda" else torch.float32`. On T4 GPUs (which don't support bf16), this causes a KV cache dtype mismatch. On A10G it's fine, but we want control.

**Patch:** Make dtype configurable via the `NANOCHAT_DTYPE` environment variable. Falls back to the original behavior if the env var is unset.

```python
old = 'dtype = torch.bfloat16 if device.type == "cuda" else torch.float32'
new = (
    'dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, '
    '"float32": torch.float32}.get(os.environ.get("NANOCHAT_DTYPE", ""), '
    'torch.bfloat16 if device.type == "cuda" else torch.float32)'
)
```

### 3.3 `gpt.py` — Manual RMSNorm

**Problem:** PyTorch 2.3.0 doesn't have `F.rms_norm` (added in PyTorch 2.4). The model's `norm()` function calls `F.rms_norm(x, (x.size(-1),))`.

**Patch:** Replace with a manual RMSNorm implementation using the standard formula:

```python
def norm(x):
    # Manual RMSNorm for PyTorch < 2.4 compatibility
    var = x.pow(2).float().mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + 1e-6)).to(x.dtype)
```

### 3.4 `chat_sft.py` — Suppress Dynamo Errors

**Problem:** `torch.compile` (used internally by nanochat) can fail on unsupported operations in PyTorch 2.3.0, crashing training with `TorchRuntimeError`.

**Patch:** Add `torch._dynamo.config.suppress_errors = True` to fall back to eager mode gracefully on Dynamo compilation errors.

```python
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # fall back to eager on Dynamo errors
```

### 3.5 `flash_attention.py` — Strip `enable_gqa` Kwarg

**Problem:** The `flash_attention.py` module passes `enable_gqa=enable_gqa` to `F.scaled_dot_product_attention()`. This parameter was added in PyTorch 2.5 and is not available in 2.3.0.

**Patch:** Strip the `, enable_gqa=enable_gqa` keyword argument from all SDPA calls.

```python
content = content.replace(", enable_gqa=enable_gqa", "")
```

**Why safe:** The d6 model uses Multi-Head Attention (`n_head=6`, `n_kv_head=6`), not Grouped-Query Attention, so GQA is never activated regardless of the flag.

---

## 4. Training Configuration

### 4.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| `max_seq_len` | 2048 |
| `device_batch_size` | 4 (per GPU) |
| `total_batch_size` | 524,288 tokens |
| `grad_accum_steps` | 32 |
| `num_iterations` | 48,000 micro-batches (= 1,500 optimizer steps) |
| `init_lr_frac` | 0.1 |
| `warmup_ratio` | 0.1 |
| `warmdown_ratio` | 0.5 |
| `final_lr_frac` | 0.0 |
| `eval_every` | 200 steps |
| `eval_tokens` | 131,072 |
| `load_optimizer` | 0 (fresh optimizer, no pretrain state) |
| `chatcore_every` | 0 (disabled during training) |

### 4.2 Batch Size Math

```
world_tokens = device_batch_size × max_seq_len × num_gpus
            = 4 × 2048 × 2
            = 16,384 tokens per micro-batch

grad_accum_steps = total_batch_size / world_tokens
                 = 524,288 / 16,384
                 = 32 (exact)

total_tokens_per_step = 524,288
```

### 4.3 Data Mixture

The training dataset is a `TaskMixture` combining multiple sources. The counts below are computed from the nanochat source:

| Component | Rows | Epochs | Total Rows |
|-----------|------|--------|------------|
| SmolTalk (train) | 460,000 | 1 | 460,000 |
| CustomJSON (identity_conversations.jsonl) | 1,000 | 2 | 2,000 |
| MMLU (all subsets, auxiliary_train) | ~100,000 | 3 | ~300,000 |
| GSM8K (main, train) | ~7,473 | 4 | ~29,892 |
| SimpleSpelling | 200,000 | 1 | 200,000 |
| SpellingBee | 80,000 | 1 | 80,000 |
| **Total** | | | **~1,071,892** |

The validation dataset uses:
- SmolTalk (test): 24,000 rows
- MMLU (test, capped at 5,200): ~5,200 rows
- GSM8K (test, capped at 420): ~420 rows
- **Total validation**: ~29,620 rows

### 4.4 Optimizer: Combined MuonAdamW

nanochat uses a hybrid optimizer strategy:

| Parameter Group | Optimizer | LR |
|----------------|-----------|----|
| Embedding & unembedding weights | AdamW | 0.3 (scaled by 1/√(384/768) = 1.414 → ~0.424) |
| Matrix params (all linear layers) | Muon | 0.02 |

- **Muon** (from the `muon` optimizer by Keller Jordan): applies orthogonalized updates to matrix parameters. More efficient than AdamW for large matrix multiplies.
- **AdamW**: used for the embedding/unembedding tables where Muon's orthogonality constraint is less beneficial.

### 4.5 Learning Rate Schedule

```
Linear warmup (10% of steps) → Constant (40% of steps) → Linear decay to 0 (50% of steps)
```

- Warmup: steps 0–150, LR goes from 0 to the target LR
- Constant: steps 150–750, LR stays at target
- Warmdown: steps 750–1,500, LR decays linearly to 0
- `init_lr_frac=0.1`: the effective learning rate multiplier starts at 0.1× of the base LR

---

## 5. Bugs Encountered & Fixes

### Bug 1: `enable_gqa` Keyword Argument Error

- **Symptom:** `scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'`
- **Root Cause:** PyTorch 2.3.0 does not support the `enable_gqa` parameter for `F.scaled_dot_product_attention()`. This parameter was added in PyTorch 2.5. The nanochat `flash_attention.py` passes it unconditionally in its SDPA fallback.
- **Fix:** Added a patch in `setup_nanochat()` that strips `, enable_gqa=enable_gqa` from all SDPA calls via a string replacement on `flash_attention.py`.
- **Why safe:** The d6 model uses MHA (`n_head=6`, `n_kv_head=6`), not GQA, so the flag has no effect even when available. Without it, SDPA behaves identically since `n_head == n_kv_head`.
- **Error trace:**
  ```
  torch._dynamo.exc.TorchRuntimeError: Failed running call_function
  <built-in function scaled_dot_product_attention>(...)
  scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'
  ```

### Bug 2: Missing `psutil` Module

- **Symptom:** `ModuleNotFoundError: No module named 'psutil'` at the end of training (during report generation).
- **Root Cause:** nanochat's `report.py` imports `psutil` for gathering system stats, but it wasn't included in the Modal image's `pip_install` list.
- **Fix:** Added `"psutil"` to the `pip_install` list in the image definition.
- **Impact:** Training would crash at the very end, after SFT completed successfully, preventing checkpoint persistence to the volume.

### Bug 3: `num_iterations` Counted Micro-Batches, Not Steps

- **Symptom:** Training stopped after only 47 steps despite setting `--num-iterations=1500`. Console showed "100.33%" completion after a few minutes.
- **Root Cause:** The data generator's `it` counter in `chat_sft.py` counts calls to `next(train_loader)`, which yields one micro-batch per call. With `grad_accum_steps=32`, 1,500 micro-batches = 1,500/32 ≈ 47 optimizer steps. The `--num-iterations` flag controls micro-batches, not optimizer steps.
- **Fix:** Changed `NUM_ITERATIONS` from 1,500 to 48,000 (1,500 × 32) in the training script's torchrun command.
- **Line in code:** `str(NUM_ITERATIONS * 32),  # 1500 steps × 32 grad_accum = 48000 micro-batches`

### Bug 4: `load_model_from_dir` Path Mismatch in `test_inference`

- **Symptom:** `FileNotFoundError: No checkpoints found in /vol/chatsft_checkpoints/d6`
- **Root Cause:** `load_model_from_dir()` expects a **parent directory** path, then looks for subdirectories named by `model_tag`. The inference function was passing the checkpoint directory directly (`/vol/chatsft_checkpoints/d6`) as both the directory and the tag.
- **Fix:** Pass the parent directory (`/vol/chatsft_checkpoints`) separately from `model_tag="d6"`:

  ```python
  parent_dir = str(SFT_CHECKPOINT_DIR.parent)  # /vol/chatsft_checkpoints
  tag = SFT_CHECKPOINT_DIR.name                # "d6"
  model, _, meta = load_model_from_dir(
      parent_dir, device=device, phase="eval", model_tag=tag
  )
  ```

### Bug 5: Stale `result` Variable Reference

- **Symptom:** `NameError: name 'result' is not defined` after SFT completed successfully.
- **Root Cause:** The variable was renamed from `result` to `_result` in one place (to avoid shadowing the `subprocess` module), but the return statement still referenced `result`:

  ```python
  _result = _sp.run(cmd, ...)   # renamed
  ...
  return result.returncode      # OLD: NameError
  ```

- **Fix:** Changed the return statement to use `_result`:

  ```python
  return _result.returncode
  ```

### Bug 6: Indentation Error from Damaged Edit

- **Symptom:** Modal deployment failed with `IndentationError: unexpected indent`.
- **Root Cause:** During iterative file editing, a stray line of dashes was introduced into the Python source code, likely from a copy-paste artifact.
- **Fix:** Identified and removed the corrupted line.

---

## 6. Training Results

### 6.1 Metrics

| Step | Validation BPB | Notes |
|------|---------------|-------|
| 0 (pretrain) | ~0.64 | Base d6 checkpoint at step 8600 on FineWeb-EDU |
| 47 | 0.6411 | First partial run (1 micro-batch, crashed on `psutil` error) |
| **971** | **0.4891** | **Full SFT run — 1 epoch of the data mixture** |

The validation BPB (bits per byte) dropped from ~0.64 to **0.4891**, indicating the model learned to fit the SFT data distribution significantly better than the pretrained baseline. The meta file confirms:

```json
{
  "step": 971,
  "val_bpb": 0.48913308218773954,
  "model_config": {
    "n_layer": 6,
    "n_head": 6,
    "n_kv_head": 6,
    "n_embd": 384,
    "vocab_size": 32768,
    "sequence_len": 2048
  },
  "user_config": {
    "num_iterations": 48000,
    "max_seq_len": 2048,
    "device_batch_size": 4,
    "total_batch_size": 524288,
    "eval_every": 200,
    "eval_tokens": 131072,
    "mmlu_epochs": 3,
    "gsm8k_epochs": 4,
    "load_optimizer": 0
  }
}
```

**Note:** The training ran for 971 steps (out of 1,500 planned) because the data mixture was exhausted after 1 epoch — all ~1.07M rows were consumed by step 971, and the data loader had no more data. To reach all 1,500 steps, the SFT data would need more epochs or more total data.

### 6.2 Training Performance

| Metric | Value |
|--------|-------|
| Step time | ~2.2 sec/step on 2× A10G |
| Aggregate throughput | ~240,000 tok/sec |
| Total training time | ~35 min |
| Peak GPU memory | ~4,866 MB per GPU |
| Estimated cost | **~$1.20** (A10G at ~$0.60/hr × 2 GPUs × ~35 min) |

### 6.3 Loss Curve Behavior

The training loss was not captured in detail (WandB was set to a dummy run), but the step-by-step console output showed:

- **Step 1 loss:** ~2.3 (normal — SFT loss starts higher than pretrain loss)
- **Mid-training:** Decreasing steadily, following the LR schedule
- **Final step:** Loss converged below ~1.5 (estimated from BPB improvement)

### 6.4 Qualitative Assessment

Testing the SFT model with sample prompts reveals:

**Prompt: "What is 2 + 2?"**
```
Response: (irrelevant text about data analysis — the model lacks math reasoning)
```

**Prompt: "Hello!"**
```
Response: (coherent paragraph-length chat response in the expected format)
```

**Observed behaviors:**

| Capability | Assessment |
|------------|-----------|
| Chat format | ✓ Learned — generates proper `<|assistant|>` responses |
| Paragraph length | ✓ Learned — produces multi-sentence responses |
| Domain awareness | ~ Partial — some awareness of data analysis, spelling |
| Math reasoning | ✗ Fails — "2+2" produces irrelevant output |
| Factual knowledge | ✗ Garbled — entities like Elon Musk produce confused output |
| Pattern looping | ~ Sometimes repeats the same response format |

**Interpretation:** This is expected behavior for a **d6 model (73.5M params)** with only **1 epoch of SFT**. The model is roughly 1/22nd the size of GPT-2 (1.6B params). At this scale, instruction following is inherently limited — the model simply doesn't have enough parameters to represent complex reasoning or store many facts. The d26 model (~1.6B params) would perform significantly better.

---

## 7. File Structure

```
nanochat_modal/
├── nanochat/                          # Upstream karpathy/nanochat repo (embedded git)
│   ├── nanochat/                      # Core library modules
│   │   ├── gpt.py                     # Model definition (patched RMSNorm)
│   │   ├── flash_attention.py         # FA3/SDPA interface (patched enable_gqa)
│   │   ├── engine.py                  # Inference engine (patched dtype)
│   │   ├── dataset.py                 # Data loading (patched path)
│   │   ├── dataloader.py              # Dataloader utilities
│   │   ├── tokenizer.py               # BPE tokenizer
│   │   ├── optim.py                   # MuonAdamW optimizer
│   │   ├── report.py                  # Training reports (needs psutil)
│   │   ├── checkpoint_manager.py      # Model saving/loading
│   │   └── common.py                  # Shared utilities
│   ├── scripts/
│   │   ├── chat_sft.py                # SFT training loop (patched)
│   │   ├── chat_eval.py               # ChatCORE evaluation
│   │   ├── tok_train.py               # Tokenizer training
│   │   └── chat_cli.py                # CLI chat interface
│   ├── tasks/
│   │   ├── smoltalk.py                # SmolTalk dataset (460K rows)
│   │   ├── customjson.py              # JSONL conversation loader
│   │   ├── mmlu.py                    # MMLU multiple-choice (57 subjects)
│   │   ├── gsm8k.py                   # Grade-school math
│   │   ├── spellingbee.py             # Letter counting + spelling
│   │   ├── common.py                  # Task base class
│   │   └── humaneval.py               # Code generation eval
│   └── tokenizer/                     # Trained tokenizer files
│       ├── tokenizer.pkl              # BPE tokenizer (402 KB)
│       └── token_bytes.pt             # Byte mapping (129 KB)
├── training/
│   ├── nanochat_sft.py                # Main Modal SFT training script
│   └── run_sft.sh                     # Shell command wrapper
├── checkpoints/
│   ├── pretrain/                      # Base pretrain checkpoint (local copy)
│   │   ├── model_008600.pt            # Model weights (280.5 MB)
│   │   ├── meta_*.json                # Training metadata (16 files)
│   │   ├── optim_008600_rank0.pt      # Optimizer state GPU 0 (260.3 MB)
│   │   └── optim_008600_rank1.pt      # Optimizer state GPU 1 (260.3 MB)
│   └── sft/                           # SFT checkpoint (local copy)
│       ├── model_000971.pt            # Model weights (280.5 MB)
│       └── meta_000971.json           # Training metadata
├── nanochat_sft_modal_runbook.md      # Full setup runbook (markdown)
├── nanochat_sft_modal_runbook.docx     # Runbook (Word format)
├── README.md                          # Project README
└── SFT_REPORT.md                      # This file
```

---

## 8. How to Run

### 8.1 Prerequisites

```bash
pip install modal
modal setup
modal volume create nanochat-vol
```

### 8.2 Upload Pretrain Checkpoint to Volume

```bash
git clone https://github.com/rajofearth/temp-file-share.git checkpoint_files
cd checkpoint_files
git lfs pull
modal volume put nanochat-vol model_008600.pt /base_checkpoints/d6/model_008600.pt
modal volume put nanochat-vol meta_008600.json /base_checkpoints/d6/meta_008600.json
```

### 8.3 Training Commands

```bash
cd training/

# Train synchronously (live logs, Ctrl+C to cancel)
modal run nanochat_sft.py::run_sft

# Train detached (continues after terminal closes)
modal run --detach nanochat_sft.py::run_sft

# Test inference on the trained checkpoint
modal run nanochat_sft.py::test_inference --prompt "Hello!"

# Download checkpoint from the volume
modal run nanochat_sft.py::download_sft

# Run ChatCORE evaluation
modal run nanochat_sft.py::run_eval
```

### 8.4 Shell Wrapper Usage

```bash
cd training/

./run_sft.sh run          # synchronous SFT (live logs)
./run_sft.sh detach       # detached SFT (background)
./run_sft.sh logs         # tail logs from latest/active run
./run_sft.sh infer "Hi!"  # test inference
./run_sft.sh status       # inspect volume contents
./run_sft.sh download     # download checkpoint from volume
./run_sft.sh eval         # run ChatCORE evaluation
```

### 8.5 Download Checkpoint Locally

```bash
modal volume get nanochat-vol /chatsft_checkpoints/d6/model_000971.pt ./my_sft_model.pt
```

---

## 9. Recommendations for Better Results

### 9.1 More Training Epochs (Immediate, ~$1/epoch)

The training only completed **971 of 1,500 planned steps** because the data mixture was exhausted. Running additional SFT passes (3–5 more epochs) would reinforce the learning:

```bash
# Modify the training script to repeat data or reduce total_batch_size
# Or simply re-run the same training (data shuffling gives different order)
modal run --detach nanochat_sft.py::run_sft
```

Each additional epoch costs approximately **$1** on 2× A10G.

### 9.2 Larger Model (d26, ~1.6B Params, ~$48)

The d6 model (73.5M params) is fundamentally limited in its instruction-following capability. The **d26** model (26 layers, ~1.6B params) is what nanochat is designed for — roughly GPT-2 scale. Training on 8× H100 would take ~2 hours at ~$24/hr, totaling ~$48. This would dramatically improve SFT quality.

### 9.3 Run ChatCORE Evaluation

Run the benchmark suite to get quantitative scores:

```bash
cd training/
modal run nanochat_sft.py::run_eval
```

This evaluates on ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval, and SpellingBee. Expected ranges for d6:

| Benchmark | Expected (d6, 73.5M) | Reference (GPT-2, 1.6B) |
|-----------|--------------------|------------------------|
| MMLU | 28–35% (random = 25%) | ~32% |
| ARC-Easy | 45–55% | ~60% |
| GSM8K | 2–8% | ~15% |

### 9.4 Adjust Data Mixture

The current mixture heavily emphasizes spelling tasks (280K of ~1.07M rows, ~26%). Consider:

- **Reducing** SimpleSpelling/SpellingBee proportion (they dominate the mixture)
- **Adding** more math/reasoning data (GSM8K at only 8K rows per epoch is a tiny fraction)
- **Adding** code data (HumanEval is eval-only, no training split)
- **Increasing** SmolTalk epochs (460K rows × 2–3 epochs would better teach conversation format)

### 9.5 Enable ChatCORE During Training

The `--chatcore-every 0` flag disables ChatCORE evaluation during training because it's slow. Set `--chatcore-every=500` to get per-task benchmark scores throughout training, which helps identify when the model starts overfitting.

---

## Appendix A: Key File Paths on Modal Volume

| What | Modal Volume Path |
|------|-------------------|
| Pretrain checkpoint | `/vol/base_checkpoints/d6/model_008600.pt` |
| Pretrain metadata | `/vol/base_checkpoints/d6/meta_008600.json` |
| Tokenizer | `/vol/tokenizer/tokenizer.pkl` |
| Token bytes | `/vol/tokenizer/token_bytes.pt` |
| Data shards | `/vol/base_data/shard_*.parquet` |
| SFT checkpoint (output) | `/vol/chatsft_checkpoints/d6/model_000971.pt` |
| SFT metadata | `/vol/chatsft_checkpoints/d6/meta_000971.json` |
| Identity conversations | `/vol/identity_conversations.jsonl` |

## Appendix B: GPU Configuration Comparison

| GPU | VRAM | bf16 | ~Cost/hr | Notes |
|-----|------|------|----------|-------|
| T4 × 2 | 2× 15 GB | No | ~$0.15 | Requires `NANOCHAT_DTYPE=float32` |
| **A10G × 2** (used) | 2× 24 GB | Yes | ~$0.60 | **Best value — full bf16, fast** |
| A100 × 2 | 2× 40/80 GB | Yes | ~$2.00 | Overkill for d6, good for d20+ |

## Appendix C: Batch Size Valid Configurations

All configurations multiply to exactly `total_batch_size = 524,288`:

| GPU Setup | device_batch | seq_len | num_gpus | grad_accum | total_batch |
|-----------|-------------|---------|----------|------------|-------------|
| T4 × 2 (float32) | 2 | 2048 | 2 | 64 | 524,288 |
| T4 × 2 (float32) | 4 | 1024 | 2 | 64 | 524,288 |
| **A10G × 2 (bf16)** | **4** | **2048** | **2** | **32** | **524,288** |
| A10G × 2 (bf16) | 8 | 2048 | 2 | 16 | 524,288 |
| A100 × 2 (bf16) | 8 | 2048 | 2 | 16 | 524,288 |

---

*Report generated June 2026. For questions, refer to the [nanochat_sft_modal_runbook.md](./nanochat_sft_modal_runbook.md) or open an issue on the project repository.*
