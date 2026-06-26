# NanoChat Training Guide

This document covers both pretraining and supervised fine-tuning (SFT) for the NanoChat d6 model.

---

## Pretraining

### Overview

Pretraining is **autoregressive language modeling** on the FineWeb-EDU dataset. The model learns to predict the next token given a sequence of previous tokens, building a general understanding of language, facts, and reasoning patterns.

The pretrained model is the starting point for SFT — it already "knows" language but doesn't follow instructions yet.

### Pretrain Checkpoint

We use the **step 8600** pretrain checkpoint as our base:

| Property | Value |
|----------|-------|
| Model | d6 (6 layers, 384 dim) |
| Steps | 8,600 |
| Sequence length | 512 (pretrain) |
| Val BPB | 0.9957 |
| Total batch size | 524,288 tokens |
| Device batch size | 32 |
| Training data | FineWeb-EDU |

This checkpoint is located at `checkpoints/pretrain/` with:
- `model_008600.pt` — model weights
- `meta_008600.json` — training metadata
- `optim_008600_rank*.pt` — optimizer states (per GPU rank)

Intermediate checkpoints at other steps (2000–8500) are also available in `checkpoints/pretrain/`.

---

## Supervised Fine-Tuning (SFT)

### Goal

Fine-tune the pretrained model on **instruction-following data** so it can respond to user requests in a conversational format. The model learns to produce useful, on-topic responses rather than merely continuing text.

### Data Mixture

The SFT dataset is a `TaskMixture` combining multiple sources:

| Component | Rows | Epochs | Total Rows Used | Description |
|-----------|------|--------|----------------|-------------|
| SmolTalk (train) | 460,000 | 1 | 460,000 | HuggingFace multi-turn conversation dataset |
| CustomJSON (identity) | 1,000 | 2 | 2,000 | Custom identity/personality conversations |
| MMLU (auxiliary_train) | ~100,000 | 3 | ~300,000 | Multiple-choice QA formatted as conversations |
| GSM8K (train) | ~7,473 | 4 | ~29,892 | Grade-school math word problems |
| SimpleSpelling | 200,000 | 1 | 200,000 | Word spelling tasks |
| SpellingBee | 80,000 | 1 | 80,000 | Letter counting / spelling challenge |
| **Total** | | | **~1,071,892** | |

Validation data:
- SmolTalk (test): 24,000 rows
- MMLU (test): ~5,200 rows (capped)
- GSM8K (test): ~420 rows (capped)
- **Total validation**: ~29,620 rows

### How to Run on Modal

The SFT pipeline lives in the `training/` directory:

```bash
cd training/

# Full pipeline (setup + training + inference test)
modal run nanochat_sft.py

# With KD disabled
modal run nanochat_sft.py --use-teacher=False

# Run SFT only (skip setup)
modal run nanochat_sft.py::run_sft

# Run detached (continues after terminal closes)
modal run --detach nanochat_sft.py

# Test inference on a trained checkpoint
modal run nanochat_sft.py::test_inference --prompt "Hello!"

# Download checkpoint from Modal volume
modal run nanochat_sft.py::download_sft

# Run ChatCORE evaluation
modal run nanochat_sft.py::run_eval
```

**Important torchrun `--` separator:** The `run_sft` command in `nanochat_sft.py` uses `"--"` to separate `torchrun` arguments from module arguments. This is critical to avoid the `--run` argument being consumed by `torchrun` (which interprets `--run` as a torchrun flag). Always include the separator when constructing SFT commands manually.

Using the shell wrapper:

```bash
./run_sft.sh run          # synchronous SFT (live logs)
./run_sft.sh detach       # detached SFT (background)
./run_sft.sh logs         # tail logs from latest/active run
./run_sft.sh infer "Hi!"  # test inference
./run_sft.sh status       # inspect volume contents
./run_sft.sh download     # download checkpoint
./run_sft.sh eval         # run ChatCORE evaluation
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MAX_SEQ_LEN` | 2048 | Context window — conversations average 800–1400 tokens |
| `DEVICE_BATCH_SIZE` | 2 (with KD) / 4 (without) | Per-GPU micro-batch size |
| `TOTAL_BATCH_SIZE` | 524,288 tokens | Effective batch across all GPUs and grad accum |
| `NUM_ITERATIONS` | 1,500 | Optimizer steps (48,000 micro-batches) |
| `INIT_LR_FRAC` | 0.1 | Starting LR as fraction of base LR |
| `WARMUP_RATIO` | 0.1 | First 10% of steps linearly ramp up LR |
| `WARMDOWN_RATIO` | 0.5 | Last 50% of steps decay LR to 0 |
| `FINAL_LR_FRAC` | 0.0 | Final LR fraction |
| `EVAL_EVERY` | 200 steps | Run validation every N steps |
| `EVAL_TOKENS` | 131,072 | Validation tokens per eval |
| `GRAD_ACCUM_STEPS` | 32 | Micro-batches per optimizer step (with device_batch=4) |
| `LOAD_OPTIMIZER` | 0 | Start fresh optimizer (do NOT load pretrain optim state) |

#### Knowledge Distillation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `KD_ALPHA` | 0.9 | Weight: `loss = KD_ALPHA × student_loss + (1-KD_ALPHA) × distillation_loss` |
| `KD_TEMPERATURE` | 2.0 | Softmax temperature for distillation |
| `KD_TOP_K` | 512 | Number of top logits to compare between teacher and student |
| Teacher model | `lfm25_350m` | ~350M parameter teacher from `/vol/teacher_models/` |

### Batch Size Math

```
world_tokens_per_microbatch = device_batch × max_seq_len × num_gpus
                            = 4 × 2048 × 2
                            = 16,384 tokens

grad_accum_steps = total_batch_size / world_tokens_per_microbatch
                 = 524,288 / 16,384
                 = 32 (exact)

total_tokens_per_step = 524,288
```

### LR Schedule

```
Linear warmup (10%) → Constant (40%) → Linear decay to 0 (50%)
```

- **Warmup**: Steps 0–150, LR ramps from 0 to target
- **Constant**: Steps 150–750, LR stays at target
- **Warmdown**: Steps 750–1,500, LR decays linearly to 0

### Optimizer: Combined MuonAdamW

NanoChat uses a hybrid optimizer strategy:

| Parameter Group | Optimizer | Base LR | Notes |
|----------------|-----------|---------|-------|
| Embedding & unembedding weights | AdamW | 0.3 | Scaled by `√(768/n_embd)` |
| Matrix params (all linear layers) | Muon | 0.02 | Orthogonalized updates |
| Scalars (gates, biases, norms) | AdamW | Various | Learned scalars |

**Muon** (from Keller Jordan) applies orthogonalized updates to matrix parameters, which is more efficient than AdamW for large linear layers. **AdamW** is used for the embedding/unembedding tables where orthogonality is less beneficial.

---

## Patches Applied for Modal Compatibility

When running on Modal, `setup_nanochat()` applies 5 patches to the upstream nanochat repo:

### 1. `dataset.py` — Data Directory Path
- **Problem**: Hardcodes `base_data_climbmix` path
- **Fix**: Replace with `base_data`
```python
content = content.replace("base_data_climbmix", "base_data")
```

### 2. `engine.py` — Configurable Dtype
- **Problem**: Hardcodes `torch.bfloat16` for CUDA, incompatible with T4
- **Fix**: Make dtype configurable via `NANOCHAT_DTYPE` env var

### 3. `gpt.py` — Manual RMSNorm
- **Problem**: `F.rms_norm` was added in PyTorch 2.4, not available in 2.3
- **Fix**: Replace with manual RMSNorm:
```python
def norm(x):
    var = x.pow(2).float().mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + 1e-6)).to(x.dtype)
```

### 4. `chat_sft.py` — Suppress Dynamo Errors
- **Problem**: `torch.compile` fails on unsupported ops in PyTorch 2.3
- **Fix**: Set `torch._dynamo.config.suppress_errors = True`

### 5. `flash_attention.py` — Strip `enable_gqa` Kwarg
- **Problem**: `enable_gqa` parameter requires PyTorch ≥ 2.5
- **Fix**: Strip the kwarg — safe since d6 uses MHA (n_head == n_kv_head)
```python
content = content.replace(", enable_gqa=enable_gqa", "")
```

---

## Infrastructure

| Component | Detail |
|-----------|--------|
| Cloud provider | [Modal](https://modal.com) |
| GPU | 2× A10G (24 GB VRAM each) |
| Container | debian-slim + Python 3.11 + PyTorch 2.3.0 (cu121) |
| Training time | ~35 minutes |
| Cost | ~$1.20 (A10G at ~$0.60/hr × 2 GPUs × 35 min) |
| Volume | `nanochat-vol` mounted at `/vol` |
| Checkpoint path (on volume) | `/vol/chatsft_checkpoints/d6/` |

### GPU Configuration Comparison

| GPU | VRAM | bf16 | ~Cost/hr | Notes |
|-----|------|------|----------|-------|
| T4 × 2 | 2× 15 GB | No | ~$0.15 | Needs `NANOCHAT_DTYPE=float32` |
| **A10G × 2** (recommended) | 2× 24 GB | Yes | ~$0.60 | Best value for d6 |
| A100 × 2 | 2× 40/80 GB | Yes | ~$2.00 | Overkill for d6, good for larger models |

### Batch Size Valid Configurations

All configs produce `total_batch_size = 524,288`:

| GPU Setup | device_batch | seq_len | num_gpus | grad_accum |
|-----------|-------------|---------|----------|------------|
| T4 × 2 (float32) | 2 | 2048 | 2 | 64 |
| T4 × 2 (float32) | 4 | 1024 | 2 | 64 |
| **A10G × 2 (bf16)** | **4** | **2048** | **2** | **32** |
| A10G × 2 (bf16) | 8 | 2048 | 2 | 16 |
| A100 × 2 (bf16) | 8 | 2048 | 2 | 16 |

---

### Curriculum Learning Strategy

Inspired by Liquid AI's approach in LFM2.5, a curriculum ordering can improve learning efficiency for small models. Instead of feeding all data in a random mix, stage the training:

1. **First third of training**: Only SmolTalk (general conversation format) + MMLU (knowledge breadth) — teaches conversation structure first
2. **Second third**: Add GSM8K (math reasoning), increase SmolTalk epochs — introduces reasoning after format is learned
3. **Final third**: Specialist tasks (spelling), GSM8K hard examples — adds precision tasks last

This is particularly valuable for small models like d6 that benefit from focused learning phases. The current nanochat `TaskMixture` doesn't natively support staged curriculum — you'd need to run separate SFT passes with different data mixtures.

## Ways to Improve

### More Training Epochs (~$1/epoch)
The current run completed only 971 of 1,500 planned steps because the data mixture exhausted. Running additional SFT passes would reinforce learning.

### Larger Model (d26, ~1.6B Params, ~$48)
The d6 model (73.5M params) is fundamentally limited. The d26 model (~1.6B params) is what nanochat targets for GPT-2 scale capability. Training on 8× H100 would take ~2 hours.

### Better Data Mixture
The current mixture has ~26% spelling tasks. A revised mixture that avoids data exhaustion and improves signal:

| Component | Current | Proposed | Change |
|-----------|---------|----------|--------|
| SimpleSpelling | 200K rows × 1 epoch | 80K rows × 1 epoch | −60% |
| SpellingBee | 80K rows × 1 epoch | 40K rows × 1 epoch | −50% |
| SmolTalk | 460K rows × 1 epoch | 460K rows × 3 epochs | +3× |
| GSM8K | ~7.5K rows × 4 epochs | ~7.5K rows × 8 epochs | +2× |
| **Total** | **~1.07M rows** | **~1.05M rows** | ≈ same size |

This doubles the math signal, reduces spelling noise, and importantly avoids data exhaustion at step 971 — enabling the full 1,500-step training plan to complete.

### Enable ChatCORE During Training
Set `--chatcore-every=500` to get per-task benchmark scores throughout training.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| NaN loss from step 1 | Pretrain optimizer state loaded | Pass `--load-optimizer=0` |
| ZERO assistant tokens | Sequence length too short | Use `--max-seq-len=2048` |
| `FileNotFoundError: base_data` | Dataset path not patched | Run `setup_nanochat()` or apply the sed replacement |
| `enable_gqa` error | PyTorch < 2.5 | Apply `flash_attention.py` patch |
| `psutil` error | Missing dependency | Add `psutil` to image dependencies |
| NCCL timeout | Rank mismatch from zero-token batch skip | Use `--load-optimizer=0` and `seq_len=2048` |
| AssertionError on batch size | total_batch_size not divisible | Use a valid configuration from the table above |

### Debugging Tips

- **Enable C++ stack traces**: Set `TORCH_SHOW_CPP_STACKTRACES=1` in the environment to get detailed CUDA error traces instead of silent failures.
- **Capture torchrun output**: In the Modal training script, use `subprocess.run(cmd, capture_output=True, text=True)` and print the last 5000 characters of stdout/stderr — otherwise torchrun swallows error messages.
- **Verify assistant tokens**: Run `ids, mask = tokenizer.render_conversation(ex); print(sum(mask[:2048]))` to check that assistant tokens are present in the first 2048 positions.
