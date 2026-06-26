# NanoChat Development Log & Decisions

A record of key decisions, milestones, bugs encountered, and future directions for the NanoChat SFT project.

---

## 2026-06 — Project Inception

### Starting Point

The project began from [karpathy/nanochat](https://github.com/karpathy/nanochat), a minimal full-stack ChatGPT clone. The upstream repo provides a complete pipeline for pretraining GPT-style models at various scales, with a focus on compute-optimal training.

### Model Choice: d6

**Decision:** Use the d6 model variant (6 layers, 384-dim).

**Rationale:** The d6 model is the smallest variant in nanochat's family. At ~73.5M parameters (with value embeddings), it's small enough to:
- Train quickly (~35 minutes on 2× A10G)
- Run inference on CPU or low-end GPUs
- Iterate rapidly on data mixtures and hyperparameters
- Keep costs low (~$1.20 per SFT run)

The trade-off is that d6 has inherently limited capacity for reasoning and knowledge — it's roughly 1/22nd the size of GPT-2 (1.6B params). The d26 variant (~1.6B params) would perform significantly better but costs ~$48 to train.

---

## Value Embeddings (ResFormer-style)

**Decision:** Retain and use value embeddings from the upstream nanochat.

Value embeddings are vocabulary-sized embedding tables (`32768 × 384`) applied on alternating layers (layers 1, 3, 5). They project input token IDs directly into the residual stream at those layers, providing token-specific information at multiple points in the forward pass.

**Impact:** Value embeddings dramatically increased the parameter count:
- Without value embeddings: ~28M parameters (d6)
- With value embeddings (3 layers): ~73.5M parameters

The ~37.7M additional parameters from value embeddings account for ~51% of the total model size. This is a deliberate trade-off — for a shallow 6-layer model, value embeddings provide multiple "injection points" for token-specific information, partially compensating for the lack of depth.

---

## Sliding Window Attention

**Decision:** Use `window_pattern="L"` (all layers use full context attention).

The upstream nanochat defaults to `"SSSL"` pattern (three short-window layers, one long-window layer) for efficiency. Our d6 model uses `"L"` for all layers — full causal attention over all 2048 tokens.

This choice prioritizes simple evaluation and debugging over efficiency. For production scaling to larger contexts, switching to a `"SSSL"` pattern would reduce memory and compute.

---

## Pretraining Checkpoint

### Step 8600 Base

The base pretrain checkpoint was trained on FineWeb-EDU for 8,600 steps with:
- Sequence length: 512 (pretrain phase)
- Total batch size: 524,288 tokens
- Device batch size: 32
- Val BPB: 0.9957

**Key constraint:** The pretrain context window is 512 tokens, but SFT training requires 2048 tokens to fit full conversations. The model architecture supports this increase (window_pattern applies to current sequence_len), but it does mean the model's position embeddings were learned for 512 positions and extrapolate to 2048.

---

## SFT Training

### Data Mixture Design

**Decision:** Combine SmolTalk, MMLU, GSM8K, spelling tasks, and identity conversations.

The mixture was designed to cover multiple capabilities:

| Component | Purpose |
|-----------|---------|
| SmolTalk (460K rows) | Conversational format learning |
| MMLU (~100K rows × 3 epochs) | Factual knowledge and MCQ answering |
| GSM8K (~7.5K rows × 4 epochs) | Math reasoning (tool use format) |
| SimpleSpelling (200K rows) | Letter-level token manipulation |
| SpellingBee (80K rows) | Spelling challenge with tool use |
| CustomJSON (1K rows × 2 epochs) | Identity/personality responses |

**Observation:** The mixture is heavily weighted toward spelling tasks (~26% of data), which may not be optimal. Future iterations should rebalance toward reasoning and factual content.

### Knowledge Distillation

**Decision:** Use LFM2.5-350M as teacher model with KD_ALPHA=0.9.

The teacher provides a richer supervision signal than raw token labels. With KD_ALPHA=0.9, the student loss is weighted at 90% and the distillation loss at 10%. The sorted top-k logit comparison (KD_TOP_K=512) handles cross-tokenizer alignment.

**Impact:** The sft checkpoint (with KD, step 1500, val_bpb=0.476) outperforms the d6 checkpoint (without KD, step 971, val_bpb=0.489), though the comparison is confounded by different training steps.

### Cross-Tokenizer Distillation Challenge

**Critical finding:** Direct online logit-level KD with LFM2.5-350M is blocked by a **vocabulary mismatch**. nanochat uses its own `rustbpe` tokenizer (32,768 vocab, trained on FineWeb-EDU) while LFM2.5-350M uses Liquid's own BPE tokenizer with a completely different vocabulary. Feeding nanochat token IDs into LFM2.5-350M would index into different vocabulary entries, making KL divergence over logits undefined.

Liquid avoided this because their teacher (LFM1-7B) and student (LFM2-230M) share the same tokenizer and architecture family.

#### Solutions Investigated

| Approach | Description | Viability |
|----------|-------------|-----------|
| **Online logit KD** | Pass same batch through both models, compute KL on logits | ❌ Blocked by vocab mismatch |
| **Offline data generation** | Run teacher on prompts, save completions as training data | ✅ Practical (~$3-5) |
| **ULD Loss** (Boizard et al., 2024) | Wasserstein distance across different vocabularies | ⚠️ Complex, hybrid distillation |
| **ALM** (Minixhofer et al., 2025) | Approximate Likelihood Matching via byte-level alignments | ⚠️ Promising but experimental |
| **Same-tokenizer KD** | Train larger nanochat model (d26) as teacher | ✅ Gold standard (~$48) |

#### Capacity Gap Research

Research on distillation scaling laws (Busbridge et al., 2025) shows an optimal teacher-student capacity ratio of roughly **linear with student scale**. For our 73.5M d6 model, the optimal teacher is in the ~150–300M parameter range:

| Teacher | Params | Ratio to d6 (73.5M) | Verdict |
|---------|--------|---------------------|---------|
| LFM2.5-350M | 350M | ~4.8× | ✅ Near-optimal zone |
| Qwen3.5-0.8B | 800M | ~10.9× | ⚠️ Borderline (mode averaging risk) |
| SmolLM2-360M | 360M | ~4.9× | ✅ Near-optimal zone |
| nanochat d26 | ~1.6B | ~22× | ❌ Needs TAID or TA chain |

Ratios exceeding 10× cause **mode averaging** — the student's limited capacity blurs together the teacher's distinct output modes (TAID paper, arXiv 2604.00626).

#### Practical Path Forward

The recommended approach is **offline hard-label distillation**: use LFM2.5-350M (or domain-split teachers like Qwen3.5-0.8B for math) to generate completions for training prompts, then train nanochat SFT on the teacher-generated data. This avoids the vocabulary mismatch entirely while still benefiting from teacher quality. See the [comparison guide](./COMPARISON.md) for how to evaluate improvements.

### Training Infrastructure: Modal Cloud

**Decision:** Move SFT training from Kaggle to Modal.

**Why:** Kaggle's free GPU quota ran out. Modal offers:
- Pay-per-use GPU access (A10G at ~$0.60/hr)
- Persistent volumes for checkpoints and data
- Reproducible container images
- Detached runs that survive terminal closure

---

## Bugs Encountered During Modal Setup

### Bug 1: `enable_gqa` Keyword Argument Error

- **Symptom:** `TypeError: scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'`
- **Root cause:** PyTorch 2.3.0 (in Modal's container) doesn't support `enable_gqa` — it was added in PyTorch 2.5.
- **Fix:** Strip the kwarg from `flash_attention.py` via string replacement.
- **Why safe:** d6 uses MHA (`n_head=6`, `n_kv_head=6`), not GQA. The flag has no effect.

### Bug 2: Missing `psutil` Module

- **Symptom:** `ModuleNotFoundError: No module named 'psutil'` at end of training.
- **Root cause:** `report.py` imports `psutil` but it wasn't in the Modal image dependencies.
- **Fix:** Added `"psutil"` to the `pip_install` list.

### Bug 3: `num_iterations` Counted Micro-Batches, Not Steps

- **Symptom:** Training stopped after 47 steps despite setting `--num-iterations=1500`.
- **Root cause:** `next(train_loader)` yields micro-batches, not optimizer steps. With `grad_accum_steps=32`, 1500 micro-batches = 47 optimizer steps.
- **Fix:** Changed NUM_ITERATIONS from 1500 to 48000 (1500 × 32).

### Bug 4: `load_model_from_dir` Path Mismatch

- **Symptom:** `FileNotFoundError: No checkpoints found in /vol/chatsft_checkpoints/d6`
- **Root cause:** The function expects a parent directory with a separate model tag, but received a full path.
- **Fix:** Pass parent directory and model tag separately.

### Bug 5: Stale `result` Variable Reference

- **Symptom:** `NameError: name 'result' is not defined` after SFT completed.
- **Root cause:** Variable renamed from `result` to `_result` in one place but return statement not updated.
- **Fix:** Updated return to reference `_result`.

### Bug 6: Indentation Error from Damaged Edit

- **Symptom:** Modal deployment failed with `IndentationError: unexpected indent`.
- **Root cause:** Stray characters introduced during iterative editing.
- **Fix:** Identified and removed the corrupted line.

---

## Compatibility Patches

Five patches are required for PyTorch 2.3 compatibility on Modal:

1. **`dataset.py`** — `base_data_climbmix` → `base_data` (path fix)
2. **`engine.py`** — Make dtype configurable via `NANOCHAT_DTYPE` env var
3. **`gpt.py`** — Manual RMSNorm (PyTorch < 2.4 doesn't have `F.rms_norm`)
4. **`chat_sft.py`** — `torch._dynamo.config.suppress_errors = True`
5. **`flash_attention.py`** — Strip `enable_gqa` kwarg

---

## Training Results

### Checkpoint Comparison

| Checkpoint | Steps | Val BPB | KD | Optimizer State |
|------------|-------|---------|----|-----------------|
| sft | 1,500 | 0.476 | Yes (lfm25_350m) | Yes (2 ranks) |
| d6 | 971 | 0.489 | No | No |

### Key Findings

**Strengths:**
- Both models learned the conversation format (proper `<|assistant|>` usage)
- Code generation works (palindrome function test)
- Spelling/tool use works (letter counting)

**Weaknesses:**
- **Hallucination**: Both models make up facts (wrong capitals, mountains)
- **Poor arithmetic**: 15 × 37 produces wrong answers
- **Verbosity**: Responses tend to ramble rather than being concise
- **Instruction following**: Format requirements (haiku, 1-sentence answers) often missed

**Interpretation:** This is expected behavior for a 73.5M parameter model with only 1 epoch of SFT. The model simply doesn't have enough capacity for complex reasoning or reliable factual recall.

---

## Training Performance

| Metric | Value |
|--------|-------|
| Step time | ~2.2 sec/step on 2× A10G |
| Throughput | ~240,000 tok/sec |
| Total training time | ~35 min |
| Peak GPU memory | ~4,866 MB per GPU |
| Cost | ~$1.20 |

---

## Data Exhaustion Issue

The training only completed **971 of 1,500 planned steps** because the data mixture was exhausted after one epoch (~1.07M rows consumed). To reach 1,500 steps, additional epochs of the mixture are needed (with reshuffling for variety), or more data must be added.

---

## Future Work

### Immediate (~$1/epoch)
- **Multiple training epochs**: Re-run SFT for 3–5 epochs to reinforce learning
- **Better data mixture**: Reduce spelling task proportion, add more math and code data
- **Enable ChatCORE during training**: Track benchmark scores throughout

### Medium-term (~$48)
- **Larger model (d26, ~1.6B params)**: Train on 8× H100 for GPT-2 scale capability
- **Extended pretraining**: Continue pretraining the d26 on FineWeb-EDU before SFT

### Long-term
- **RLHF / DPO**: Align model with human preferences after SFT
- **RL fine-tuning**: Use the nanochat `chat_rl.py` pipeline for RL training
- **Custom data generation**: Use the teacher model to generate higher-quality SFT data
- **Systematic evaluation**: Run full ChatCORE benchmarks on all checkpoints

---

## Architecture Version History

| Date | Change | Impact |
|------|--------|--------|
| — | Upstream nanochat d6 baseline | ~28M params, no value embeddings |
| — | Value embeddings added | ~73.5M params (+37.7M from VE layers) |
| — | Smear mechanism | Previous-token embedding mixing |
| — | Backout residual | Learnable residual subtraction |
| — | Logit softcap | Tanh-based logit clamping for stability |
| — | QK normalization | RMSNorm on QK for stable attention |
| — | ReLU² activation | Squared ReLU for sparser activations |

These architectural choices (value embeddings, smear, backout, softcap, QK norm) are inherited from the upstream nanochat research codebase. Our contribution is the SFT pipeline, Modal infrastructure, and the specific training configuration documented here.
