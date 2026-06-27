---
license: gpl-2.0
tags:
  - nanochat
  - tool-use
  - small-llm
  - research
  - pytorch
  - custom-architecture
language:
  - en
pipeline_tag: text-generation
library_name: nanochat
---

# Chinchilla-1-73M

**Chinchilla-1** is a ~73.5M parameter instruction-tuned language model built on the [nanochat](https://github.com/karpathy/nanochat) d6 architecture. The name references [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) — this is a **research-scale** model, not a production LLM.

This model was created as a learning exercise in SFT infrastructure (Modal, torchrun, MuonAdamW) and to explore nanochat's custom architecture (value embeddings, smear mechanism, backout residuals). It should be treated as a baseline for understanding how small transformers behave after SFT — not as a usable chatbot.

## What it does well

- **Tool-augmented reasoning** — correctly orchestrates `<|python_start|>` / `<|python_end|>` tokens to invoke Python for tasks like letter counting. Both checkpoints correctly emit `'strawberry'.count('r')` and interpret the result. See [evaluation Q7](evaluations/checkpoints/report.md#44-structured-tool-use-q7).
- **Basic Python generation** — produces correct, idiomatic code for simple algorithmic tasks (e.g. palindrome: `s == s[::-1]` with alphanumeric filtering). See [evaluation Q6](evaluations/checkpoints/report.md#43-code-generation-q6).
- **Chat format fluency** — consistently generates proper `<|user_start|>` / `<|assistant_start|>` conversation markers with no format breakages. Fluent, grammatically correct English prose. See [evaluation §5.3–5.4](evaluations/checkpoints/report.md#5-strengths).
- **Lightweight** — 73.5M params, runs on CPU at ~2 tok/s, fits in ~280 MB (bf16) or ~140 MB (fp16 via `NANOCHAT_DTYPE`).

## What it does poorly

This model has **severe limitations** that make it unsuitable for any factual or reasoning task without augmentation.

- **Pervasive hallucination** — fabricates facts with high confidence. Examples: claims Paris is "located in the northern United States" (Q1), invents a mountain called "Krasnarol" (Q8), makes up fictional US government branches like "Rentalist" and "Operationalist" (Q10). See [evaluation §6.1](evaluations/checkpoints/report.md#61-pervasive-hallucination).
- **Arithmetic failure** — cannot reliably perform multi-digit multiplication (15 × 37 was answered as 375 or avoided entirely). See [evaluation §6.2](evaluations/checkpoints/report.md#62-poor-arithmetic).
- **Poor instruction following** — ignores formatting constraints like "in one sentence" or "write a haiku", producing verbose multi-sentence paragraphs instead. See [evaluation §6.4](evaluations/checkpoints/report.md#64-poor-instruction-following).
- **Rambling** — responses degrade after ~50 tokens, becoming repetitive, self-contradictory, and off-topic. See [evaluation §6.3](evaluations/checkpoints/report.md#63-verbosity-and-rambling).
- **Factual knowledge is absent** — at 73.5M params the model lacks capacity to store reliable world knowledge. It falls back to plausible-sounding generation.

## Architecture

| Property | Value |
|----------|-------|
| Params | 73,531,646 |
| Layers | 6 |
| Embed dim | 384 |
| Attention heads | 6 (full MHA, no GQA) |
| Context (SFT) | 2048 tokens |
| Context (pretrain) | 512 tokens |
| Vocab | 32,768 BPE (rustbpe + tiktoken) |
| Activation | ReLU² (`relu(x)²`) |
| Position encoding | RoPE (rotary) |
| Norm | RMSNorm (manual implementation) |
| Window pattern | `"L"` — full causal attention on all layers |

Custom innovations (see [`ARCHITECTURE.md`](archive/nanochat_modal/docs/ARCHITECTURE.md) for full details):

- **Value embeddings** on alternating layers (layers 1, 3, 5) — full `32768 × 384` embedding tables injected into the residual stream via learned gates. Account for ~51% of total parameters (~37.7M of 73.5M).
- **Smear mechanism** — blends previous token's embedding into the current position for cheap bigram-like coherence.
- **Backout residual subtraction** — learns to subtract mid-layer features before the final LM head.
- **QK normalization** — RMSNorm on query and key vectors before attention.
- **Logit softcap** (`tanh` at 15) — prevents overconfident predictions.

**Not weight-tied.** Embedding and LM head are separate parameter matrices (2 × 12.6M params).

Parameter breakdown:

| Component | Parameters |
|-----------|------------|
| Token embedding | 12,582,912 |
| Value embeddings (3 layers) | 37,748,736 |
| LM head (unembedding) | 12,582,912 |
| 6× Transformer (attention + MLP) | ~10,616,832 |
| Scalars & gates | 254 |
| **Total** | **73,531,646** |

## Training

1. **Pretrain (base):** nanochat d6 on 11 FineWeb-EDU shards (~1 GB), 32,768 BPE merges, trained to step 8600 with 512 sequence length on Kaggle GPUs.
2. **SFT (this checkpoint):** Supervised fine-tuning on a mixture of SmolTalk (460K rows), MMLU (~100K), GSM8K (~7.5K), SimpleSpelling (200K), SpellingBee (80K), and identity conversations (1K). Trained on **Modal** with 2× A10G GPUs for ~35 minutes at ~$1.20 total.
3. **Knowledge distillation (optional):** The `model_001500.pt` checkpoint was trained with KD from [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM-40B) (cross-tokenizer KD — limitations documented in the [SFT report](archive/nanochat_modal/docs/SFT_REPORT.md)).

Hyperparameters: `max_seq_len=2048`, `total_batch_size=524,288` tokens, `device_batch_size=2` (KD run), AdamW for embeddings (LR 0.424) + Muon for matrix params (LR 0.02), linear warmup→constant→decay schedule over 1,500 steps.

## Usage

> ⚠️ **Not compatible with llama.cpp, Ollama, LM Studio, or HuggingFace Transformers' `from_pretrained`.** This model uses a custom nanochat architecture that requires the nanochat inference engine.

```bash
# Install dependencies
pip install torch tiktoken rustbpe

# Run inference
python inference.py --prompt "Write a Python function to reverse a string."
```

See [`inference.py`](inference.py) — it loads the checkpoint and tokenizer from this repo.

### Prompt format

```
<|bos|><|user_start|>Your message here<|user_end|><|assistant_start|>
```

Tool calls use `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>` tokens for Python REPL integration.

## Checkpoints included

| File | Description |
|------|-------------|
| `checkpoints/sft/model_001500.pt` | Best SFT checkpoint (KD run, step 1500, val_bpb 0.4763) |
| `checkpoints/sft/meta_001500.json` | Training metadata (config, metrics) |
| `checkpoints/tokenizer/tokenizer.pkl` | rustbpe tokenizer for tiktoken |

The alternative non-KD checkpoint (`sft-d6`, step 971, val_bpb 0.4891) is available on the [Modal volume](https://modal.com) archive if needed.

## Evaluation

A manual qualitative evaluation across 12 questions spanning factual knowledge, reasoning, math, code, tool use, and creative writing was conducted. Full results and example outputs in [`evaluations/checkpoints/report.md`](evaluations/checkpoints/report.md).

**Scorecard (sft at step 1500 vs non-KD checkpoint at step 971):**

| Metric | Count |
|--------|-------|
| sft wins | 2 |
| d6 wins | 3 |
| Ties | 7 |

The additional 529 SFT steps improved fluency modestly without improving factual accuracy. See the full [checkpoint evaluation report](evaluations/checkpoints/report.md) for detailed question-by-question analysis.

## Limitations

- **Do not use for any factual Q&A** without retrieval augmentation (RAG).
- **Do not rely on for math or arithmetic** — use the built-in Python tool instead.
- **Does not respect format instructions** reliably.
- **Rambles** on open-ended questions — consider setting `max_tokens=64` for production use.
- **73.5M parameters** is small even by 2020 standards (GPT-2 Small was 124M). Temper expectations accordingly.

## Citation

```bibtex
@misc{chinchilla1_73m,
  title={Chinchilla-1: A Research-Scale nanochat SFT Experiment},
  author={Yashraj Maher},
  year={2026},
  howpublished={\url{https://huggingface.co/rajofearth/Chinchilla-1-73M}}
}
```

## Acknowledgments

Built on [karpathy/nanochat](https://github.com/karpathy/nanochat) (MIT license). Training on [Modal](https://modal.com) cloud GPUs. SFT data includes SmolTalk by [HuggingFaceH4](https://huggingface.co/HuggingFaceH4), MMLU by [hendrycks](https://github.com/hendrycks/test), GSM8K by [OpenAI](https://github.com/openai/grade-school-math). The nanochat architecture and training recipe are the work of Andrej Karpathy and contributors.
