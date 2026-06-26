# NanoChat Model Architecture

## Overview

NanoChat is a **GPT-style decoder-only transformer** with several custom innovations inspired by ResFormer, sliding window attention, and modern training techniques. Our custom build uses a d6 configuration (6 layers, 384-dim) that totals ~73.5M parameters — roughly GPT-2 Small scale but with a significantly different internal design.

The model is built on [karpathy/nanochat](https://github.com/karpathy/nanochat) with modifications including value embeddings on alternating layers, a smear mechanism, backout residual connections, ReLU² activation, QK normalization, and logit softcap.

---

## GPTConfig Parameters

Defined in `nanochat/gpt.py` as a `@dataclass`:

| Parameter | Default | Our d6 Value | Description |
|-----------|---------|-------------|-------------|
| `sequence_len` | 2048 | 2048 | Maximum context length in tokens |
| `vocab_size` | 32768 | 32768 | Vocabulary size (BPE tokens) |
| `n_layer` | 12 | 6 | Number of transformer layers |
| `n_head` | 6 | 6 | Number of query attention heads |
| `n_kv_head` | 6 | 6 | Number of key/value heads (no GQA — full 6:6 ratio) |
| `n_embd` | 768 | 384 | Embedding / hidden dimension |
| `window_pattern` | `"SSSL"` | `"L"` | Sliding window attention pattern string, tiled across layers |

---

## Model Variants

NanoChat uses a single dial of complexity — **depth** (`n_layer`). The depth value automatically determines all other hyperparameters (width, number of heads, learning rates, training horizon) for compute-optimal training.

| Variant | Layers | Embed Dim | Total Params (approx) | Scale |
|---------|--------|-----------|----------------------|-------|
| d6 | 6 | 384 | ~73.5M | GPT-2 Small class |
| d12 | 12 | 768 | ~300M | GPT-1 scale |
| d26 | 26 | ~1,536 | ~1.6B | GPT-2 scale |

Our project uses **d6** exclusively — chosen for fast experimentation on 2× A10G GPUs.

---

## Layer-by-Layer Breakdown

### 1. Token Embedding + Norm

```
input_ids → Embedding(vocab_size=32768, n_embd=384) → norm(x)
```

The token embedding layer maps vocabulary indices (0–32767) to 384-dimensional vectors. Input token IDs first pass through the embedding lookup, then through **RMSNorm** (implemented manually for PyTorch 2.3 compatibility).

The embedding and LM head are **not weight-tied**; they are separate parameter matrices.

### 2. Smear Mechanism

```
x_smeared = smear(x) = x + smear_gate * norm(embed(input_ids_prev))
```

Before entering the transformer blocks, the model applies a **smear** mechanism: it mixes in the embedding of the **previous token** (position `t-1`) into the current token's representation (position `t`). This provides each token with explicit awareness of its immediate predecessor, improving coherence.

The smear is controlled by a learned scalar gate (initialized to 0), allowing the model to learn how much of the previous token's embedding to blend in.

### 3. Transformer Blocks (×6)

Each block follows this structure:

```
x_prev = x

# Attention sub-block
x = x + QK_norm(attention(norm(x)))

# Value embeddings (applied on alternating layers: 1, 3, 5)
x = x + value_embedding(input_ids) * ve_gate

# MLP sub-block
x = x + backout_residual(norm(x))

# Optional: backout residual subtraction
x = x - resid_lambda * x_prev
```

#### Attention

Each attention layer uses **multi-head attention** with:
- **QK normalization**: Query and key vectors are normalized before the attention dot product, improving training stability at higher learning rates.
- **Sliding window attention**: Controlled by `window_pattern`.
- **Rotary Position Embeddings (RoPE)**: Applied to query and key vectors for position encoding.

#### MLP

Each MLP layer uses:
- **ReLU² activation**: `relu(x)²` — squared ReLU activation, which provides sparser activations and better training dynamics compared to standard ReLU or GELU.
- Standard linear projection → activation → output projection.

#### QK Normalization

```
q = norm(q) * scale
k = norm(k) * scale
```

Query and key vectors are normalized using RMSNorm before the attention computation. This prevents the attention logits from growing too large and stabilizes training, especially with sliding window attention.

#### Value Embeddings (ResFormer-style)

On **alternating layers** (layers 1, 3, 5 in our 6-layer model), the model adds a **value embedding** — a full vocabulary-sized embedding table (`32768 × 384`) that projects the **input token IDs** directly into the residual stream. This is inspired by ResFormer, where each layer can inject token-specific information directly into the representation.

Each value embedding is gated by a learned scalar gate (initialized to 0), allowing the model to learn how much of each value embedding to include.

The value embeddings contribute the majority of the model's parameters (~37.7M of ~73.5M total).

### 4. Backout Residual Subtraction

After each transformer block, a learned scalar (`resid_lambda`) is used to subtract a fraction of the **input to the block** from its output:

```
x = x - resid_lambda * x_prev
```

This "backout" mechanism creates a residual pathway that can explicitly forget or dampen information from earlier layers, effectively creating a learnable high-pass filter on the residual stream.

### 5. Final Norm + LM Head with Logit Softcap

```
logits = norm(x) @ lm_head.weight.T
logits = logit_softcap * tanh(logits / logit_softcap)
```

After all transformer blocks:
- **RMSNorm**: Final normalization
- **LM Head**: Linear projection from `n_embd` (384) to `vocab_size` (32768) — produces logits over the vocabulary
- **Logit Softcap**: Applies `tanh` scaling to clamp logits to `[-logit_softcap, logit_softcap]`. This prevents the model from becoming overconfident and improves training stability with knowledge distillation.

---

## Attention: Sliding Window

The `window_pattern` string controls which layers use which attention pattern:

| Character | Pattern | Description |
|-----------|---------|-------------|
| `L` | Long | Full causal attention over all `sequence_len` tokens |
| `S` | Short | Quarter-context sliding window (`sequence_len / 4` tokens) |

The pattern is **tiled across layers** from input to output. For example, `"L"` means all layers use full context, `"SL"` alternates short/long, `"SSL"` uses two short then one long.

Our d6 model uses `"L"` — all 6 layers attend to the full 2048-token context. The linear sliding window pattern from the upstream nanochat is `"SSSL"` (repeated as needed), where 4-layer patterns cycle: short, short, short, long.

The linear sliding window reduces memory and computation from O(T²) to O(T·w) where w is the window size, while maintaining long-range communication through the `L` layers.

---

## Parameter Count Breakdown

The model has **73,531,646** total parameters (~73.5M):

| Component | Calculation | Parameters |
|-----------|------------|------------|
| Token embedding | `vocab_size × n_embd = 32768 × 384` | 12,582,912 |
| Value embeddings (3 layers) | `3 × 32768 × 384` | 37,748,736 |
| LM head (unembedding) | `384 × 32768` | 12,582,912 |
| 6× Transformer (attention + MLP) | ~10,616,832 total | ~10,616,832 |
| Other (scalars, VE gates, smear gate, resid_lambdas) | 254 scalars | 254 |
| **Total** | | **73,531,646** |

Key observation: **Value embeddings dominate** the parameter count. They account for ~51% of all parameters. Without value embeddings, the d6 model would be only ~28M parameters (the size quoted in the upstream nanochat docs for d6).

The significant investment in value embeddings is a deliberate design choice — it allows the small 6-layer model to effectively store and retrieve token-specific information at multiple points in the forward pass, partially compensating for the shallow depth.

---

## Tokenizer

- **Type**: BPE (Byte Pair Encoding) via tiktoken / RustBPE
- **Vocabulary size**: 32,768 tokens
- **Training data**: 11 FineWeb-EDU shards (~1GB)
- **Special tokens**:

| Token | Purpose |
|-------|---------|
| `<|endoftext|>` | BOS (Beginning of Sequence) — token ID 0 |
| `<|user_start|>` | User message delimiter |
| `<|user_end|>` | End of user message |
| `<|assistant_start|>` | Assistant message delimiter |
| `<|assistant_end|>` | End of assistant response |
| `<|python_start|>` | Tool call delimiter (code execution) |
| `<|python_end|>` | End of tool call |
| `<|output_start|>` | Tool output delimiter |
| `<|output_end|>` | End of tool output |

Conversation format:
```
bos + user_start + user_text + user_end + assistant_start + tokens (+ assistant_end)
```

---

## KV Cache for Inference

During autoregressive generation, the model caches key (K) and value (V) tensors for each attention layer to avoid recomputing past context:

```python
# The Engine caches KV pairs during generation
past_kv = self.kv_cache[l]  # Cached from previous steps
k = torch.cat([past_kv[0], k], dim=2)  # Append new keys
v = torch.cat([past_kv[1], v], dim=2)  # Append new values
```

The KV cache resides on the same device as the model (GPU or CPU). For the d6 model with 6 heads, 384-dim, and 2048 context, the cache uses approximately:
- Per layer: `2 × 2 × 6 × 2048 × 64` bytes (bf16) = ~3 MB
- Total (6 layers): ~18 MB
- With batch size 4: ~72 MB

The cache is flushed on each new generation call.
