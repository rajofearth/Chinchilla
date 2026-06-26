# Checkpoint Comparison Guide

This guide explains how to compare two NanoChat checkpoints side-by-side using the `compare_checkpoints.py` script.

---

## Overview

The comparison script loads two different checkpoint versions, runs the same set of questions through both models, and prints the responses with timing information. This is useful for:

- **Regression testing** — Did a training run make things better or worse?
- **Iterative improvement** — Compare SFT vs RL checkpoints, or different training steps
- **Qualitative analysis** — See where models differ in reasoning, tone, and accuracy

---

## Usage

### Basic Command

```bash
uv run python -m scripts.compare_checkpoints \
    --ckpt1 /mnt/p/temp-file-share/nanochat_modal/checkpoints/sft \
    --ckpt2 /mnt/p/temp-file-share/nanochat_modal/nanochat/chatsft_checkpoints/d6 \
    --tokenizer /mnt/p/temp-file-share/nanochat_modal/checkpoints/tokenizer
```

This loads the `sft` checkpoint (step 1500) and the `d6` checkpoint (step 971) and runs the default 12 questions through both.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt1 PATH` | required | First checkpoint directory (must contain `model_*.pt`) |
| `--ckpt2 PATH` | required | Second checkpoint directory (must contain `model_*.pt`) |
| `--tokenizer PATH` | required | Tokenizer directory (must contain `tokenizer.pkl` and `token_bytes.pt`) |
| `--questions FILE` | (built-in 12) | JSON file with a list of question strings |
| `--max-tokens N` | 256 | Max tokens to generate per response |
| `--temperature T` | 0.7 | Sampling temperature |
| `--top-k N` | 50 | Top-k sampling |
| `--device TYPE` | autodetect | `cuda`, `cpu`, or `mps` |

### Custom Questions File

Create a JSON file with your own questions:

```json
[
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
    "What is the difference between ai and ml?"
]
```

Then pass it:

```bash
uv run python -m scripts.compare_checkpoints \
    --ckpt1 checkpoints/sft \
    --ckpt2 nanochat/chatsft_checkpoints/d6 \
    --tokenizer checkpoints/tokenizer \
    --questions my_questions.json
```

### Default Questions

The 12 built-in questions cover diverse categories:

| # | Question | Category |
|---|----------|----------|
| 1 | What is the capital of France? | Factual |
| 2 | Explain why the sky is blue in one paragraph. | Science |
| 3 | Write a short poem about artificial intelligence. | Creative |
| 4 | What is 15 × 37? | Math |
| 5 | Explain the difference between a CPU and a GPU in one sentence. | Technical |
| 6 | Write a Python function that checks if a string is a palindrome. | Coding |
| 7 | How many 'r' are in the word 'strawberry'? | Tool use / spelling |
| 8 | What is the tallest mountain in the world? | Factual |
| 9 | Explain quantum computing in simple terms. | Science |
| 10 | What are the three branches of the US government? | Civics |
| 11 | Write a haiku about programming. | Creative |
| 12 | What is the meaning of life? | Philosophy |

---

## How the Script Works Internally

### Loading

1. **Tokenizer**: Sets `NANOCHAT_BASE_DIR` to the parent of the tokenizer directory so `get_tokenizer()` can find `tokenizer.pkl`.
2. **Checkpoints**: Scans each checkpoint directory for `model_*.pt` files, picks the one with the highest step number, and calls `build_model()` to load it.
3. **Device**: Auto-detects GPU availability, or uses the specified device.

### Inference

For each question, the script:

1. Builds the conversation format:
   ```
   bos + user_start + question + user_end + assistant_start
   ```
2. Runs `engine.generate()` in a streaming loop, collecting tokens until:
   - `<|assistant_end|>` is generated (normal completion)
   - BOS token is generated (fallback stopping signal)
   - `max_tokens` is reached
3. Decodes the token sequence back to text.

### Output

The script prints a side-by-side comparison for each question:

```
──────────────────────────────────────────────────────────────────────
Q1: What is the capital of France?
──────────────────────────────────────────────────────────────────────

  ┌─ [sft - step 1500]  (1.2s)
  │ The capital of France is Paris.

  └─

  ┌─ [d6 - step 971]  (1.5s)
  │ Paris is the capital of France.

  └─
```

Each response shows:
- **Label**: Which checkpoint and training step
- **Timing**: Generation time in seconds
- **Response**: The model's output (indented for readability)

---

## Reading the Results

When comparing outputs, look for:

| Signal | What It Means |
|--------|--------------|
| Shorter, more relevant response | Better instruction following |
| Correct facts | Better knowledge retention |
| Proper format (poem, code, etc.) | Better format learning |
| Faster generation | Same architecture — variance from GPU throttling |
| Coherent multi-sentence structure | Better language modeling |
| No repetition / looping | Better training quality |

### Known Baseline (sft vs d6)

From our comparison runs:

| Capability | Assessment |
|------------|-----------|
| Chat format | ✓ Both learned — proper `<|assistant|>` responses |
| Code generation | ✓ Both produce working Python code |
| Spelling / tool use | ✓ Both handle "count the r in strawberry" correctly |
| Math reasoning | ✗ Both fail at arithmetic (15 × 37) |
| Factual knowledge | ✗ Both hallucinate (wrong capitals, wrong mountains) |
| Instruction following | ~ Partial — haiku format, concise answers often missed |

---

## Using Comparisons for Iterative Improvement

### Workflow

1. **Baseline**: Run comparison on your current best checkpoint
2. **Change**: Modify training data, hyperparameters, or model architecture
3. **Train**: Produce a new checkpoint
4. **Compare**: Run the same questions against old vs new
5. **Evaluate**: Decide if the changes helped or hurt

### Example: Testing a Training Improvement

```bash
# Compare the old checkpoint vs a new one
uv run python -m scripts.compare_checkpoints \
    --ckpt1 checkpoints/sft/old \
    --ckpt2 checkpoints/sft/new \
    --tokenizer checkpoints/tokenizer \
    --temperature 0.7 \
    --max-tokens 256

# Also run at temperature 0.0 for deterministic comparison
uv run python -m scripts.compare_checkpoints \
    --ckpt1 checkpoints/sft/old \
    --ckpt2 checkpoints/sft/new \
    --tokenizer checkpoints/tokenizer \
    --temperature 0.0 \
    --max-tokens 256
```

At `temperature=0.0`, outputs are deterministic (greedy decoding), making it easier to see whether the model's preferred answer changed.

### Automated Evaluation

For quantitative comparison in addition to qualitative, run the benchmark suite:

```bash
export NANOCHAT_BASE_DIR=/mnt/p/temp-file-share/nanochat_modal/checkpoints

# Old checkpoint
uv run python -m scripts.chat_eval -i sft -a "ARC-Easy|MMLU|GSM8K"

# New checkpoint
uv run python -m scripts.chat_eval -i rl -a "ARC-Easy|MMLU|GSM8K"
```

Combine quantitative benchmark scores with qualitative comparison responses for a complete picture.
