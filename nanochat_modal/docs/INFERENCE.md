# NanoChat Inference Guide

NanoChat provides multiple ways to run inference: a web server with a chat UI, a CLI interactive chat, evaluation benchmarks, and programmatic API access.

---

## Web Server

The web server (`scripts/chat_web.py`) launches a FastAPI server with a built-in chat UI and an OpenAI-compatible streaming API.

### Quick Start

```bash
# Set the base directory (where checkpoints/ and tokenizer/ live)
export NANOCHAT_BASE_DIR=/mnt/p/temp-file-share/nanochat_modal/checkpoints

# Launch with the sft checkpoint
uv run python -m scripts.chat_web -i sft

# Or with custom options
uv run python -m scripts.chat_web -i sft -t 0.8 -m 512 -p 8000
```

The server automatically detects GPU availability — it will use CUDA if available, otherwise falls back to CPU.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI (HTML/CSS/JS single-page frontend) |
| `/chat/completions` | POST | Streaming chat completions (OpenAI-compatible) |
| `/health` | GET | Health check with worker pool status |
| `/stats` | GET | Worker pool statistics and GPU utilization |

The API supports **streaming only** via Server-Sent Events.

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --source` | `sft` | Model source: `sft`, `rl`, or `base` |
| `-n, --num-gpus` | 1 | Number of GPUs for data parallelism |
| `-t, --temperature` | 0.8 | Default sampling temperature |
| `-k, --top-k` | 50 | Default top-k sampling |
| `-m, --max-tokens` | 512 | Default max generation tokens |
| `-p, --port` | 8000 | Server port |
| `-s, --step` | (latest) | Specific checkpoint step to load |

### Abuse Prevention Limits

| Limit | Value |
|-------|-------|
| Max messages per request | 500 |
| Max characters per message | 8,000 |
| Max total conversation length | 32,000 characters |
| Temperature range | 0.0 – 2.0 |
| Top-k range | 0 – 200 (0 = full vocabulary) |
| Max tokens range | 1 – 4,096 |

### How `NANOCHAT_BASE_DIR` Maps Sources

| Source | Subdirectory under `NANOCHAT_BASE_DIR` |
|--------|--------------------------------------|
| `sft` | `{base_dir}/sft/` |
| `rl` | `{base_dir}/chatrl_checkpoints/` |
| `base` | `{base_dir}/base_checkpoints/` |

If `NANOCHAT_BASE_DIR` is not set, it defaults to `~/.cache/nanochat/`.

### Example API Call

```python
import requests

response = requests.post(
    "http://localhost:8000/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": True,
    },
    stream=True,
)

for chunk in response:
    print(chunk.decode(), end="")
```

---

## CLI Chat

The interactive terminal chat (`scripts/chat_cli.py`) provides a conversation interface in your terminal.

```bash
uv run python -m scripts.chat_cli
```

You can also send a single prompt:

```bash
uv run python -m scripts.chat_cli -p "What is the capital of France?"
```

---

## Programmatic API

### Quick Start

```python
from nanochat.checkpoint_manager import build_model
from nanochat.engine import Engine
import torch

# Load model and tokenizer
checkpoint_dir = "/mnt/p/temp-file-share/nanochat_modal/checkpoints/sft"
step = 1500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer, meta = build_model(checkpoint_dir, step, device, "eval")

# Create inference engine
engine = Engine(model, tokenizer)

# Build conversation tokens
def build_conversation_tokens(tokenizer, user_message):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")

    tokens = [bos]
    tokens.append(user_start)
    tokens.extend(tokenizer.encode(user_message))
    tokens.append(user_end)
    tokens.append(assistant_start)
    return tokens

tokens = build_conversation_tokens(tokenizer, "Hello! How are you?")

# Generate response (streaming)
assistant_end = tokenizer.encode_special("<|assistant_end|>")
accumulated = []

for token_column, token_masks in engine.generate(
    tokens,
    num_samples=1,
    max_tokens=512,
    temperature=0.7,
    top_k=50,
    seed=42,
):
    token = token_column[0]
    if token in (assistant_end, tokenizer.get_bos_token_id()):
        break
    accumulated.append(token)

response = tokenizer.decode(accumulated)
print(response)
```

### Generation Parameters

| Parameter | Effect |
|-----------|--------|
| `temperature=0.0` | Greedy/deterministic — best for evaluation |
| `temperature=0.7` | Balanced — good for chat |
| `temperature=0.9` | Creative — more diverse outputs |
| `top_k=50` | Default; limits sampling to top 50 logits |
| `max_tokens=256` | Maximum tokens to generate |
| `num_samples=1` | Number of parallel completions |
| `seed=42` | Random seed for reproducibility |

---

## Conversation Format

The model uses special tokens to delimit conversation turns:

```
<|endoftext|>                ← BOS token (always at the start)
<|user_start|>user message<|user_end|>
<|assistant_start|>assistant response<|assistant_end|>
```

For multi-turn conversations:

```
<|endoftext|>
<|user_start|>What's 2+2?<|user_end|>
<|assistant_start|>4<|assistant_end|>
<|user_start|>What's 3+5?<|user_end|>
<|assistant_start|>8<|assistant_end|>
```

The model is trained to predict all tokens, but during inference, we only generate tokens after `<|assistant_start|>` and stop at `<|assistant_end|>` (or BOS as a fallback).

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<|endoftext|>` | Beginning of sequence (BOS) |
| `<|user_start|>` / `<|user_end|>` | User message delimiters |
| `<|assistant_start|>` / `<|assistant_end|>` | Assistant message delimiters |
| `<|python_start|>` / `<|python_end|>` | Tool call (code execution) delimiters |
| `<|output_start|>` / `<|output_end|>` | Tool output delimiters |

### Tokenizer API

```python
from nanochat.checkpoint_manager import get_tokenizer

tokenizer = get_tokenizer("/path/to/tokenizer")

# Encode/decode
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")
```

---

## Engine Internals

The `Engine` class (`nanochat/engine.py`) provides:

- **KV Cache**: Caches key/value tensors during autoregressive generation for efficiency
- **Parallel sampling**: Generates multiple completions with `num_samples`
- **Tool use state machine**: Integrates Python code execution with a calculator
- **Configurable generation**: temperature, top-k, max tokens, stop tokens

```python
class Engine:
    def generate(self, prompt_tokens, num_samples=1, max_tokens=256, 
                 temperature=0.7, top_k=50, seed=42):
        """Generator that yields (token_column, token_masks) for each step."""
        ...

    def generate_batch(self, prompt_tokens, num_samples=1, max_tokens=256,
                       temperature=0.0, top_k=50):
        """Batch generation for evaluation. Returns (result_tokens, token_masks)."""
        ...
```

Use `generate()` for streaming (token by token) and `generate_batch()` for batched evaluation (returns all tokens at once).

---

## Benchmark Evaluation

Run standardized benchmarks with `scripts/chat_eval.py`:

```bash
export NANOCHAT_BASE_DIR=/mnt/p/temp-file-share/nanochat_modal/checkpoints

# Single task
uv run python -m scripts.chat_eval -i sft -a ARC-Easy -x 50

# Multiple tasks
uv run python -m scripts.chat_eval -i sft -a "ARC-Easy|MMLU|GSM8K"

# All tasks (default)
uv run python -m scripts.chat_eval -i sft -a "ARC-Easy|ARC-Challenge|MMLU|GSM8K|HumanEval|SpellingBee"
```

### Benchmarks

| Task | Type | Description | Random Baseline |
|------|------|-------------|-----------------|
| ARC-Easy | Categorical | Grade-school science MC (Easy) | 25% |
| ARC-Challenge | Categorical | Grade-school science MC (Challenge) | 25% |
| MMLU | Categorical | Massive Multitask Language Understanding | 25% |
| GSM8K | Generative | Grade-school math word problems | 0% |
| HumanEval | Generative | Python code synthesis | 0% |
| SpellingBee | Generative | Letter counting / spelling tasks | 0% |

**Categorical** tasks compare log-probabilities of answer choices (fast, deterministic).  
**Generative** tasks sample completions and check correctness (slower, stochastic).

### Evaluation Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --source` | required | Model source: `sft` or `rl` |
| `-a, --task-name` | all | Task(s) to evaluate |
| `-x, --max-problems` | all | Limit number of problems |
| `-t, --temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `-n, --num-samples` | 1 | Samples per problem (generative tasks) |
| `-k, --top-k` | 50 | Top-k sampling |
| `-b, --batch-size` | 8 | Batch size (categorical tasks) |
| `-s, --step` | latest | Specific step to load |
