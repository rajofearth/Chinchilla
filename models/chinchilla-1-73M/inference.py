"""
Minimal inference script for Chinchilla-1-73M.

Loads the checkpoint + tokenizer from this repository and runs
autoregressive generation with the nanochat engine.

Usage:
    python inference.py --prompt "Write a Python function to reverse a string."
    python inference.py --prompt "Hello!" --temperature 0.7 --max-tokens 128

Requirements:
    pip install torch tiktoken rustbpe
"""

import argparse
import json
import os
import pickle
import sys

import torch

# ── Paths relative to this repo ────────────────────────────────────────────
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(REPO_DIR, "checkpoints", "sft")
TOKENIZER_DIR = os.path.join(REPO_DIR, "checkpoints", "tokenizer")
META_PATH = os.path.join(CHECKPOINT_DIR, "chinchilla-1-73M.json")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "chinchilla-1-73M.pt")
TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")

# ── Nanochat source (bundled in repo) ──────────────────────────────────────
NANOCHAT_SRC = os.path.join(REPO_DIR, "nanochat")
if os.path.isdir(NANOCHAT_SRC):
    sys.path.insert(0, NANOCHAT_SRC)
else:
    raise FileNotFoundError(
        f"nanochat source not found at {NANOCHAT_SRC}. "
        "Please clone https://github.com/karpathy/nanochat into this directory."
    )

from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer


def load_tokenizer(tokenizer_dir: str) -> RustBPETokenizer:
    """Load the rustbpe tokenizer from the saved pickle."""
    if not os.path.isfile(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    return RustBPETokenizer.from_directory(tokenizer_dir)


def load_model(checkpoint_dir: str, device: torch.device):
    """Load model weights and metadata from a nanochat checkpoint directory."""
    with open(META_PATH) as f:
        meta = json.load(f)

    model_config_kwargs = meta["model_config"]
    # Patch missing keys that old checkpoints may lack
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"

    model_config = GPTConfig(**model_config_kwargs)
    model_data = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    # Remove torch.compile prefix if present
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model on meta device then load weights
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    # Convert bf16 -> fp32 for CPU inference
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()
    return model, meta


def build_prompt(tokenizer: RustBPETokenizer, user_text: str) -> list[int]:
    """Build a chat-format prompt: <bos><|user_start|>text<|user_end|><|assistant_start|>"""
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    user_ids = tokenizer.encode(user_text)
    return [bos, user_start] + user_ids + [user_end, assistant_start]


def main():
    parser = argparse.ArgumentParser(description="Chinchilla-1-73M inference")
    parser.add_argument("--prompt", type=str, default="Hello!")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype_str = "bfloat16"
    else:
        device = torch.device("cpu")
        dtype_str = "float32"
    print(f"Device: {device} ({dtype_str})", file=sys.stderr)

    # Load
    print(f"Loading tokenizer from {TOKENIZER_PATH}...", file=sys.stderr)
    tokenizer = load_tokenizer(TOKENIZER_DIR)

    print(f"Loading model from {MODEL_PATH}...", file=sys.stderr)
    model, meta = load_model(CHECKPOINT_DIR, device)
    print(f"Model: {meta['model_config']}", file=sys.stderr)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}", file=sys.stderr)

    # Prompt
    prompt_ids = build_prompt(tokenizer, args.prompt)
    print(f"\nUser: {args.prompt}", file=sys.stderr)
    print("Assistant: ", end="", flush=True)

    # Generate
    engine = Engine(model, tokenizer)
    stream = engine.generate(
        prompt_ids,
        num_samples=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    for token_column, _ in stream:
        token = token_column[0]
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
