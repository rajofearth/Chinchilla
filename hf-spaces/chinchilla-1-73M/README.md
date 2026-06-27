---
title: Chinchilla-1-73M Demo
emoji: 🐹
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
license: gpl-2.0
---

# 🐹 Chinchilla-1-73M Demo

A ~73.5M parameter instruction-tuned language model running entirely on CPU.
Built on the [nanochat](https://github.com/karpathy/nanochat) d6 architecture.

> **⚠️ Research-scale model — not a usable chatbot.** Temper expectations accordingly.
> At 73.5M parameters this is smaller than GPT-2 Small (124M). It hallucinates,
> struggles with arithmetic, and rambles beyond ~50 tokens.

### Links

- [Model on Hugging Face Hub](https://huggingface.co/rajofearth/Chinchilla-1-73M)
- [Training code & SFT report](https://github.com/rajofearth/chinchilla)

### Usage Tips

- Keep prompts short and specific
- Set **Max Tokens ≤ 64** to avoid rambling
- The model supports Python tool calls via `<|python_start|>` tokens
- **Not compatible** with llama.cpp, Ollama, or HuggingFace Transformers
