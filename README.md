# Chinchilla

Supervised Fine-Tuning (SFT) of [karpathy/nanochat](https://github.com/karpathy/nanochat) language models — trained on [Modal](https://modal.com) cloud GPUs with ResFormer-style value embeddings, sliding window attention, and knowledge distillation.

**d6 model:** 6-layer, 384-dim, ~ 73.5M parameters. Quick to train (~$1.20 on 2× A10G), runs on CPU.

## Quick Links

| Resource | Description |
|---|---|
| [`nanochat_modal/README.md`](nanochat_modal/README.md) | Project setup, SFT pipeline, and commands |
| [`nanochat_modal/docs/ARCHITECTURE.md`](nanochat_modal/docs/ARCHITECTURE.md) | Model architecture and parameter breakdown |
| [`nanochat_modal/docs/TRAINING.md`](nanochat_modal/docs/TRAINING.md) | Full training guide: setup, data, hyperparameters |
| [`nanochat_modal/docs/SFT_REPORT.md`](nanochat_modal/docs/SFT_REPORT.md) | Technical report from the SFT run |
| [`nanochat_modal/docs/INFERENCE.md`](nanochat_modal/docs/INFERENCE.md) | Web server, CLI, and programmatic inference |
| [`nanochat_modal/docs/COMPARISON.md`](nanochat_modal/docs/COMPARISON.md) | Checkpoint comparison script usage |
| [`nanochat_modal/docs/DEVLOG.md`](nanochat_modal/docs/DEVLOG.md) | Development log, decisions, and bug history |

## Repository Structure

```
├── nanochat_modal/
│   ├── nanochat/            # Core library: GPT model, tokenizer, engine, scripts
│   ├── checkpoints/         # Trained checkpoints (pretrain, sft, sft-d6, tokenizer)
│   ├── training/            # Modal cloud training scripts
│   ├── evaluations/         # Evaluation results and analysis
│   └── docs/                # Comprehensive documentation
├── README.md                # This file
└── LICENSE                  # MIT
```
