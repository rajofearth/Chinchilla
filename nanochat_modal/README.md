# NanoChat SFT — Supervised Fine-Tuning of NanoChat Language Models

Supervised Fine-Tuning (SFT) pipeline running on [Modal](https://modal.com) cloud GPUs. Trains [karpathy/nanochat](https://github.com/karpathy/nanochat) models on a SmolTalk + MMLU + GSM8K mixture.

## Project Structure

```
nanochat_modal/
├── nanochat/               # Upstream karpathy/nanochat repo (Python package + scripts)
├── checkpoints/             # Model checkpoints: tokenizer, pretrain, sft
│   ├── tokenizer/
│   ├── pretrain/
│   └── sft/
├── evaluations/             # Evaluation results and checkpoint metadata
│   ├── checkpoints/         # Checkpoint JSON metadata + evaluation reports
│   └── 2026-06-26_sft-vs-d6/  # (created by the evaluation agent)
├── training/                # Modal SFT training scripts
│   ├── nanochat_sft.py      # Main Modal app
│   ├── run_sft.sh           # Shell wrapper for common operations
│   └── check_billing.py     # Modal billing checker
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # Model architecture and checkpoint guide
│   ├── TRAINING.md          # Training guide (setup, pipeline, evaluation)
│   ├── INFERENCE.md         # Web/CLI inference guide
│   ├── COMPARISON.md        # Checkpoint comparison guide
│   ├── SFT_REPORT.md        # Technical SFT report
│   └── SFT_MODAL_RUNBOOK.md # Full Modal runbook (also .docx)
└── old/                     # Staging for unclear items
```

## Quick Start

```bash
cd training/

# Run SFT synchronously (live logs)
modal run nanochat_sft.py::run_sft

# Run SFT detached (continues after terminal closes)
modal run --detach nanochat_sft.py

# Test inference on the trained checkpoint
modal run nanochat_sft.py::test_inference --prompt "Hello!"

# Download latest checkpoint from volume
modal run nanochat_sft.py::download_sft

# Run ChatCORE evaluation
modal run nanochat_sft.py::run_eval
```

## Model

The d6 checkpoint is a 6-layer, 384-dim GPT-style model with ~73.5M parameters. It uses:

- Sliding window linear attention (`"L"` pattern)
- ResFormer-style value embeddings on alternating layers
- Knowledge distillation during SFT (teacher: LFM2.5-350M)
- BPE tokenizer with 32,768 vocab and chat special tokens

## Documentation

Detailed documentation lives in `docs/`:

| Document | Description |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Model architecture, checkpoints, file structure, loading/running |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Full training guide: setup, data pipeline, hyperparameters, evaluation |
| [`docs/SFT_REPORT.md`](docs/SFT_REPORT.md) | Technical report from the SFT run |
| [`docs/SFT_MODAL_RUNBOOK.md`](docs/SFT_MODAL_RUNBOOK.md) | Modal runbook with troubleshooting and batch size math |

## Prerequisites

- A [Modal](https://modal.com) account with billing enabled
- Modal CLI installed and configured:

```bash
pip install modal
modal setup
```

- A Modal Volume named `nanochat-vol` containing:
  - Pretrain base checkpoint at `/base_checkpoints/d6/`
  - Tokenizer files at `/tokenizer/`

## License

MIT — same as [nanochat](https://github.com/karpathy/nanochat).
