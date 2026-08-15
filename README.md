# Chinchilla — Model Training Project

![MIT License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

This repo houses **two model tracks** trained on [Modal](https://modal.com) cloud GPUs:

| Track | Model | Size | Stage |
|-------|-------|------|-------|
| **Chinchilla-1** | SFT of [karpathy/nanochat](https://github.com/karpathy/nanochat) | 73M params | ✅ Published to HF |
| **Mars-1.0** | Agent finetune of [LiquidAI/LFM2.5-8B-A1B-Base](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base) | ~8B (active) | 🔄 In progress |

---

## 📦 Chinchilla-1 (73M)

A small, CPU-runnable chat model — fine-tuned from Karpathy's nanochat with ResFormer-style value embeddings and sliding window attention. Trained for ~$1.20 on 2× A10G.

**Quick links:**

| Resource | Location |
|----------|----------|
| Model card & artifacts | [`models/chinchilla-1/`](models/chinchilla-1/) |
| Source code & training pipeline | [`archive/nanochat_modal/`](archive/nanochat_modal/) |
| Technical report | [`archive/nanochat_modal/docs/SFT_REPORT.md`](archive/nanochat_modal/docs/SFT_REPORT.md) |
| Architecture doc | [`archive/nanochat_modal/docs/ARCHITECTURE.md`](archive/nanochat_modal/docs/ARCHITECTURE.md) |

> **Status:** Published to HuggingFace Hub. See the [model card](models/chinchilla-1/README.md) for usage and inference.

---

## 🚀 Mars-1.0 (8B-A1B Agent)

A Liquid Foundation Model finetune — targeting tool-use and agentic benchmarks. Built on the LFM2.5 mixture-of-experts architecture.

**Quick links:**

| Resource | Location |
|----------|----------|
| Training pipeline | [`train/`](train/) |
| SFT configs | [`train/configs/`](train/configs/) |
| Eval harness | [`train/evals/`](train/evals/) |
| Realistic eval guide | [`train/evals/README.md`](train/evals/README.md) |

The current evaluation is an offline, deterministic agent regression suite covering repository tool use, destructive-action safety, secret handling, prompt injection, uncertainty grounding, constrained output, practical Python generation, and source-grounded research. The default runner explicitly starts llama-server with reasoning enabled and an unrestricted reasoning budget; see the [evaluation guide](train/evals/README.md).

> **Status:** In development. Pipeline scaffold lives under [`train/`](train/).

---

## 📁 Repository Layout

```
temp-file-share/
├── README.md                   # This file
├── notes.md                    # Master plan & research notes
├── LICENSE                     # MIT
├── models/
│   └── chinchilla-1/           # Chinchilla-1 HF publish artifacts
│       ├── README.md           #   Model card
│       ├── config.json         #   Model config
│       ├── inference.py        #   Inference script
│       └── evals/              #   Evaluation results
├── train/                      # Mars-1.0 LFM training pipeline
│   ├── modal_app.py            #   Modal entrypoint
│   ├── configs/                #   Training configs (YAML)
│   ├── scripts/                #   Training & export scripts
│   └── evals/                  #   Benchmark suites
├── archive/
│   └── nanochat_modal/         # Chinchilla-1 source (preserved)
│       ├── checkpoints/        #   All trained checkpoints
│       ├── evaluations/        #   Evaluation results
│       ├── docs/               #   Documentation
│       └── training/           #   Modal training scripts
└── .gitignore
```

## 🔧 Setup

```bash
git clone https://github.com/<your-org>/temp-file-share
cd temp-file-share

# For Chinchilla-1 (runs on CPU)
cd archive/nanochat_modal
pip install -e nanochat

# For Mars-1.0 (Modal cloud)
cd train
pip install modal fastapi
modal setup
```

## 📄 License

MIT — see [LICENSE](LICENSE).
