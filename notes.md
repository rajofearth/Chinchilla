# Model Training Pivot — Research Notes

> **Last updated:** 2026-06-26 (rev 2 — finetune-first, Ornith-lite, Chinchilla-1 HF release)

---

## ★ CURRENT PLAN (read this first)

### Decision (rev 2)

1. **Do NOT train from scratch.** Fine-tune the shit out of an open base — Ornith-style multi-stage post-training, scaled to your Modal budget (~$30/mo).
2. **Primary base model:** [LiquidAI/LFM2.5-8B-A1B-Base](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base) (start from **Base**, not the instruct checkpoint).
3. **Stretch / Phase 2 base:** [google/gemma-4-E4B-pt](https://huggingface.co/google/gemma-4-E4B-pt) if you get more compute and need stronger raw coding (LiveCodeBench 52% on IT vs LFM's weaker heavy-code profile).
4. **Publish nanochat d6 to HF** as **Chinchilla-1** — honest research artifact, not a product model.
5. **Restructure repo** — archive `nanochat_modal/`, new `train/` pipeline for LFM finetuning.

### Why LFM2.5-8B-A1B wins for you (vs Gemma 4 E4B-it)

| Criterion | [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) | [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it) |
|-----------|------------------------------------------------------------------|------------------------------------------------------------------|
| **Active params at inference** | **1.5B** (8.3B MoE total) | **4.5B effective** (~8B w/ embeddings) |
| **Context** | 128K | 128K |
| **Agent / tool native** | ✅ Built for it; BFCLv4 ~48.5, tool tokens documented | ✅ Tau² ~42%, function calling |
| **Heavy codebase coding** | ⚠️ Liquid docs say *not best for heavy programming* | ✅ LiveCodeBench **52%**, Codeforces ELO 940 |
| **Fine-tune docs** | ✅ Official [Liquid TRL/Unsloth/GRPO](https://docs.liquid.ai/lfm/models/lfm25-8b-a1b) notebooks | ⚠️ Standard HF; multimodal (vision/audio) adds complexity |
| **Modal A10G budget** | ✅ LoRA + GRPO feasible (1.5B active) | ⚠️ Tight; needs LoRA + checkpointing; full FT needs A100 |
| **Local apps** | ✅ GGUF, llama.cpp, vLLM, SGLang, LM Studio day-one | ✅ Transformers; GGUF ecosystem catching up |
| **License for derivatives** | [LFM 1.0](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) — allows FT & deploy; **must attribute Liquid**; read before commercial rebrand | **Apache 2.0** — cleanest for naming & redistribution (like Ornith's MIT approach) |
| **Maintain base capabilities** | Strong tool/agent; weaker on deep code | Stronger coding/reasoning; more risk of forgetting with bad SFT mix |

**Verdict:** Start **LFM2.5-8B-A1B-Base** for Hermes-style tool agents on Modal budget. Add **Gemma 4 E4B** only if coding benchmarks stall after GRPO (needs ~2–3× more GPU budget).

**Ornith reference:** [Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B) is MIT-licensed, post-trained on Qwen3.5/Gemma 4, RL co-trains scaffold + solution. You copy the **stages and eval discipline**, not the 9B scale or scaffold RL (too expensive).

---

### Ornith-lite pipeline (your budget, ~8–12 weeks)

```
Week 1–2   Setup + baseline eval (LFM base vs LFM instruct vs your target tasks)
Week 3–4   SFT-1: agent + tool format (Synth-APIGen, tool-call traces, Hermes-style)
Week 5     SFT-2: coding (SWE-smith subset, CodeFeedback, repo-aware prompts) — small mix, don't drown tools
Week 6     DPO: UltraFeedback + coding preference pairs (chosen=concise correct, rejected=verbose wrong)
Week 7–8   GRPO: verifiable rewards (HumanEval sandbox, IFEval programmatic, tool JSON schema)
Week 9     Merge best SFT + GRPO checkpoints; regression eval vs base
Week 10    Export GGUF + vLLM smoke test + HF publish
Week 11–12 Buffer / DPO v2 if regression on tool-use
```

**Techniques to use (best ROI per dollar):**

| Technique | Use? | Why |
|-----------|------|-----|
| Multi-stage SFT (narrow mixes) | ✅ | Ornith/LFM both do staged specialization |
| DPO / ORPO | ✅ | Cheap on A10G; fixes verbosity and format |
| GRPO on verifiable tasks | ✅ | Ornith core idea, scaled down |
| LoRA r=16–64 first | ✅ | Saves VRAM; merge for export if needed |
| Full fine-tune | ⚠️ Phase 2 only | Better but 3–5× cost; try after LoRA plateaus |
| Online RL on SWE-bench | ❌ | OpenHands harness = $$$; use HumanEval + mini repo tasks |
| Scaffold co-training (Ornith) | ❌ | Requires RL env + many rollouts |
| KD from Claude/GPT-4o | ⚠️ Optional | Offline traces for coding only (~$20–50 API); high leverage |
| Model merge (SFT + GRPO) | ✅ | Free; often recovers general cap while keeping RL gains |
| CPT on base | ❌ for now | Burns budget; base already saw 38T tokens |

**Preserve existing capabilities:** Always keep **10–20% general chat** (SmolTalk slice) in every SFT stage. Run **regression eval** after each stage against base on BFCL + IFEval + 12 qualitative questions. If MMLU/IFEval drops >5 pts, reduce coding mix or merge with earlier checkpoint.

**Realistic benchmark deltas (LFM2.5-8B-A1B, LoRA post-train):**

| Benchmark | Base (~) | Your stretch goal | Ornith 9B (context) |
|-----------|----------|-------------------|---------------------|
| BFCLv4 | ~48.5 | **55–60** | — |
| IFEval | high | **+3–5 pts** | — |
| HumanEval | moderate | **+5–10 pts** | — |
| Tau²-Bench | moderate | **+5 pts** | — |
| SWE-bench Verified | low | **skip or lite** | **69.4%** |

Shoot for **Mars** (meaningful +5–15 pt gains on agent/code evals), not **Moon** (Ornith-tier SWE-bench).

---

### Naming & license — "call it my model" like Ornith

**Ornith pattern:** MIT license on derivative weights + model card clearly states base model + training method + eval harness.

**Your naming convention:**

| Asset | Suggested HF repo | License |
|-------|-------------------|---------|
| LFM derivative | `YOUR_ORG/Mars-1.0-8B-A1B` or `YOUR_ORG/Chinchilla-Agent-8B` | LFM 1.0 (derivative) + your README |
| nanochat d6 | `YOUR_ORG/Chinchilla-1-73M` | Match nanochat upstream (MIT) |

**LFM 1.0 rules (read full license before publish):**
- Fine-tuning and deployment allowed
- **Must credit Liquid AI** and note derived from LFM2.5-8B-A1B
- Commercial use may have size/terms constraints — read [LFM license](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) before monetizing
- You CAN publish under your org name on HF (Liquid does this themselves with model variants)

**Gemma 4 (if Phase 2):** Apache 2.0 — publish as `YOUR_ORG/Mars-Coder-E4B`, attribute Google, no "Gemma" in product name without following [Gemma terms](https://ai.google.dev/gemma/terms).

---

### Chinchilla-1 — publish your 73M nanochat model to HF

**Naming note:** "Chinchilla-1" is a fun nod to scaling laws; the model is **~73.5M params (d6)**, not 1B. Say that upfront in the model card to avoid confusion.

**What it actually is:** nanochat d6 (ResFormer value embeddings, 6L/384d) SFT'd on SmolTalk+MMLU+GSM8K+spelling, with optional KD from LFM2.5-350M. Best at **tool-call format + simple Python**; bad at facts and math.

**Presentable release checklist:**

- [ ] Pick best checkpoint: `sft` step 1500 (KD) unless eval says otherwise
- [ ] Include: `model.safetensors` or `.pt` + `tokenizer.pkl` + `token_bytes.pt` + `config.json` (write from ARCHITECTURE.md)
- [ ] Add `inference.py` (Transformers-style or minimal torch script) — **will NOT run in llama.cpp** (custom arch)
- [ ] Add `evals/` folder with your `evaluations/checkpoints/report.md` summary + scores
- [ ] Quantization: optional; 73M is tiny — bf16/fp32 is fine
- [ ] Space: optional Gradio demo on HF Spaces (CPU ok)
- [ ] Honest limitations section (hallucination, no LM Studio path)

**Do NOT claim:** competitive with LFM/Qwen/Gemma, LM Studio ready, or general-purpose chatbot.

---

### Repo restructure (do before LFM work)

```
temp-file-share/
├── README.md                    # New top-level: two tracks (Chinchilla-1 + Mars agent FT)
├── notes.md                     # This file — master plan
├── models/
│   └── chinchilla-1/            # HF publish artifacts
│       ├── README.md            # Model card (HF)
│       ├── config.json
│       ├── inference.py
│       └── evals/
├── train/                       # NEW — LFM Ornith-lite pipeline
│   ├── modal_app.py
│   ├── configs/
│   │   ├── sft_agent.yaml
│   │   ├── sft_code.yaml
│   │   ├── dpo.yaml
│   │   └── grpo.yaml
│   ├── scripts/
│   │   ├── download_base.py
│   │   ├── train_sft.py
│   │   ├── train_dpo.py
│   │   ├── train_grpo.py
│   │   ├── eval_agent.py
│   │   └── export_hf.py
│   └── evals/
│       └── benchmark_suites.yaml
├── archive/
│   └── nanochat_modal/          # Move existing project here unchanged
└── LICENSE
```

**Migrate:** `git mv nanochat_modal archive/nanochat_modal` when ready. Keep checkpoints in `archive/.../checkpoints/` or LFS.

---

### Docs audit — are the agent-written docs any good?

| Doc | Quality | Action |
|-----|---------|--------|
| `SFT_REPORT.md` | ⭐⭐⭐⭐ Excellent technical report | **Keep** in archive; cite in Chinchilla-1 model card |
| `DEVLOG.md` | ⭐⭐⭐⭐ Good decisions log | **Keep** in archive |
| `ARCHITECTURE.md` | ⭐⭐⭐⭐ Accurate for d6 | **Keep**; source for Chinchilla-1 config.json |
| `COMPARISON.md` | ⭐⭐⭐ Useful eval methodology | **Keep**; reuse 12-question suite for Mars eval |
| `TRAINING.md` | ⭐⭐⭐ Good but nanochat-specific | **Keep** in archive only |
| `INFERENCE.md` | ⭐⭐⭐ Modal/nanochat specific | **Keep** in archive only |
| Top `README.md` | ⭐⭐⭐ Fine | **Replace** with new top-level README pointing to both tracks |

**Do NOT follow old notes §7 Path A (Qwen3-0.6B)** — superseded by LFM2.5-8B-A1B plan above.

**Do NOT invest more in nanochat docs** except copying facts into Chinchilla-1 model card.

---

### Budget (rev 2 — finetune only)

| Month | Spend | Work |
|-------|-------|------|
| M1 | $30 | LFM setup, baseline eval, SFT-1 agent (~20 A10G hr) |
| M2 | $30 | SFT-2 code + DPO |
| M3 | $30 | GRPO + merge + export |
| **Total** | **~$90** | Shippable Mars-1.0 agent model on HF + GGUF |

Burst A100 ($2.50/hr) for 2–4 hr if LoRA GRPO OOMs on long context — save from one month's budget.

---

## AGENT PROMPTS (copy-paste into Cursor for each task)

### Prompt 1 — Repo restructure

```
Restructure temp-file-share per notes.md "Repo restructure":
- Create train/, models/chinchilla-1/, archive/
- git mv nanochat_modal → archive/nanochat_modal
- Write new top-level README.md with two tracks: Chinchilla-1 (73M HF release) and Mars-1.0 (LFM finetune)
- Do not delete checkpoints or evaluations
- Update .gitignore for large weights
```

### Prompt 2 — Chinchilla-1 HF model card + publish prep

```
Read archive/nanochat_modal/docs/SFT_REPORT.md, ARCHITECTURE.md, evaluations/checkpoints/report.md.
Create models/chinchilla-1/README.md as a HuggingFace model card for "Chinchilla-1-73M":
- Honest about 73.5M params, nanochat architecture, no llama.cpp support
- Strengths: tool tokens, simple Python, chat format
- Weaknesses: hallucination, math, facts (with eval citations)
- Include inference.py minimal script loading checkpoints/sft/model_001500.pt + tokenizer
- Include config.json derived from ARCHITECTURE.md
- License: (match this project)
- Base model: nanochat d6 pretrain step 8600 + SFT
Do NOT oversell capabilities.
```

### Prompt 3 — LFM Modal training scaffold

```
Create train/ Modal app for fine-tuning LiquidAI/LFM2.5-8B-A1B-Base:
- Volume layout per notes.md
- Download base from HF to /vol/bases/
- TRL SFTTrainer with LoRA (peft), Liquid chat template preserved
- Checkpoint resume every N steps with vol.commit()
- A10G default, 14hr timeout, detach-friendly
- configs/sft_agent.yaml: Synth-APIGen + SmolTalk 20% + agent traces
Follow Liquid docs: https://docs.liquid.ai/lfm/models/lfm25-8b-a1b
Use transformers>=5.2, trl, peft, datasets
```

### Prompt 4 — Agent eval harness

```
Create train/scripts/eval_agent.py running after each training stage:
- BFCL lite or JSON tool-call format validation
- IFEval subset via lm-eval-harness
- HumanEval pass@1 (Modal sandbox or local)
- 12 questions from archive/nanochat_modal/docs/COMPARISON.md
- Save JSON to /vol/evals/{stage}_{step}.json
- Compare against LiquidAI/LFM2.5-8B-A1B baseline
```

### Prompt 5 — DPO + GRPO stages

```
Add train/scripts/train_dpo.py and train_grpo.py for LFM2.5-8B-A1B LoRA:
- DPO: UltraFeedback binarized + optional coding preference pairs
- GRPO: TRL GRPOTrainer, rewards = HumanEval pass + IFEval check + tool JSON valid
- Load from previous SFT checkpoint on Volume
- Regression guard: abort if IFEval drops >5 vs base
Reference Liquid TRL notebooks on HF model card
```

### Prompt 6 — Export & HF publish (Mars model)

```
Create train/scripts/export_hf.py:
- Merge LoRA into base if needed
- Push to HF as YOUR_ORG/Mars-1.0-8B-A1B with model card:
  - Base: LiquidAI/LFM2.5-8B-A1B-Base
  - Training: staged SFT → DPO → GRPO on Modal
  - Eval table vs base
  - LFM 1.0 license + Liquid attribution
  - GGUF link or conversion instructions via LiquidAI/LFM2.5-8B-A1B-GGUF recipe
- vLLM serve command in README
```

### Prompt 7 — Archive nanochat docs

```
In archive/nanochat_modal/docs/, add ARCHIVE.md at top:
"This project is archived. See repo root notes.md for current direction."
List which docs are historical reference vs obsolete.
Do not rewrite content unless factual errors.
```

---

## DRAFT — Chinchilla-1 HuggingFace model card

Use this as `models/chinchilla-1/README.md` (edit YOUR_ORG):

```markdown
---
license: mit
base_model: karpathy/nanochat
tags:
  - nanochat
  - tool-use
  - small-llm
  - research
language:
  - en
pipeline_tag: text-generation
---

# Chinchilla-1-73M

**Chinchilla-1** is a ~73.5M parameter instruction-tuned language model built on the [nanochat](https://github.com/karpathy/nanochat) d6 architecture. The name references [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) — this is a **research-scale** model, not a 1B production LLM.

## What it does well

- Chat format with custom tool tokens (`<|python_start|>`, etc.)
- Simple Python generation (e.g. palindrome, string ops)
- SpellingBee-style tool orchestration

## What it does poorly

- Factual QA (heavy hallucination)
- Multi-digit arithmetic
- Strict instruction formatting (haiku, one-sentence)

See our [evaluation report](evals/report.md): 12-question qualitative eval, d6-class limitations.

## Architecture

| Property | Value |
|----------|-------|
| Params | ~73.5M |
| Layers | 6 |
| Embed dim | 384 |
| Context | 2048 (SFT); pretrain 512 |
| Vocab | 32,768 BPE |
| Special tokens | user/assistant/python tool format |

Value embeddings on alternating layers (~51% of params). Full details in bundled `ARCHITECTURE.md`.

## Training

- **Base:** FineWeb-EDU pretrain, step 8600
- **SFT:** SmolTalk, MMLU, GSM8K, spelling tasks on Modal (2× A10G)
- **Optional KD:** LFM2.5-350M (cross-tokenizer limitations documented)

## Usage

> ⚠️ **Not compatible with llama.cpp, Ollama, or LM Studio.** Requires custom inference (nanochat engine).

\`\`\`python
# See inference.py — loads checkpoint + tokenizer from this repo
python inference.py --prompt "Write a Python function to reverse a string."
\`\`\`

## Checkpoints included

| File | Description |
|------|-------------|
| `sft-step1500/` | Best SFT (KD run, val_bpb 0.476) |
| `tokenizer/` | rustbpe tokenizer |

## Citation

\`\`\`bibtex
@misc{chinchilla1_73m,
  title={Chinchilla-1: A Research-Scale nanochat SFT Experiment},
  author={YOUR_NAME},
  year={2026},
  howpublished={\\url{https://huggingface.co/YOUR_ORG/Chinchilla-1-73M}}
}
\`\`\`

## Acknowledgments

Built on [karpathy/nanochat](https://github.com/karpathy/nanochat). Training infra inspired by Liquid LFM and Modal.
```

---

## DRAFT — Mars-1.0 agent model card (LFM derivative)

Template for after finetune completes:

```markdown
---
license: other
license_name: lfm1.0
base_model: LiquidAI/LFM2.5-8B-A1B-Base
tags:
  - agent
  - tool-use
  - lfm2.5
  - moe
language:
  - en
pipeline_tag: text-generation
---

# Mars-1.0-8B-A1B

Agent-focused fine-tune of [LiquidAI/LFM2.5-8B-A1B-Base](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base) for tool calling, coding assistance, and long-context agent workflows. Post-trained with staged SFT → DPO → GRPO on Modal.

**Not affiliated with Liquid AI or DeepReinforce (Ornith).** Derived weights with attribution.

## Base model

| Property | Value |
|----------|-------|
| Architecture | LFM2.5 MoE (8.3B total / 1.5B active) |
| Context | 128K |
| Pretrain | 38T tokens (Liquid) |

## Post-training

| Stage | Data | Purpose |
|-------|------|---------|
| SFT-1 | Synth-APIGen, agent traces | Tool format |
| SFT-2 | CodeFeedback, repo prompts | Coding |
| DPO | UltraFeedback | Preferences |
| GRPO | HumanEval, IFEval, tool schema | Verifiable RL |

## Eval vs base

| Benchmark | LFM2.5-8B-A1B | Mars-1.0 | Δ |
|-----------|---------------|----------|---|
| BFCLv4 | 48.5 | TBD | TBD |
| IFEval | TBD | TBD | TBD |
| HumanEval | TBD | TBD | TBD |

## Quick start (vLLM)

\`\`\`bash
vllm serve YOUR_ORG/Mars-1.0-8B-A1B --enable-auto-tool-choice --tool-call-parser lfm2
\`\`\`

## Quick start (llama.cpp)

\`\`\`bash
llama-cli -hf LiquidAI/LFM2.5-8B-A1B-GGUF  # convert from merged HF weights
\`\`\`

Sampling: `temperature=0.2`, `top_k=80`, `repetition_penalty=1.05` (per Liquid docs).

## License

Derived from LFM2.5-8B-A1B under [LFM 1.0](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B). See Liquid AI license terms.
```

---

## Immediate action list (this week)

1. [ ] Run **Prompt 1** (repo restructure)
2. [ ] Run **Prompt 2** (Chinchilla-1 model card + inference.py)
3. [ ] Upload Chinchilla-1 to HF (private first, test inference.py)
4. [ ] Download `LFM2.5-8B-A1B-Base` locally or Modal Volume; confirm LM Studio / llama.cpp works with stock Liquid GGUF
5. [ ] Run baseline eval (Prompt 4) on stock LFM before any training
6. [ ] Run **Prompt 3** (Modal SFT scaffold) — one cheap LoRA smoke test (~$2)
7. [ ] Read LFM 1.0 license end-to-end before picking public model name

---

# Historical notes (2026-06-26 rev 1 — mostly superseded)

> **Original decision:** Ditch nanochat for from-scratch ~1B. **Superseded by rev 2** (finetune LFM, publish Chinchilla-1 separately).

---
## 1. Why nanochat was the wrong choice

| Problem | Impact |
|---------|--------|
| Custom architecture (value embeddings, smear, backout) | ~50% params in embedding tables; poor param efficiency vs Llama/Qwen |
| Custom tokenizer + chat format | No vLLM/llama.cpp/Ollama compatibility out of the box |
| 2048 context, MHA not GQA | Worse inference vs modern 32K–262K GQA models |
| GPT-2-era pretrain target (CORE ~0.26) | Competitive with 2019 GPT-2, not 2025–2026 1B instruct models |
| Your d6 SFT results | Format/tool-use OK; facts, math, instruction-following broken |
| Ecosystem lock-in | Can't drop into LM Studio; everything is bespoke |

**Verdict:** nanochat is a research speedrun platform (train GPT-2 in 2 hours). It is not a path to a usable local AI app model.

---

## 2. What "decent high-school level" actually means

Calibrate expectations against public benchmarks (not vibes):

| Capability | High-school bar (rough) | Reference scores |
|------------|-------------------------|------------------|
| General knowledge | MMLU / MMLU-Pro ~35–45% | Qwen3.5-0.8B: MMLU-Pro 29.7% (non-thinking); Gemma 3 1B IT: MMLU-Pro 14.7% |
| Math word problems | GSM8K ~30–50% | Gemma 3 1B IT: GSM8K 62.8%; LFM2.5-350M: weak on knowledge, strong on IFEval (76.96%) |
| Instruction following | IFEval ~55–70% | LFM2.5-350M: 76.96%; Qwen3-0.6B: strong for size |
| Basic coding | HumanEval ~25–40%, MBPP ~30–50% | Gemma 3 1B IT: HumanEval 41.5%; Ornith-9B: SWE-bench 69.4% (different league) |
| Writing / chat | Subjective but coherent multi-paragraph | Needs SFT + DPO, not pretrain alone |

**Honest target for your budget:** A **~0.8–1.2B instruct model** that feels like a **capable study assistant** — good at writing, basic code, following instructions, tool calls — but **not** competitive with Ornith-9B or Qwen3.5 on agentic coding. That's still a real, usable model.

**Ornith-1.0-9B** ([HF](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B)) is a different category: 9B, Qwen3.5-based, RL-trained for agentic coding (SWE-bench Verified 69.4%, Terminal-Bench 43.1%). Use it as a **north star for post-training methodology**, not as a size target.

---

## 3. Reference model comparison (your benchmarks)

| Model | Params | Architecture | Pretrain tokens | Context | Ecosystem | Best at |
|-------|--------|--------------|-----------------|---------|-----------|---------|
| [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m) | 270M | Gemma3 hybrid local/global attn | **6T** | 32K | ✅ llama.cpp, vLLM, Ollama | Tiny baseline; IF Eval 51.2% IT |
| [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) | 350M | Hybrid conv + GQA (16 layers) | **28T** | 32K | ✅ GGUF, MLX, vLLM, LM Studio | Tool use, IFEval, edge inference; **not** knowledge-heavy |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | 28L dense GQA | **36T** (reported) | 32K | ✅ Full stack + thinking mode | Reasoning, agents, multilingual |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.8B | Gated DeltaNet + sparse MoE hybrid | Large (undisclosed) | **262K** | ✅ vLLM, SGLang (recent versions) | Multimodal, agents; complex arch |
| [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it) | 1B | 5 local + 1 global attn, QK-norm | **2T** | 32K | ✅ Full stack | Balanced instruct; GSM8K 62.8% |
| [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | 1.7B | **Llama2-style** dense | **11T** | 8K→128K | ✅ Full stack | Best open recipe for 1–2B; SFT+DPO |
| [Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B) | 9B | Qwen3.5 derivative | Unknown | 262K | ✅ vLLM, SGLang, GGUF | Agentic coding via RL scaffold co-training |

**Key insight from [Liquid Nanos press release](https://www.liquid.ai/press/liquid-unveils-nanos-extremely-small-foundation-models-that-match-frontier-model-quality--running-directly-on-everyday-devices):** Small models beat larger ones on **specialized tasks** via distillation + RL + merging — not by matching frontier models on everything. LFM2-350M-Extract beats Gemma 3 4B on extraction with 11× fewer params.

**Key insight from LFM2.5:** 350M model trained on **28T tokens** + multi-stage RL. Your budget cannot replicate this at 1B from scratch. You must either (a) start from a strong base checkpoint, or (b) accept a weaker from-scratch model trained on 20–100B tokens over many months.

---

## 4. Architecture recommendation: what to actually build

### ❌ Do NOT train from scratch

| Architecture | Why skip |
|--------------|----------|
| nanochat custom | Already ditched |
| Qwen3.5 hybrid (DeltaNet + MoE) | Too complex; needs custom kernels; hard to train solo |
| LFM2 hybrid (conv + attn) | Architecture search result; no open training recipe at 1B |
| Gemma 3 (local/global + multimodal) | Multimodal overhead; Google trained on 2T tokens |

### ✅ Recommended: **Llama 3.2 1B-style dense transformer**

Reason: Maximum ecosystem compatibility. SmolLM2, TinyLlama, Llama 3.2 all prove this works in llama.cpp / Ollama / vLLM / LM Studio.

**Target spec (~1B params):**

```
Architecture:    Decoder-only Llama-style
Layers:          16–20
Hidden dim:      2048
FFN dim:         8192 (SwiGLU)
Heads:           32 Q / 8 KV (GQA 4:1)
Vocab:           128256 (Llama 3 tokenizer) OR 49152 (SmolLM2 tokenizer)
Context:         8192 pretrain → extend to 32K via YaRN/NPI in post-training
Pos emb:         RoPE (base 500000 for Llama 3)
Norm:            RMSNorm pre-attention + pre-FFN
Activation:      SwiGLU
Tied embeddings: Yes
```

**Closest open recipes:**
- [SmolLM2 paper](https://arxiv.org/abs/2502.02737) — 1.7B Llama2 arch, Nanotron, WSD schedule, 11T tokens
- [LitGPT](https://github.com/Lightning-AI/litgpt) — `tiny-llama-1.1b`, `llama-3.2-1b` configs, pretrain + finetune + export
- [Nanotron](https://github.com/huggingface/nanotron) — what HF used for SmolLM2; multi-GPU pretrain

**Export path (non-negotiable):**
```
HF safetensors → GGUF (llama.cpp) → Ollama Modelfile → LM Studio
              → vLLM / SGLang for serving
Chat template:  Llama 3.x or ChatML (pick one, stick with it)
Tool format:    OpenAI-style tool_calls (what vLLM expects)
```

---

## 5. Training pipeline (LFM-style flexibility)

Mirror what makes LFM/Qwen models "good by default":

```
Stage 0: Tokenizer (skip if using Llama 3 / SmolLM2 tokenizer)
Stage 1: Pretrain (CPT or from scratch)     — base LM
Stage 2: SFT (supervised fine-tuning)       — chat, instructions, tools
Stage 3: DPO / ORPO                         — preference alignment
Stage 4: GRPO / RL on verifiable tasks      — math, code, tool-use rewards
Stage 5: (Optional) Model merge             — combine SFT + RL checkpoints
Stage 6: Quantize + export GGUF             — local deployment
```

**Tools per stage:**

| Stage | Framework | Modal GPU |
|-------|-----------|-----------|
| Pretrain | LitGPT or Nanotron | A100 80GB or 2× A10G (small runs) |
| SFT | TRL `SFTTrainer` or Unsloth | A10G ($1.10/hr) |
| DPO | TRL `DPOTrainer` | A10G |
| GRPO/RL | TRL `GRPOTrainer` or OpenRLHF | A10G–A100 |
| Eval | lm-eval-harness, lighteval | A10G or CPU |
| Export | llama.cpp `convert_hf_to_gguf.py` | CPU on Modal |

**LFM2.5 fine-tuning docs** (if you start from their base): [Liquid TRL/Unsloth/GRPO notebooks](https://huggingface.co/LiquidAI/LFM2.5-350M) — DPO, GRPO, SFT all documented.

**SmolLM2 recipe** (if you fork their stack): [github.com/huggingface/smollm](https://github.com/huggingface/smollm) — full pretrain + SmolTalk SFT + UltraFeedback DPO.

---

## 6. Budget math — the hard truth

### Modal pricing (list / preemptible, [modal.com/pricing](https://modal.com/pricing))

| GPU | $/hr | Your $30/month | ~hours/month |
|-----|------|----------------|--------------|
| A10G | $1.10 | $30 | **27 hr** |
| L40S | $1.95 | $30 | 15 hr |
| A100 80GB | $2.50 | $30 | 12 hr |
| H100 | $3.95 | $30 | 7.5 hr |

Volume storage is extra (~$0.10/GB/month). Checkpoint a 1B bf16 model ≈ 2 GB + optimizer ≈ 4 GB per checkpoint.

### Chinchilla-optimal 1B pretrain (minimum viable from scratch)

- **Tokens:** ~20B (20:1 ratio, [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556))
- **FLOPs:** 6 × 1B × 20B = **1.2 × 10²³ FLOPs**
- **Single A10G** (~9 TFLOPS effective @ 30% MFU): ~**3,700 GPU-hours** ≈ **$4,000** at list price
- **8× A100** (~40% MFU): ~**520 GPU-hours** ≈ **$1,300** wall-clock ~65 hr

**Your $30/month = 27 A10G hours.** At that rate, Chinchilla-optimal 1B pretrain takes **~137 months (~11 years)** on a single A10G.

Even **$100/month × 6 months = 540 GPU-hours** is only ~14% of single-GPU Chinchilla budget. You cannot pretrain 1B from scratch to LFM/Qwen quality on this budget in a reasonable timeline.

### What modern models actually use (overtraining)

| Model | Params | Tokens | Token:Param ratio |
|-------|--------|--------|-------------------|
| Chinchilla optimal 1B | 1B | 20B | 20:1 |
| SmolLM2-1.7B | 1.7B | 11T | **6,470:1** |
| LFM2.5-350M | 350M | 28T | **80,000:1** |
| Gemma 3 1B | 1B | 2T | **2,000:1** |
| Qwen3-0.6B | 0.6B | ~36T | **60,000:1** |

They overtrain by **100–4000×** beyond Chinchilla. That's why they feel smart despite small size.

---

## 7. Three viable paths (ranked)

### Path A — RECOMMENDED: Fine-tune an open base (fastest to usable model)

**Start from:** `Qwen/Qwen3-0.6B-Base`, `LiquidAI/LFM2.5-350M-Base`, or `HuggingFaceTB/SmolLM2-360M-Base`

**Why:** Base already has billions–trillions of tokens of knowledge baked in. You spend budget on **what actually makes it feel smart**: SFT + DPO + GRPO on your target tasks (writing, dev, high-school QA).

**Pipeline:**
1. Download base to Modal Volume
2. SFT on SmolTalk + OpenHermes + coding mix (~$5–15 on A10G)
3. DPO on UltraFeedback or custom preferences (~$5–10)
4. GRPO on GSM8K/HumanEval rewards (~$10–20)
5. Merge best checkpoints
6. Export GGUF → test in LM Studio

**Expected result:** Near Qwen3-0.6B / LFM2.5-350M quality, customized to your data. **Achievable in 1–2 months on $30/month.**

**Caveat:** It's not "your" pretrained model — it's your aligned variant. License check: LFM (LFM License), Qwen (Apache 2.0), SmolLM2 (Apache 2.0).

---

### Path B — AMBITIOUS: Continue-pretrain (CPT) a base toward 1B behavior

**Start from:** SmolLM2-360M-Base or train a custom 1B then CPT

**Add:** 5–20B tokens of curated data (FineWeb-Edu, FineMath, StarCoder, Cosmopedia) on Modal with checkpoint resume

**Cost estimate:** 20B tokens on 1× A10G at ~50K tok/s effective ≈ 111 GPU-hours ≈ **$122** (stretch over 4–5 months at $30/mo)

**Expected result:** Better domain knowledge than Path A, still below SmolLM2-1.7B unless you run 50B+ tokens.

---

### Path C — MOONSHOT: Train 1B Llama-arch from scratch (multi-month)

Only if you accept **won't match LFM/Qwen** but want a fully owned model.

**Minimal viable from-scratch:**
- 20B tokens (Chinchilla floor) → ~$122–400 depending on GPU
- 50B tokens (better) → ~$300–1000
- Architecture: LitGPT `tiny-llama-1.1b` or custom 1B config
- Data: FineWeb-Edu + StarCoder + OpenWebMath mix (SmolLM2-style)
- **Timeline:** 6–12 months at $30–100/month with checkpoint resume

**Stack:** LitGPT or Nanotron on Modal, Volume checkpoints every 1000 steps, `torchrun` on 2× A10G or burst 8× A100 for a weekend.

**Expected result:** GPT-2+ to early SmolLM1 tier. Usable for writing/dev **after heavy SFT+DPO**. Not competitive with Qwen3.5-0.8B on benchmarks.

---

## 8. Better ideas than multi-account Modal (don't undermine ambition)

### ⚠️ Multi-account checkpoint hopping

Likely violates [Modal ToS](https://modal.com/legal/terms). Risk: account bans, lost volumes. **Don't rely on this.**

### ✅ Legitimate ways to stretch compute

| Source | What you get | Notes |
|--------|--------------|-------|
| **Modal free $30/mo** | ~27 A10G hr | Your baseline; use for SFT/DPO/RL |
| **Hugging Face Grants** | GPU credits | Apply: [hf.co/support](https://huggingface.co/support) — SmolLM team got 256× H100 |
| **Google Colab Pro+** | ~100 compute units/mo | Good for SFT/DPO experiments |
| **Kaggle** | ~30 hr/week T4/P100 | Free; slower but fine for SFT |
| **Lambda / RunPod spot** | A10G ~$0.40–0.70/hr | Cheaper than Modal for long pretrain bursts |
| **University / lab cluster** | Free if available | Worth asking |
| **Startups for Open Source** | Cloud credits | Various programs |

### ✅ Smart budget allocation (maximize impact per dollar)

```
Month 1–2:  Path A — SFT + DPO on Qwen3-0.6B or SmolLM2-360M     (~$30)
Month 3:    Eval suite + GRPO on verifiable tasks                  (~$30)
Month 4:    CPT add 5B domain tokens if base feels weak            (~$30)
Month 5–6:    DPO v2 + merge + GGUF export + local app testing      (~$30)
```

**Total ~$150 over 6 months → usable local model in LM Studio.** This beats $150 spent on 12 hours of from-scratch pretrain.

---

## 9. Modal infrastructure plan

### Volume layout

```
/vol/
├── bases/              # Downloaded HF weights (Qwen3-0.6B-Base, etc.)
├── checkpoints/
│   ├── pretrain/       # step_XXXXX/ (model + optim + scheduler + dataloader state)
│   ├── sft/
│   ├── dpo/
│   └── rl/
├── data/               # Tokenized shards or parquet
├── tokenized/          # Preprocessed binary chunks
├── exports/            # GGUF, merged HF repos
└── evals/              # lm-eval results JSON
```

### Checkpoint resume (critical for multi-month)

Every training script must save:
- `model.safetensors` (or sharded)
- `optimizer.pt`
- `scheduler.pt`
- `rng_state.pt`
- `dataloader_state.pt` (shard index + offset)
- `meta.json` (step, loss, config, git hash)

Modal pattern:
```python
@app.function(gpu="A10G", volumes={"/vol": vol}, timeout=86400)
def train_step(resume_from: str | None = None):
    if resume_from:
        load_checkpoint(resume_from)
    # train until budget exhausted or step limit
    save_checkpoint("/vol/checkpoints/...")
    vol.commit()
```

Use `modal run --detach` so runs survive terminal close. Poll with `modal app logs`.

### GPU selection by stage

| Stage | GPU | Why |
|-------|-----|-----|
| Pretrain 1B | A100 80GB or 2× A10G | Memory for 1B + batch |
| SFT/DPO LoRA | A10G | 1B LoRA fits easily |
| SFT full | A10G | ~4 GB model + activations |
| GRPO | A10G | Multiple rollouts; may need A100 if long context |
| GGUF convert | CPU | No GPU needed |

---

## 10. Data plan

### Pretrain mix (if doing CPT or from-scratch)

| Source | HF dataset | Weight | Purpose |
|--------|-----------|--------|---------|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 45% | General knowledge |
| DCLM | `mlfoundations/dclm-baseline-1.0` | 25% | Web quality |
| StarCoder | `bigcode/starcoderdata` | 15% | Code |
| OpenWebMath | `open-web-math/open-web-math` | 10% | Math |
| Cosmopedia | `HuggingFaceTB/cosmopedia` | 5% | Synthetic quality text |

SmolLM2 used this style of mix over **11T tokens**. You'd use the same mix over **5–50B tokens** depending on budget.

### SFT mix

| Source | Purpose |
|--------|---------|
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | General chat (460K) |
| [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | Instruction diversity |
| [MetaMath](https://huggingface.co/datasets/meta-math/MetaMathQA) | Math reasoning |
| [CodeFeedback](https://huggingface.co/datasets/HuggingFaceH4/codefeedback-filtered) | Code |
| [Argilla Synth-APIGen](https://huggingface.co/datasets/argilla/Synth-APIGen-v0.1) | Tool calling |

### DPO / preference

| Source | Purpose |
|--------|---------|
| [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | General preferences |
| Custom pairs | Writing style, verbosity, safety |

### RL (GRPO) rewards

| Task | Reward |
|------|--------|
| GSM8K | Exact match on `\boxed{}` answer |
| HumanEval | Pass@1 via sandbox |
| IFEval | Programmatic instruction-check |
| Tool call | JSON schema validation |

---

## 11. Evaluation suite (track progress monthly)

Run after every stage:

```bash
# lm-eval-harness
lm_eval --model hf --model_args pretrained=/vol/exports/my-model \
  --tasks mmlu,gsm8k,hellaswag,arc_easy,arc_challenge,ifeval,humaneval

# Qualitative (same 12 questions from your nanochat comparison)
# Plus: "Explain photosynthesis", "Write a Python merge sort", "Solve x²+5x+6=0"
```

**Target milestones:**

| Stage | MMLU | GSM8K | IFEval | HumanEval |
|-------|------|-------|--------|-----------|
| Base (Qwen3-0.6B) | ~25% | ~15% | ~50% | ~10% |
| After SFT | ~30% | ~25% | ~60% | ~20% |
| After DPO | ~32% | ~30% | ~65% | ~25% |
| After GRPO | ~35% | ~40% | ~70% | ~30% |
| **Your goal** | **35%+** | **35%+** | **65%+** | **25%+** |

Compare against LFM2.5-350M and Qwen3-0.6B at each stage.

---

## 12. Concrete first steps (week 1)

- [ ] **Pick path:** Path A (recommended) — start from `Qwen/Qwen3-0.6B-Base` (Apache 2.0, best ecosystem)
- [ ] **Verify local inference:** Download Qwen3-0.6B-Instruct GGUF → LM Studio → confirm "it just works"
- [ ] **Scaffold new repo** (not nanochat_modal): `llm-train/` with Modal app, Volume, TRL scripts
- [ ] **Phase 1 Modal run:** SFT Qwen3-0.6B on SmolTalk subset (1 epoch, LoRA) — validate pipeline end-to-end (~$2–5)
- [ ] **Export test:** HF → GGUF → Ollama → confirm chat template works
- [ ] **Eval baseline:** Run lm-eval on base vs your SFT checkpoint
- [ ] **Archive nanochat_modal:** Keep for reference, stop investing

### Week 2–4

- [ ] Full SFT (SmolTalk + code + math)
- [ ] DPO with UltraFeedback
- [ ] Qualitative comparison vs LFM2.5-350M, Qwen3-0.6B, Gemma 3 270M in LM Studio

### Month 2+

- [ ] GRPO on GSM8K + HumanEval
- [ ] Optional CPT if knowledge gap remains
- [ ] If still want custom 1B: start LitGPT pretrain with 5B token pilot on Modal

---

## 13. Repo structure (new project)

```
llm-train/
├── modal_app.py           # Volume, image, GPU configs
├── configs/
│   ├── sft.yaml
│   ├── dpo.yaml
│   └── grpo.yaml
├── scripts/
│   ├── download_base.py
│   ├── train_sft.py       # TRL SFTTrainer
│   ├── train_dpo.py       # TRL DPOTrainer
│   ├── train_grpo.py      # TRL GRPOTrainer
│   ├── eval.py            # lm-eval wrapper
│   └── export_gguf.sh
├── data/
│   └── mixtures.yaml
└── README.md
```

**Dependencies:** `transformers>=4.51`, `trl`, `peft`, `datasets`, `modal`, `lm-eval`, `flash-attn` (optional)

---

## 14. What success looks like

**Minimum viable (3 months, ~$90):**
- Custom SFT+DPO variant of Qwen3-0.6B or SmolLM2-360M
- Runs in LM Studio / Ollama / llama.cpp
- Decent writing, basic code, high-school-level QA
- Full RL pipeline proven

**Stretch (6–12 months, ~$180–500):**
- Custom 1B Llama-arch model (from scratch or CPT)
- GRPO-polished for math + code
- Published to HF with GGUF
- Competitive with Gemma 3 1B IT on subset of tasks

**Not achievable on this budget:**
- Ornith-9B tier agentic coding
- LFM2.5-350M tier with 28T token pretrain
- Matching Qwen3.5-0.8B on MMLU-Pro (29.7%) from scratch

---

## 15. Archived nanochat todos (superseded)

These applied to the old nanochat project and are **cancelled**:

1. ~~Multi-stage SFT → DPO on nanochat d6~~ → Do on standard base model
2. ~~Non-KD vs KD SFT comparison on d6~~ → Irrelevant after pivot
3. ~~chat_rl.py on nanochat~~ → Use TRL GRPO on Qwen/SmolLM base
4. ~~Compare d6 vs OSS~~ → Compare your aligned model vs OSS in LM Studio
5. ~~Update nanochat docs~~ → Archive nanochat_modal, write new llm-train docs
6. ~~Explore modern architecture in nanochat~~ → Use Llama 3.2 1B in LitGPT/Nanotron instead

---

## 16. References

- [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m) — 6T tokens, 32K ctx, llama.cpp ready
- [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) — 28T tokens, TRL/Unsloth/GRPO fine-tuning docs
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — thinking mode, Apache 2.0, agentic
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) — 262K ctx, hybrid arch (reference only)
- [Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B) — RL scaffold co-training for coding agents
- [Liquid Nanos press release](https://www.liquid.ai/press/liquid-unveils-nanos-extremely-small-foundation-models-that-match-frontier-model-quality--running-directly-on-everyday-devices) — task-specific small models via distillation + RL
- [SmolLM2 paper](https://arxiv.org/abs/2502.02737) — best open 1–2B training recipe
- [SmolLM repo](https://github.com/huggingface/smollm) — full pipeline code
- [LitGPT](https://github.com/Lightning-AI/litgpt) — pretrain 1.1B from scratch tutorials
- [Modal pricing](https://modal.com/pricing) — GPU $/hr reference
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) — 20 tokens/param minimum

---

*Historical section last updated: 2026-06-26 rev 1. Current plan: see top of file (rev 2).*
