# Mars-1.0 evaluation

The default Mars evaluation is `cases/agent_realistic_v1.json`. It is a deterministic, offline regression suite designed around tasks an agent would encounter in real repository and operations work rather than trivia prompts.

## What it tests

The 14 cases cover:

- **Tool use:** scoped repository search and running tests without modifying files.
- **Safety:** production deletion, ambiguous email sending, and credential disclosure.
- **Security:** untrusted email prompt injection and forbidden external actions.
- **Reliability:** timeouts, unknown state, and conflicting deployment evidence.
- **Instruction following:** exact JSON contracts and constrained bullet lists.
- **Coding:** practical CSV parsing and retry-backoff utilities with Python syntax checks.
- **Research:** source-grounded explanation and exact fact extraction.

The suite is inspired by BFCL, IFEval, HumanEval/LiveCodeBench, BrowseComp, AgentDojo, OWASP Excessive Agency guidance, and NIST agent-hijacking evaluation guidance. It is **not** an official score for any of those benchmarks; it is a local regression suite using deterministic cases and local scoring.

The case file records the source URLs under `provenance.sources`.

## Run the default evaluation

From the repository root:

```bash
python train/scripts/eval_runner.py \
  --cases train/evals/cases/agent_realistic_v1.json \
  --output train/evals/results/realistic_run.json
```

On Windows, the paths in `models.yaml` point to the local llama.cpp executable, base GGUF, and optional LoRA GGUF checkpoints. Override `models.yaml` for another machine.

Run selected checkpoints with repeated `--model` arguments:

```bash
python train/scripts/eval_runner.py \
  --model grpo-5500 \
  --model grpo-5600 \
  --cases train/evals/cases/agent_realistic_v1.json
```

The dashboard uses the same realistic case file by default.

## Reasoning configuration

Reasoning is explicitly enabled in `models.yaml`:

```yaml
reasoning: "on"
reasoning_budget: -1
```

The runner passes these through to llama-server as:

```text
--reasoning on
--reasoning-budget -1
```

`-1` means an unrestricted reasoning budget. The runner records any separate `reasoning_content` stream but scores only visible model output. This prevents hidden reasoning from satisfying a tool, safety, or formatting check.

A model may still emit zero separate reasoning tokens even when reasoning is enabled. That is model/checkpoint behavior, not the evaluator disabling thinking. In the August 15, 2026 run, the base model emitted a reasoning stream while the GRPO checkpoints did not.

## Scoring

The runner supports the legacy smoke expectations and the stronger realistic expectations:

- Structured LFM and JSON-style tool-call parsing.
- Exact tool names and argument values.
- Forbidden tool detection.
- JSON object key/value contracts.
- Python syntax and named-function checks.
- Required and forbidden safety language/content.
- Grounded responses that preserve uncertainty.
- Exact source-value extraction.
- Constrained list and sentence checks.

Safety cases are intentionally strict. A refusal that avoids the dangerous action may still fail if it does not communicate the required authorization, scope, backup, uncertainty, or redaction behavior.

The current runner does not execute generated code, invoke real tools, browse the web, or mutate a real database. Those capabilities require a separate sandboxed or agent-loop evaluator. The realistic suite uses offline evidence and simulated tool descriptions so runs remain reproducible and safe.

## Legacy smoke suite

`cases/smoke_v1.json` is retained for historical comparison with older result files. It should not be used as the primary quality signal because it contains simple weather/trivia prompts and declaration-only coding checks.

## Recent recorded run

`results/realistic_reasoning_on_20260815.json` was run with explicit reasoning enabled across the configured base, GRPO-5400, GRPO-5500, and GRPO-5600 models. All 56 cases completed. The best observed score was 5/14 (35.7%) for GRPO-5500 and GRPO-5600. These numbers are regression results for this local suite, not general model capability claims.
