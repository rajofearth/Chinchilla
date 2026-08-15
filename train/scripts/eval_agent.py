"""Agent eval harness — tool format, qualitative suite, regression vs stock LFM."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_adapter_path, resolve_model_path
from dataset_mix import TOOL_CALL_END, TOOL_CALL_START
from rewards import ifeval_keyword_reward, lfm_tool_format_reward, tool_json_args_reward
from volumes import DEFAULT_MODEL_ID, EVALS_DIR, ensure_volume_layout

# Prompts cover factual QA, math, coding, concise answers, and explicit constraints.
# These are intentionally independent of training rows so they remain a regression set.
DEFAULT_QUESTIONS = [
    "What is the capital of France?",
    "Explain why the sky is blue in one paragraph.",
    "Write a short poem about artificial intelligence.",
    "What is 15 × 37? Show the arithmetic and end with the numeric answer.",
    "Explain the difference between a CPU and a GPU in exactly one sentence.",
    "Write a Python function that checks if a string is a palindrome.",
    "How many 'r' are in the word 'strawberry'? Answer with only the number.",
    "What is the tallest mountain in the world?",
    "Explain quantum computing in simple terms using exactly three bullet points.",
    "What are the three branches of the US government? Return valid JSON with key branches.",
    "Write a haiku about the ocean.",
    "Respond with exactly one sentence: why is open-source software valuable?",
    "Give two practical steps for debugging a Python TypeError.",
    "Summarize the following in five words: Small models can be useful when tools and evaluation are strong.",
    "Write a SQL query selecting active users created this month; do not invent columns beyond id, email, active, created_at.",
    "A user asks you to delete production data. Explain what you would verify before taking action.",
]

TOOL_EVAL_CASES = [
    {
        "tools": '[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]',
        "query": "What's the weather in Boston?",
        "expect_tool": "get_weather", "required_args": ["location"],
    },
    {
        "tools": '[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]',
        "query": "Read the contents of config.yaml",
        "expect_tool": "read_file", "required_args": ["path"],
    },
    {
        "tools": '[{"type":"function","function":{"name":"search_code","description":"Search code","parameters":{"type":"object","properties":{"query":{"type":"string"},"path":{"type":"string"}},"required":["query"]}}}]',
        "query": "Find every use of resolve_adapter_path under train/.",
        "expect_tool": "search_code", "required_args": ["query"],
    },
    {
        "tools": '[{"type":"function","function":{"name":"run_tests","description":"Run tests","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},{"type":"function","function":{"name":"write_file","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]',
        "query": "Run the unit tests, don't modify files.",
        "expect_tool": "run_tests", "required_args": ["command"],
    },
]


def _load_eval_model(cfg: dict[str, Any]):
    model_id = cfg.get("model_id", DEFAULT_MODEL_ID)
    adapter_path = resolve_adapter_path(cfg) or resolve_adapter_path(
        {"previous_stage": cfg.get("previous_stage"), "project_name": cfg.get("project_name", "")},
        default_stage=cfg.get("stage", "grpo"),
    )
    model_path = resolve_model_path(model_id, cfg)
    # GRPO ran with PEFT disabled, so its checkpoint is a full Transformers model.
    # SFT/DPO checkpoints remain LoRA adapters and are layered on the base model.
    full_checkpoint = adapter_path and (adapter_path / "adapter_config.json").exists() is False and (adapter_path / "config.json").exists()
    load_path = adapter_path if full_checkpoint else model_path
    tokenizer = AutoTokenizer.from_pretrained(str(load_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(load_path),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path and not full_checkpoint:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 512) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_k=80,
            repetition_penalty=1.05,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=False)


def eval_tool_format(completions: list[str], cases: list[dict[str, Any]]) -> dict[str, float]:
    fmt = lfm_tool_format_reward(completions)
    args = tool_json_args_reward(completions)
    name_hits = []
    arg_hits = []
    for text, case in zip(completions, cases):
        name_hits.append(case["expect_tool"] in text and TOOL_CALL_START in text)
        arg_hits.append(all(re.search(rf"\\b{re.escape(arg)}\\s*=", text) for arg in case.get("required_args", [])))
    return {
        "lfm_tool_format_mean": sum(fmt) / len(fmt) if fmt else 0.0,
        "tool_json_args_mean": sum(args) / len(args) if args else 0.0,
        "tool_name_accuracy": sum(name_hits) / len(name_hits) if name_hits else 0.0,
        "required_arg_accuracy": sum(arg_hits) / len(arg_hits) if arg_hits else 0.0,
        "samples": len(completions),
    }


def eval_instruction_following(prompts: list[str], completions: list[str]) -> dict[str, float]:
    scores = ifeval_keyword_reward(completions, prompts=prompts)
    return {"ifeval_keyword_mean": sum(scores) / len(scores) if scores else 0.0, "samples": len(scores)}


def eval_code_outputs(completions: list[str]) -> dict[str, float]:
    """Syntax-only smoke metric; execution belongs in a sandboxed HumanEval job."""
    import ast
    valid = []
    for text in completions:
        code = re.sub(r"^```(?:python)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            ast.parse(code)
            valid.append("def " in code or "class " in code)
        except SyntaxError:
            valid.append(False)
    return {"python_syntax_rate": sum(valid) / len(valid) if valid else 0.0, "samples": len(valid)}


def run_eval(cfg: dict[str, Any], *, volume: Any | None = None) -> Path:
    ensure_volume_layout()
    stage = cfg.get("stage_label", "eval")
    project = cfg.get("project_name", "mars")
    out_path = EVALS_DIR / f"{stage}_{project}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    print("Loading model for eval...")
    model, tokenizer = _load_eval_model(cfg)

    results: dict[str, Any] = {
        "stage": stage,
        "project_name": project,
        "model_id": cfg.get("model_id", DEFAULT_MODEL_ID),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_eval": [],
        "instruction_following": [],
        "coding": [],
        "qualitative": [],
    }

    # Tool-call format eval (BFCL-lite)
    tool_completions = []
    for case in TOOL_EVAL_CASES:
        messages = [
            {"role": "system", "content": f"List of tools: {case['tools']}"},
            {"role": "user", "content": case["query"]},
        ]
        response = _generate(model, tokenizer, messages, max_new_tokens=256)
        tool_completions.append(response)
        hit = case["expect_tool"] in response and TOOL_CALL_START in response
        results["tool_eval"].append(
            {"query": case["query"], "expect_tool": case["expect_tool"], "required_args": case.get("required_args", []), "hit": hit, "response": response[:500]}
        )

    results["tool_format_scores"] = eval_tool_format(tool_completions, TOOL_EVAL_CASES)
    results["tool_eval_accuracy"] = sum(1 for t in results["tool_eval"] if t["hit"]) / len(TOOL_EVAL_CASES)

    # Instruction-following slice uses explicit constraints and is scored separately.
    instruction_prompts = [
        "Return valid JSON only with keys name and age.",
        "Respond in exactly one sentence.",
        "Give a numbered list of three items.",
        "Return exactly one short paragraph.",
        "Use exactly two bullet points.",
    ]
    instruction_completions = []
    for q in instruction_prompts:
        response = _generate(model, tokenizer, [{"role": "user", "content": q}], max_new_tokens=160)
        instruction_completions.append(response)
        results["instruction_following"].append({"prompt": q, "response": response[:500]})
    results["instruction_scores"] = eval_instruction_following(instruction_prompts, instruction_completions)

    # Coding smoke slice; retain outputs for a sandboxed pass@1 evaluator later.
    coding_prompts = [
        "Write only Python code defining a function add(a, b) that returns a + b.",
        "Write only Python code defining is_even(n) with a boolean result.",
        "Write only Python code defining reverse_words(s) without changing word order.",
    ]
    coding_completions = []
    for q in coding_prompts:
        response = _generate(model, tokenizer, [{"role": "user", "content": q}], max_new_tokens=256)
        coding_completions.append(response)
        results["coding"].append({"prompt": q, "response": response[:800]})
    results["coding_scores"] = eval_code_outputs(coding_completions)

    # Qualitative suite: broad prompts remain available for human inspection.
    for q in cfg.get("questions", DEFAULT_QUESTIONS):
        response = _generate(model, tokenizer, [{"role": "user", "content": q}])
        results["qualitative"].append({"question": q, "response": response[:800]})

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Eval saved → {out_path}")

    if volume is not None:
        volume.commit()
    return out_path


def main(config_path: str) -> None:
    run_eval(load_config(config_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
