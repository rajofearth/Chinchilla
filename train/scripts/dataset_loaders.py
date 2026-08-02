"""DPO and GRPO dataset loaders with Liquid explicit preference / prompt formats."""

from __future__ import annotations

import json
import random
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from dataset_mix import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    build_mixed_dataset,
    format_apigen_row,
    json_tool_call_to_lfm,
    normalize_messages,
)


def _as_assistant_message(content: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": content}]


def _load_split(path: str, split: str = "train", subset: str | None = None, limit: int | None = None) -> Dataset:
    kwargs: dict[str, Any] = {"path": path, "split": split}
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(**kwargs)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _format_ultrafeedback_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert UltraFeedback binarized → explicit DPO format (Liquid docs)."""
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not all([prompt, chosen, rejected]):
        return None

    if isinstance(prompt, list):
        prompt_msgs = normalize_messages(prompt)
    else:
        prompt_msgs = [{"role": "user", "content": str(prompt)}]

    if isinstance(chosen, list):
        chosen_msgs = normalize_messages(chosen)
    else:
        chosen_msgs = _as_assistant_message(str(chosen))

    if isinstance(rejected, list):
        rejected_msgs = normalize_messages(rejected)
    else:
        rejected_msgs = _as_assistant_message(str(rejected))

    return {"prompt": prompt_msgs, "chosen": chosen_msgs, "rejected": rejected_msgs}


def _format_coding_pref_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """HuggingFaceH4/Code-Feedback → DPO pairs (concise correct vs verbose).

    The dataset ships a pre-formatted `messages` column (user/assistant turns);
    falls back to instruction/answer columns if messages is missing.
    """
    messages = row.get("messages")
    if messages:
        normalized = normalize_messages(messages)
        user_msgs = [m for m in normalized if m["role"] == "user"]
        asst_msgs = [m for m in normalized if m["role"] == "assistant"]
        if not user_msgs or not asst_msgs:
            return None
        instruction = user_msgs[0]["content"]
        good = asst_msgs[-1]["content"]
    else:
        instruction = row.get("instruction") or row.get("query")
        good = row.get("answer") or row.get("response")
        if not instruction or not good:
            return None
    bad = row.get("bad_answer") or row.get("rejected")
    if not bad:
        # Synthesize verbose/wrong rejected for alignment signal
        bad = good + "\n\nAdditionally, here's extra unrelated context and a potentially incorrect alternative approach."
    return {
        "prompt": [{"role": "user", "content": str(instruction)}],
        "chosen": _as_assistant_message(str(good)),
        "rejected": _as_assistant_message(str(bad)),
    }


def build_dpo_dataset(dataset_cfg: dict[str, Any]) -> Dataset:
    """Build preference dataset from config `dataset.sources`."""
    sources = dataset_cfg.get("sources") or [
        {"name": "ultrafeedback", "path": "HuggingFaceH4/ultrafeedback_binarized", "weight": 0.7},
        {"name": "codefeedback", "path": "HuggingFaceH4/codefeedback-filtered", "weight": 0.3},
    ]
    seed = dataset_cfg.get("seed", 42)
    total_limit = dataset_cfg.get("limit")
    parts: list[tuple[Dataset, float]] = []

    for spec in sources:
        name = spec["name"]
        raw = _load_split(
            spec["path"],
            split=spec.get("split", "train"),
            subset=spec.get("subset"),
            limit=spec.get("limit"),
        )
        rows = []
        formatter = _format_ultrafeedback_row if name == "ultrafeedback" else _format_coding_pref_row
        for row in raw:
            sample = formatter(row)
            if sample:
                rows.append(sample)
        if not rows:
            print(f"  Warning: no valid rows from {name}")
            continue
        ds = Dataset.from_list(rows)
        parts.append((ds, float(spec.get("weight", 1.0))))
        print(f"  {name}: {len(ds)} preference pairs")

    if not parts:
        raise RuntimeError("No DPO samples loaded")

    if total_limit:
        return _weighted_sample(parts, total_limit, seed)
    max_len = max(len(d) for d, _ in parts)
    total = sum(int(max_len * w / sum(w for _, w in parts)) for _, w in parts)
    return _weighted_sample(parts, max(total, 1), seed)


def _weighted_sample(parts: list[tuple[Dataset, float]], total: int, seed: int) -> Dataset:
    weights = [w for _, w in parts]
    weight_sum = sum(weights)
    counts = [max(1, int(total * w / weight_sum)) for w in weights]
    rng = random.Random(seed)
    out = []
    for (ds, _), count in zip(parts, counts):
        if count >= len(ds):
            out.append(ds)
        else:
            out.append(ds.select(rng.sample(range(len(ds)), count)))
    return concatenate_datasets(out).shuffle(seed=seed)


def _apigen_to_grpo(row: dict[str, Any]) -> dict[str, Any] | None:
    formatted = format_apigen_row(row)
    if not formatted:
        return None
    msgs = formatted["messages"]
    prompt = msgs[:-1]
    solution = msgs[-1]["content"]
    return {"prompt": prompt, "solution": solution}


def _tool_eval_to_grpo(row: dict[str, Any]) -> dict[str, Any] | None:
    """Simple tool-use eval rows with expected LFM tool call in solution."""
    query = row.get("question") or row.get("query")
    tools = row.get("tools")
    expected = row.get("expected") or row.get("answer")
    if not query or not expected:
        return None
    prompt = [{"role": "system", "content": f"List of tools: {tools or '[]'}"}, {"role": "user", "content": str(query)}]
    return {"prompt": prompt, "solution": str(expected)}


def build_grpo_dataset(dataset_cfg: dict[str, Any]) -> Dataset:
    """Build prompt(+solution) dataset for GRPO verifiable rewards."""
    sources = dataset_cfg.get("sources") or [
        {"name": "synth_apigen", "path": "argilla/Synth-APIGen-v0.1", "weight": 0.5},
        {"name": "gsm8k", "path": "openai/gsm8k", "subset": "main", "weight": 0.25},
        {"name": "humaneval", "path": "openai/openai_humaneval", "weight": 0.25},
    ]
    seed = dataset_cfg.get("seed", 42)
    total_limit = dataset_cfg.get("limit")
    parts: list[tuple[Dataset, float]] = []

    for spec in sources:
        name = spec["name"]
        raw = _load_split(
            spec["path"],
            split=spec.get("split", "train"),
            subset=spec.get("subset"),
            limit=spec.get("limit"),
        )
        rows = []
        for row in raw:
            sample = None
            if name == "synth_apigen":
                sample = _apigen_to_grpo(row)
            elif name == "gsm8k":
                sample = _gsm8k_to_grpo(row)
            elif name == "humaneval":
                sample = _humaneval_to_grpo(row)
            elif name == "agent_traces":
                sample = _agent_trace_to_grpo(row)
            if sample:
                rows.append(sample)
        if rows:
            ds = Dataset.from_list(rows)
            parts.append((ds, float(spec.get("weight", 1.0))))
            print(f"  {name}: {len(ds)} GRPO prompts")

    if not parts:
        raise RuntimeError("No GRPO samples loaded")

    if total_limit:
        return _weighted_sample(parts, total_limit, seed)
    max_len = max(len(d) for d, _ in parts)
    total = sum(int(max_len * w / sum(w for _, w in parts)) for _, w in parts)
    return _weighted_sample(parts, max(total, 1), seed)


def _gsm8k_to_grpo(row: dict[str, Any]) -> dict[str, Any] | None:
    question = row.get("question")
    answer = row.get("answer")
    if not question or not answer:
        return None
    return {
        "prompt": [
            {"role": "system", "content": "Solve the math problem step by step. Put the final numeric answer after ####."},
            {"role": "user", "content": question},
        ],
        "solution": answer.split("####")[-1].strip(),
    }


def _humaneval_to_grpo(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    entry = row.get("entry_point") or row.get("task_id")
    if not prompt:
        return None
    return {
        "prompt": [
            {
                "role": "system",
                "content": "Write a complete Python function. Return only valid Python code in a fenced block.",
            },
            {"role": "user", "content": prompt},
        ],
        "solution": str(entry),
    }


def _agent_trace_to_grpo(row: dict[str, Any]) -> dict[str, Any] | None:
    """Extract first user turn + expected tool call from agent traces."""
    conversations = row.get("conversations") or row.get("messages") or []
    if not conversations:
        return None
    msgs = normalize_messages(conversations)
    if len(msgs) < 2:
        return None
    # Use first user message as prompt; last assistant tool call as solution
    user_msgs = [m for m in msgs if m["role"] == "user"]
    asst_msgs = [m for m in msgs if m["role"] == "assistant"]
    if not user_msgs or not asst_msgs:
        return None
    tools = row.get("tools")
    prompt = []
    if tools:
        prompt.append({"role": "system", "content": f"List of tools: {tools}"})
    prompt.append(user_msgs[0])
    solution = asst_msgs[0]["content"]
    if TOOL_CALL_START not in solution:
        return None
    return {"prompt": prompt, "solution": solution}


def build_sft_dataset(dataset_cfg: dict[str, Any]) -> Dataset:
    """Delegate to mixed SFT builder."""
    return build_mixed_dataset(dataset_cfg)
