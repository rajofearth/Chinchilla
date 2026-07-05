"""Build mixed SFT datasets with Liquid LFM2.5 chat / tool-call formatting."""

from __future__ import annotations

import json
import random
import re
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

# LFM2.5 tool tokens (see https://docs.liquid.ai/lfm/key-concepts/tool-use)
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"

SHAREGPT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
}

HERMES_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
HERMES_TOOL_RESPONSE_RE = re.compile(
    r"<tool_response>\s*(.*?)\s*</tool_response>",
    re.DOTALL,
)
THINKING_RE = re.compile(
    r"<\s*redacted_thinking\s*>.*?</\s*redacted_thinking\s*>", re.DOTALL
)


def _pythonic_args(arguments: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in arguments.items():
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'{key}="{escaped}"')
        else:
            parts.append(f"{key}={json.dumps(value)}")
    return ", ".join(parts)


def json_tool_call_to_lfm(name: str, arguments: dict[str, Any]) -> str:
    args = _pythonic_args(arguments)
    inner = f"{name}({args})" if args else f"{name}()"
    return f"{TOOL_CALL_START}[{inner}]{TOOL_CALL_END}"


def convert_inline_tool_calls(text: str) -> str:
    """Convert Hermes-style <tool_call>{json}</tool_call> blocks to LFM bracket notation."""

    def _replace(match: re.Match[str]) -> str:
        payload = json.loads(match.group(1))
        if isinstance(payload, list):
            return "".join(
                json_tool_call_to_lfm(item["name"], item.get("arguments", {}))
                for item in payload
            )
        return json_tool_call_to_lfm(payload["name"], payload.get("arguments", {}))

    return HERMES_TOOL_CALL_RE.sub(_replace, text)


def strip_hermes_wrappers(text: str) -> str:
    text = THINKING_RE.sub("", text)
    text = HERMES_TOOL_RESPONSE_RE.sub(lambda m: m.group(1).strip(), text)
    return text.strip()


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Ensure role/content strings and apply LFM tool-call conversion on assistant turns."""
    out: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role") or msg.get("from")
        content = msg.get("content") or msg.get("value") or msg.get("text")
        if role is None or content is None:
            continue
        role = SHAREGPT_ROLE_MAP.get(str(role), str(role))
        content = str(content)

        if role == "assistant":
            content = strip_hermes_wrappers(content)
            content = convert_inline_tool_calls(content)
        elif role == "tool":
            content = strip_hermes_wrappers(content)
        elif role == "user" and content.lstrip().startswith("<tool_response>"):
            # Some Hermes traces replay tool output as user turns.
            m = HERMES_TOOL_RESPONSE_RE.search(content)
            if m:
                out.append({"role": "tool", "content": m.group(1).strip()})
                continue
            content = strip_hermes_wrappers(content)

        if content:
            out.append({"role": role, "content": content})
    return out


def format_apigen_row(row: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    """Convert argilla/Synth-APIGen row → LFM messages with tool definitions in system prompt."""
    answers_raw = row.get("answers") or ""
    if not answers_raw.startswith("["):
        return None

    try:
        calls = json.loads(answers_raw)
    except json.JSONDecodeError:
        return None
    if not calls:
        return None

    tools = row.get("tools") or "[]"
    query = row.get("query") or ""
    if not query:
        return None

    assistant_parts = [
        json_tool_call_to_lfm(call["name"], call.get("arguments", {})) for call in calls
    ]
    return {
        "messages": [
            {"role": "system", "content": f"List of tools: {tools}"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "".join(assistant_parts)},
        ]
    }


def format_smoltalk_row(row: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    messages = row.get("messages")
    if not messages:
        return None
    normalized = normalize_messages(messages)
    if len(normalized) < 2:
        return None
    return {"messages": normalized}


def format_hermes_trace(row: dict[str, Any]) -> dict[str, list[dict[str, str]]] | None:
    conversations = row.get("conversations") or row.get("messages")
    if not conversations:
        return None

    normalized = normalize_messages(conversations)
    if not normalized:
        return None

    # Ensure tool definitions appear in system prompt when provided separately.
    tools = row.get("tools")
    if tools and normalized[0]["role"] == "system":
        if "List of tools:" not in normalized[0]["content"]:
            normalized[0]["content"] = (
                normalized[0]["content"].rstrip()
                + f"\n\nList of tools: {tools if isinstance(tools, str) else json.dumps(tools)}"
            )
    elif tools:
        normalized.insert(
            0,
            {
                "role": "system",
                "content": f"List of tools: {tools if isinstance(tools, str) else json.dumps(tools)}",
            },
        )

    if not any(m["role"] == "assistant" for m in normalized):
        return None
    return {"messages": normalized}


def _load_source(name: str, spec: dict[str, Any]) -> Dataset:
    path = spec["path"]
    split = spec.get("split", "train")
    subset = spec.get("subset")
    kwargs: dict[str, Any] = {"path": path, "split": split}
    if subset is not None:
        kwargs["name"] = subset
    ds = load_dataset(**kwargs)
    limit = spec.get("limit")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def format_codefeedback_row(
    row: dict[str, Any],
) -> dict[str, list[dict[str, str]]] | None:
    """HuggingFaceH4/Code-Feedback → messages.

    The dataset has a pre-formatted `messages` column already in OpenAI
    chat format (user/assistant turns). Falls back to instruction/answer
    columns if messages is missing.
    """
    messages = row.get("messages")
    if messages:
        normalized = normalize_messages(messages)
        if len(normalized) >= 2:
            return {"messages": normalized}
    instruction = row.get("instruction") or row.get("query")
    answer = row.get("answer") or row.get("response")
    if not instruction or not answer:
        return None
    return {
        "messages": [
            {"role": "user", "content": str(instruction)},
            {"role": "assistant", "content": str(answer)},
        ]
    }


def _format_source(name: str, ds: Dataset) -> Dataset:
    formatters = {
        "synth_apigen": format_apigen_row,
        "smoltalk": format_smoltalk_row,
        "agent_traces": format_hermes_trace,
        "codefeedback": format_codefeedback_row,
    }
    fn = formatters.get(name)
    if fn is None:
        raise ValueError(f"Unknown dataset source: {name}")

    formatted = []
    for row in ds:
        sample = fn(row)
        if sample is not None:
            formatted.append(sample)
    if not formatted:
        raise RuntimeError(f"No valid rows after formatting source {name!r}")
    return Dataset.from_list(formatted)


def _sample_by_weight(
    datasets: list[tuple[Dataset, float]], total: int, seed: int
) -> Dataset:
    weights = [w for _, w in datasets]
    weight_sum = sum(weights)
    counts = [max(1, int(total * w / weight_sum)) for w in weights]

    rng = random.Random(seed)
    parts: list[Dataset] = []
    for (ds, _), count in zip(datasets, counts):
        if count >= len(ds):
            parts.append(ds)
        else:
            indices = rng.sample(range(len(ds)), count)
            parts.append(ds.select(indices))
    return concatenate_datasets(parts).shuffle(seed=seed)


def build_mixed_dataset(dataset_cfg: dict[str, Any]) -> Dataset:
    """Build weighted mix from config `dataset.mix` block."""
    mix = dataset_cfg.get("mix")
    if not mix:
        raise ValueError("dataset.mix is required")

    seed = dataset_cfg.get("seed", 42)
    total_limit = dataset_cfg.get("limit")

    formatted_sets: list[tuple[Dataset, float]] = []
    for spec in mix:
        name = spec["name"]
        weight = float(spec.get("weight", 1.0))
        raw = _load_source(name, spec)
        formatted = _format_source(name, raw)
        formatted_sets.append((formatted, weight))
        print(f"  {name}: {len(formatted)} rows (weight={weight})")

    if total_limit is not None:
        return _sample_by_weight(formatted_sets, total_limit, seed)

    # No global limit — concatenate all sources scaled by relative weight caps.
    max_len = max(len(ds) for ds, _ in formatted_sets)
    total = sum(
        int(max_len * w / sum(w for _, w in formatted_sets)) for _, w in formatted_sets
    )
    return _sample_by_weight(formatted_sets, max(total, 1), seed)
