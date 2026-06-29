"""Shared training utilities for Mars-1.0 LFM post-training pipeline."""

from __future__ import annotations

import hashlib

# Monkey-patch: PEFT 0.19.1's build_peft_weight_mapping passes unexpected kwargs
# (distributed_operation, quantization_operation, etc.) to WeightConverter.__init__
# which doesn't accept them. This is a PEFT bug; we strip unrecognized kwargs.
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import peft.utils.transformers_weight_conversion as _peft_twc

_orig_wc_init = _peft_twc.WeightConverter.__init__
_wc_params = set(inspect.signature(_orig_wc_init).parameters.keys())


def _patched_wc_init(self, *args, **kwargs):
    filtered = {k: v for k, v in kwargs.items() if k in _wc_params}
    return _orig_wc_init(self, *args, **filtered)


_peft_twc.WeightConverter.__init__ = _patched_wc_init

import torch
import yaml
from datasets import Dataset, load_from_disk
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from volumes import (
    DEFAULT_MODEL_ID,
    TOKENIZED_DIR,
    base_model_dir,
    ensure_volume_layout,
    stage_output_dir,
)

# LiquidAI/LFM2.5-8B-A1B — instruct checkpoint (tool-native, BFCL-ready).
# Post-train from here rather than Base for faster agent gains.
# https://docs.liquid.ai/lfm/models/lfm25-8b-a1b


class VolumeCommitCallback(TrainerCallback):
    """Persist Modal Volume after each checkpoint save."""

    def __init__(self, volume: Any | None = None) -> None:
        self._volume = volume

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        if self._volume is not None:
            print(f"Committing volume after checkpoint step {state.global_step}...")
            self._volume.commit()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def find_latest_final_adapter(stage: str, project_name: str) -> Path | None:
    root = stage_output_dir(stage, project_name)
    if not root.exists():
        return None
    finals = list(root.glob("*/final"))
    if finals:
        return max(finals, key=lambda p: p.stat().st_mtime)
    # Fall back to latest numbered checkpoint in most recent run dir
    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not run_dirs:
        latest_ckpt = find_latest_checkpoint(root)
        return Path(latest_ckpt) if latest_ckpt else None
    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    ckpt = find_latest_checkpoint(latest_run)
    return Path(ckpt) if ckpt else None


def resolve_adapter_path(
    cfg: dict[str, Any], *, default_stage: str | None = None
) -> Path | None:
    """Resolve LoRA adapter from explicit path, previous stage, or 'latest'."""
    explicit = cfg.get("adapter_path")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"adapter_path not found: {path}")

    prev = cfg.get("previous_stage")
    if prev:
        stage = prev.get("stage", default_stage or "sft")
        project = prev.get("project_name", cfg["project_name"])
        adapter = find_latest_final_adapter(stage, project)
        if adapter:
            return adapter
        raise FileNotFoundError(f"No adapter found for stage={stage} project={project}")

    resume = cfg.get("resume_adapter")
    if resume == "latest" and default_stage:
        adapter = find_latest_final_adapter(default_stage, cfg["project_name"])
        if adapter:
            return adapter
    return None


def resolve_model_path(model_id: str, cfg: dict[str, Any]) -> Path:
    from download_base import download_base

    explicit = cfg.get("model_path") or cfg.get("base_path")
    if explicit:
        return Path(explicit)
    local = base_model_dir(model_id)
    if local.exists() and any(local.iterdir()):
        return local
    return download_base(model_id, token=os.environ.get("HF_TOKEN"))


def build_moe_lora_config(peft_cfg: dict[str, Any]) -> LoraConfig:
    """MoE LoRA per leap-finetune MOE_LORA / Liquid TRL docs."""
    return LoraConfig(
        r=int(peft_cfg.get("r", 16)),
        lora_alpha=int(peft_cfg.get("lora_alpha", 32)),
        lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
        bias="none",
        target_modules=peft_cfg.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(
    model_id: str,
    cfg: dict[str, Any],
    *,
    adapter_path: Path | None = None,
    peft_cfg: dict[str, Any] | None = None,
    trainable_adapter: bool = True,
):
    """Load LFM2.5 MoE with optional existing LoRA adapter."""
    model_path = resolve_model_path(model_id, cfg)
    print(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = None
    if adapter_path is not None:
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=trainable_adapter,
        )
    elif peft_cfg and peft_cfg.get("enabled", True):
        lora_config = build_moe_lora_config(peft_cfg)

    return model, tokenizer, lora_config


def write_run_meta(
    output_dir: Path,
    cfg: dict[str, Any],
    *,
    stage: str,
    resume_from: str | None = None,
    adapter_path: str | None = None,
) -> None:
    meta = {
        "stage": stage,
        "project_name": cfg["project_name"],
        "model_id": cfg.get("model_id", DEFAULT_MODEL_ID),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resume_from": resume_from,
        "adapter_path": adapter_path,
        "training": cfg.get("training", {}),
        "peft": cfg.get("peft", {}),
        "dataset": cfg.get("dataset", {}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def prepare_run(
    cfg: dict[str, Any], stage: str
) -> tuple[Path, str | None, Path | None]:
    ensure_volume_layout()
    run_name = cfg.get("run_name") or datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    output_dir = stage_output_dir(stage, cfg["project_name"], run_name)

    training = cfg.get("training", {})
    resume_cfg = training.get("resume_from_checkpoint")
    resume_from: str | None = None
    if resume_cfg == "latest":
        resume_from = find_latest_checkpoint(output_dir)
    elif resume_cfg:
        resume_from = str(resume_cfg)

    adapter_path = resolve_adapter_path(cfg, default_stage=_prev_stage(stage))
    return output_dir, resume_from, adapter_path


def _prev_stage(stage: str) -> str | None:
    return {"dpo": "sft", "grpo": "dpo"}.get(stage)


def save_final(trainer, output_dir: Path, tokenizer, volume: Any | None = None) -> Path:
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if volume is not None:
        volume.commit()
    return final_dir


def get_or_cache_sft_dataset(
    dataset_cfg: dict[str, Any],
    tokenizer,
    max_length: int,
    model_id: str,
    project_name: str = "mars_sft_agent",
    force_rebuild: bool = False,
) -> Dataset:
    """Build and cache tokenized SFT dataset.

    Cache key = MD5 of (dataset mix config + model_id + max_length).
    Saves/loads tokenized dataset to/from /vol/tokenized/<project>/<cache_key>/
    """
    # Build deterministic cache key
    cache_input = {
        "mix": dataset_cfg.get("mix", []),
        "model_id": model_id,
        "max_length": max_length,
    }
    cache_key = hashlib.md5(
        json.dumps(cache_input, sort_keys=True, default=str).encode()
    ).hexdigest()

    cache_dir = TOKENIZED_DIR / project_name / cache_key

    if not force_rebuild and cache_dir.exists() and any(cache_dir.iterdir()):
        print(f"Loading tokenized dataset from cache: {cache_dir}")
        return load_from_disk(str(cache_dir))

    # Build raw formatted dataset
    print("Building formatted dataset...")
    from dataset_mix import build_mixed_dataset

    raw_dataset = build_mixed_dataset(dataset_cfg)
    print(f"Formatted dataset: {len(raw_dataset)} samples")

    # Tokenize with apply_chat_template
    def tokenize_fn(batch):
        texts = tokenizer.apply_chat_template(
            batch["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print(f"Tokenizing {len(raw_dataset)} samples (max_length={max_length})...")
    tokenized_dataset = raw_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["messages"],
        desc="Tokenizing",
    )

    # Save to cache
    print(f"Saving tokenized dataset to cache: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(str(cache_dir))

    return tokenized_dataset
