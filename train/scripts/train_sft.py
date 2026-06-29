"""TRL SFTTrainer with LoRA for Liquid LFM2.5 MoE models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common import (
    VolumeCommitCallback,
    get_or_cache_sft_dataset,
    load_config,
    load_model_and_tokenizer,
    prepare_run,
    save_final,
    write_run_meta,
)
from trl import SFTConfig, SFTTrainer
from volumes import DEFAULT_MODEL_ID


def run_sft(cfg: dict[str, Any], *, volume: Any | None = None) -> Path:
    """Run LoRA SFT from a parsed YAML config dict."""
    output_dir, resume_from, adapter_path = prepare_run(cfg, "sft")
    model_id = cfg.get("model_id", DEFAULT_MODEL_ID)
    training = cfg.get("training", {})
    peft_cfg = cfg.get("peft", {})

    if resume_from:
        print(f"Resuming SFT from {resume_from}")
    elif not adapter_path:
        write_run_meta(output_dir, cfg, stage="sft", resume_from=None)

    # Load model first (need tokenizer for dataset caching)
    print("Loading model and tokenizer...")
    model, tokenizer, lora_config = load_model_and_tokenizer(
        model_id,
        cfg,
        adapter_path=adapter_path,
        peft_cfg=peft_cfg if not adapter_path else None,
    )

    # Build/cache tokenized dataset
    dataset_cfg = cfg.get("dataset", {})
    max_length = int(training.get("max_length", 4096))
    force_rebuild = dataset_cfg.get("force_rebuild", False)
    print("Building/caching SFT dataset...")
    train_dataset = get_or_cache_sft_dataset(
        dataset_cfg,
        tokenizer,
        max_length=max_length,
        model_id=model_id,
        project_name=cfg.get("project_name", "mars_sft_agent"),
        force_rebuild=force_rebuild,
    )
    print(f"Training on {len(train_dataset)} tokenized samples")

    save_steps = int(training.get("save_steps", 200))
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 16)
        ),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        logging_steps=int(training.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(training.get("save_total_limit", 5)),
        bf16=True,
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        max_length=max_length,
        packing=bool(training.get("packing", False)),
        dataset_text_field=None,
        report_to=training.get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[VolumeCommitCallback(volume)],
    )

    trainer.train(resume_from_checkpoint=resume_from)
    final_dir = save_final(trainer, output_dir, tokenizer, volume)
    print(f"SFT complete → {final_dir}")
    return output_dir


def main(config_path: str) -> None:
    run_sft(load_config(config_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML training config")
    args = parser.parse_args()
    main(args.config)
