"""TRL DPOTrainer with LoRA for LFM2.5 MoE — preference alignment stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trl import DPOConfig, DPOTrainer

from common import (
    VolumeCommitCallback,
    load_config,
    load_model_and_tokenizer,
    prepare_run,
    save_final,
    write_run_meta,
)
from dataset_loaders import build_dpo_dataset
from volumes import DEFAULT_MODEL_ID


def run_dpo(cfg: dict[str, Any], *, volume: Any | None = None) -> Path:
    output_dir, resume_from, adapter_path = prepare_run(cfg, "dpo")
    model_id = cfg.get("model_id", DEFAULT_MODEL_ID)
    training = cfg.get("training", {})
    peft_cfg = cfg.get("peft", {})

    if not adapter_path and not cfg.get("previous_stage"):
        raise ValueError(
            "DPO requires `previous_stage` or `adapter_path` pointing to SFT LoRA adapter"
        )

    if resume_from:
        print(f"Resuming DPO from {resume_from}")
    else:
        write_run_meta(
            output_dir,
            cfg,
            stage="dpo",
            adapter_path=str(adapter_path) if adapter_path else None,
        )

    print("Building DPO dataset...")
    train_dataset = build_dpo_dataset(cfg["dataset"])
    print(f"Training on {len(train_dataset)} preference pairs")

    model, tokenizer, lora_config = load_model_and_tokenizer(
        model_id,
        cfg,
        adapter_path=adapter_path,
        peft_cfg=peft_cfg if not adapter_path else None,
    )

    save_steps = int(training.get("save_steps", 200))
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 16)),
        learning_rate=float(training.get("learning_rate", 5e-7)),
        beta=float(training.get("beta", 0.1)),
        lr_scheduler_type=training.get("lr_scheduler_type", "linear"),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        logging_steps=int(training.get("logging_steps", 10)),
        save_strategy=training.get("save_strategy", "steps"),
        save_steps=save_steps,
        save_total_limit=int(training.get("save_total_limit", 3)),
        bf16=True,
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        max_length=int(training.get("max_length", 4096)),
        report_to=training.get("report_to", "none"),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[VolumeCommitCallback(volume)],
    )

    trainer.train(resume_from_checkpoint=resume_from)
    final_dir = save_final(trainer, output_dir, tokenizer, volume)
    print(f"DPO complete → {final_dir}")
    return output_dir


def main(config_path: str) -> None:
    run_dpo(load_config(config_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)
