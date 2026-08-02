"""Modal app for Mars-1.0 full post-training: SFT → DPO → GRPO → eval → export.

Starting model: LiquidAI/LFM2.5-8B-A1B (instruct, tool-native)
Volume layout: notes.md §9 under /vol/

Prerequisites:
  modal setup
  modal secret create huggingface-secret HF_TOKEN=<token>

Usage:
  # Download instruct model once
  modal run train/modal_app.py --download-only

  # Individual stages (recommended — 14hr timeout each)
  modal run --detach train/modal_app.py --stage sft --config sft_agent.yaml
  modal run --detach train/modal_app.py --stage sft --config sft_code.yaml
  modal run --detach train/modal_app.py --stage dpo  --config dpo.yaml
  modal run --detach train/modal_app.py --stage grpo --config grpo.yaml
  modal run train/modal_app.py --stage eval   --config eval_export.yaml
  modal run train/modal_app.py --stage export --config eval_export.yaml

  # Full RL pipeline (SFT agent → SFT code → DPO → GRPO) — chain detached spawns
  modal run train/modal_app.py --stage full --detach

Monitor:
  modal app logs mars-lfm-train
  modal volume ls mars-train-vol
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal
import yaml

_TRAIN_DIR = Path(__file__).parent
_SCRIPTS_DIR = _TRAIN_DIR / "scripts"
_PIPELINE_CONFIG = _TRAIN_DIR / "configs" / "pipeline.yaml"

VOL_ROOT = Path("/vol")

_pipeline = yaml.safe_load(_PIPELINE_CONFIG.read_text(encoding="utf-8"))
_modal_defaults = _pipeline.get("modal", {})
DEFAULT_VOLUME = _modal_defaults.get("volume", "mars-train-vol")
DEFAULT_GPU = _modal_defaults.get("gpu", "A10G")
DEFAULT_TIMEOUT = int(_modal_defaults.get("timeout_hours", 14) * 3600)
DEFAULT_MODEL = _pipeline.get("model_id", "LiquidAI/LFM2.5-8B-A1B")

app = modal.App(_modal_defaults.get("app_name", "mars-lfm-train"))
volume = modal.Volume.from_name(DEFAULT_VOLUME, create_if_missing=True)

try:
    hf_secret = modal.Secret.from_name("huggingface-secret")
except modal.exception.NotFoundError:
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": ""})

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=5.2.0",
        "trl>=0.19.0",
        "peft>=0.14.0",  # weight will monkey-patch WeightConverter.__init__ to fix distributed_operation bug
        "datasets>=3.0.0",
        "accelerate>=1.2.0",
        "huggingface_hub>=0.27.0",
        "safetensors>=0.5.0",
        "pyyaml>=6.0",
        "sentencepiece",
    )
    .env({"HF_HOME": "/vol/data/hf-cache", "TRANSFORMERS_CACHE": "/vol/data/hf-cache"})
    .add_local_dir(str(_SCRIPTS_DIR), remote_path="/root/scripts")
    .add_local_dir(str(_TRAIN_DIR / "configs"), remote_path="/root/configs")
    .add_local_dir(str(_TRAIN_DIR / "evals"), remote_path="/root/evals")
)


def _ensure_scripts_path() -> None:
    scripts = "/root/scripts"
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def _config_path(name: str) -> str:
    return f"/root/configs/{Path(name).name}"


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={str(VOL_ROOT): volume},
    timeout=3600,
)
def download_model(model_id: str = DEFAULT_MODEL) -> str:
    _ensure_scripts_path()
    from download_base import download_base

    dest = download_base(model_id)
    volume.commit()
    return str(dest)


@app.function(
    image=image,
    secrets=[hf_secret],
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT,
    volumes={str(VOL_ROOT): volume},
)
def train_sft(config: str = "sft_agent.yaml") -> str:
    _ensure_scripts_path()
    from common import load_config
    from train_sft import run_sft

    out = run_sft(load_config(_config_path(config)), volume=volume)
    volume.commit()
    return str(out)


@app.function(
    image=image,
    secrets=[hf_secret],
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT,
    volumes={str(VOL_ROOT): volume},
)
def train_dpo(config: str = "dpo.yaml") -> str:
    _ensure_scripts_path()
    from common import load_config
    from train_dpo import run_dpo

    out = run_dpo(load_config(_config_path(config)), volume=volume)
    volume.commit()
    return str(out)


@app.function(
    image=image,
    secrets=[hf_secret],
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT,
    volumes={str(VOL_ROOT): volume},
)
def train_grpo(config: str = "grpo.yaml") -> str:
    _ensure_scripts_path()
    from common import load_config
    from train_grpo import run_grpo

    out = run_grpo(load_config(_config_path(config)), volume=volume)
    volume.commit()
    return str(out)


@app.function(
    image=image,
    secrets=[hf_secret],
    gpu=DEFAULT_GPU,
    timeout=3600,
    volumes={str(VOL_ROOT): volume},
)
def eval_agent(config: str = "eval_export.yaml") -> str:
    _ensure_scripts_path()
    from common import load_config
    from eval_agent import run_eval

    out = run_eval(load_config(_config_path(config)), volume=volume)
    volume.commit()
    return str(out)


@app.function(
    image=image,
    secrets=[hf_secret],
    gpu=DEFAULT_GPU,
    timeout=3600,
    volumes={str(VOL_ROOT): volume},
)
def export_model(config: str = "eval_export.yaml") -> str:
    _ensure_scripts_path()
    from common import load_config
    from export_hf import export_merged

    out = export_merged(load_config(_config_path(config)), volume=volume)
    volume.commit()
    return str(out)


@app.function(
    image=image,
    secrets=[hf_secret],
    gpu=DEFAULT_GPU,
    timeout=min(int(DEFAULT_TIMEOUT * 4), 86400),
    volumes={str(VOL_ROOT): volume},
)
def run_full_pipeline() -> dict[str, str]:
    """Run SFT agent → SFT code → DPO → GRPO sequentially (long-running)."""
    _ensure_scripts_path()
    from common import load_config
    from download_base import download_base
    from train_dpo import run_dpo
    from train_grpo import run_grpo
    from train_sft import run_sft

    stages = _pipeline.get(
        "pipeline_stages", ["sft_agent.yaml", "sft_code.yaml", "dpo.yaml", "grpo.yaml"]
    )
    download_base(DEFAULT_MODEL)
    volume.commit()

    results: dict[str, str] = {}
    runners = {
        "sft_agent.yaml": lambda c: run_sft(c, volume=volume),
        "sft_code.yaml": lambda c: run_sft(c, volume=volume),
        "dpo.yaml": lambda c: run_dpo(c, volume=volume),
        "grpo.yaml": lambda c: run_grpo(c, volume=volume),
    }
    for cfg_name in stages:
        print(f"\n{'=' * 60}\nStage: {cfg_name}\n{'=' * 60}")
        cfg = load_config(_config_path(cfg_name))
        runner = runners.get(Path(cfg_name).name)
        if not runner:
            raise ValueError(f"No runner for {cfg_name}")
        out = runner(cfg)
        results[cfg_name] = str(out)
        volume.commit()

    return results


@app.local_entrypoint()
def main(
    config: str = "sft_agent.yaml",
    stage: str = "sft",
    detach: bool = False,
    download_only: bool = False,
):
    """Run a training stage on Modal.

    stage: sft | dpo | grpo | eval | export | full | download
    """
    stage_map = {
        "download": download_model,
        "sft": train_sft,
        "dpo": train_dpo,
        "grpo": train_grpo,
        "eval": eval_agent,
        "export": export_model,
        "full": run_full_pipeline,
    }

    if download_only or stage == "download":
        with modal.enable_output():
            result = download_model.remote(DEFAULT_MODEL)
        print(f"Model at: {result}")
        return

    fn = stage_map.get(stage)
    if fn is None:
        raise ValueError(f"Unknown stage {stage!r}. Choose from: {list(stage_map)}")

    # Always use .spawn(): spawned Function Calls are durable and are not
    # canceled when the local caller disconnects (unlike .remote()). Combined
    # with `modal run --detach`, long training runs survive terminal closes
    # or the local client being killed. .get() blocks until the run finishes.
    with modal.enable_output():
        if stage == "full":
            result = fn.spawn().get()
        else:
            result = fn.spawn(config).get()

    if detach:
        print(f"Detached ({stage}) — monitor: modal app logs mars-lfm-train")
    else:
        print(f"Done ({stage}): {result}")
