"""Modal Volume path layout for Mars-1.0 LFM training.

Matches notes.md §9:
  /vol/bases, /vol/checkpoints/{sft,dpo,rl}, /vol/data, /vol/exports, /vol/evals
"""

import os
from pathlib import Path

# Modal mounts the volume at /vol; local runs can override this without changing configs.
VOL_ROOT = Path(os.environ.get("MARS_VOL_ROOT", "/vol"))

BASES_DIR = VOL_ROOT / "bases"
CHECKPOINTS_DIR = VOL_ROOT / "checkpoints"
SFT_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "sft"
DPO_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "dpo"
RL_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "rl"
DATA_DIR = VOL_ROOT / "data"
TOKENIZED_DIR = VOL_ROOT / "tokenized"
EXPORTS_DIR = VOL_ROOT / "exports"
EVALS_DIR = VOL_ROOT / "evals"

STAGE_DIRS = {
    "sft": SFT_CHECKPOINTS_DIR,
    "dpo": DPO_CHECKPOINTS_DIR,
    "grpo": RL_CHECKPOINTS_DIR,
    "rl": RL_CHECKPOINTS_DIR,
}

VOLUME_DIRS = (
    BASES_DIR,
    SFT_CHECKPOINTS_DIR,
    DPO_CHECKPOINTS_DIR,
    RL_CHECKPOINTS_DIR,
    DATA_DIR,
    TOKENIZED_DIR,
    EXPORTS_DIR,
    EVALS_DIR,
)

# Instruct checkpoint — tool-native starting point for agent harnesses (Cursor, Zed, etc.)
DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-8B-A1B"


def ensure_volume_layout() -> None:
    for path in VOLUME_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def base_model_dir(model_id: str = DEFAULT_MODEL_ID) -> Path:
    """HF repo id → stable directory under /vol/bases/."""
    return BASES_DIR / model_id.replace("/", "__")


def stage_output_dir(stage: str, project_name: str, run_name: str | None = None) -> Path:
    root = STAGE_DIRS[stage] / project_name
    return root / run_name if run_name else root
