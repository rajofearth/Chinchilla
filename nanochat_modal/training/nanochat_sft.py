"""Supervised Fine-Tuning of nanochat d6 on Modal.

Trains the d6 (6-layer, 384-dim, ~73.5M param) model on SmolTalk + MMLU + GSM8K
for 1500 steps. Uses 2x A10G GPUs for best cost/performance.

Optionally uses knowledge distillation (sorted-distribution KD) from a teacher
model (default: LFM2.5-350M) to improve the student's output distributions.
The teacher runs a frozen forward pass per batch; its logits are sorted and
compared with the student's at proportionally-aligned positions.

The pretrain checkpoint is cloned from GitHub in-container (fast), no local upload needed.

Prerequisites:
    pip install modal
    modal setup
    modal volume create nanochat-vol

Usage:
    # Full pipeline with teacher KD (checkpoint → tokenizer → teacher → SFT):
    modal run --detach nanochat_sft.py

    # Full pipeline without KD (cheaper, ~$1):
    modal run --detach nanochat_sft.py --use-teacher=False

    # Just SFT (assumes checkpoint + tokenizer + teacher already on volume):
    modal run --detach nanochat_sft.py::run_sft

    # Download the SFT checkpoint:
    modal run nanochat_sft.py::download_sft

    # Test inference with the SFT model:
    modal run nanochat_sft.py::test_inference --prompt "What is 2+2?"
"""

from pathlib import Path

import modal

# ── Script-relative paths for local file mounting ──────────────────────────
_SCRIPT_DIR = Path(__file__).parent.parent  # nanochat_modal/

# ── Paths ──────────────────────────────────────────────────────────────────

VOLUME_DIR = Path("/vol")
VOLUME_CHECKPOINT_DIR = VOLUME_DIR / "base_checkpoints" / "d6"
TOKENIZER_DIR = VOLUME_DIR / "tokenizer"
DATA_DIR = VOLUME_DIR / "base_data"
SFT_CHECKPOINT_DIR = VOLUME_DIR / "chatsft_checkpoints" / "d6"
IDENTITY_CONVOS = VOLUME_DIR / "identity_conversations.jsonl"
TEACHER_MODEL_DIR = VOLUME_DIR / "teacher_models" / "lfm25_350m"
TEACHER_HF_ID = "LiquidAI/LFM2.5-350M"  # safetensors, 354M params, Lfm2ForCausalLM arch

# ── Hyperparameters ────────────────────────────────────────────────────────

MAX_SEQ_LEN = 2048
DEVICE_BATCH_SIZE = 2  # reduced from 4 for KD (student + teacher need more memory)
TOTAL_BATCH_SIZE = 524288
NUM_ITERATIONS = 1500
EVAL_EVERY = 200
SAVE_EVERY = 200  # save checkpoint every N optimizer steps (0 = only at end)
EVAL_TOKENS = 131072

# ── Knowledge Distillation (optional) ──────────────────────────────────────
# Set USE_TEACHER=False to skip teacher download and run plain SFT.
USE_TEACHER = True
KD_ALPHA = 0.9  # CE loss weight (0.9 = 90% CE, 10% KD)
KD_TEMPERATURE = 2.0  # softmax temperature for KD
KD_TOP_K = 512  # top-k logits to compare

# ── Modal App & Volumes ────────────────────────────────────────────────────

app = modal.App("nanochat-sft")
volume = modal.Volume.from_name("nanochat-vol", create_if_missing=True)

# ── Container Image ────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "torchaudio==2.3.0",
        "rustbpe",
        "tiktoken",
        "huggingface_hub",
        "accelerate",  # for device_map in teacher model loading
        "datasets",
        "transformers>=4.48.0,<5.0.0",
        "tokenizers",
        "httpx",
        "pyarrow",
        "sentencepiece",
        "wandb",
        "psutil",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_file(
        str(_SCRIPT_DIR / "nanochat" / "nanochat" / "gpt.py"),
        "/patched/nanochat/gpt.py",
    )
    .add_local_file(
        str(_SCRIPT_DIR / "nanochat" / "scripts" / "chat_sft.py"),
        "/patched/scripts/chat_sft.py",
    )
)


# ── Helper: Setup nanochat repo ────────────────────────────────────────────


def setup_nanochat():
    """Clone nanochat if needed and apply patches."""
    import os
    import shutil
    import subprocess

    repo_path = "/nanochat"
    if not os.path.exists(repo_path):
        subprocess.run(
            ["git", "clone", "https://github.com/karpathy/nanochat.git", repo_path],
            check=True,
            capture_output=True,
        )
        print("✓ Cloned nanochat repo")

    os.chdir(repo_path)

    # Patch dataset.py: base_data_climbmix → base_data
    dset_path = os.path.join(repo_path, "nanochat", "dataset.py")
    with open(dset_path) as f:
        content = f.read()
    if "base_data_climbmix" in content:
        content = content.replace("base_data_climbmix", "base_data")
        with open(dset_path, "w") as f:
            f.write(content)
        print("✓ Patched dataset.py: climbmix → base_data")

    # Patch engine.py: make dtype configurable
    engine_path = os.path.join(repo_path, "nanochat", "engine.py")
    with open(engine_path) as f:
        content = f.read()
    old_line = 'dtype = torch.bfloat16 if device.type == "cuda" else torch.float32'
    new_line = (
        'dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, '
        '"float32": torch.float32}.get(os.environ.get("NANOCHAT_DTYPE", ""), '
        'torch.bfloat16 if device.type == "cuda" else torch.float32)'
    )
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(engine_path, "w") as f:
            f.write(content)
        print("✓ Patched engine.py: dtype respects NANOCHAT_DTYPE")
    elif "NANOCHAT_DTYPE" not in content:
        print("⚠ engine.py pattern mismatch")

    # Patch gpt.py: replace F.rms_norm (PyTorch 2.4+) with manual impl for 2.3.0 compat
    gpt_path = os.path.join(repo_path, "nanochat", "gpt.py")
    with open(gpt_path) as f:
        content = f.read()
    old_norm = "def norm(x):\n    return F.rms_norm(x, (x.size(-1),))"
    new_norm = (
        "def norm(x):\n"
        "    # Manual RMSNorm for PyTorch < 2.4 compatibility\n"
        "    var = x.pow(2).float().mean(-1, keepdim=True)\n"
        "    return (x.float() * torch.rsqrt(var + 1e-6)).to(x.dtype)"
    )
    if old_norm in content:
        content = content.replace(old_norm, new_norm)
        with open(gpt_path, "w") as f:
            f.write(content)
        print("✓ Patched gpt.py: F.rms_norm → manual RMSNorm")
    else:
        print("⚠ gpt.py norm function already patched or pattern changed")

    # Patch chat_sft.py: suppress torch.compile Dynamo errors
    sft_path = os.path.join(repo_path, "scripts", "chat_sft.py")
    with open(sft_path) as f:
        content = f.read()
    old_sft = "# ── Batch sizes"
    new_sft = (
        "import torch._dynamo\n"
        "torch._dynamo.config.suppress_errors = True  # fall back to eager on Dynamo errors\n"
        "# ── Batch sizes"
    )
    if "suppress_errors" not in content:
        content = content.replace(old_sft, new_sft)
        with open(sft_path, "w") as f:
            f.write(content)
        print("✓ Patched chat_sft.py: suppress_errors=True")
    else:
        print("⚠ chat_sft.py already patched")

    # Patch flash_attention.py: strip enable_gqa for PyTorch < 2.5 compat
    fa_path = os.path.join(repo_path, "nanochat", "flash_attention.py")
    with open(fa_path) as f:
        content = f.read()
    if "enable_gqa=enable_gqa" in content:
        content = content.replace(", enable_gqa=enable_gqa", "")
        with open(fa_path, "w") as f:
            f.write(content)
        print("✓ Patched flash_attention.py: stripped enable_gqa for PyTorch < 2.5")
    else:
        print("⚠ flash_attention.py already patched or pattern changed")

    # Copy patched files from image (return_logits flag, KD support, data mixture fix)
    patched_gpt = "/patched/nanochat/gpt.py"
    patched_sft = "/patched/scripts/chat_sft.py"
    if os.path.exists(patched_gpt):
        shutil.copy(patched_gpt, os.path.join(repo_path, "nanochat", "gpt.py"))
        print("✓ Deployed patched gpt.py (return_logits flag)")
    if os.path.exists(patched_sft):
        shutil.copy(patched_sft, os.path.join(repo_path, "scripts", "chat_sft.py"))
        print("✓ Deployed patched chat_sft.py (KD, data mixture, teacher loading)")

    print("nanochat setup complete")


# ── SFT (self-contained: handles checkpoint + tokenizer internally) ────────


@app.function(
    image=image,
    volumes={str(VOLUME_DIR): volume},
    gpu="A10G:2",
    memory=32768,
    timeout=86400,
)
def run_sft(
    use_teacher: bool = USE_TEACHER,
    kd_alpha: float = KD_ALPHA,
    kd_temperature: float = KD_TEMPERATURE,
    kd_top_k: int = KD_TOP_K,
    save_every: int = SAVE_EVERY,
    resume_step: int = -1,
):
    """Run SFT for 1500 steps on 2x A10G.

    Self-contained: downloads checkpoint + trains tokenizer + optionally teacher model.

    Args:
        use_teacher: If True, downloads teacher model and enables KD during SFT.
        kd_alpha: Weight for CE loss (0.9 = 90%% CE, 10%% KD).
        kd_temperature: Softmax temperature for sorted KD loss.
        kd_top_k: Number of top logits to compare in sorted KD loss.
    """
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    import urllib.request

    sys.path.insert(0, "/nanochat")
    setup_nanochat()

    os.environ["NANOCHAT_BASE_DIR"] = str(VOLUME_DIR)
    os.environ["WANDB_RUN"] = "dummy"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    # ── Step 0: Checkpoint ──
    if not (VOLUME_CHECKPOINT_DIR / "model_008600.pt").exists():
        print("\n=== Step 0: Checkpoint not found on volume ===")
        print("The pretrain checkpoint was not uploaded to the volume.")
        print("Run through the main entrypoint to auto-upload:")
        print("  modal run --detach nanochat_sft.py")
        print("")
        print("Or upload manually:")
        print(
            "  modal volume put nanochat-vol <path-to-model_008600.pt> /base_checkpoints/d6/model_008600.pt"
        )
        print(
            "  modal volume put nanochat-vol <path-to-meta_008600.json> /base_checkpoints/d6/meta_008600.json"
        )
        raise FileNotFoundError(
            "Pretrain checkpoint not found on volume. See instructions above."
        )

    # ── Step 1: Train tokenizer ──
    if not (TOKENIZER_DIR / "tokenizer.pkl").exists():
        print("\n=== Step 1: Training tokenizer ===")
        env = os.environ.copy()
        env["PYTHONPATH"] = "/nanochat"

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        existing = len(list(DATA_DIR.glob("*.parquet")))
        if existing < 10:
            print("Downloading 10 data shards...")
            subprocess.run(
                ["python", "-m", "nanochat.dataset", "-n", "10"],
                cwd="/nanochat",
                env=env,
                check=True,
                capture_output=True,
            )
            climbmix = VOLUME_DIR / "base_data_climbmix"
            if climbmix.exists():
                for f in climbmix.glob("*.parquet"):
                    shutil.move(str(f), str(DATA_DIR / f.name))

        print("Training tokenizer (~2-3 min)...")
        result = subprocess.run(
            ["python", "scripts/tok_train.py", "--max-chars=2000000000"],
            cwd="/nanochat",
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"tok_train.py stdout:\n{result.stdout[-2000:]}")
            print(f"tok_train.py stderr:\n{result.stderr[-2000:]}")
            result.check_returncode()
        volume.commit()
        print("✓ Tokenizer ready")

    # ── Step 2: Download identity conversations ──
    if not IDENTITY_CONVOS.exists():
        print("\n=== Downloading identity conversations ===")
        urllib.request.urlretrieve(
            "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl",
            str(IDENTITY_CONVOS),
        )
        volume.commit()

    # ── Step 3: Download teacher model (optional, for KD) ──
    teacher_path_str = ""
    if use_teacher:
        if not (TEACHER_MODEL_DIR / "config.json").exists():
            print(f"\n=== Step 3: Downloading teacher model {TEACHER_HF_ID} ===")
            from huggingface_hub import snapshot_download

            TEACHER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                TEACHER_HF_ID,
                local_dir=str(TEACHER_MODEL_DIR),
                ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
            )
            volume.commit()
            print("\u2713 Teacher model downloaded to volume")
        teacher_path_str = str(TEACHER_MODEL_DIR)

    # ── Step 4: Run SFT ──
    print("\n" + "=" * 60)
    print("Starting SFT on 2× A10G")
    print(f"  Steps: {NUM_ITERATIONS}")
    print(
        f"  Batch: {DEVICE_BATCH_SIZE}×{MAX_SEQ_LEN}×2 GPUs = {DEVICE_BATCH_SIZE * MAX_SEQ_LEN * 2} tok/microbatch"
    )
    print(f"  Grad accum: {TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * 2)}")
    print(f"  Total tokens/step: {TOTAL_BATCH_SIZE:,}")
    if teacher_path_str:
        print(
            f"  Teacher: {TEACHER_HF_ID} (KD enabled, alpha={KD_ALPHA}, T={KD_TEMPERATURE}"
        )
    if resume_step > 0:
        print(f"  Resume from step {resume_step} (auto-detect)")
    elif resume_step == -1:
        print(f"  Auto-resume: yes (will detect latest checkpoint)")
    if not teacher_path_str:
        print(f"  No teacher (plain SFT)")
    print(f"  load_optimizer=0 | eval_every={EVAL_EVERY} | save_every={save_every}")
    print("=" * 60)

    _grad_accum = TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * 2)
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "-m",
        "scripts.chat_sft",
        "--",  # ← CRITICAL: separates torchrun args from module args (avoids --run ambiguity)
        "--max-seq-len",
        str(MAX_SEQ_LEN),
        "--device-batch-size",
        str(DEVICE_BATCH_SIZE),
        "--total-batch-size",
        str(TOTAL_BATCH_SIZE),
        "--num-iterations",
        str(NUM_ITERATIONS * _grad_accum),
        "--eval-every",
        str(EVAL_EVERY),
        "--save-every",
        str(save_every),
        "--resume-step",
        str(resume_step),
        "--eval-tokens",
        str(EVAL_TOKENS),
        "--init-lr-frac",
        "0.1",
        "--warmup-ratio",
        "0.1",
        "--warmdown-ratio",
        "0.5",
        "--chatcore-every",
        "0",  # disabled during training (too slow), run eval separately
        "--final-lr-frac",
        "0.0",
        "--load-optimizer",
        "0",
        "--run",
        "dummy",
    ]
    if teacher_path_str:
        cmd.extend(["--teacher-model-path", teacher_path_str])
        cmd.extend(["--kd-alpha", str(kd_alpha)])
        cmd.extend(["--kd-temperature", str(kd_temperature)])
        cmd.extend(["--kd-top-k", str(kd_top_k)])

    print(f"Running: {' '.join(cmd)}")
    sys.stdout.flush()

    # Run with streaming so Modal logs show live progress
    import subprocess as _sp
    import threading as _threading
    import time as _time

    _process = _sp.Popen(
        cmd, cwd="/nanochat", stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1
    )

    # Background thread: periodically commit the volume so checkpoints survive
    # container kills. Check for new .pt files and commit when found.
    _last_ckpt_count = len(list(SFT_CHECKPOINT_DIR.glob("model_*.pt")))
    _commit_active = True

    def _volume_committer():
        nonlocal _last_ckpt_count
        while _commit_active:
            _time.sleep(30)  # check every 30s
            if not _commit_active:
                break
            try:
                cur = len(list(SFT_CHECKPOINT_DIR.glob("model_*.pt")))
                if cur > _last_ckpt_count:
                    volume.commit()
                    print(
                        f"[volume] Committed {cur - _last_ckpt_count} new checkpoint(s)"
                    )
                    _last_ckpt_count = cur
            except Exception:
                pass  # best-effort

    _committer = _threading.Thread(target=_volume_committer, daemon=True)
    _committer.start()

    # Stream every line in real-time so Modal logs show live progress
    for line in _process.stdout:
        print(line, end="", flush=True)
    _process.wait()
    _commit_active = False
    _committer.join(timeout=10)
    _result_returncode = _process.returncode

    volume.commit()
    print(f"\nSFT exited with code {_result_returncode}")

    if _result_returncode != 0:
        raise RuntimeError(f"SFT failed with code {_result_returncode}")

    # Show output checkpoints
    print("\n✓ SFT checkpoints:")
    for f in sorted(SFT_CHECKPOINT_DIR.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB)")

    return _result_returncode


@app.function(
    image=image,
    volumes={str(VOLUME_DIR): volume},
    gpu="T4",
    timeout=300,
)
def test_inference(prompt: str = "What is 2 + 2?"):
    """Test the SFT model by generating a response."""
    import os
    import sys

    import torch

    sys.path.insert(0, "/nanochat")
    setup_nanochat()
    os.environ["NANOCHAT_BASE_DIR"] = str(VOLUME_DIR)

    from nanochat.checkpoint_manager import load_model_from_dir
    from nanochat.tokenizer import get_tokenizer

    device = torch.device("cuda")
    tokenizer = get_tokenizer()

    sft_dir = str(SFT_CHECKPOINT_DIR)
    parent_dir = str(SFT_CHECKPOINT_DIR.parent)  # /vol/chatsft_checkpoints
    tag = SFT_CHECKPOINT_DIR.name  # "d6"
    if not os.path.exists(sft_dir) or not any(
        f.endswith(".pt") for f in os.listdir(sft_dir)
    ):
        print(f"✗ No SFT checkpoint at {sft_dir}")
        # Try fallback path
        sft_dir = os.path.join("/nanochat", "chatsft_checkpoints", "d6")
        parent_dir = str(Path(sft_dir).parent)
        tag = "d6"
        if not os.path.exists(sft_dir):
            print(f"✗ No SFT checkpoint found anywhere")
            return

    print(f"Loading {tag} from {parent_dir}...")
    model, _, meta = load_model_from_dir(
        parent_dir, device=device, phase="eval", model_tag=tag
    )
    model.eval()
    print(f"✓ Loaded SFT step {meta.get('step', '?')}")

    ids, _ = tokenizer.render_conversation(
        {"messages": [{"role": "user", "content": prompt}]},
        max_tokens=MAX_SEQ_LEN,
    )
    x = torch.tensor([ids], dtype=torch.long, device=device)

    print(f"\nPrompt: {prompt}")
    print("Response: ", end="", flush=True)

    with torch.no_grad():
        for _ in range(300):
            logits = model(x)[:, -1, :]
            next_tok = torch.multinomial(torch.softmax(logits / 0.8, dim=-1), 1)
            x = torch.cat([x, next_tok], dim=1)
            print(tokenizer.decode([next_tok.item()]), end="", flush=True)
    print()
    return prompt


# ── Download SFT Checkpoint ────────────────────────────────────────────────


@app.function(
    image=image,
    volumes={str(VOLUME_DIR): volume},
    timeout=120,
)
def download_sft():
    """Download SFT checkpoint from volume to /nanochat_local_checkpoints."""
    import shutil

    volume.reload()

    if not SFT_CHECKPOINT_DIR.exists():
        print(f"✗ No checkpoint at {SFT_CHECKPOINT_DIR}")
        return

    local_dir = Path("/nanochat_local_checkpoints")
    local_dir.mkdir(parents=True, exist_ok=True)

    for f in SFT_CHECKPOINT_DIR.iterdir():
        dest = local_dir / f.name
        shutil.copy(str(f), str(dest))
        print(f"Downloaded: {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    print(f"\n✓ Saved to {local_dir}")
    return [str(local_dir / f) for f in SFT_CHECKPOINT_DIR.iterdir()]


# ── Post-Training Evaluation ────────────────────────────────────────────


@app.function(
    image=image,
    volumes={str(VOLUME_DIR): volume},
    gpu="A10G:2",
    timeout=7200,
    memory=32768,
)
def run_eval():
    """Run ChatCORE evaluation on the SFT checkpoint (2 GPUs)."""
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/nanochat")
    setup_nanochat()
    os.environ["NANOCHAT_BASE_DIR"] = str(VOLUME_DIR)
    os.environ["PYTHONUNBUFFERED"] = "1"

    if not SFT_CHECKPOINT_DIR.exists() or not any(
        SFT_CHECKPOINT_DIR.glob("model_*.pt")
    ):
        print(f"✗ No SFT checkpoint at {SFT_CHECKPOINT_DIR}")
        return

    print("\n" + "=" * 60)
    print("Running ChatCORE evaluation on SFT checkpoint")
    print("Tasks: ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval, SpellingBee")
    print("=" * 60)

    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "-m",
        "scripts.chat_eval",
        "--",
        "-i",
        "sft",
        "-a",
        "all",
        "-b",
        "8",
    ]
    result = subprocess.run(cmd, cwd="/nanochat")
    print(f"\nEval exited with code {result.returncode}")
    return result.returncode


# ── Local helpers for checkpoint upload ─────────────────────────────────


def _checkpoint_local_paths():
    """Return (pt_path, json_path) for the pretrain checkpoint on the local machine."""
    ckpt_dir = _SCRIPT_DIR / "checkpoints" / "pretrain"
    return ckpt_dir / "model_008600.pt", ckpt_dir / "meta_008600.json"


def _ensure_checkpoint_on_volume():
    """Upload the pretrain checkpoint to the Modal volume if it's not already there.

    Runs locally (inside @app.local_entrypoint), so it can access the local filesystem.
    """
    import subprocess
    import sys

    vol_path_pt = "/base_checkpoints/d6/model_008600.pt"
    vol_path_json = "/base_checkpoints/d6/meta_008600.json"

    # Quick check via modal volume ls (avoids re-uploading every time)
    result = subprocess.run(
        [sys.executable, "-m", "modal", "volume", "ls", "nanochat-vol", vol_path_pt],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 0 and "model_008600.pt" in result.stdout:
        print("✓ Pretrain checkpoint already on volume, skipping upload.")
        return

    local_pt, local_json = _checkpoint_local_paths()
    if not local_pt.exists():
        print(f"✗ Local checkpoint not found at: {local_pt}")
        print("  Upload the checkpoint manually with:")
        print(
            f"    modal volume put nanochat-vol <path-to-model_008600.pt> {vol_path_pt}"
        )
        print(
            f"    modal volume put nanochat-vol <path-to-meta_008600.json> {vol_path_json}"
        )
        sys.exit(1)

    print("Uploading pretrain checkpoint to Modal volume...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "put",
            "nanochat-vol",
            str(local_pt),
            vol_path_pt,
        ],
        check=True,
        timeout=600,
    )
    print("✓ Uploaded model_008600.pt")

    if local_json.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "volume",
                "put",
                "nanochat-vol",
                str(local_json),
                vol_path_json,
            ],
            check=True,
            timeout=30,
        )
        print("✓ Uploaded meta_008600.json")
    else:
        print("⚠ meta_008600.json not found locally, skipping.")

    print("Pretrain checkpoint ready on volume.")


# ── CLI Entrypoint ─────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    inference_prompt: str = "",
    use_teacher: bool = USE_TEACHER,
    kd_alpha: float = KD_ALPHA,
    kd_temperature: float = KD_TEMPERATURE,
    resume_step: int = -1,
):
    """Run the full SFT pipeline on nanochat d6.

    Spawns both tokenizer training and SFT as a single detached job.
    Optionally enables knowledge distillation from a larger teacher model.

    Args:
        inference_prompt: Prompt for test inference (optional).
        use_teacher: If True, download LFM2.5-350M and enable KD.
        kd_alpha: Weight for CE loss (0.9 = 90%% CE, 10%% KD).
        kd_temperature: Softmax temperature for sorted KD loss.
    """
    # Step 0: ensure the pretrain checkpoint is on the volume
    _ensure_checkpoint_on_volume()

    print("Spawning SFT on 2× A10G (detached, self-contained)...")
    if use_teacher:
        print(f"  Teacher: {TEACHER_HF_ID} | alpha={kd_alpha}, T={kd_temperature}")
    run_sft.spawn(
        use_teacher=use_teacher,
        kd_alpha=kd_alpha,
        kd_temperature=kd_temperature,
        resume_step=resume_step,
    )
    print()
    print("=" * 60)
    print("  SFT SPAWNED — running in the cloud.")
    print("  Monitor:    modal app logs nanochat-sft")
    print("  Download:   modal run nanochat_sft.py::download_sft")
    print("  Inference:  modal run nanochat_sft.py::test_inference --prompt 'Hi!'")
    print("  Eval:       modal run nanochat_sft.py::run_eval")
    cost_est = "~$3-5 total" if use_teacher else "~$1-2 total"
    print(f"  Estimated:  ~60-90 min, {cost_est}")
    print("=" * 60)
