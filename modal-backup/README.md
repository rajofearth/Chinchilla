# Modal backup — mars-train-vol (workspace: maheryashrajtest)

Taken 2026-08-09 before switching accounts. All files verified **byte-exact**
against the volume via `exact_compare.py`.

## Contents (all byte-exact matches)

| Tree         | Files | Size    | Notes |
|--------------|-------|---------|-------|
| `checkpoints/` | 391 | 3.28 GB | SFT (agent + code) and DPO adapters + optimizer states; `final/` = merged adapter for each run |
| `tokenized/`   | 17  | 5.69 GB | Tokenized SFT datasets (HF Arrow shards) |
| `evals/`       | 2   | ~18 KB  | Final eval JSONs |

## Deliberately NOT backed up (re-downloadable from Hugging Face)

- `bases/` — `LiquidAI/LFM2.5-8B-A1B` base model (~17 GB). Get with:
  `hf download LiquidAI/LFM2.5-8B-A1B --local-dir <dest>`
- `data/hf-cache/` — source datasets (smoltalk, ultrafeedback_binarized,
  Code-Feedback, hermes-agent-traces-filtered, Synth-APIGen-v0.1) + model cache
  (~9.3 GB). Re-fetched automatically on next training run.

## Restoring on the new account

```sh
modal profile activate <new-profile>
modal volume create mars-train-vol
cd modal-backup
modal volume put mars-train-vol /checkpoints   # re-uploads tree
modal volume put mars-train-vol /tokenized
modal volume put mars-train-vol /evals
```

Then recreate the HF secret (the token value cannot be exported from Modal —
grab it from https://huggingface.co/settings/tokens):

```sh
modal secret create huggingface-secret HF_TOKEN=<token>
```

## Verification

```sh
# needs the python env that provides the modal SDK (e.g. the uv tool env)
/home/yrm/.local/share/uv/tools/modal/bin/python3 exact_compare.py mars-train-vol modal-backup /checkpoints
```

The important files were also validated by fully loading every Arrow shard
with pyarrow (`validate_arrow.py`).
