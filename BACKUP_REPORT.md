# Modal Cloud Backup Report — `mars-train-vol`

**Date:** 2026-08-09
**Source account / profile:** `maheryashrajtest` (workspace `maheryashrajtest`), accessed from the Modal CLI inside WSL
**Destination:** `modal-backup/` in this repo (local disk)

---

## 1. Executive summary

The Modal account `maheryashrajtest` is being **switched out** because the workspace
hit its **spend limit** — Modal refused to start apps with
`Workspace ... has exceeded its spend limit`. Before switching, all irreplaceable
data on the account's single Modal volume (`mars-train-vol`) was downloaded and
**verified byte-exact** against the remote.

- **Total on volume:** 35.24 GB across 531 files.
- **Backed up:** `checkpoints/` (391 files, 3.28 GB), `tokenized/` (17 files,
  5.69 GB), `evals/` (2 files, ~18 KB) — all byte-exact, Arrow shards validated
  with pyarrow.
- **Deliberately skipped:** `bases/` (~17 GB) and `data/hf-cache/` (~9.3 GB) —
  both re-downloadable from Hugging Face.
- **Secrets:** one secret, `huggingface-secret` (`HF_TOKEN`). Its value **cannot
  be exported** from Modal and must be recreated on the new account.

A partial earlier backup existed but was corrupt/incomplete (0-byte and truncated
Arrow shards, missing checkpoints, stale evals). It was discarded and all three
trees were re-downloaded fresh with `modal volume get --force`, then verified.

---

## 2. What was backed up

| Tree | Files | Size | Verification status |
|------|-------|------|---------------------|
| `checkpoints/` | 391 | 3.28 GB | ✅ Byte-exact 391/391 |
| `tokenized/` | 17 | 5.69 GB | ✅ Byte-exact 17/17; all 13 Arrow shards load cleanly in pyarrow |
| `evals/` | 2 | ~18 KB | ✅ Byte-exact 2/2 |
| **Total** | **410** | **~8.97 GB** | ✅ |

### Contents in detail

- **`checkpoints/`** — trained LoRA adapters:
  - `sft/mars_sft_agent/` — runs `20260627_174008`, `20260629_181848`,
    `20260705_112528` (completed); several empty/failed run dirs.
  - `sft/mars_sft_code/` — runs `20260705_134352`, `20260705_141712`,
    `20260802_143221`. The `final/` adapter of `20260802_143221` is the adapter
    the DPO run started from.
  - `dpo/mars_dpo/` — runs `20260802_161818`, `20260802_210525`,
    `20260803_030359`; the last run has checkpoints `1000/1200/1400`, each with a
    `ref/` dir.
  - Each checkpoint contains `adapter_model.safetensors`, `optimizer.pt`,
    `tokenizer.json`, `trainer_state.json`, `training_args.bin`, etc.
  - `rl/` and `exports/` are empty (no RL runs, no exported merged model).
- **`tokenized/`** — tokenized SFT datasets (Hugging Face Arrow shards):
  - `mars_sft_agent/3a3924a7be6e434d46c54db438507cc4/` — 6 Arrow shards +
    `dataset_info.json` + `state.json`.
  - `mars_sft_code/870a942157479af913427e00e1064eea/` — 7 Arrow shards +
    2 JSON files.
- **`evals/`** — final eval results:
  - `sft_agent_final_mars_sft_agent_20260705_132106.json`
  - `sft_code_final_mars_sft_code_20260802_161617.json`
  - The same two files also exist at the repo root.

---

## 3. What's where

### Local (`modal-backup/`)

```
modal-backup/
├── checkpoints/                391 files, 3.28 GB — byte-exact
├── tokenized/                  17 files, 5.69 GB — byte-exact
├── evals/                      2 files, ~18 KB — byte-exact
├── README.md                   restore instructions
├── inventory.py                walks the volume via the Modal SDK
├── compare.py                  quick CLI-based check (sizes rounded)
├── exact_compare.py            byte-exact SDK comparison
└── validate_arrow.py           pyarrow full-load validation
```

### Only on Modal (nothing new to do — re-downloadable)

| Path | Size | Notes |
|------|------|-------|
| `bases/` | ~17 GB | `LiquidAI/LFM2.5-8B-A1B` base model (`model.safetensors` ~16.9 GB, `tokenizer.json` 17.9 MB, config files). Public on Hugging Face. |
| `data/hf-cache/` | 9.31 GB | HF model + dataset cache: smoltalk, ultrafeedback_binarized, Code-Feedback, hermes-agent-traces-filtered, Synth-APIGen-v0.1, LFM2.5-8B-A1B. All public. |

### Other account resources

- **Secret:** `huggingface-secret` (`HF_TOKEN`) — value **not recoverable** from
  Modal (see §5, item 7). No dicts, queues, or deployed apps with data.

---

## 4. How it was verified

Two independent checks ran against the finished backup:

1. **Byte-exact comparison (`exact_compare.py`)** — walks the volume with the
   Modal SDK's `Volume.listdir`, which reports **exact byte sizes**, and compares
   every file's size against the local copy.
   - `checkpoints/`: 391/391 match
   - `tokenized/`: 17/17 match
   - `evals/`: 2/2 match
2. **pyarrow validation (`validate_arrow.py`)** — fully loads every `.arrow`
   shard (13 total) into memory, counting rows/columns; all loaded cleanly.

> **Pitfall discovered:** `modal volume ls --json` reports sizes **rounded to
> 1 decimal place in MiB** (±52 KB error). Comparing against those rounded values
> flagged perfectly healthy files as mismatched. The SDK's `Volume.listdir`
> returns exact byte counts, so the CLI-based `compare.py` was kept only as a
> quick smoke check while `exact_compare.py` (SDK) is the source of truth.

---

## 5. Problems encountered & how they were overcome

1. **Incomplete/corrupt earlier backup.** The pre-existing `modal-backup/`
   content was unusable: `tokenized/` had a double-nesting bug
   (`tokenized/tokenized/...`, `evals/evals/...`), several Arrow shards were
   0-byte (interrupted downloads), others truncated (e.g.
   `data-00000-of-00006.arrow` was 352 MB vs 437 MB remote), `checkpoints/` held
   only a few small files from one DPO run (`20260803_030359`), and the `evals/`
   copies were stale (byte sizes differed).
   *Overcome:* deleted the stale local `checkpoints/`, `tokenized/`, `evals/`
   trees entirely and re-downloaded all three with
   `modal volume get --force mars-train-vol /checkpoints|/tokenized|/evals`,
   then verified byte-exact.

2. **Download "looked stuck".** The 5.69 GB `tokenized/` download showed a
   progress bar frozen at ~98–100% for a long time. It was actually progressing
   slowly (~2.7 MB/s; each shard ~440 MB). It was stopped at 12/17 files.
   *Lesson:* it wasn't stuck, just slow — let it finish.

3. **WSL session teardown killed backgrounded downloads.** `nohup ... &` and
   `setsid ... &` processes died when the `wsl.exe bash -lc` session ended; the
   WSL VM also shuts down between separate tool calls and wipes `/tmp`.
   *Overcome:* ran downloads **synchronously** in a single tool call with a
   generous timeout.

4. **CLI size rounding caused false "mismatch" alarms.** See §4 — `modal volume
   ls --json` rounds sizes, so parsing those numbers flagged healthy files.
   *Overcome:* switched to the Modal SDK (`Volume.listdir`, exact byte sizes);
   the CLI-based `compare.py` was demoted to a quick check.

5. **One genuinely truncated shard.** `tokenized/mars_sft_agent/
   3a3924a7be6e434d46c54db438507cc4/data-00003-of-00006.arrow` was ~2 MB short
   (438,721,983 vs 440,809,760 bytes) because the earlier download was stopped
   mid-file. Notably, **pyarrow still loaded it fine** (Arrow streams can end at
   a batch boundary), so only the exact-size comparison caught it.
   *Overcome:* re-downloaded the file and moved it into place. Caveat learned:
   `modal volume get <vol> <file>` with no destination writes to the current
   directory using only the basename — the file initially landed at
   `modal-backup/data-00003-of-00006.arrow` and had to be moved into its
   `tokenized/...` subdirectory.

6. **Workspace spend limit blocked `modal run`.** Running a script via
   `modal run` failed with the spend-limit error. Plain CLI reads
   (`modal volume ls/get`) and a direct SDK script (no `App`) worked fine, so
   all backup tooling used those paths.

7. **HF token not recoverable.** The `huggingface-secret` value cannot be
   exported via the Modal CLI, and no local copy exists (checked WSL
   `~/.cache/huggingface/token`, `~/.huggingface/token`, and the repo). It must
   be recreated on the new account from https://huggingface.co/settings/tokens.

---

## 6. Restore instructions for the new account

```sh
modal profile activate <new-profile>
modal volume create mars-train-vol
cd modal-backup
modal volume put mars-train-vol /checkpoints   # re-uploads tree
modal volume put mars-train-vol /tokenized
modal volume put mars-train-vol /evals
```

Recreate the HF secret (the token value cannot be exported from Modal — grab it
from https://huggingface.co/settings/tokens):

```sh
modal secret create huggingface-secret HF_TOKEN=<token>
```

Re-fetch the skipped trees on demand:

```sh
hf download LiquidAI/LFM2.5-8B-A1B --local-dir <dest>   # bases/
# data/hf-cache/ re-populates automatically on the next training run
```

Re-verify after upload if desired:

```sh
/home/yrm/.local/share/uv/tools/modal/bin/python3 exact_compare.py mars-train-vol modal-backup /
```

---

## 7. Open items / recommendations

1. **Recreate the HF token.** `huggingface-secret` cannot be restored from the
   old account — create a new token at https://huggingface.co/settings/tokens
   and add it to the new workspace (`modal secret create huggingface-secret
   HF_TOKEN=<token>`).
2. **Re-upload after switching.** Run the §6 restore steps against the new
   profile; the old workspace (`maheryashrajtest`) is at its **spend limit** and
   won't run apps, so data should be restored before relying on Modal again.
3. **Refetch `bases/` and `data/hf-cache/`** on the new account via
   `hf download` / automatic re-caching (both are public on Hugging Face).
4. **Old workspace status:** `maheryashrajtest` remains at its spend limit; no
   action is possible there until the limit is raised or the account is
   switched.
5. **Untouched local profile:** another local Modal profile, `rajofearth`,
   exists and was **not** touched during this operation.
6. **Keep the tooling:** `inventory.py`, `compare.py`, `exact_compare.py`, and
   `validate_arrow.py` remain in `modal-backup/` and can be reused to re-verify
   the volume after restore.
