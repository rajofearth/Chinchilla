"""Byte-exact volume-vs-local comparison using the Modal SDK directly.

Unlike `modal volume ls`, the SDK's `Volume.listdir` reports exact byte sizes,
so this catches any real truncation. Run with the python that has the modal
SDK installed (e.g. the uv tool env that provides the `modal` CLI):

  /home/yrm/.local/share/uv/tools/modal/bin/python3 modal-backup/exact_compare.py \
      <volume> <local_root> [remote_prefix]

Usage example:
  python3 modal-backup/exact_compare.py mars-train-vol modal-backup /
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

VOLUME = sys.argv[1] if len(sys.argv) > 1 else "mars-train-vol"
LOCAL_ROOT = Path(sys.argv[2] if len(sys.argv) > 2 else "modal-backup")
REMOTE_PREFIX = (sys.argv[3] if len(sys.argv) > 3 else "/").strip("/")

vol = modal.Volume.from_name(VOLUME, create_if_missing=False)


def walk(prefix: str) -> dict[str, int]:
    files: dict[str, int] = {}
    for entry in vol.listdir(prefix):
        if entry.type.name == "DIRECTORY":
            files.update(walk(entry.path.rstrip("/") + "/"))
        else:
            files[entry.path] = entry.size
    return files


def main() -> None:
    prefix = REMOTE_PREFIX.rstrip("/") + "/"
    print(f"Walking volume '{VOLUME}' under '{prefix}' ...")
    remote = walk(prefix)
    if not remote:
        print("No files found.")
        return

    total = sum(remote.values())
    print(f"Remote files: {len(remote)}  Total: {total / 1e9:.2f} GB")

    local: dict[str, int] = {}
    for p in LOCAL_ROOT.rglob("*"):
        if p.is_file():
            local[str(p.relative_to(LOCAL_ROOT)).replace(os.sep, "/")] = p.stat().st_size

    missing = [(r, s) for r, s in remote.items() if r not in local]
    mismatch = [
        (r, s, local[r])
        for r, s in remote.items()
        if r in local and local[r] != s
    ]
    ok = len(remote) - len(missing) - len(mismatch)
    extra = [r for r in local if r not in remote]

    print(f"Byte-exact matches: {ok}")
    if missing:
        print(f"\nMISSING locally ({len(missing)} files, {sum(s for _, s in missing) / 1e6:.1f} MB):")
        for r, s in sorted(missing, key=lambda kv: -kv[1]):
            print(f"  {s:>12,} B  {r}")
    if mismatch:
        print(f"\nREAL SIZE MISMATCH ({len(mismatch)} files):")
        for r, s, l in sorted(mismatch, key=lambda kv: -abs(kv[1] - kv[2])):
            print(f"  remote={s:>12,} local={l:>12,} diff={l - s:+,}  {r}")
    if extra:
        print(f"\nLocal-only (not on volume): {len(extra)}")
        for r in sorted(extra):
            print(f"  {r}")
    if missing or mismatch:
        sys.exit(1)


if __name__ == "__main__":
    main()
