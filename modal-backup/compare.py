"""Compare a Modal Volume against a local backup directory.

Walks the volume via the `modal volume ls --json` CLI (works even when the
workspace is at its spend limit, unlike `modal run`) and reports:
  - every remote file and its size (recursive)
  - which remote files are missing locally / differ in size
  - total bytes per top-level remote directory

Usage (from WSL, cwd = repo root):
  python3 modal-backup/compare.py <volume> <local_root> [remote_prefix]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

VOLUME = sys.argv[1] if len(sys.argv) > 1 else "mars-train-vol"
LOCAL_ROOT = Path(sys.argv[2] if len(sys.argv) > 2 else "modal-backup")
REMOTE_PREFIX = (sys.argv[3] if len(sys.argv) > 3 else "/").strip("/")


def ls(path: str) -> list[dict]:
    for attempt in range(4):
        out = subprocess.run(
            ["modal", "volume", "ls", "--json", VOLUME, path],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0:
            return json.loads(out.stdout)
        if attempt == 3:
            print(f"ls failed for {path}:\n{out.stdout}\n{out.stderr}", file=sys.stderr)
            raise SystemExit(1)
        time.sleep(2 ** attempt)


def walk_remote(prefix: str) -> dict[str, int]:
    """Return {remote_path: size_bytes} for every file under prefix."""
    files: dict[str, int] = {}

    def walk(p: str):
        for entry in ls(p):
            name = entry["filename"]
            is_dir = entry["type"] == "dir"
            size = entry["size"]
            if is_dir:
                walk(name.rstrip("/") + "/")
            else:
                files[name] = parse_size(size)

    walk(prefix)
    return files


def parse_size(s: str) -> int:
    parts = s.strip().split()
    num = float(parts[0])
    unit = parts[1] if len(parts) > 1 else "B"
    mult = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    return int(num * mult.get(unit, 1))


def main() -> None:
    prefix = REMOTE_PREFIX.rstrip("/") + "/"
    print(f"Walking volume '{VOLUME}' under '{prefix}' ...")
    remote = walk_remote(prefix)
    if not remote:
        print("No files found.")
        return

    total = sum(remote.values())
    print(f"Remote files: {len(remote)}  Total size: {total / 1e9:.2f} GB")

    per_dir: dict[str, int] = defaultdict(int)
    for path, size in remote.items():
        rel = path
        top = rel.split("/")[0] if "/" in rel else "(root)"
        per_dir[top] += size
    print("\nPer top-level directory:")
    for d, sz in sorted(per_dir.items(), key=lambda kv: -kv[1]):
        n = len([p for p in remote if p.split("/")[0] == d])
        print(f"  {d:30s} {sz / 1e9:8.2f} GB  ({n:5d} files)")

    print(f"\nLocal backup root: {LOCAL_ROOT.resolve()}")
    local: dict[str, int] = {}
    for p in LOCAL_ROOT.rglob("*"):
        if p.is_file():
            local[str(p.relative_to(LOCAL_ROOT)).replace(os.sep, "/")] = p.stat().st_size

    missing, mismatch, ok = [], [], 0
    for path, size in remote.items():
        rel = path
        local_path = LOCAL_ROOT / rel
        if not local_path.exists():
            missing.append((rel, size))
        elif local_path.stat().st_size != size:
            mismatch.append((rel, size, local_path.stat().st_size))
        else:
            ok += 1

    print(f"\nMatch by size: {ok} files")
    if missing:
        print(f"\nMISSING locally ({len(missing)} files, {sum(s for _, s in missing) / 1e9:.2f} GB):")
        for rel, size in sorted(missing, key=lambda kv: -kv[1]):
            print(f"  {size / 1e6:9.2f} MB  {rel}")
    if mismatch:
        print(f"\nSIZE MISMATCH ({len(mismatch)} files):")
        for rel, remote_size, local_size in mismatch:
            print(f"  {rel}: remote={remote_size} local={local_size}")


if __name__ == "__main__":
    main()
