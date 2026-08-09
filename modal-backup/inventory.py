"""Walk a Modal Volume recursively and print every file with its size.

Usage: VOLUME=mars-train-vol PATH=/checkpoints modal run modal-backup/inventory.py
"""
from __future__ import annotations

import os

import modal

app = modal.App("volume-inventory")

VOLUME = os.environ.get("VOLUME", "mars-train-vol")
ROOT = os.environ.get("REMOTE_PATH", "/")


@app.local_entrypoint()
def main():
    vol = modal.Volume.from_name(VOLUME, create_if_missing=False)

    def walk(prefix: str):
        for entry in vol.listdir(prefix):
            name, is_dir, size = entry.path, entry.type == "directory", entry.size
            if is_dir:
                walk(name.rstrip("/") + "/")
            else:
                print(f"{size}\t{name}")

    walk(ROOT.rstrip("/") + "/")


if __name__ == "__main__":
    main()
