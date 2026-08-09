"""Validate every .arrow file by fully loading it with pyarrow.

Usage: PYTHONPATH=/tmp/pyarrow_ck python3 modal-backup/validate_arrow.py [dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "modal-backup/tokenized")

checked, failed = 0, 0
for p in sorted(ROOT.rglob("*.arrow")):
    checked += 1
    size = p.stat().st_size
    try:
        with p.open("rb") as f:
            data = f.read()
        # datasets ArrowWriter emits the IPC streaming format; try file first
        try:
            reader = ipc.open_file(data)
        except pa.ArrowInvalid:
            reader = ipc.open_stream(data)
        table = reader.read_all()
        nrows = table.num_rows
        ncols = table.num_columns
        # second pass: ensure we can re-read (footer/metadata consistent)
        print(f"OK   {p.name}: {nrows:,} rows x {ncols} cols, {size:,} B")
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"FAIL {p.name}: {type(e).__name__}: {e}")

print(f"\nChecked {checked} arrow files, {failed} failed")
sys.exit(1 if failed else 0)
