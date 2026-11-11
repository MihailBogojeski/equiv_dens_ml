#!/usr/bin/env python3
"""Merge polythiophene dipole shards and run IR convergence (R3.6)."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/merge_polythiophene_dipoles_ir.py'


import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("results/revision/polythiophene_dipoles_10mer"),
    )
    ap.add_argument(
        "--merged-npy",
        type=Path,
        default=Path("results/revision/polythiophene_dipoles_10mer/rep0_merged.npy"),
    )
    ap.add_argument(
        "--ir-out",
        type=Path,
        default=Path("results/revision/ir_convergence_polythiophene_10mer"),
    )
    ap.add_argument("--windows", default="25,50,100,200")
    ap.add_argument("--dt-fs", type=float, default=1.0)
    args = ap.parse_args()

    shards = sorted(args.shard_dir.glob("rep0_*.npy"))
    if not shards:
        raise SystemExit(f"no shards in {args.shard_dir}")

    dipoles = []
    positions = atom_numbers = None
    dt_fs = args.dt_fs
    for shard in shards:
        blob = np.load(shard, allow_pickle=True).item()
        dipoles.append(blob["dipole_moment"])
        if positions is None:
            positions = blob["positions"]
            atom_numbers = blob["atom_numbers"]
            dt_fs = float(blob.get("time_step_fs", dt_fs))
        else:
            positions = np.concatenate([positions, blob["positions"]], axis=0)
            atom_numbers = np.concatenate([atom_numbers, blob["atom_numbers"]], axis=0)

    merged = {
        "positions": positions,
        "atom_numbers": atom_numbers,
        "dipole_moment": np.concatenate(dipoles, axis=0),
        "time_step_fs": dt_fs,
        "n_shards": len(shards),
    }
    args.merged_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.merged_npy, merged, allow_pickle=True)
    print(f"merged {merged['dipole_moment'].shape[0]} frames -> {args.merged_npy}")

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts/revision/analysis/ir_vs_length_npy.py"),
        "--npy",
        str(args.merged_npy),
        "--label",
        "polythiophene_10mer_rep0",
        "--windows",
        args.windows,
        "--dt-fs",
        str(dt_fs),
        "--out",
        str(args.ir_out),
    ]
    subprocess.run(cmd, check=True)
    summary = args.ir_out / "summary.json"
    if summary.is_file():
        print(summary.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
