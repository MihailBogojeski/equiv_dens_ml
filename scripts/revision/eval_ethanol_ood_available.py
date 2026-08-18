#!/usr/bin/env python3
"""Score 96w7KyGG on whatever ethanol OOD DFT frames exist (R1.2)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="paper/models/ethanol/2024-03-22_96w7KyGG")
    parser.add_argument("--dens", default="datasets/revision/ood/ethanol_ood_pyscf_augccpvdz_pbe.npy")
    parser.add_argument("--geom", default="datasets/revision/ood/ethanol_ood_npy.npy")
    parser.add_argument("--out-dir", default="datasets/revision/ood")
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    dens_path = root / args.dens
    if not dens_path.exists():
        print(json.dumps({"error": f"missing {dens_path}", "n_frames": 0}))
        return 2
    n = len(np.load(dens_path, allow_pickle=True))
    if n < args.min_frames:
        print(json.dumps({"error": "too few labeled OOD frames", "n_frames": n}))
        return 2

    out_dir = root / args.out_dir
    sliced_geom = out_dir / f"ethanol_ood_partial_{n}_npy.npy"
    sliced_dens = out_dir / f"ethanol_ood_partial_{n}_pyscf_augccpvdz_pbe.npy"
    subprocess.check_call(
        [
            sys.executable,
            str(root / "scripts/revision/slice_labeled_pair.py"),
            "--dens",
            str(dens_path),
            "--geom",
            str(root / args.geom),
            "--out-dens",
            str(sliced_dens),
            "--out-geom",
            str(sliced_geom),
        ]
    )

    cfg = (root / "config/training/eval_ethanol_ood.txt").read_text()
    lines = []
    replacements = {
        "--np_dataset=": f"--np_dataset={sliced_geom}",
        "--dens_dataset=": f"--dens_dataset={sliced_dens}",
        "--np_dataset_valid=": f"--np_dataset_valid={sliced_geom}",
        "--dens_dataset_valid=": f"--dens_dataset_valid={sliced_dens}",
        "--np_dataset_test=": f"--np_dataset_test={sliced_geom}",
        "--dens_dataset_test=": f"--dens_dataset_test={sliced_dens}",
        "--num_test=": f"--num_test={n}",
        "--use_gpu=": f"--use_gpu={'True' if args.use_gpu else 'False'}",
        "--restart=": f"--restart={args.model}",
    }
    for line in cfg.splitlines():
        stripped = line.strip()
        replaced = False
        for prefix, new in replacements.items():
            if stripped.startswith(prefix):
                lines.append(new)
                replaced = True
                break
        if not replaced:
            lines.append(line)
    # eval_model_npy.py sets restart = parent of this args file.
    model_dir = root / args.model
    tmp_cfg = model_dir / f"eval_ood_partial_{n}.txt"
    tmp_cfg.write_text("\n".join(lines) + "\n")

    cmd = [
        sys.executable,
        str(root / "scripts/training/eval_model_npy.py"),
        str(tmp_cfg),
        str(sliced_geom),
        "--density_path",
        str(sliced_dens),
    ]
    print(f"evaluating {args.model} on {n} OOD frames", flush=True)
    proc = subprocess.run(cmd, cwd=root)
    summary = {
        "model": args.model,
        "n_frames": n,
        "geom": str(sliced_geom),
        "dens": str(sliced_dens),
        "config": str(tmp_cfg),
        "eval_returncode": proc.returncode,
        "note": "Partial OOD until DFT reaches 190/190. Energy-head architecture mismatch still applies.",
    }
    out_json = root / "results/revision" / f"eval_ethanol_ood_partial_{n}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
