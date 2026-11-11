#!/usr/bin/env python3
"""Merge ethanol DFT relabel shards into datasets/ethanol_dft_pyscf_ccpvdz_train.npy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("datasets/revision/ethanol_dft_shards"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/ethanol_dft_pyscf_ccpvdz_train.npy"),
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/revision/ethanol_dft_relabel_manifest.json"),
    )
    args = ap.parse_args()

    shards = sorted(args.shard_dir.glob("shard_*.npy"))
    if not shards:
        raise SystemExit(f"no shards under {args.shard_dir}")

    positions, atom_numbers, energies, forces, dipoles = [], [], [], [], []
    for shard in shards:
        geo = np.load(shard, allow_pickle=True).item()
        positions.extend(list(geo["positions"]))
        atom_numbers.extend(list(geo["atom_numbers"]))
        energies.append(geo["energy"])
        forces.extend(list(geo["forces"]))
        if "dipole_moment" in geo:
            dipoles.append(geo["dipole_moment"])

    out = {
        "positions": np.asarray(positions, dtype=object),
        "atom_numbers": np.asarray(atom_numbers, dtype=object),
        "energy": np.concatenate([np.atleast_1d(e) for e in energies]),
        "forces": np.asarray(forces, dtype=object),
    }
    if dipoles:
        out["dipole_moment"] = np.concatenate([np.atleast_1d(d) for d in dipoles])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out, allow_pickle=True)
    manifest = {
        "n_frames": len(positions),
        "n_shards": len(shards),
        "shard_dir": str(args.shard_dir),
        "output": str(args.out),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
