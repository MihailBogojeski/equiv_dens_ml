#!/usr/bin/env python3
"""Merge ORCA shard outputs into DensNet dens_dataset + np_dataset npy files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import gto

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from qm7x_orca_common import pad_frames  # noqa: E402


def load_shard_results(shard_dir: Path) -> list[dict]:
    results_path = shard_dir / "results.npy"
    if not results_path.exists():
        return []
    rows = list(np.load(results_path, allow_pickle=True))
    out = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
        else:
            mol_pack, calc = row
            out.append({"index": None, "mol": mol_pack, "calc": calc})
    return out


def mol_from_pack(mol_pack: dict):
    mol = gto.Mole()
    mol.unpack(mol_pack)
    mol.build(False, False)
    return mol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Directory of shard_XXXX folders")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--dens-out",
        default=None,
        help="DensNet dens_dataset path (list of (mol.pack(), calc_dict))",
    )
    parser.add_argument("--base-out", default=None, help="DensNet np_dataset / *_base.npy path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    shard_dirs = sorted(p for p in results_dir.glob("shard_*") if p.is_dir())
    records = []
    for shard_dir in shard_dirs:
        records.extend(load_shard_results(shard_dir))
    records = [r for r in records if r.get("calc") is not None]
    records.sort(key=lambda r: (-1 if r.get("index") is None else int(r["index"])))

    dens = []
    frames = []
    for rec in records:
        calc = rec["calc"]
        mol_pack = rec["mol"]
        mol = mol_from_pack(mol_pack)
        dens.append((mol_pack, calc))
        z = [int(mol.atom_charge(i)) for i in range(mol.natm)]
        xyz = mol.atom_coords(unit="Angstrom")
        frames.append(
            {
                "index": rec.get("index"),
                "atom_numbers": z,
                "positions": np.asarray(xyz, dtype=float),
                "energy": calc["energy"],
                "forces": calc["forces"],
                "dipole": calc.get("dipole"),
            }
        )

    dens_out = Path(args.dens_out) if args.dens_out else _REPO_ROOT / f"datasets/qm7x_{args.split}_dft_augccpvdz.npy"
    base_out = (
        Path(args.base_out)
        if args.base_out
        else _REPO_ROOT / f"datasets/qm7x_{args.split}_dft_augccpvdz_orca_base.npy"
    )
    dens_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(dens_out, np.array(dens, dtype=object), allow_pickle=True)
    if frames:
        np.save(base_out, pad_frames(frames), allow_pickle=True)
    summary = {
        "split": args.split,
        "n_frames": len(dens),
        "n_shards": len(shard_dirs),
        "dens_out": str(dens_out),
        "base_out": str(base_out),
    }
    (results_dir / "assemble.json").write_text(json.dumps(summary, indent=2))
    print(f"assembled {len(dens)} frames from {len(shard_dirs)} shards")
    print(f"wrote {dens_out}")
    print(f"wrote {base_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
