#!/usr/bin/env python3
"""Merge ORCA shard outputs into DensNet dens_dataset + np_dataset npy files.

Reports completeness per cluster size, not just in total. Shards are packed
smallest-first, so an interrupted campaign loses its *largest* frames -- exactly
the ones a size-extrapolation claim rests on. A split that is 90 percent
complete overall but missing every 24-water cluster would look healthy under a
single total and would quietly narrow the size range being reported on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    parser.add_argument(
        "--results-dir",
        required=True,
        nargs="+",
        help="directories of shard_XXXX folders; several are merged into one dataset",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--dens-out",
        default=None,
        help="DensNet dens_dataset path (list of (mol.pack(), calc_dict))",
    )
    parser.add_argument("--base-out", default=None, help="DensNet np_dataset / *_base.npy path")
    parser.add_argument(
        "--shard-dir",
        default=None,
        nargs="+",
        help="shard directories, so assembled frames can be checked against what was queued",
    )
    parser.add_argument(
        "--min-complete",
        type=float,
        default=0.0,
        help="refuse to write unless this fraction of every size bucket is present",
    )
    args = parser.parse_args()

    results_dirs = [Path(p) for p in args.results_dir]
    shard_dirs = [
        p for results_dir in results_dirs for p in sorted(results_dir.glob("shard_*")) if p.is_dir()
    ]
    records = []
    for shard_dir in shard_dirs:
        records.extend(load_shard_results(shard_dir))
    records = [r for r in records if r.get("calc") is not None]
    records.sort(key=lambda r: (-1 if r.get("index") is None else int(r["index"])))

    # Size tiers are separate arrays over one XYZ, and build_shards_from_xyz.py
    # numbers frames before it filters by atom count, so merging tiers must not
    # produce a repeat. A repeat would mean a frame counted twice in the size
    # histogram below and silently duplicated in the training set.
    seen, deduped = set(), []
    for rec in records:
        idx = rec.get("index")
        if idx is not None:
            if idx in seen:
                continue
            seen.add(idx)
        deduped.append(rec)
    if len(deduped) != len(records):
        print(f"dropped {len(records) - len(deduped)} duplicate frame indices across tiers")
    records = deduped

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

    got = Counter(len(f["atom_numbers"]) for f in frames)
    expected: Counter = Counter()
    if args.shard_dir:
        for shard_root in args.shard_dir:
            for shard_path in sorted(Path(shard_root).glob("shard_*.json")):
                for frame in json.loads(shard_path.read_text())["frames"]:
                    expected[len(frame["atom_numbers"])] += 1

    completeness = {}
    shortfalls = []
    for n_atoms in sorted(set(expected) | set(got)):
        have, want = got.get(n_atoms, 0), expected.get(n_atoms, 0)
        fraction = have / want if want else float("nan")
        completeness[str(n_atoms)] = {"assembled": have, "queued": want, "fraction": fraction}
        if want and fraction < args.min_complete:
            shortfalls.append(f"{n_atoms} atoms: {have}/{want} ({fraction:.0%})")

    print(f"assembled {len(dens)} frames from {len(shard_dirs)} shards")
    if expected:
        print("  completeness by cluster size:")
        for n_atoms, entry in completeness.items():
            flag = "" if entry["fraction"] >= 0.999 else "   <-- incomplete"
            print(f"    {n_atoms:>4s} atoms: {entry['assembled']:5d}/{entry['queued']:<5d}{flag}")
    if shortfalls:
        raise SystemExit(
            f"refusing to write: below --min-complete {args.min_complete:.0%} for\n  "
            + "\n  ".join(shortfalls)
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
        "results_dirs": [str(p) for p in results_dirs],
        "completeness_by_size": completeness,
        "dens_out": str(dens_out),
        "base_out": str(base_out),
    }
    for results_dir in results_dirs:
        (results_dir / "assemble.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {dens_out}")
    print(f"wrote {base_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
