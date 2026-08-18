#!/usr/bin/env python3
"""Package density-fitted CSH shards into the NPY pair the DenSNet loader expects.

`AtomsDensityData` reads two files: a geometry dict and an object array of
(mol.pack(), calc_dict) pairs. Producing exactly that format means the training
code needs no modification.

`mo_coeff` and `mo_occ` are read unconditionally by the loader, but the real MO
coefficients would be nao^2 per structure - tens of megabytes each at these
system sizes - so placeholders are stored instead. They are never dereferenced
when the reference density comes from the fitted coefficients
(`--projected_density=True`), and any run that turns that off will fail loudly
rather than train on zeros.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import BASIS, ECP  # noqa: E402


def load_shards(outdir: Path):
    records, failures = [], []
    for path in sorted(outdir.glob("shard_*.npz")):
        data = np.load(path, allow_pickle=True)
        records.extend(list(data["records"]))
        failures.extend(list(data["failures"]))
    return records, failures


def build_pair(records, auxbasis: str, with_mo: bool):
    positions = np.empty(len(records), dtype=object)
    numbers = np.empty(len(records), dtype=object)
    dens = []
    for i, rec in enumerate(records):
        positions[i] = np.asarray(rec["positions"], dtype=np.float64)
        numbers[i] = np.asarray(rec["atom_numbers"], dtype=np.int64)

        atom = [
            (int(z), tuple(float(x) for x in xyz))
            for z, xyz in zip(rec["atom_numbers"], rec["positions"])
        ]
        mol = gto.Mole()
        mol.atom = atom
        mol.basis = BASIS
        mol.ecp = ECP
        mol.charge = int(rec["charge"])
        mol.spin = 0
        mol.unit = "Angstrom"
        mol.verbose = 0
        mol.build()

        calc = {
            "df_coeff": np.asarray(rec["df_coeff"], dtype=np.float64),
            "auxbasis": auxbasis,
            "xc": "wb97m-v",
            "energy": 0.0,
            "n_elec": int(rec["n_elec"]),
            "charge": int(rec["charge"]),
        }
        if with_mo:
            calc["mo_coeff"] = np.asarray(rec["mo_coeff"], dtype=np.float64)
            calc["mo_occ"] = np.asarray(rec["mo_occ"], dtype=np.float64)
        else:
            calc["mo_coeff"] = np.zeros((1, 1), dtype=np.float64)
            calc["mo_occ"] = np.zeros((1,), dtype=np.float64)
        dens.append((mol.pack(), calc))

    geo = {"positions": positions, "atom_numbers": numbers}
    return geo, np.array(dens, dtype=object)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--auxbasis", default="augccpvqzjkfit")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--val-prefix", default=None)
    parser.add_argument("--max-atoms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-mo", action="store_true")
    args = parser.parse_args()

    records, failures = load_shards(Path(args.shards))
    print(f"loaded {len(records)} records, {len(failures)} failures from {args.shards}")
    if args.max_atoms:
        records = [r for r in records if len(r["atom_numbers"]) <= args.max_atoms]
        print(f"after natoms<={args.max_atoms}: {len(records)}")
    if not records:
        raise SystemExit("no records")

    natoms = np.array([len(r["atom_numbers"]) for r in records])
    print(f"natoms med/mean/max: {np.median(natoms):.0f} {natoms.mean():.0f} {natoms.max()}")
    drift = np.array([abs(r["n_elec_err"]) for r in records])
    print(f"electron-count drift: max {drift.max():.2e}, mean {drift.mean():.2e}")

    val_records = []
    if args.val_fraction > 0:
        # Stratify by size so the validation set spans the same range of system
        # sizes as training rather than concentrating in one tier.
        order = np.argsort(natoms)
        stride = max(int(round(1.0 / args.val_fraction)), 2)
        val_idx = set(order[::stride].tolist())
        val_records = [records[i] for i in sorted(val_idx)]
        records = [r for i, r in enumerate(records) if i not in val_idx]
        print(f"split: {len(records)} train / {len(val_records)} val")

    for tag, recs, prefix in (
        ("train", records, args.out_prefix),
        ("val", val_records, args.val_prefix or f"{args.out_prefix}_val"),
    ):
        if not recs:
            continue
        geo, dens = build_pair(recs, args.auxbasis, args.with_mo)
        geo_path = Path(f"{prefix}_npy.npy")
        dens_path = Path(f"{prefix}_pyscf_{args.auxbasis}_wb97mv.npy")
        geo_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(geo_path, geo, allow_pickle=True)
        np.save(dens_path, dens, allow_pickle=True)
        print(f"{tag}: {len(recs)} -> {geo_path} , {dens_path}")

    summary = {
        "n_records": len(records),
        "n_val": len(val_records),
        "n_failures": len(failures),
        "auxbasis": args.auxbasis,
        "max_electron_drift": float(drift.max()),
    }
    Path(f"{args.out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
