#!/usr/bin/env python3
"""Work out which AO convention the OMol_CSH Fock matrices are stored in.

The HDF5 ships no overlap matrix, energy or orbital metadata, so the convention
has to be inferred. Two signals are used, cheapest first:

1. ``corr(|F|, |H_core|)`` over the AO matrix. F and the one-electron core
   Hamiltonian share the same AO indexing and the same block structure, and the
   large elements of F are core-dominated, so a mis-permutation shows up as a
   collapse in correlation. H_core costs two one-electron integrals.
2. The lowest generalised eigenvalues, which must land on the atomic 1s levels
   of the elements present (C ~ -10, O ~ -19, F ~ -25, S ~ -89, Cl ~ -101 Ha).

Usage:
  python scripts/revision/omol_csh_identify_convention.py \
      datasets/revision/omol_csh/omol_csh_1k_test_common.h5 --max-atoms 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import (  # noqa: E402
    build_ao_transform,
    build_mol,
    inflate_triangle,
    parse_charge_mult,
    triangle_dim,
)

CANDIDATES = [
    ("pyscf", False),
    ("orca", False),
    ("orca", True),
    ("m_ascending", False),
    ("m_ascending", True),
]


def iter_entries(handle):
    for top in handle.keys():
        group = handle[top]
        for name in group.keys():
            yield top, name, group[name]


def pick_small_entries(path: str, max_atoms: int, count: int):
    picked = []
    with h5py.File(path, "r") as fh:
        for top, name, node in iter_entries(fh):
            natoms = node["elements"].shape[0]
            if natoms > max_atoms:
                continue
            picked.append(
                {
                    "top": top,
                    "name": name,
                    "elements": np.asarray(node["elements"][()]),
                    "coords": np.asarray(node["coords"][()]),
                    "fock": np.asarray(node["fock"][()]),
                }
            )
            if len(picked) >= count:
                break
    return picked


def score_entry(entry: dict) -> dict:
    charge, mult = parse_charge_mult(entry["name"])
    spin = mult - 1
    mol = build_mol(entry["elements"], entry["coords"], charge, spin)
    n_stored = triangle_dim(entry["fock"].shape[0])

    result = {
        "name": entry["name"],
        "subset": entry["top"],
        "natoms": int(entry["elements"].shape[0]),
        "elements": sorted(set(int(z) for z in entry["elements"])),
        "charge": charge,
        "multiplicity": mult,
        "nao_stored": int(n_stored),
        "nao_pyscf": int(mol.nao),
        "basis_match": bool(n_stored == mol.nao),
    }
    if n_stored != mol.nao:
        return result

    fock_raw = inflate_triangle(entry["fock"])
    ovlp = mol.intor("int1e_ovlp")
    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    if mol.has_ecp():
        hcore = hcore + mol.intor("ECPscalar")

    nelec = int(mol.nelectron)
    result["nelectron"] = nelec

    scores = []
    for source, flip in CANDIDATES:
        transform = build_ao_transform(mol, source, flip)
        fock = transform.apply(fock_raw)
        corr = float(np.corrcoef(np.abs(fock).ravel(), np.abs(hcore).ravel())[0, 1])
        eps = scipy.linalg.eigh(fock, ovlp, eigvals_only=True)
        scores.append(
            {
                "source": source,
                "flip_high_m": flip,
                "corr_abs_hcore": corr,
                "eps_min": float(eps[0]),
                "eps_homo": float(eps[nelec // 2 - 1]),
                "eps_lumo": float(eps[nelec // 2]),
                "gap_ev": float((eps[nelec // 2] - eps[nelec // 2 - 1]) * 27.211386245988),
            }
        )
    result["candidates"] = scores
    result["best"] = max(scores, key=lambda s: s["corr_abs_hcore"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--max-atoms", type=int, default=12)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    entries = pick_small_entries(args.path, args.max_atoms, args.count)
    print(f"selected {len(entries)} entries with <= {args.max_atoms} atoms")

    results = []
    for entry in entries:
        res = score_entry(entry)
        results.append(res)
        print(f"\n=== {res['subset']}/{res['name']} ===")
        print(
            f"natoms={res['natoms']} charge={res['charge']} mult={res['multiplicity']} "
            f"Z={res['elements']}"
        )
        print(f"nao stored={res['nao_stored']} pyscf={res['nao_pyscf']} match={res['basis_match']}")
        if not res["basis_match"]:
            continue
        print(f"nelectron={res['nelectron']}")
        for cand in res["candidates"]:
            print(
                f"  {cand['source']:<12} flip_f={str(cand['flip_high_m']):<5} "
                f"corr={cand['corr_abs_hcore']:+.6f} eps_min={cand['eps_min']:+.4f} "
                f"HOMO={cand['eps_homo']:+.4f} gap={cand['gap_ev']:.3f} eV"
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
