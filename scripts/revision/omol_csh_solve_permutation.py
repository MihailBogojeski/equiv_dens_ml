#!/usr/bin/env python3
"""Solve exactly for the stored->PySCF AO permutation using anchored columns.

Matching AOs on their diagonal or on whole rows fails because polarization
shells are near-degenerate: many assignments look equally good and the solver
settles on a wrong one. The core s functions, by contrast, are unambiguous -
their diagonals are well separated and they already sit at identical positions
in both orderings.

So those are used as anchors. An unknown AO is identified by its couplings to
the already-mapped columns, which is a non-degenerate fingerprint, and each
newly resolved AO becomes an anchor for the next round.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import build_mol, inflate_triangle, parse_charge_mult  # noqa: E402


def find_anchors(diag_stored, diag_ref, slices, natm, tol=5e-3, sep=5e-2):
    """AOs whose diagonal already agrees and is well separated within its atom."""
    anchors = []
    for atom in range(natm):
        s0, s1 = slices[atom][2], slices[atom][3]
        for i in range(s0, s1):
            if abs(diag_stored[i] - diag_ref[i]) > tol:
                continue
            others = np.abs(diag_ref[s0:s1] - diag_ref[i])
            others[i - s0] = np.inf
            if others.min() > sep:
                anchors.append(i)
    return anchors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-rounds", type=int, default=12)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    name = str(data["name"])
    elements = np.asarray(data["elements"])
    mol = build_mol(
        elements, np.asarray(data["coords"]), *reversed([parse_charge_mult(name)[1] - 1,
                                                         parse_charge_mult(name)[0]])
    )
    fock_ref = np.asarray(data["fock"])
    fock_stored = inflate_triangle(np.asarray(data["fock_stored"]))
    slices = mol.aoslice_by_atom()
    nao = mol.nao

    anchors = find_anchors(np.diag(fock_stored), np.diag(fock_ref), slices, mol.natm)
    print(f"entry={name} nao={nao}")
    print(f"initial anchors: {len(anchors)}")

    # perm[i] = stored index of the AO PySCF places at i
    perm = -np.ones(nao, dtype=int)
    for i in anchors:
        perm[i] = i

    for rnd in range(args.max_rounds):
        known_ref = np.flatnonzero(perm >= 0)
        known_stored = perm[known_ref]
        resolved = 0
        for atom in range(mol.natm):
            s0, s1 = slices[atom][2], slices[atom][3]
            unknown = [i for i in range(s0, s1) if perm[i] < 0]
            if not unknown:
                continue
            free = sorted(set(range(s0, s1)) - set(perm[perm >= 0]))
            ref_block = fock_ref[np.ix_(unknown, known_ref)]
            stored_block = fock_stored[np.ix_(free, known_stored)]
            cost = np.linalg.norm(ref_block[:, None, :] - stored_block[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                perm[unknown[r]] = free[c]
            resolved += len(unknown)
        err = np.abs(fock_stored[np.ix_(perm, perm)] - fock_ref).max()
        print(f"  round {rnd}: resolved {resolved}, max|diff| = {err:.4e}")
        if resolved == 0:
            break

    recovered = fock_stored[np.ix_(perm, perm)]
    err = np.abs(recovered - fock_ref)
    print(f"\nfinal max|F_perm - F_ref|  = {err.max():.4e}")
    print(f"final mean|F_perm - F_ref| = {err.mean():.4e}")
    print(f"valid permutation: {sorted(perm.tolist()) == list(range(nao))}")

    labels = mol.ao_labels()
    by_element = defaultdict(list)
    for atom in range(mol.natm):
        s0, s1 = slices[atom][2], slices[atom][3]
        by_element[int(elements[atom])].append(list((perm[s0:s1] - s0).tolist()))

    summary = {}
    print("\nper-element local map (PySCF slot -> stored slot):")
    for z, maps in sorted(by_element.items()):
        same = all(m == maps[0] for m in maps)
        summary[str(z)] = {"perm": maps[0], "consistent": same, "n_atoms": len(maps)}
        print(f"  Z={z:>2} n={len(maps)} consistent={same}")
        print(f"     {maps[0]}")

    print("\nannotated (first atom of each element):")
    seen = set()
    for atom in range(mol.natm):
        z = int(elements[atom])
        if z in seen:
            continue
        seen.add(z)
        s0, s1 = slices[atom][2], slices[atom][3]
        for i in range(s0, s1):
            print(f"   {labels[i]:<20} <- stored {perm[i] - s0:>3}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "entry": name,
                    "max_abs_error": float(err.max()),
                    "mean_abs_error": float(err.mean()),
                    "per_element": summary,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
