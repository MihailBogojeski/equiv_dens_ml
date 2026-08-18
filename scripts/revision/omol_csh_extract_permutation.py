#!/usr/bin/env python3
"""Extract the exact stored->PySCF AO permutation from a converged reference.

Matching on diagonal values alone is ambiguous because polarization shells have
near-degenerate diagonals. Each AO instead gets a permutation-invariant
fingerprint: the sorted vector of |F[i, :]| over the whole row. That is
insensitive to how the *columns* are ordered, so stored rows can be matched to
reference rows even before the permutation is known, and it is essentially
unique per AO.

Prints the recovered map per element and checks that it is consistent across
atoms of the same element, which is what makes it a usable convention rather
than a per-structure fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import build_mol, inflate_triangle, parse_charge_mult  # noqa: E402


def row_fingerprints(mat: np.ndarray) -> np.ndarray:
    return np.sort(np.abs(mat), axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    name = str(data["name"])
    elements = np.asarray(data["elements"])
    charge, mult = parse_charge_mult(name)
    mol = build_mol(elements, np.asarray(data["coords"]), charge, mult - 1)

    fock_ref = np.asarray(data["fock"])
    fock_stored = inflate_triangle(np.asarray(data["fock_stored"]))
    labels = mol.ao_labels()
    slices = mol.aoslice_by_atom()

    fp_ref = row_fingerprints(fock_ref)
    fp_stored = row_fingerprints(fock_stored)

    # perm[i] = stored index holding the AO that PySCF places at index i.
    perm = np.empty(mol.nao, dtype=int)
    residuals = []
    for atom in range(mol.natm):
        s0, s1 = slices[atom][2], slices[atom][3]
        cost = np.linalg.norm(
            fp_ref[s0:s1, None, :] - fp_stored[None, s0:s1, :], axis=2
        )
        rows, cols = linear_sum_assignment(cost)
        perm[s0 + rows] = s0 + cols
        residuals.append(cost[rows, cols].max())

    print(f"entry={name}  nao={mol.nao}")
    print(f"worst per-atom fingerprint residual: {max(residuals):.3e}")

    recovered = fock_stored[np.ix_(perm, perm)]
    print(f"max|F_recovered - F_ref| = {np.abs(recovered - fock_ref).max():.3e}")
    print(f"mean|F_recovered - F_ref| = {np.abs(recovered - fock_ref).mean():.3e}")

    print("\nper-element local permutation (PySCF slot -> stored slot):")
    by_element: dict[int, list[list[int]]] = {}
    for atom in range(mol.natm):
        s0, s1 = slices[atom][2], slices[atom][3]
        by_element.setdefault(int(elements[atom]), []).append(list(perm[s0:s1] - s0))

    consistent = True
    summary = {}
    for z, maps in sorted(by_element.items()):
        same = all(m == maps[0] for m in maps)
        consistent &= same
        summary[str(z)] = {"perm": maps[0], "consistent_across_atoms": same}
        print(f"  Z={z:>2}  n_atoms={len(maps)}  identical across atoms: {same}")
        print(f"       {maps[0]}")

    print(f"\npermutation consistent across all like atoms: {consistent}")

    # Show the map annotated with AO labels for the first atom of each element.
    print("\nannotated map (first atom per element):")
    seen = set()
    for atom in range(mol.natm):
        z = int(elements[atom])
        if z in seen:
            continue
        seen.add(z)
        s0, s1 = slices[atom][2], slices[atom][3]
        for i in range(s0, s1):
            print(f"   {labels[i]:<20} <- stored slot {perm[i] - s0:>3}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "entry": name,
                    "consistent": bool(consistent),
                    "max_abs_error": float(np.abs(recovered - fock_ref).max()),
                    "per_element": summary,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
