#!/usr/bin/env python3
"""Compare a stored CSH Fock matrix against an independently converged one.

With ground truth in hand the failure mode is identifiable rather than guessed:

  permutation   the diagonals are the same multiset per atom but in a
                different order
  scaling       F_stored[i,j] = lam_i lam_j F_true[i,j], so the elementwise
                ratio has outer-sum structure in log|R| and no reindexing can
                ever fix it
  agreement     the matrix is fine and the problem is downstream

Run scripts/revision/omol_csh_reference_scf.py first to produce the npz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import build_mol, inflate_triangle, parse_charge_mult  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz")
    parser.add_argument("--rows", type=int, default=45)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    name = str(data["name"])
    elements = data["elements"]
    coords = data["coords"]
    charge, mult = parse_charge_mult(name)
    mol = build_mol(elements, coords, charge, mult - 1)

    fock_ref = data["fock"]
    fock_stored = inflate_triangle(data["fock_stored"])
    labels = mol.ao_labels()

    print(f"entry={name}  E={float(data['energy']):.6f}  converged={bool(data['converged'])}")
    print(f"nao={mol.nao}  xc={str(data['xc'])}")

    diff = fock_stored - fock_ref
    print(f"\nmax|F_stored - F_ref| = {np.abs(diff).max():.6f}")
    print(f"mean|F_stored - F_ref| = {np.abs(diff).mean():.6f}")
    print(f"corr(F_stored, F_ref)  = {np.corrcoef(fock_stored.ravel(), fock_ref.ravel())[0, 1]:+.6f}")

    ds, dr = np.diag(fock_stored), np.diag(fock_ref)
    print(f"corr(diag)             = {np.corrcoef(ds, dr)[0, 1]:+.6f}")

    # Is the diagonal a per-atom permutation of the reference diagonal?
    slices = mol.aoslice_by_atom()
    print("\nper-atom diagonal: is stored a permutation of reference?")
    for atom in range(min(mol.natm, 6)):
        s0, s1 = slices[atom][2], slices[atom][3]
        a = np.sort(ds[s0:s1])
        b = np.sort(dr[s0:s1])
        print(
            f"  atom {atom} Z={int(elements[atom]):>2}  "
            f"max|sorted diff|={np.abs(a - b).max():.4f}  "
            f"max|unsorted diff|={np.abs(ds[s0:s1] - dr[s0:s1]).max():.4f}"
        )

    # Multiplicative structure: lam_i from the diagonal, then test off-diagonals.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_diag = ds / dr
    good = np.isfinite(ratio_diag) & (np.abs(dr) > 1e-3) & (ratio_diag > 0)
    lam = np.ones(mol.nao)
    lam[good] = np.sqrt(ratio_diag[good])
    predicted = fock_ref * lam[:, None] * lam[None, :]
    mask = np.abs(fock_ref) > 1e-3
    print(
        f"\nscaling hypothesis: max|F_stored - lam F_ref lam| on |F_ref|>1e-3 = "
        f"{np.abs((fock_stored - predicted)[mask]).max():.6f}"
    )
    print(f"  lam range: {lam.min():.4f} .. {lam.max():.4f}")

    print(f"\n{'idx':>4} {'label':<20} {'stored':>11} {'ref':>11} {'diff':>10}")
    for i in range(min(args.rows, mol.nao)):
        print(f"{i:>4} {labels[i]:<20} {ds[i]:>11.5f} {dr[i]:>11.5f} {ds[i] - dr[i]:>10.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
