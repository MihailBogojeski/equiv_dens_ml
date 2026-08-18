#!/usr/bin/env python3
"""Compare the stored CSH Fock diagonal against PySCF's AO layout.

The diagonal is the most legible part of the matrix: each AO's diagonal Fock
element sits near the corresponding atomic level, so lining it up against
``mol.ao_labels()`` immediately shows whether the packing (upper vs lower
triangle) and the shell ordering agree with PySCF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import build_mol, parse_charge_mult, triangle_dim  # noqa: E402


def upper_row_major_diag(vec: np.ndarray, n: int) -> np.ndarray:
    idx = np.array([i * n - i * (i - 1) // 2 for i in range(n)])
    return vec[idx]


def lower_row_major_diag(vec: np.ndarray, n: int) -> np.ndarray:
    idx = np.array([i * (i + 1) // 2 + i for i in range(n)])
    return vec[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--subset", default="ani1xbb")
    parser.add_argument("--entry", default=None)
    parser.add_argument("--rows", type=int, default=60)
    args = parser.parse_args()

    with h5py.File(args.path, "r") as fh:
        group = fh[args.subset]
        name = args.entry or sorted(group.keys())[0]
        node = group[name]
        elements = np.asarray(node["elements"][()])
        coords = np.asarray(node["coords"][()])
        fock = np.asarray(node["fock"][()])

    charge, mult = parse_charge_mult(name)
    mol = build_mol(elements, coords, charge, mult - 1)
    n = triangle_dim(fock.shape[0])
    print(f"entry {name}  natoms={len(elements)}  Z={list(elements)}")
    print(f"charge={charge} mult={mult}  nao stored={n} pyscf={mol.nao} nelec={mol.nelectron}")

    hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    ovlp = mol.intor("int1e_ovlp")
    s_eig = np.linalg.eigvalsh(ovlp)
    print(f"overlap eigenvalue range: {s_eig[0]:.3e} .. {s_eig[-1]:.3e}")
    print(f"overlap eigenvalues below 1e-3: {(s_eig < 1e-3).sum()} of {n}")

    diag_upper = upper_row_major_diag(fock, n)
    diag_lower = lower_row_major_diag(fock, n)
    labels = mol.ao_labels()

    print(f"\ncorr(diag_upper, diag_hcore) = {np.corrcoef(diag_upper, np.diag(hcore))[0, 1]:+.6f}")
    print(f"corr(diag_lower, diag_hcore) = {np.corrcoef(diag_lower, np.diag(hcore))[0, 1]:+.6f}")

    print(f"\n{'idx':>4} {'pyscf ao label':<22} {'F_up':>12} {'F_low':>12} {'Hcore':>12}")
    for i in range(min(args.rows, n)):
        print(
            f"{i:>4} {labels[i]:<22} {diag_upper[i]:>12.5f} "
            f"{diag_lower[i]:>12.5f} {np.diag(hcore)[i]:>12.5f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
