#!/usr/bin/env python3
"""Converge a reference SCF at the CSH level of theory for convention checking.

The stored CSH matrices are wB97M-V/def2-TZVPD Fock matrices. Reproducing one
independently in PySCF gives a ground-truth matrix in known (PySCF) AO order,
which is what pins down the stored ordering: an atomic-density guess is too
crude to resolve the polarization shells, whereas a converged Fock matrix has a
non-degenerate diagonal that identifies every AO.

Writes the converged Fock matrix, density matrix and energy to an npz.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from pyscf import dft

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import build_mol, parse_charge_mult  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--subset", default="ani1xbb")
    parser.add_argument("--entry", default=None)
    parser.add_argument("--xc", default="wb97m-v")
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--density-fit", action="store_true")
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with h5py.File(args.path, "r") as fh:
        group = fh[args.subset]
        name = args.entry or sorted(group.keys())[0]
        node = group[name]
        elements = np.asarray(node["elements"][()])
        coords = np.asarray(node["coords"][()])
        fock_stored = np.asarray(node["fock"][()])

    charge, mult = parse_charge_mult(name)
    mol = build_mol(elements, coords, charge, mult - 1)
    print(f"entry={name} natoms={len(elements)} nao={mol.nao} nelec={mol.nelectron}", flush=True)
    print(f"xc={args.xc} charge={charge} mult={mult}", flush=True)

    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.grids.level = args.grid
    mf.chkfile = None
    mf.verbose = 4
    mf.max_cycle = args.max_cycle
    if args.density_fit:
        mf = mf.density_fit()

    start = time.time()
    energy = mf.kernel()
    print(f"converged={mf.converged} E={energy:.9f} in {time.time() - start:.1f}s", flush=True)

    dm = mf.make_rdm1()
    fock = mf.get_fock(dm=dm)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        fock=fock,
        dm=dm,
        mo_energy=mf.mo_energy,
        energy=energy,
        converged=mf.converged,
        elements=elements,
        coords=coords,
        fock_stored=fock_stored,
        charge=charge,
        multiplicity=mult,
        name=name,
        xc=args.xc,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
