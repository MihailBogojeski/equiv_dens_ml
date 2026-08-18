#!/usr/bin/env python3
"""Build a free-atom SAD prior at a chosen XC (PBE or PBE0).

Follows notebooks/coreless_density.ju.py: spherically averaged atomic RKS
densities, stored as mo_coeff / mo_occ plus a spline interpolant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pyscf import gto

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from equiv_dens.utils.hirshfeld_analysis import (  # noqa: E402
    free_atom_spline,
    get_atm_nrks,
)
from equiv_dens.utils import base as utils  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xc", default="pbe0")
    parser.add_argument("--elements", default="H,C,N,O,S")
    parser.add_argument("--basis", default="augccpvdz")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_pbe0.npy"),
    )
    args = parser.parse_args()

    symbols = [s.strip() for s in args.elements.split(",") if s.strip()]
    atom_str = "; ".join(f"{el} 0 0 {3 * i}" for i, el in enumerate(symbols))
    mol = gto.M(atom=atom_str, basis=args.basis, charge=0, spin=0)
    mf_elems = get_atm_nrks(mol, xc=args.xc)

    result = {}
    for key, mf_elem in mf_elems.items():
        el = "".join(c for c in key if c.isalpha())
        anum = int(utils.symbols_to_numbers([el])[0])
        atom_mol = gto.M(atom=[[anum, [0, 0, 0]]], basis=args.basis, spin=anum % 2)
        result[anum] = {
            "symbol": el,
            "xc": args.xc,
            "mo_coeff": np.asarray(mf_elem.mo_coeff),
            "mo_occ": np.asarray(mf_elem.mo_occ),
            "spline_interp": free_atom_spline(mf_elem),
            "mo_basis": {anum: atom_mol._basis[el]},
        }
        print(f"{el} Z={anum} mo_coeff={result[anum]['mo_coeff'].shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result, allow_pickle=True)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
