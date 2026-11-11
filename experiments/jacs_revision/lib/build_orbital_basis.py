#!/usr/bin/env python3
"""Regenerate the auxiliary orbital-basis and radial-coefficient tables.

These two files define the per-element (n, l) channel layout DenSNet predicts
into. The shipped pair is inconsistent: the radial coefficients cover
C, Cl, F, H, N, O, S while the orbital basis covers only C, H, N, O, S, because
the two were generated from different atom lists. Any element missing from the
orbital basis cannot be trained on at all.

This follows scripts/md/extract_orbital_coeffs.py but takes the element list as
an argument and, before writing, checks that every element already present
regenerates bit-identically, so extending the table cannot perturb existing
models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols
from pyscf import gto


def extract(symbols: list[str], basis: str, libcint: bool):
    # One atom of each element, spaced far enough apart that nothing overlaps;
    # only the per-element shell structure is read back out.
    atom = "; ".join(f"{sym} 0 0 {3 * i}" for i, sym in enumerate(symbols))
    mol = gto.M(atom=atom, basis=basis, spin=None)

    ao_basis: dict[str, list] = {}
    ao_coeffs: dict[str, list] = {}
    first_atom: dict[str, int] = {}
    for shell in range(mol._bas.shape[0]):
        atom_index = mol.bas_atom(shell)
        z = int(mol._atm[atom_index, 0])
        symbol = chemical_symbols[z]
        if symbol not in ao_basis:
            ao_basis[symbol] = []
            ao_coeffs[symbol] = []
            first_atom[symbol] = atom_index
        if first_atom[symbol] != atom_index:
            continue
        l = mol.bas_angular(shell)
        nprim = mol.bas_nprim(shell)
        nctr = mol.bas_nctr(shell)
        exps = mol.bas_exp(shell)
        ctr = mol._libcint_ctr_coeff(shell) if libcint else mol.bas_ctr_coeff(shell)
        for j in range(nctr):
            ao_basis[symbol].append((np.int32(z), np.int32(nprim), np.int32(l)))
            ao_coeffs[symbol].append((np.array(exps), np.array(ctr[:, j])))
    return ao_basis, ao_coeffs


def same_basis(a, b) -> bool:
    if len(a) != len(b):
        return False
    return all(tuple(int(x) for x in s) == tuple(int(x) for x in t) for s, t in zip(a, b))


def same_coeffs(a, b) -> bool:
    if len(a) != len(b):
        return False
    for (e1, c1), (e2, c2) in zip(a, b):
        if not np.allclose(np.asarray(e1, float), np.asarray(e2, float)):
            return False
        if not np.allclose(np.asarray(c1, float), np.asarray(c2, float)):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="augccpvqzjkfit")
    parser.add_argument("--elements", default="H,C,N,O,F,S,Cl,Br")
    parser.add_argument("--basis-out", required=True)
    parser.add_argument("--coeffs-out", required=True)
    parser.add_argument("--compare-basis", default=None)
    parser.add_argument("--compare-coeffs", default=None)
    parser.add_argument("--write", action="store_true", help="write only if checks pass")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.elements.split(",") if s.strip()]
    ao_basis, ao_coeffs = extract(symbols, args.basis, libcint=True)

    print(f"basis={args.basis} elements={symbols}")
    for sym in symbols:
        naux = sum(2 * int(s[2]) + 1 for s in ao_basis[sym])
        print(f"  {sym:<3} shells={len(ao_basis[sym]):>3}  naux={naux:>3}")

    ok = True
    if args.compare_basis:
        ref = np.load(args.compare_basis, allow_pickle=True).item()
        for sym in ref:
            if sym not in ao_basis:
                print(f"  MISSING {sym} in regenerated basis")
                ok = False
            elif not same_basis(ref[sym], ao_basis[sym]):
                print(f"  CHANGED orbital basis for {sym}")
                ok = False
        print(f"orbital basis identical for all pre-existing elements: {ok}")

    if args.compare_coeffs:
        ref = np.load(args.compare_coeffs, allow_pickle=True).item()
        coeff_ok = True
        for sym in ref:
            if sym not in ao_coeffs:
                print(f"  MISSING {sym} in regenerated coeffs")
                coeff_ok = False
            elif not same_coeffs(ref[sym], ao_coeffs[sym]):
                print(f"  CHANGED radial coeffs for {sym}")
                coeff_ok = False
        print(f"radial coeffs identical for all pre-existing elements: {coeff_ok}")
        ok = ok and coeff_ok

    if not args.write:
        print("dry run; pass --write to save")
        return 0 if ok else 1
    if not ok:
        raise SystemExit("refusing to write: pre-existing elements changed")

    Path(args.basis_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.basis_out, ao_basis, allow_pickle=True)
    np.save(args.coeffs_out, ao_coeffs, allow_pickle=True)
    print(f"wrote {args.basis_out}")
    print(f"wrote {args.coeffs_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
