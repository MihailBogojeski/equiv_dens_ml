#!/usr/bin/env python3
"""
Generate an even-tempered density (auxiliary) basis derived from a PySCF basis (default augccpvqzjkfit)
and the files a training config consumes for it.

For each element and angular momentum l in the base basis with n_l shells, the new basis has
round(size_factor * n_l) uncontracted shells (python round) with exponents in a geometric progression
spanning [e_min / range_ext, e_max * range_ext] of the base l-shells. --keep_base uses the base basis
unchanged (reproduction test of the existing jkfit files).

Outputs in --out_dir (NAME = --name):
  NAME_basis.npy                                  {symbol: [[l, [exp, 1.0]], ...]} (PySCF format, usable as mol.basis)
  NAME_orbital_basis_df.npy                       {symbol: [(Z, nctr, l), ...]}
  NAME_radial_coeffs_libcint_df.npy               {symbol: [(exponents, libcint coeffs), ...]}
  free_atom_densities_augccpvdz_NAME_pyscf.npy    free-atom file with df_basis/df_coeffs refit in the new basis
The init_L0 file is not generated: it holds learned values of a trained model and no config uses it.
"""
import argparse
import os
import sys

import numpy as np
import torch
from pyscf import gto, df

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import equiv_dens.compat  # noqa: F401,E402  (before cupy/gpu4pyscf)

ELEMENTS = ["O", "H", "N", "C", "S", "F", "Cl"]


def even_tempered(base, size_factor, range_ext):
    new = []
    for l in sorted({sh[0] for sh in base}):
        exps = np.array([sh[1][0] for sh in base if sh[0] == l])
        n = round(size_factor * len(exps))
        new_exps = np.geomspace(exps.max() * range_ext, exps.min() / range_ext, n)
        new += [[l, [float(e), 1.0]] for e in new_exps]
    return new


def libcint_radial(sym, basis):
    z = gto.charge(sym)
    mol = gto.M(atom=sym, basis={sym: basis}, spin=z % 2)
    orb = [(np.int32(z), np.int32(mol.bas_nctr(i)), np.int32(mol.bas_angular(i))) for i in range(mol.nbas)]
    rad = [(mol.bas_exp(i), mol._libcint_ctr_coeff(i)[:, 0]) for i in range(mol.nbas)]
    return orb, rad


def refit_free_atom(z, atom, basis):
    sym = gto.mole.ELEMENTS[z]
    mol = gto.M(atom=sym, basis={sym: atom["mo_basis"][z]}, spin=z % 2)
    mo, occ = atom["mo_coeff"], atom["mo_occ"]
    dm = (mo * occ) @ mo.T
    auxmol = df.addons.make_auxmol(mol, {sym: basis})
    rho_aux = np.einsum("ijP,ij->P", df.incore.aux_e2(mol, auxmol, intor="int3c2e"), dm)
    return np.linalg.solve(auxmol.intor("int2c2e"), rho_aux)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="augccpvqzjkfit")
    p.add_argument("--name", default="augccpvqzjkfit_et125")
    p.add_argument("--size_factor", type=float, default=1.25)
    p.add_argument("--range_ext", type=float, default=1.0, help="multiplicative extension of the exponent range on both ends")
    p.add_argument("--keep_base", action="store_true", help="use the base basis unchanged (reproduction test)")
    p.add_argument("--elements", nargs="+", default=ELEMENTS)
    p.add_argument("--out_dir", default="/workspace/datasets")
    p.add_argument("--atom_dens_in", default="/workspace/datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy")
    args = p.parse_args()

    basis, orbital_basis, radial = {}, {}, {}
    nf = lambda b: sum(2 * sh[0] + 1 for sh in b)
    for sym in args.elements:
        base = gto.basis.load(args.base, sym)
        basis[sym] = base if args.keep_base else even_tempered(base, args.size_factor, args.range_ext)
        orbital_basis[sym], radial[sym] = libcint_radial(sym, basis[sym])
        print(sym, "base: %d shells %d funcs" % (len(base), nf(base)), "new: %d shells %d funcs" % (len(basis[sym]), nf(basis[sym])))

    out = lambda f: os.path.join(args.out_dir, f)
    np.save(out(f"{args.name}_basis.npy"), basis, allow_pickle=True)
    np.save(out(f"{args.name}_orbital_basis_df.npy"), orbital_basis, allow_pickle=True)
    np.save(out(f"{args.name}_radial_coeffs_libcint_df.npy"), radial, allow_pickle=True)

    atoms = np.load(args.atom_dens_in, allow_pickle=True).item()
    for z, atom in atoms.items():
        sym = gto.mole.ELEMENTS[z]
        atom["df_basis"] = {z: basis[sym]}
        atom["df_coeffs"] = torch.from_numpy(refit_free_atom(z, atom, basis[sym])).reshape(1, -1)
    np.save(out(f"free_atom_densities_augccpvdz_{args.name}_pyscf.npy"), atoms, allow_pickle=True)


if __name__ == "__main__":
    main()
