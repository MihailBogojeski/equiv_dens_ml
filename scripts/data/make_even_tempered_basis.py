#!/usr/bin/env python3
"""
Generate an even-tempered density (auxiliary) basis derived from a PySCF basis (default augccpvqzjkfit)
and the files a training config consumes for it.

Uncontracted shells in a geometric progression with ratio beta, going down from the top of each L's exponent range,
n_L = ceil(log(emax/emin)/log(beta)) + 1. A single global beta is used for all L and elements and is searched so that the
function count averaged over the elements is --size_factor times the base basis (--range_ext widens every range by that
factor at both ends). Recipes (--recipe):
  jkrange  range of the base basis shells per L and element
  aoprod   range from sums of AO (--ao_basis) primitive exponents per L, as pyscf.df.addons._aug_etb_element, for the L
           present in the base basis (L above 2*Lmax_ao falls back to the base range)
  perL     base ranges, ratio beta_L = (base mean exponent ratio for L and element)^(1/s) with a single global s searched
--keep_base uses the base basis unchanged (reproduction test of the existing jkfit files).

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


def l_ranges(basis):
    """{l: (emin, emax)} of the shells of a PySCF basis."""
    r = {}
    for l in sorted({sh[0] for sh in basis}):
        exps = [sh[1][0] for sh in basis if sh[0] == l]
        r[l] = (min(exps), max(exps))
    return r


def ao_product_ranges(ao_basis, l_aux_max):
    """Per-L exponent ranges from sums of AO primitive exponents (as pyscf.df.addons._aug_etb_element)."""
    lmax = max(b[0] for b in ao_basis)
    emin_l, emax_l = [1e99] * (lmax + 1), [0.0] * (lmax + 1)
    for b in ao_basis:
        e_c = np.array(b[2:] if isinstance(b[1], (int, np.integer)) else b[1:])
        es = e_c[:, 0][np.abs(e_c[:, 1:]).max(axis=1) > 1e-3]
        emax_l[b[0]] = max(es.max(), emax_l[b[0]])
        emin_l[b[0]] = min(es.min(), emin_l[b[0]])
    emin_l, emax_l = np.array(emin_l), np.array(emax_l)
    emin, emax = emin_l[:, None] + emin_l, emax_l[:, None] + emax_l
    lsum = np.arange(lmax + 1)[:, None] + np.arange(lmax + 1)
    r = {}
    for L in range(min(l_aux_max, 2 * lmax) + 1):
        r[L] = (emin[lsum == L].min(), emax[lsum == L].max())
    return r


def build_basis(sym, recipe, beta, range_ext, base, ao_basis):
    """Even-tempered basis for one element. beta is the global ratio (jkrange, aoprod) or the global factor s (perL)."""
    jk = l_ranges(base)
    if recipe == "aoprod":
        ranges = {**jk, **ao_product_ranges(ao_basis, max(jk))}
        ranges = {L: ranges[L] for L in jk}
    else:
        ranges = jk
    new = []
    for l, (emin, emax) in ranges.items():
        emin, emax = emin / range_ext, emax * range_ext
        if recipe == "perL":
            n_jk = sum(1 for sh in base if sh[0] == l)
            if n_jk == 1:
                new += [[l, [float(emax), 1.0]]]
                continue
            beta_l = (jk[l][1] / jk[l][0]) ** (1 / (n_jk - 1) / beta)
        else:
            beta_l = beta
        n = int(np.ceil(np.log(emax / emin) / np.log(beta_l) - 1e-12)) + 1
        new += [[l, [float(emax * beta_l ** -k), 1.0]] for k in range(n)]
    return new


def nfuncs(b):
    return sum(2 * sh[0] + 1 for sh in b)


def find_beta(recipe, size_factor, range_ext, elements, bases, ao_bases):
    """Scan the global beta (or s) and take the value whose mean per-element function ratio is closest to size_factor."""
    grid = np.linspace(1.0, 4.0, 3001)[1:] if recipe == "perL" else np.geomspace(1.02, 6.0, 3000)
    best = None
    for b in grid:
        ratio = np.mean([nfuncs(build_basis(s, recipe, b, range_ext, bases[s], ao_bases[s])) / nfuncs(bases[s]) for s in elements])
        if best is None or abs(ratio - size_factor) < best[0] - 1e-12:
            best = (abs(ratio - size_factor), b, ratio)
    return best[1], best[2]


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
    p.add_argument("--recipe", choices=["jkrange", "aoprod", "perL"], required=True)
    p.add_argument("--name", default=None, help="default: <base>_et_<recipe>")
    p.add_argument("--size_factor", type=float, default=1.25, help="target mean (over elements) total function count relative to the base")
    p.add_argument("--beta", type=float, default=None, help="global ratio (s for perL); default: searched to hit --size_factor")
    p.add_argument("--range_ext", type=float, default=1.0, help="multiplicative extension of the exponent range on both ends")
    p.add_argument("--keep_base", action="store_true", help="use the base basis unchanged (reproduction test)")
    p.add_argument("--ao_basis", default="augccpvdz", help="orbital basis for the aoprod recipe")
    p.add_argument("--elements", nargs="+", default=ELEMENTS)
    p.add_argument("--out_dir", default="/workspace/datasets")
    p.add_argument("--atom_dens_in", default="/workspace/datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy")
    args = p.parse_args()

    name = args.name or f"{args.base}_et_{args.recipe}"
    bases = {sym: gto.basis.load(args.base, sym) for sym in args.elements}
    ao_bases = {sym: gto.basis.load(args.ao_basis, sym) for sym in args.elements}
    if not args.keep_base:
        beta = args.beta
        if beta is None:
            beta, ratio = find_beta(args.recipe, args.size_factor, args.range_ext, args.elements, bases, ao_bases)
            print("recipe %s: beta (s for perL) = %.5f, mean function ratio = %.4f" % (args.recipe, beta, ratio))

    basis, orbital_basis, radial = {}, {}, {}
    for sym in args.elements:
        base = bases[sym]
        basis[sym] = base if args.keep_base else build_basis(sym, args.recipe, beta, args.range_ext, base, ao_bases[sym])
        orbital_basis[sym], radial[sym] = libcint_radial(sym, basis[sym])
        count = lambda b: {l: sum(1 for sh in b if sh[0] == l) for l in sorted({sh[0] for sh in b})}
        print(sym, "jkfit: %d funcs" % nfuncs(base), count(base), "| new: %d funcs (x%.3f)" % (nfuncs(basis[sym]), nfuncs(basis[sym]) / nfuncs(base)), count(basis[sym]))

    out = lambda f: os.path.join(args.out_dir, f)
    np.save(out(f"{name}_basis.npy"), basis, allow_pickle=True)
    np.save(out(f"{name}_orbital_basis_df.npy"), orbital_basis, allow_pickle=True)
    np.save(out(f"{name}_radial_coeffs_libcint_df.npy"), radial, allow_pickle=True)

    atoms = np.load(args.atom_dens_in, allow_pickle=True).item()
    for z, atom in atoms.items():
        sym = gto.mole.ELEMENTS[z]
        atom["df_basis"] = {z: basis[sym]}
        atom["df_coeffs"] = torch.from_numpy(refit_free_atom(z, atom, basis[sym])).reshape(1, -1)
    np.save(out(f"free_atom_densities_augccpvdz_{name}_pyscf.npy"), atoms, allow_pickle=True)


if __name__ == "__main__":
    main()
