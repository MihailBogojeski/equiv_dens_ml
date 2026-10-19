#!/usr/bin/env python3
"""
Free-atom densities (spherically averaged atomic KS) in the format of
free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf{,_minimized}.npy: mo_coeff/mo_occ/mo_basis, a radial
spline of the density, and density-fitting coefficients (Coulomb metric) in the auxiliary basis.
The minimized file keeps the occupied (and all s) orbitals and only the l=0 auxiliary functions.
"""
import argparse

import numpy as np
import torch
from pyscf import gto, df

from equiv_dens.utils import orbitals
from equiv_dens.utils.base import numbers_to_symbols
from equiv_dens.utils.hirshfeld_analysis import get_atm_nrks, free_atom_spline

p = argparse.ArgumentParser()
p.add_argument("--basis", default="augccpvdz")
p.add_argument("--auxbasis", default="augccpvqzjkfit")
p.add_argument("--xc", default="PBE")
p.add_argument("--elements", type=int, nargs="*", default=[1, 6, 7, 8, 9, 16, 17])
p.add_argument("--output", required=True)
p.add_argument("--output_minimized", required=True)
args = p.parse_args()


def minimize(d, Z):
    """Same truncation as notebooks/coreless_density.ju.py."""
    mo_basis = d["mo_basis"][Z]
    spec = orbitals.combine_pyscf_basis(d["mo_basis"], max(b[0] for b in mo_basis))[0][Z]
    occ = d["mo_occ"]
    nz = np.nonzero(occ)[0]
    num_min_orb = max(nz[-1] + 1, spec[0][1])
    curr, ncoeff = 0, 0
    for _, n_L, L in spec:
        if curr + n_L > num_min_orb:
            ncoeff += (num_min_orb - curr) * (2 * L + 1)
            break
        ncoeff += n_L * (2 * L + 1)
        curr += n_L
    basis_min, curr = [], 0
    for orb in mo_basis:
        n = len(orb[1]) - 1
        if curr + n < num_min_orb:
            basis_min.append(orb)
            curr += n
        else:
            basis_min.append([orb[0]] + [orb[i][:(num_min_orb - curr + 1)] for i in range(1, len(orb))])
            break
    aux_spec = orbitals.combine_pyscf_basis(d["df_basis"], max(b[0] for b in d["df_basis"][Z]))[0][Z]
    n_s = aux_spec[0][1]
    return {"mo_basis": {Z: basis_min}, "df_basis": {Z: d["df_basis"][Z][:n_s]},
            "mo_coeff": d["mo_coeff"][:ncoeff, :ncoeff], "mo_occ": occ[:ncoeff],
            "spline_interp": d["spline_interp"], "df_coeffs": d["df_coeffs"][:, :n_s]}


full, mini = {}, {}
for Z in args.elements:
    sym = numbers_to_symbols([Z])[0]
    mol = gto.M(atom=[[Z, (0, 0, 0)]], basis=args.basis, spin=Z % 2, verbose=0)
    mf = list(get_atm_nrks(mol, xc=args.xc).values())[0]
    auxmol = gto.M(atom=[[Z, (0, 0, 0)]], basis=args.auxbasis, spin=Z % 2, verbose=0)
    dm = mf.make_rdm1()
    rho_P = np.einsum("ijP,ij->P", df.incore.aux_e2(mf.mol, auxmol, intor="int3c2e"), dm)
    c = np.linalg.solve(auxmol.intor("int2c2e"), rho_P)
    full[Z] = {"mo_coeff": mf.mo_coeff, "mo_occ": mf.mo_occ, "mo_basis": {Z: mol._basis[sym]},
               "spline_interp": free_atom_spline(mf), "df_basis": {Z: auxmol._basis[sym]},
               "df_coeffs": torch.tensor(c[None])}
    mini[Z] = minimize(full[Z], Z)
    print(Z, "electrons", np.einsum("ij,ji->", dm, mol.intor("int1e_ovlp")), flush=True)

np.save(args.output, full, allow_pickle=True)
np.save(args.output_minimized, mini, allow_pickle=True)
