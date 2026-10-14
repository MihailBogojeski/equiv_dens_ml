#!/usr/bin/env python3
"""
Refit the density-fitting coefficients of an existing label file in a different auxiliary basis.

Input/output: object array (n, 2) of [mol_kwargs_dict, calc_dict] with calc_dict holding mo_coeff, mo_occ, ...
The DF coefficients are c = J^-1 (P|ij) D_ij (Coulomb metric) with D from the stored MO coefficients, so no new DFT
is needed. 'df_coeff' and 'auxbasis' are replaced; everything else is kept.

--basis is either a PySCF basis name (stored as a string) or a .npy file with a {symbol: pyscf basis} dict
(e.g. make_even_tempered_basis.py output; stored as that dict, which Mole.basis accepts directly).
"""
import argparse
import os
import sys

import numpy as np
from pyscf import gto, df

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import equiv_dens.compat  # noqa: F401,E402  (before cupy/gpu4pyscf)


def fit_df_coeff(mol, auxmol, dm, gpu):
    if gpu:
        import cupy
        from gpu4pyscf.df.int3c2e_bdiv import contract_int3c2e_dm, int2c2e
        rho_aux = contract_int3c2e_dm(mol, auxmol, cupy.asarray(dm))
        return cupy.linalg.solve(int2c2e(auxmol), rho_aux).get()
    rho_aux = np.einsum("ijP,ij->P", df.incore.aux_e2(mol, auxmol, intor="int3c2e"), dm)
    return np.linalg.solve(auxmol.intor("int2c2e"), rho_aux)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", nargs="+", required=True, help="one output path per input")
    p.add_argument("--basis", default="augccpvqzjkfit")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.basis.endswith(".npy"):
        auxbasis = np.load(args.basis, allow_pickle=True).item()
    else:
        auxbasis = args.basis

    if not args.cpu:
        import cupy
        cupy.get_default_memory_pool().set_limit(size=2.5 * 1024**3)

    for inp, outp in zip(args.input, args.output):
        data = np.load(inp, allow_pickle=True)
        for mol_dict, calc_dict in data:
            mol_dict = dict(mol_dict, verbose=0)
            mol = gto.Mole(**mol_dict)
            mol.build()
            auxmol = df.addons.make_auxmol(mol, auxbasis)
            mo, occ = calc_dict["mo_coeff"], calc_dict["mo_occ"]
            dm = (mo * occ) @ mo.T
            calc_dict["df_coeff"] = fit_df_coeff(mol, auxmol, dm, not args.cpu)
            calc_dict["auxbasis"] = auxbasis
        np.save(outp, data, allow_pickle=True)
        print(inp, "->", outp, data[0][1]["df_coeff"].shape)


if __name__ == "__main__":
    main()
