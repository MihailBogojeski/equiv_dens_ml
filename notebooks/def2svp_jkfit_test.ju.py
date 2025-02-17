# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..
# %%
import ase
import numpy as np
import pyscf
import time
import os
from pyscf.scf import hf
from pyscf import gto, dft, df, lib
from pyscf.gto import mole
import scipy

import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.errors import ErrorDict
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
     CubicalGrid, spherical_grid, spherical_radial_sampling
from equiv_dens.training.model_loader import load_model
import equiv_dens.utils.base as utils

from functools import partial

import ase.io
import dftd4.pyscf as d4disp
from pyscf.dft import gen_grid, radi, numint
from pyscf.lib import param
from equiv_dens.training import utils as train_utils
from argparse import Namespace
from equiv_dens.training import model_loader
from pymbd.fortran import MBDGeom
from vdw import to_mbd
from equiv_dens.utils import hirshfeld_analysis, orbitals
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

hf.MUTE_CHKFILE = True

# %%
data = np.load('datasets/ethanol_dft_test.npy', allow_pickle=True).item()
print(data.keys())
print(data['atom_numbers'])
# %%
atom = [[data['atom_numbers'][i], data['positions'][0, i]] for i in range(data['atom_numbers'].shape[0])]
basis_sets = ['def2svp']
mols = []
mfs = []
for bas in basis_sets:
    mol = gto.M(atom=atom, basis=bas, charge=1, spin=1)
    mf = dft.RKS(mol)
    mf.xc = 'PBE'
    mf.chkfile = False
    mf.kernel()
    mols.append(mol)
    mfs.append(mf)
    print('mo occ', mf.mo_occ)
# %%
auxbasis_sets = ['def2svpjkfit', 'def2tzvpjkfit', 'def2universaljkfit', 'augccpvqzjkfit', ]

for abas in auxbasis_sets:
    auxmol = gto.M(atom=atom, basis=abas)
    for i in range(len(mols)):
        mol = mols[i]
        mf = mfs[i]
        dm1 = mf.make_rdm1()
        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
        ints_2c2e = auxmol.intor('int2c2e')
        nao = mol.nao
        naux = auxmol.nao
        df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao * nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

        grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=1)
        coords, weights = gen_grid.get_partition(mol, grid_spec)
        ao = numint.eval_ao(mol, coords, deriv=0)
        rho_dm1 = numint.eval_rho(mol, ao, dm1)

        ao = numint.eval_ao(auxmol, coords, deriv=0)
        rho_df = np.einsum('ij,j->i', ao, df_basis)

        rho_diff = np.sum(np.abs(rho_dm1 - rho_df) * weights) / np.sum(mol.atom_charges())

        print('basis', mol.basis, 'auxbasis', abas, 'diff', rho_diff)


# %% 
auxmol = gto.M(atom=atom, basis='augccpvqzjkfit')
print(auxmol._basis)
print({key: sum([2 * b[0] + 1 for b in auxmol._basis[key]]) for key in auxmol._basis.keys()})
# %%
mol_tmp = gto.M(atom=atom, basis='def2tzvp')
auxbasis = df.addons.aug_etb(mol_tmp, beta=2)
auxmol = gto.M(atom=mol_tmp.atom, basis=auxbasis)
print('auxmol basis', auxmol.basis)
print({key: sum([2 * b[0] + 1 for b in auxmol._basis[key]]) for key in auxmol._basis.keys()})
for i in range(len(mols)):
    mol = mols[i]
    mf = mfs[i]
    dm1 = mf.make_rdm1()
    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
    ints_2c2e = auxmol.intor('int2c2e')
    nao = mol.nao
    naux = auxmol.nao
    df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao * nao, naux).T)
    df_coef = df_coef.reshape(naux, nao, nao)
    df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

    grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=1)
    coords, weights = gen_grid.get_partition(mol, grid_spec)
    ao = numint.eval_ao(mol, coords, deriv=0)
    rho_dm1 = numint.eval_rho(mol, ao, dm1)

    ao = numint.eval_ao(auxmol, coords, deriv=0)
    rho_df = np.einsum('ij,j->i', ao, df_basis)

    rho_diff = np.sum(np.abs(rho_dm1 - rho_df) * weights) / np.sum(mol.atom_charges())

    print('basis', mol.basis, 'diff', rho_diff)

# %%
mol_tmp = gto.M(atom=atom, basis='def2qzvp')
auxmol = df.make_auxmol(mol_tmp)
print('auxmol basis', auxmol.basis)
for i in range(len(mols)):
    mol = mols[i]
    mf = mfs[i]
    dm1 = mf.make_rdm1()
    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c1e')
    ints_2c2e = auxmol.intor('int1e_ovlp')
    nao = mol.nao
    naux = auxmol.nao
    df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao * nao, naux).T)
    df_coef = df_coef.reshape(naux, nao, nao)
    df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

    grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=1)
    coords, weights = gen_grid.get_partition(mol, grid_spec)
    ao = numint.eval_ao(mol, coords, deriv=0)
    rho_dm1 = numint.eval_rho(mol, ao, dm1)

    ao = numint.eval_ao(auxmol, coords, deriv=0)
    rho_df = np.einsum('ij,j->i', ao, df_basis)

    rho_diff = np.sum(np.abs(rho_dm1 - rho_df) * weights) / np.sum(mol.atom_charges())

    print('basis', mol.basis, 'diff', rho_diff)
# %%
auxbasis_sets = ['def2svpjkfit', 'def2tzvpjkfit', 'def2universaljkfit', 'augccpvqzjkfit', ]

for abas in auxbasis_sets:
    auxmol = gto.M(atom=atom, basis=abas)
    for i in range(len(mols)):
        mol = mols[i]
        mf = mfs[i]
        dm1 = mf.make_rdm1()
        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c1e')
        ints_2c2e = auxmol.intor('int1e_ovlp')
        nao = mol.nao
        naux = auxmol.nao
        df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao * nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

        grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=1)
        coords, weights = gen_grid.get_partition(mol, grid_spec)
        ao = numint.eval_ao(mol, coords, deriv=0)
        rho_dm1 = numint.eval_rho(mol, ao, dm1)

        ao = numint.eval_ao(auxmol, coords, deriv=0)
        rho_df = np.einsum('ij,j->i', ao, df_basis)

        rho_diff = np.sum(np.abs(rho_dm1 - rho_df) * weights) / np.sum(mol.atom_charges())

        print('basis', mol.basis, 'auxbasis', abas, 'diff', rho_diff)


