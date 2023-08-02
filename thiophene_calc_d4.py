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

hf.MUTE_CHKFILE = True

# %load_ext autoreload
# %autoreload 2

# %%
xyz_data = list(ase.io.iread('datasets/small_thiophene_d4_2.xyz'))
save_file = 'datasets/thiophene_123_small_pyscf_d4_xyz_augccpvdz.npy'

# %%
# idx = np.arange(0, 21)
idx = np.arange(0, 21)

results = []
# for mol in xyz_data:
for i in idx:
    mol = xyz_data[i]
    npy_data = utils.ase_to_npy2([mol])
    pos = npy_data['positions'][0]
    atom_nums = npy_data['atom_numbers'][0]
    print('npy data atoms', npy_data['atom_numbers'][0])

    start = time.time()
    atom = []
    for j in range(len(atom_nums)):
        if atom_nums[j] < 1:
            continue
        atom.append((atom_nums[j], pos[j, :]))
    mol = gto.M(atom=atom, basis='augccpvdz')

    mf = dft.RKS(mol)
    mf.chkfile = False
    mf.xc = 'pbe'
    mf.max_cycle = 1000
    d4mf = d4disp.energy(mf).run()
    grad = d4mf.nuc_grad_method()
    gradients = grad.kernel()
    # print('combined gradient', grad.kernel())
    res = []
    res.append(mol.pack())
    calc_dict = {}
    print('mo occ', d4mf.mo_occ)
    calc_dict['mo_coeff'] = d4mf.mo_coeff
    calc_dict['mo_occ'] = d4mf.mo_occ
    calc_dict['energy'] = d4mf.e_tot
    calc_dict['forces'] = -gradients/ase.units.Bohr
    res.append(calc_dict)
    results.append(res)
    print('elapsed time', time.time() - start)

# %%
idx = np.concatenate([np.arange(0, 10), np.arange(1000, 1010), np.arange(2000, 2001)])
thio_poly = np.load('datasets/thiophene_all_test_d4.npy', allow_pickle=True).item()
for res_i, i in enumerate(idx):
    nonzero = thio_poly['atom_numbers'][i] > 0
    # print('energy old', utils.hartree_to_kcal(thio_poly['energy'][i]))
    # print('forces old', utils.hartree_to_kcal(thio_poly['forces'][i, nonzero]))
    # print('energy new', utils.hartree_to_kcal(results[res_i][1]['energy']))
    # print('forces new', utils.hartree_to_kcal(results[res_i][1]['forces']))
    print('energy diff', utils.hartree_to_kcal(results[res_i][1]['energy'] - thio_poly['energy'][i]))
    print('forces diff', utils.hartree_to_kcal(results[res_i][1]['forces'] - thio_poly['forces'][i, nonzero]))
    # print('mo coeffs diff', results[0][1]['mo_coeff'][res_i] - thio_poly[i][1]['mo_coeff'][0])
