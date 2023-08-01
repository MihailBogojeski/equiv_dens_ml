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

thio_poly = np.load('datasets/thiophene_all_test_pyscf_d4_augccpvdz.npy', allow_pickle=True)
save_file = 'datasets/thiophene_123_small_pyscf_d4_augccpvdz.npy'

idx = np.concatenate([np.arange(0, 5), np.arange(1000, 1005)])
idx = np.concatenate([np.arange(0, 5), np.arange(1000, 1005)])

results = []
for i in idx:
    calc_dict_old = thio_poly[i]
    print(i)
    start = time.time()
    mol = gto.Mole.unpack(calc_dict_old[0])
    mol.build()

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
    print('mo occ', mf.mo_occ)
    calc_dict['mo_coeff'] = d4mf.mo_coeff
    calc_dict['mo_occ'] = d4mf.mo_occ
    calc_dict['energy'] = d4mf.e_tot
    calc_dict['forces'] = -gradients/ase.units.Bohr
    res.append(calc_dict)
    results.append(res)
    if (i+1) % 1000 == 0:
        print('i=', i, 'saving file')
        # if (i+1) == 4000:
        #     break
        np.save(save_file, results, allow_pickle=True)
    print('elapsed time', time.time() - start)

np.save(save_file, results, allow_pickle=True)

for i in idx:
    print('energy diff', utils.hartree_to_kcal(results[0][1]['energy'] - thio_poly[0][1]['energy']))
    print('forces diff', utils.hartree_to_kcal(results[0][1]['forces'] - thio_poly[0][1]['forces']))
    print('mo coeffs diff', results[0][1]['mo_coeff'][0] - thio_poly[0][1]['mo_coeff'][0])
