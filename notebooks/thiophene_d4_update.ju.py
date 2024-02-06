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
thio_poly = np.load('datasets/thiophene_all_test_pyscf_augccpvdz.npy', allow_pickle=True)
save_file = 'datasets/thiophene_all_test_pyscf_d4_augccpvdz.npy'

# %%
if os.path.exists(save_file):
    results = np.load(save_file, allow_pickle=True)
else:
    results = []

for i, calc_dict in enumerate(thio_poly):
    print(i)
    start = time.time()
    mol = gto.Mole.unpack(calc_dict[0])
    mol.build()

    disp = d4disp.DFTD4Dispersion(mol, xc='pbe').kernel()
    print('calc time', time.time() - start)
    print('D4 energy', disp[0])
    print('D4 gradient', disp[1])
    print('calc energy', calc_dict[1]['energy'])
    print('calc forces', calc_dict[1]['forces'])
    if 'df_coeff' in calc_dict[1].keys():
        calc_dict[1].pop('df_coeff')

    calc_dict[1]['energy'] += disp[0]
    calc_dict[1]['forces'] += -disp[1]/ase.units.Bohr
    results.append(calc_dict)
    print((i+1) % 1000)
    if (i+1) % 1000 == 0:
        print('i=', i, 'saving file')
        # if (i+1) == 4000:
        #     break
        np.save(save_file, results, allow_pickle=True)
        

    # #print(mol.pack())
    # mf = dft.RKS(mol)
    # mf.chkfile=False
    # mf.xc = 'pbe'
    # #mf.max_cycle = 1000
    # mf.kernel()
    # g = mf.nuc_grad_method()
    # gradients = g.grad()
    # #print(mfs[i].mo_coeff)
    # res = []
    # res.append(mol.pack())
    # calc_dict = {}
    # print('mo occ', mf.mo_occ)
    # calc_dict['mo_coeff'] = mf.mo_coeff
    # calc_dict['mo_occ'] = mf.mo_occ
    # calc_dict['energy'] = mf.e_tot
    # calc_dict['forces'] = -gradients/ase.units.Bohr 

# pos = thio_3_mer['positions'][-3]
# atom_nums = thio_3_mer['atom_numbers'][-3]
#
# start = time.time()
# atom = []
# for j in range(len(atom_nums)):
#     atom.append((atom_nums[j], pos[j, :])) 
# basis = ''
# mol = gto.M(atom=atom, basis='augccpvdz')
# #print(mol.pack())
# mf = dft.RKS(mol)
# mf.chkfile=False
# mf.xc = 'pbe'
# mf.max_cycle = 1000
# mf.kernel()
# g = mf.nuc_grad_method()
# gradients = g.grad()
#
# print('dft energy', mf.e_tot)
# print('dft gradient', gradients)
#
# disp = d4disp.DFTD4Dispersion(mol, xc='pbe') 
# dkr = disp.kernel()
# print('D4 energy', dkr[0])
# print('D4 gradient', dkr[1])
np.save(save_file, results, allow_pickle=True)

# %%
print(i+1 %1000)

# %%
mf = dft.RKS(mol)
mf.chkfile=False
mf.xc = 'pbe'
mf.max_cycle = 1000
d4mf = d4disp.energy(mf).run()
grad = d4mf.nuc_grad_method()
print('combined gradient', grad.kernel())

# %%
thio_no6 = np.load('datasets/thiophene_not_all_train_pyscf_d4_augccpvdz.npy', allow_pickle=True)
print(len(thio_no6))

# %%
thio_6 = np.load('datasets/thiophene6mer_train_pyscf_d4_augccpvdz.npy', allow_pickle=True)
print(len(thio_6))

# %%
print(thio_6.shape)
print(thio_no6.shape)

# %%
thio = np.concatenate([thio_no6, thio_6], axis=0)
np.save('datasets/thiophene_all_train_pyscf_d4_augccpvdz.npy', thio, allow_pickle=True)

# %%
thio_d4 = np.load('datasets/thiophene_all_test_pyscf_d4_augccpvdz.npy', allow_pickle=True)
thio_poly = np.load('datasets/thiophene_all_test_pyscf_augccpvdz.npy', allow_pickle=True)

# %%
indices = [214]
for i in indices:
    print(i)
    start = time.time()
    calc_dict = thio_d4[i]
    mol = gto.Mole.unpack(calc_dict[0])
    mol.build()

    mf = dft.RKS(mol)
    mf.chkfile = False
    mf.xc = 'pbe'
    mf.max_cycle = 1000
    d4mf = d4disp.energy(mf).run()
    grad = d4mf.nuc_grad_method()
    g = grad.kernel()
    print('d4mf.etot', d4mf.e_tot)
    print('calc en', calc_dict[1]['energy'])
    print('grad to angstrom', -g/ase.units.Bohr)
    print('calc forces', calc_dict[1]['forces'])

# %%
indices = [214, 1523, 2894]
for i in indices:
    print(i)
    start = time.time()
    calc_dict = thio_d4[i]
    mol = gto.Mole.unpack(calc_dict[0])
    mol.build()

    mf = dft.RKS(mol)
    mf.chkfile = False
    mf.xc = 'pbe'
    mf.max_cycle = 1000
    d4mf = d4disp.energy(mf).run()
    grad = d4mf.nuc_grad_method()
    g = grad.kernel()
    print('d4mf.etot', d4mf.e_tot)
    print('calc en', calc_dict[1]['energy'])

    print('- grad to angstrom', -g/ase.units.Bohr)
    print('calc forces', calc_dict[1]['forces'])

    mf.kernel()
    g = mf.nuc_grad_method()
    gradients = g.grad()
    #print(mfs[i].mo_coeff)
    print('mf etot', mf.e_tot)
    print('mf grads', -gradients/ase.units.Bohr)
    print('calc etot', thio_poly[i][1]['energy'])
    print('calc forces', thio_poly[i][1]['forces'])

    disp = d4disp.DFTD4Dispersion(mol, xc='pbe').kernel()
    print('disp en', disp[0])
    print('disp f', -disp[1]/ase.units.Bohr)

# %%
thio_d4 = np.load('datasets/thiophene_all_test_pyscf_d4_augccpvdz.npy', allow_pickle=True)

thiod4 = utils.calc_dict_to_npy(thio_d4, convert_forces=False, compress_atoms=True)
np.save('datasets/thiophene_all_test_d4.npy', thiod4, allow_pickle=True)
# %%
thio_d4 = np.load('datasets/thiophene_all_train_pyscf_d4_augccpvdz.npy', allow_pickle=True)
thiod4 = np.load('datasets/thiophene_all_train_d4.npy', allow_pickle=True).item()

# %%
for i in [100, 1100, 2100, 3100, 4100]:
    print('calc en', thio_d4[i][1]['energy'])
    print('npy en', thiod4['energy'][i])
    print('calc f', thio_d4[i][1]['forces'])
    print('npy f', thiod4['forces'][i])
    print('calc pos', thio_d4[i][0]['atom'])
    print('npy pos', thiod4['positions'][i])
