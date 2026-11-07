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
thio_3_mer = np.load('datasets/thiophene123mer_train.npy', allow_pickle=True).item()
thio3mer = np.load('datasets/thiophene123mer_train_pyscf_augccpvdz.npy', allow_pickle=True)

pos = thio_3_mer['positions'][-3]
atom_nums = thio_3_mer['atom_numbers'][-3]

start = time.time()
atom = []
for j in range(len(atom_nums)):
    atom.append((atom_nums[j], pos[j, :])) 
basis = ''
mol = gto.M(atom=atom, basis='augccpvdz')
#print(mol.pack())
mf = dft.RKS(mol)
mf.chkfile=False
mf.xc = 'pbe'
mf.max_cycle = 1000
mf.kernel()
g = mf.nuc_grad_method()
gradients = g.grad()

print('dft energy', mf.e_tot)
print('dft gradient', gradients)

disp = d4disp.DFTD4Dispersion(mol, xc='pbe') 
dkr = disp.kernel()
print('D4 energy', dkr[0])
print('D4 gradient', dkr[1])

# %%
print('energy sum', mf.e_tot + dkr[0])
print('forces sum', gradients + dkr[1])

# %%

mf = dft.RKS(mol)
mf.chkfile=False
mf.xc = 'pbe'
mf.max_cycle = 1000
d4mf = d4disp.energy(mf).run()
grad = d4mf.nuc_grad_method()
print('combined gradient', grad.kernel())

# %%


