# %%
import os
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling
import equiv_dens.utils.base as utils
from equiv_dens.training.model_loader import load_model
from equiv_dens.utils.misc import generate_id
from datetime import datetime

import numpy as np
from functools import partial
import time
from pyscf.dft import gen_grid, radi
from pyscf import gto
import ase.io
import equiv_dens.utils.cubetools as cubetools

# %load_ext autoreload
# %autoreload 2

# %%
pbe_data = np.load('datasets/fci_xc/PBE/FeS.npy', allow_pickle=True).item()

print(pbe_data.keys())
print(pbe_data['positions'].shape)
dict_data = np.load('datasets/fci_xc/PBE/submit-test/_pyscf_ccpvdz_npy.npy', allow_pickle=True).item()
print(dict_data[0])
# %%
