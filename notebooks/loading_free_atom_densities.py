# %%
from pyscf import gto, dft, df, lib
from pyscf.scf import hf
from pyscf.lib import param
import scipy

import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling, \
     CubicalGrid, spherical_grid, spherical_radial_sampling
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from equiv_dens.utils.misc import generate_id

from functools import partial
from argparse import Namespace
from equiv_dens.training import density_errors
import matplotlib.pyplot as plt
import numpy as np
from equiv_dens.training import model_loader
from equiv_dens.utils.hirshfeld_analysis import get_atm_nrks, free_atom_spline,\
    eval_spline_density, hirshfeld_partitioning
from equiv_dens.utils.grids import spherical_grid
from pyscf.dft import gen_grid, radi
import os

# %load_ext autoreload
# %autoreload 2
# %cd /home/mihail/Documents/workspace/equiv_dens/
