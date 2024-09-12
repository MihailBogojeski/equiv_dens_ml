#!/usr/bin/env python3
import os
import torch
from datetime import datetime
import wandb
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.utils.misc import generate_id
from equiv_dens.training.errors import ErrorDict
from equiv_dens.training.trainer import Trainer
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.training.lookahead import Lookahead
from equiv_dens.training.model_loader import load_model
from equiv_dens.data.custom_samplers import set_up_data_loader
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling
from equiv_dens.utils import orbitals
from equiv_dens.utils import base as utils
from equiv_dens.training import utils as train_utils
import argparse

import numpy as np
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('args_file', type=str)
parser.add_argument('target', type=str)
parser.add_argument('--is_dir', action='store_true', default=False)
parser.add_argument('--res_dpm', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=1)

main_args = parser.parse_args()


args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)

args, hyperparam_args, test_vars = train_utils.init_training_vars(args, hyperparam_args)
checkpoint = test_vars['checkpoint']
args_dict = vars(args)

print('model code:', test_vars['model_code'])

# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
args.use_gpu = False

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

grid_vars = train_utils.init_grid_vars(args, test=True)
print('grid vars', grid_vars)

required_properties = ['energy', 'forces', 'coords']

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=None,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000,
                           required_properties=required_properties,
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           L0_coeffs_file=args.L0_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_vars['grid_fn'],
                           pyscf_grid=args.pyscf_grid,
                           pyscf_rotate=grid_vars['rotate'],
                           sampling_fn=grid_vars['sampling_fn'],
                           grid_extent=grid_vars['grid_extent'],
                           grid_origin=grid_vars['grid_origin'],
                           verbose=args.verbose,
                           timing=args.timing,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           atom_dens_path=args.atom_dens_path,
                           atom_dens_type=args.atom_dens_type,
                           projected_density=False,
                           density_grad=False,
                           )

model = load_model(args, dataset, train=False)

if args.use_gpu:
    model.cuda()
model.to(args.dtype)

if main_args.is_dir:
    files = os.listdir(main_args.target)
    for file in files:
        file = os.path.join(main_args.target, file)
else:
    files = [main_args.target]

for file in files:
    data_npy = np.load(file, allow_pickle=True).item()
    data_pos = 0
    while data_pos < data_npy['positions'].shape[0]:
        if data_pos + main_args.batch_size > data_npy['positions'].shape[0]:
            max_pos = data_npy['positions'].shape[0]
        else:
            max_pos = data_pos + main_args.batch_size
        batch_npy = {'positions': data_npy['positions'][data_pos:max_pos],
                     'atom_numbers': data_npy['atom_numbers'][data_pos:max_pos]}
        data = orbitals.model_input_from_atoms(batch_npy, use_gpu=args.use_gpu,
                                               density_expansion=True,
                                               pyscf_grid=True,
                                               grid_spec=dataset.grid_spec,
                                               atom_dens_type=args.atom_dens_type,
                                               cutoff=args.cutoff,
                                               grid_sampling_fn=dataset.sampling_fn,
                                               dtype=args.dtype,
                                               free_atom_densities=dataset.atom_dens,
                                               )
        for key in data.keys():
            print('key', key)
            if isinstance(data[key], torch.Tensor):
                print('type', data[key].type())
            else:
                print('type', type(data[key]))
        res = model(data)
        print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
        print('num electrons', orbitals.get_n_electrons(res['atom_numbers']))
        print('dipole_moment', res['dipole_moment'])
        data_pos += main_args.batch_size
