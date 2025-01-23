#!/usr/bin/env python3
import os
import torch
from datetime import datetime
import time
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
parser.add_argument('--dpm_intor', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=1)

main_args = parser.parse_args()


args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)
print('args use gpu', args.use_gpu)

args.timing = True
args.energy_weight = 0
args.forces_weight = 0
args.density_weight = 0
args.dipole_moment_weight = 1
args.dpm_intor = main_args.dpm_intor
if args.dpm_intor:
    args.integral_constraint = 'coeffs_in_coeffs_net'
# args.integral_constraint = None 
args, hyperparam_args, test_vars = train_utils.init_training_vars(args, hyperparam_args)
checkpoint = test_vars['checkpoint']
args_dict = vars(args)

print('model code:', test_vars['model_code'])

# determine whether GPU is used for training
print('args use gpu', args.use_gpu)

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

grid_vars = train_utils.init_grid_vars(args, test=True)
print('grid vars', grid_vars)

required_properties = ['dipole_moment']
args.dens_dataset = None

required_properties = ['dipole_moment']
# args.np_dataset = "datasets/ethanethiol_md_traj_every1000_dft_augccpvdz.npy"
# args.dens_dataset = "datasets/ethanethiol_md_traj_every1000_dft_augccpvdz_df_augccpvqzjkfit.npy"
# args.np_dataset = '/home/ml-dft/equiv_dens/datasets/8mer_all-every20_pyscf_d4_augccpvdz_npy.npy'
# args.dens_dataset = '/home/ml-dft/equiv_dens/datasets/8mer_all-every20_pyscf_d4_augccpvdz.npy'
print('pyscf grid', args.pyscf_grid)
dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000,
                           required_properties=required_properties,
                           center_positions=True,
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
                           dpm_intor=args.dpm_intor,
                           )

model = load_model(args, dataset, train=False)

model.eval()
if args.use_gpu:
    model.cuda()
print('use gpu', args.use_gpu)
model.to(args.dtype)
for param in model.parameters():
    param.requires_grad = False

is_dir = os.path.isdir(main_args.target)

if is_dir:
    files = os.listdir(main_args.target)
    print(files)
    for i in range(len(files)):
        files[i] = os.path.join(main_args.target, files[i])
    print(files)
else:
    files = [main_args.target]

for file in files:
    dpm_errors = []
    dir = '/'.join(file.split('/')[:-1])
    fname = file.split('/')[-1]
    if args.dpm_intor:
        suffix = '_dpm_intor.npy'
    else:
        suffix = '_dpm.npy'

    ref_file_name = fname[:-4] + '_dm.txt'
    fname = fname[:-4] + suffix
    print('fname', fname)
    print('filetype', file[-3:])
    with open(os.path.join(dir, ref_file_name), 'r') as f:
        ref_file_lines = f.readlines()
    out_exists = os.path.exists(os.path.join('results', fname))
    if 'npy' != file[-3:] or out_exists:
        print('skipping file')
        continue
    data_npy = np.load(file, allow_pickle=True).item()
    if data_npy['atom_numbers'].ndim == 1:
        data_npy['atom_numbers'] = np.tile(data_npy['atom_numbers'][None, :],
                                           (data_npy['positions'].shape[0], 1))
    data_pos = 0
    data_npy['dipole_moment'] = None
    print('data len', data_npy['positions'].shape)

    count = 0
    while data_pos < data_npy['positions'].shape[0]:
        count += 1
        # if count > 10:
        #     break
        if data_pos + main_args.batch_size > data_npy['positions'].shape[0]:
            max_pos = data_npy['positions'].shape[0]
        else:
            max_pos = data_pos + main_args.batch_size

        batch_npy = {'positions': data_npy['positions'][data_pos:max_pos],
                     'atom_numbers': data_npy['atom_numbers'][data_pos:max_pos]}
        start = time.time()
        data = orbitals.model_input_from_atoms(batch_npy,
                                               density_expansion=(not args.dpm_intor),
                                               pyscf_grid=True,
                                               grid_spec=dataset.grid_spec,
                                               atom_dens_type=args.atom_dens_type,
                                               cutoff=args.cutoff,
                                               grid_sampling_fn=dataset.sampling_fn,
                                               dtype=args.dtype,
                                               free_atom_densities=dataset.atom_dens,
                                               )
        for key in data.keys():
            if isinstance(data[key], torch.Tensor) and args.use_gpu:
                data[key] = data[key].cuda()
        if args.timing:
            print('data from npy time', time.time() - start)
        res = model(data)
        # print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
        # print('center positions', torch.mean(res['positions'],))
        # print('num electrons', orbitals.get_n_electrons(res['atom_numbers']))
        # print('dipole_moment', res['dipole_moment'])
        # print('dipole magnitude', torch.norm(res['dipole_moment'], dim=-1))
        np_dpm = utils.internal_to_debye(res['dipole_moment'].numpy(force=True))
        print('dipole_moment converted', np_dpm)
        dipole_ref = [ref_file_lines[i].split(' ') for i in range(data_pos, max_pos)]
        dipole_ref = np.array(dipole_ref).astype(float)
        print('dipole ref', dipole_ref)
        print('dipole error', np.linalg.norm(dipole_ref - np_dpm, axis=-1))
        dpm_errors.append(np.linalg.norm(dipole_ref - np_dpm, axis=-1))

        if data_npy['dipole_moment'] is None:
            data_npy['dipole_moment'] = np_dpm
        else:
            data_npy['dipole_moment'] = np.concatenate([data_npy['dipole_moment'], np_dpm], axis=0)

        # samp = dataset.get_properties(np.arange(data_pos, max_pos))
        # for key in samp.keys():
        #     if isinstance(samp[key], torch.Tensor) and args.use_gpu:
        #         samp[key] = samp[key].cuda()
        # print('samp dipole_moment', samp['dipole_moment'])
        # dpm_err = utils.internal_to_debye(torch.norm(samp['dipole_moment'] - res['dipole_moment'], dim=-1)).numpy(force=True)
        # print('dpm_errors', dpm_err)
        # if dpm_errors is None:
        #     dpm_errors = dpm_err
        # else:
        #     dpm_errors = np.concatenate([dpm_errors, dpm_err], axis=0)

        data_pos += main_args.batch_size
        allocated_memory = torch.cuda.memory_allocated()
        print(f"Memory allocated: {allocated_memory / (1024**2):.2f} MB")
        res = None
    np.save(os.path.join('results', fname), data_npy, allow_pickle=True)
    print('average dpm error', np.mean(np.concatenate(dpm_errors, axis=-1)))

    # np.save(os.path.join('results', 'dpm_errors_' + fname), dpm_errors, allow_pickle=True)
    # print('mean dpm error', np.mean(dpm_errors))
