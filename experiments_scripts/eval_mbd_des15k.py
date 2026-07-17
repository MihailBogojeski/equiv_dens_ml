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
from vdw import to_mbd
from equiv_dens.utils import hirshfeld_analysis, orbitals
import pickle
import socket

_hostname = socket.gethostname()
DATA_ROOT = '/home/ml-dft/equiv_dens/datasets' if (_hostname == 'hydra' or (_hostname.startswith('head') and _hostname[4:].isdigit())) else 'datasets'

hf.MUTE_CHKFILE = True
# %%
main_args = Namespace()

# main_args.args_file = "args/resorcinol_all_001.txt"
# main_args.args_file = "args/ethanethiol_all_006_test.txt"
# main_args.args_file = "args/ethanethiol_all_106_test.txt"
# main_args.args_file = "args/h2o_small_all_001.txt"
# main_args.args_file = "args/ethanethiol_df_coeffs_001_test.txt"
# main_args.args_file = "args/ethanethiol_all_001_coreless_test.txt"
# main_args.args_file = "args/ethanethiol_all_004_coreless.txt"
# main_args.args_file = "args/qm7x250_dens_001_coreless.txt"
main_args.args_file = "args/qm7x250_dens_001_coreless.txt"
# main_args.args_file = "args/ethanethiol_all_001_SH_even.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
# main_args.res_load_file = f'{DATA_ROOT}/ethanethiol_all_006_test.pt'
# main_args.res_load_file = f'{DATA_ROOT}/ethanethiol_all_001_coreless_test_results.npy'
# main_args.res_load_file = None
# main_args.res_load_file = f'{DATA_ROOT}/resorcinol_all_005_test.pt'
# main_args.save_file = 'ethanethiol_all_006'
# main_args.save_file = 'ethanethiol_all_106'
# main_args.save_file = 'h2o_small_all_001'
# main_args.save_file = 'resorcinol_all_005'
# main_args.save_file = 'ethanethiol_all_001_coreless'
main_args.save_file = 'qm7x250_dens_001_coreless'
# main_args.save_file = 'ethanethiol_all_001_SH_even'
# main_args.save_file = 'ethanethiol_df_coeffs_001'
main_args.df_error = True
main_args.use_gpu = True 
main_args.num_samples = 100
main_args.make_plots = True
main_args.expansion_volumes = False 

df_losses = None

# %%
args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)

# print('type dtype', type(args.dtype))
args.fix_arguments = True
# print('args np dir', args.np_dataset)
# args.restart = None
# args.pred_radial_coeffs = False

args, hyperparam_args, train_vars = train_utils.init_training_vars(args, hyperparam_args)
checkpoint = train_vars['checkpoint']

# determine whether GPU is used for training

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

args.verbose = 0
args.use_gpu = main_args.use_gpu
print('args use gpu', args.use_gpu)
args.cube_grid = False
args.radii_adjust = True
args.expansion_constraint = None
args.integral_constraint = 'coeffs_in_coeffs_net'
args.ignore_missing_keywords = True
args.spherical_grid_level = 1
grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
sampling_fn = partial(spherical_radial_sampling, rotate=False)
grid_origin = 0
grid_extent = None
rotate = False

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file

# args.np_dataset_test = f'{DATA_ROOT}/qm7x_test_dft_augccpvdz_small_base.npy'
# args.dens_dataset_test = f'{DATA_ROOT}/qm7x_test_dft_augccpvdz_small.npy'
# data = np.load(f'{DATA_ROOT}/qm7x_test_dft_augccpvdz_small.npy', allow_pickle=True)
args.np_dataset_test = f'{DATA_ROOT}/s66x8_pyscf_augccpvdz_base.npy'
args.dens_dataset_test = f'{DATA_ROOT}/s66x8_pyscf_augccpvdz_calc.npy'
data = np.load(f'{DATA_ROOT}/s66x8_pyscf_augccpvdz_calc.npy', allow_pickle=True)

print('pyscf_grid', args.pyscf_grid)
dataset = AtomsDensityData(np_path=args.np_dataset, density_path=None,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000000000000,
                           required_properties=[],
                           center_positions=True,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           pyscf_grid=args.pyscf_grid,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
                           radii_adjust=args.radii_adjust,
                           calc_data=True,
                           atom_dens_path=f'{DATA_ROOT}/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy',
                           atom_dens_type='mo_coeffs',
                           split_atom_dens=True,
                           density_grad=args.density_grad,
                           all_atom_numbers=np.array([1, 6, 7, 8, 16, 17]),
                           )

print('dataset length', len(dataset))
print('sample pos shape', dataset.get_properties([0])['positions'].shape)
# print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
print('num samples', main_args.num_samples)
print('args use gpu', args.use_gpu)

torch.cuda.empty_cache()
np.random.seed(42)
model = model_loader.load_model(args, dataset)
ml_mbd_dict_1 = {}
ml_mbd_dict_2 = {}
if args.use_gpu:
    model.cuda()

volumes = orbitals.free_atom_volumes(dataset.atom_dens, 'spline',
                                     grid_spec=dataset.grid_spec,
                                     to_bohr=False)
print(volumes)
exit()
# print('len dataset', len(dataset))
# for i in range(len(dataset)):
with open(f'{DATA_ROOT}/opt_monomers.pickle', 'rb') as f:
    opt_monomers1, opt_monomers2 = pickle.load(f)
with open(f'{DATA_ROOT}/md_monomers.pickle', 'rb') as f:
    md_monomers1, md_monomers2 = pickle.load(f)

monomers_1 = {'opt': opt_monomers1, 'md': md_monomers1}
monomers_2 = {'opt': opt_monomers2, 'md': md_monomers2}
for mode in monomers_1.keys():
    mono1 = monomers_1[mode]
    mono2 = monomers_2[mode]
    ovlps_ml = {}
    calc_num = len(mono1.keys())
    for i, calc_key in enumerate(mono1.keys()):
        print(f'mode {mode}: {i}/{calc_num}')
        # if i > 5:
        #     break
        ovlps_ml[calc_key] = []
        pos1 = []
        z1 = []
        pos2 = []
        z2 = []

        p1 = mono1[calc_key].get_positions()
        a1 = mono1[calc_key].get_atomic_numbers()
        p2 = mono2[calc_key].get_positions()
        a2 = mono2[calc_key].get_atomic_numbers()
        pos1.append(p1)
        pos2.append(p2)
        z1.append(a1)
        z2.append(a2)

        pos1 = np.stack(pos1, axis=0)
        pos2 = np.stack(pos2, axis=0)
        z1 = np.stack(z1, axis=0)
        z2 = np.stack(z2, axis=0)
        input1 = {'atom_numbers': z1, 'positions': pos1}
        input2 = {'atom_numbers': z2, 'positions': pos2}
        coord_params = None

        samp1 = orbitals.model_input_from_atoms(input1,
                                                density_expansion=True,
                                                skip_compress=True,
                                                grid_spec=dataset.grid_spec,
                                                cutoff=args.cutoff,
                                                dtype=torch.float32,
                                                atom_dens_type="mo_coeffs",
                                                free_atom_densities=dataset.atom_dens,
                                                split_atom_densities=True,
                                                basis=None,
                                                all_atom_coeffs=False,
                                                coord_params=None,
                                                valence=False,
                                                full_valence=False,
                                                )
        samp2 = orbitals.model_input_from_atoms(input2,
                                                density_expansion=True,
                                                skip_compress=True,
                                                grid_spec=dataset.grid_spec,
                                                cutoff=args.cutoff,
                                                dtype=torch.float32,
                                                atom_dens_type="mo_coeffs",
                                                free_atom_densities=dataset.atom_dens,
                                                split_atom_densities=True,
                                                basis=None,
                                                all_atom_coeffs=False,
                                                coord_params=None,
                                                valence=False,
                                                full_valence=False,
                                                )
        if args.use_gpu:
            for key in samp1.keys():
                if isinstance(samp1[key], torch.Tensor):
                    samp1[key] = samp1[key].cuda()
                    samp2[key] = samp2[key].cuda()

        res1 = model(samp1)
        res2 = model(samp2)

        wA1, atomic_charges1, dipoles1, volume_ratio1, r3_volume1, r3_volume_free1 = hirshfeld_analysis.hirshfeld_partitioning(
                res1['density'],
                samp1['atom_density_split'],
                samp1['batch_positions'], samp1['batch_atom_numbers'],
                samp1['coords'], samp1['coord_weights'],
                to_bohr=True)
        wA2, atomic_charges2, dipoles2, volume_ratio2, r3_volume2, r3_volume_free2 = hirshfeld_analysis.hirshfeld_partitioning(
                res2['density'],
                samp2['atom_density_split'],
                samp2['batch_positions'], samp2['batch_atom_numbers'],
                samp2['coords'], samp2['coord_weights'],
                to_bohr=True)

        ml_mbd_dict_1[calc_key] = {'atom_numbers': [], 'positions': [], 'hirshfeld_charges': [], 'volume_ratio': []}
        ml_mbd_dict_1[calc_key]['atom_numbers'].append(res1['batch_atom_numbers'].numpy(force=True))
        ml_mbd_dict_1[calc_key]['positions'].append(res1['batch_positions'].numpy(force=True))
        ml_mbd_dict_1[calc_key]['hirshfeld_charges'].append(atomic_charges1.numpy(force=True))
        ml_mbd_dict_1[calc_key]['volume_ratio'].append(volume_ratio1.numpy(force=True))
        ml_mbd_dict_2[calc_key] = {'atom_numbers': [], 'positions': [], 'hirshfeld_charges': [], 'volume_ratio': []}
        ml_mbd_dict_2[calc_key]['atom_numbers'].append(res2['batch_atom_numbers'].numpy(force=True))
        ml_mbd_dict_2[calc_key]['positions'].append(res2['batch_positions'].numpy(force=True))
        ml_mbd_dict_2[calc_key]['hirshfeld_charges'].append(atomic_charges2.numpy(force=True))
        ml_mbd_dict_2[calc_key]['volume_ratio'].append(volume_ratio2.numpy(force=True))
        allocated_memory = torch.cuda.memory_allocated()
        # print('n_electrons1', torch.sum(samp1['batch_atom_numbers']))
        # print('dens int 1', torch.sum(res1['density'] * res1['coord_weights']) - torch.sum(samp1['batch_atom_numbers']))
        # print('n_electrons2', torch.sum(samp2['batch_atom_numbers']))
        # print('dens int 2', torch.sum(res2['density'] * res2['coord_weights']) - torch.sum(samp2['batch_atom_numbers']))

        np.save('results/ml_mbd_dict_1_' + mode + '_des15k.npy', ml_mbd_dict_1, allow_pickle=True)
        np.save('results/ml_mbd_dict_2_' + mode + '_des15k.npy', ml_mbd_dict_2, allow_pickle=True)

    np.save('results/ml_mbd_dict_1_' + mode + '_des15k.npy', ml_mbd_dict_1, allow_pickle=True)
    np.save('results/ml_mbd_dict_2_' + mode + '_des15k.npy', ml_mbd_dict_2, allow_pickle=True)

