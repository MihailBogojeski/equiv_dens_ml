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
from equiv_dens.utils import hirshfeld_analysis, orbitals, sapt0
import pickle

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
# main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
# main_args.res_load_file = 'datasets/ethanethiol_all_001_coreless_test_results.npy'
# main_args.res_load_file = None
# main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
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
args.dpm_intor = True
args.density_weight = 0

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file

# args.np_dataset_test = "/home/ml-dft/equiv_dens/datasets/qm7x_test_dft_augccpvdz_small_base.npy"
# args.dens_dataset_test = "/home/ml-dft/equiv_dens/datasets/qm7x_test_dft_augccpvdz_small.npy"
# data = np.load('/home/ml-dft/equiv_dens/datasets/qm7x_test_dft_augccpvdz_small.npy', allow_pickle=True)
args.np_dataset_test = "/home/ml-dft/equiv_dens/datasets/s66x8_pyscf_augccpvdz_base.npy"
args.dens_dataset_test = "/home/ml-dft/equiv_dens/datasets/s66x8_pyscf_augccpvdz_calc.npy"
data = np.load('/home/ml-dft/equiv_dens/datasets/s66x8_pyscf_augccpvdz_calc.npy', allow_pickle=True)

print('pyscf_grid', args.pyscf_grid)
dataset = AtomsDensityData(np_path=args.np_dataset, density_path=None,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000000000000,
                           required_properties=['dipole_moment'],
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
                           atom_dens_path='/home/ml-dft/equiv_dens/datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy',
                           atom_dens_type='mo_coeffs',
                           split_atom_dens=True,
                           density_grad=args.density_grad,
                           all_atom_numbers=np.array([1, 6, 7, 8, 16, 17]),
                           dpm_intor=args.dpm_intor,
                           )

print('dataset length', len(dataset))
# print('sample pos shape', dataset.get_properties([0])['positions'].shape)
# print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
print('num samples', main_args.num_samples)
print('args use gpu', args.use_gpu)

torch.cuda.empty_cache()
np.random.seed(42)
model = model_loader.load_model(args, dataset)
for param in model.parameters():
    param.requires_grad = False

if args.use_gpu:
    model.cuda()

"""
Each calculation name is a key for the dicts monomer1 and monomer2. monomer1[key] is an ase.Atoms object of just the first monomer in each calc and monomer2[key] is the corresponding Atoms object for monomer2.
"""
with open("/home/ml-dft/equiv_dens/datasets/opt_monomers.pickle", 'rb') as f:
    opt_monomers1, opt_monomers2 = pickle.load(f)
with open("/home/ml-dft/equiv_dens/datasets/md_monomers.pickle", 'rb') as f:
    md_monomers1, md_monomers2 = pickle.load(f)
monomers_1 = {'opt': opt_monomers1, 'md': md_monomers1}
monomers_2 = {'opt': opt_monomers2, 'md': md_monomers2}
for mode in monomers_1.keys():
    mono1 = monomers_1[mode]
    mono2 = monomers_2[mode]
    sapt0_ml = {}
    calc_num = len(mono1.keys())
    for count, calc_key in enumerate(mono1.keys()):
        print(f'mode {mode}: {count}/{calc_num}')
        sapt0_ml[calc_key] = {}
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

        input12 = {key: np.concatenate([input1[key], input2[key]], axis=1) for key in input1.keys()}

        samp12 = orbitals.model_input_from_atoms(input12,
                                                 density_expansion=False,
                                                 skip_compress=True,
                                                 grid_spec=None,
                                                 cutoff=args.cutoff,
                                                 dtype=torch.float32,
                                                 atom_dens_type="mo_coeffs",
                                                 free_atom_densities=dataset.atom_dens,
                                                 split_atom_densities=False,
                                                 all_atom_coeffs=True,
                                                 coord_params=None,
                                                 valence=False,
                                                 full_valence=False,
                                                 )
        samp1 = orbitals.model_input_from_atoms(input1,
                                                density_expansion=False,
                                                skip_compress=True,
                                                grid_spec=None,
                                                cutoff=args.cutoff,
                                                dtype=torch.float32,
                                                atom_dens_type="mo_coeffs",
                                                free_atom_densities=dataset.atom_dens,
                                                split_atom_densities=False,
                                                all_atom_coeffs=True,
                                                coord_params=None,
                                                valence=False,
                                                full_valence=False,
                                                )
        samp2 = orbitals.model_input_from_atoms(input2,
                                                density_expansion=False,
                                                skip_compress=True,
                                                grid_spec=None,
                                                cutoff=args.cutoff,
                                                dtype=torch.float32,
                                                atom_dens_type="mo_coeffs",
                                                free_atom_densities=dataset.atom_dens,
                                                split_atom_densities=False,
                                                all_atom_coeffs=True,
                                                coord_params=None,
                                                valence=False,
                                                full_valence=False,
                                                )

        if args.use_gpu:
            for key in samp1.keys():
                if isinstance(samp1[key], torch.Tensor):
                    samp1[key] = samp1[key].cuda()
                    samp2[key] = samp2[key].cuda()
                    samp12[key] = samp12[key].cuda()
        res1 = model(samp1)
        res2 = model(samp2)
        print('sum atom numbers', torch.sum(samp12['batch_atom_numbers'], dim=1))

        df_coeffs_ml1 = orbitals.coeffs_dict_to_vector(res1, dataset.orbital_basis_num, res1['batch_atom_numbers'],
                                                        radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

        df_coeffs_ml2 = orbitals.coeffs_dict_to_vector(res2, dataset.orbital_basis_num, res2['batch_atom_numbers'],
                                                        radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

        print('intor res1', orbitals.calculate_1e_intor_ml(res1, dataset.orbital_basis_num, 'int1e_ovlp'))
        print('intor res2', orbitals.calculate_1e_intor_ml(res2, dataset.orbital_basis_num, 'int1e_ovlp'))

        input12 = {key: np.concatenate([input1[key], input2[key]], axis=1) for key in input1.keys()}
        res12 = samp12
        res12['spherical_coeffs'] = res1['spherical_coeffs'] + res2['spherical_coeffs']
        res12['radial_scale'] = res1['radial_scale'] + res2['radial_scale']
        res12['radial_width'] = res1['radial_width'] + res2['radial_width']
        print('intor res12', orbitals.calculate_1e_intor_ml(res12, dataset.orbital_basis_num, 'int1e_ovlp'))

        # sapt0 electrosatic values 
        sapt0_ml_elst, sapt0_ml_efield12, sapt0_ml_efield21, sapt0_ml_ovlp = sapt0.calculate_sapt0_ml(res1, res2, res12, dataset.orbital_basis_num, precalc_basis=False)
        sapt0_ml[calc_key]['elst'] = sapt0_ml_elst
        sapt0_ml[calc_key]['efield12'] = sapt0_ml_efield12
        sapt0_ml[calc_key]['efield21'] = sapt0_ml_efield21
        sapt0_ml[calc_key]['ovlp'] = sapt0_ml_ovlp

        np.save('results/' + main_args.save_file + '_sapt0_ml_' + mode + '_des15k.npy', sapt0_ml, allow_pickle=True)
