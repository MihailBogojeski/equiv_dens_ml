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
from pymbd.fortran import MBDGeom
from vdw import to_mbd
from equiv_dens.utils import hirshfeld_analysis, orbitals

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
print('args use gpu', args.use_gpu)
args.use_gpu = False

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

args.verbose = 0
args.use_gpu = main_args.use_gpu
print('args use gpu', args.use_gpu)
args.cube_grid = False
args.radii_adjust = True
args.expansion_constraint = None
args.integral_constraint = 'coeffs_in_coeff_net'
args.ignore_missing_keywords = True
if args.cube_grid:
    args.cube_origin = -2
    args.cube_extent = 4
    args.cube_size = 50
    args.radii_adjust = False
    grid_origin = args.cube_origin
    grid_extent = np.array([args.cube_extent] * 3)
    grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                      extent=grid_extent,
                      origin=np.array([grid_origin] * 3))
    sampling_fn = cubical_sampling
else:
    args.spherical_grid_level = 1
    grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
    sampling_fn = partial(spherical_radial_sampling, rotate=False)
    grid_origin = 0
    grid_extent = None
    rotate = False

required_properties = ['density', 'dipole_moment']

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
dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000000000000,
                           required_properties=required_properties,
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
                           atom_dens_type='spline',
                           split_atom_dens=True,
                           density_grad=args.density_grad,
                           all_atom_numbers=np.array([1, 6, 7, 8, 16, 17]),
                           )

print('dataset length', len(dataset))
print('sample pos shape', dataset.get_properties([0])['positions'].shape)
print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
print('num samples', main_args.num_samples)
print('args use gpu', args.use_gpu)

# %%
# model = model_loader.load_model(args, dataset)
# idx = [3]
# samp = dataset.get_properties(idx)
# if args.use_gpu:
#     for key in samp.keys():
#         if isinstance(samp[key], torch.Tensor):
#             samp[key] = samp[key].cuda()
#
# res = model(samp)
#
#
# # print('res_radial width', res['radial_width'])
# # print('dataset radial coeffs', dataset.radial_coeffs)
# print('sum atom numbers', torch.sum(samp['batch_atom_numbers'], dim=1))
# print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
# print('true density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
# print('density error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1))
# %%
# mol = dataset.mols[idx[0]]
# disp = d4disp.DFTD4Dispersion(mol, xc='pbe')
# dkr = disp.kernel()
# print('D4 energy', utils.hartree_to_kcal(dkr[0]))
# mf = to_mbd(dft.RKS(mol, xc="PBE"), variant="rsscs").run()
# print(utils.hartree_to_kcal(mf.e_vdw))
# # %%
# wA, atomic_charges, dipoles, volume_ratio = hirshfeld_analysis.hirshfeld_partitioning(samp['density'],
#                                                                                       samp['atom_density_split'],
#                                                                                       samp['batch_positions'], samp['batch_atom_numbers'],
#                                                                                       samp['coords'], samp['coord_weights'],
#                                                                                       to_bohr=True)
# pos = utils.angstrom_to_bohr(samp['batch_positions'][0].numpy(force=True))
# nums = samp['batch_atom_numbers'][0].numpy(force=True)
# print('v ratio', volume_ratio[0])
# energy = MBDGeom(pos).mbd_energy_species(utils.numbers_to_symbols(nums), volume_ratio[0].numpy(force=True), 0.83)
# print('MBD energy', utils.hartree_to_kcal(energy))
# # %%
# wA, atomic_charges, dipoles, volume_ratio = hirshfeld_analysis.hirshfeld_partitioning(res['density'],
#                                                                                       samp['atom_density_split'],
#                                                                                       samp['batch_positions'], samp['batch_atom_numbers'],
#                                                                                       samp['coords'], samp['coord_weights'],
#                                                                                       to_bohr=True)
# print('v ratio', volume_ratio[0])
# energy = MBDGeom(pos).mbd_energy_species(utils.numbers_to_symbols(nums), volume_ratio[0].numpy(force=True), 0.83)
# print('MBD energy', utils.hartree_to_kcal(energy))
# %%
torch.cuda.empty_cache()
np.random.seed(42)
model = model_loader.load_model(args, dataset)
true_mbd_dict = {'atomic_charges': [], 'dipoles': [], 'volume_ratio': [], 'mbd_energy': []}
ml_mbd_dict = {'atomic_charges': [], 'dipoles': [], 'volume_ratio': [], 'mbd_energy': []}
density_err = []
dpm_err = []
fname_dict = {}

volumes = orbitals.free_atom_volumes(dataset.atom_dens, 'spline',
                                     grid_spec=dataset.grid_spec,
                                     to_bohr=False)
print(volumes)
# for i in range(5):
print('len dataset', len(dataset))
for i in range(len(dataset)):
        idx = i
        print('idx', idx)
        print('filename', data[idx][0]['filename'])
        samp = dataset.get_properties([idx])
        print('sum atom numbers', torch.sum(samp['batch_atom_numbers'], dim=1))
        if args.use_gpu:
            for key in samp.keys():
                if isinstance(samp[key], torch.Tensor):
                    samp[key] = samp[key].cuda()

        # print('samp atom numbers', samp['batch_atom_numbers'])
        # print('samp num atoms', torch.sum(samp['batch_atom_numbers'] > 0, dim=1))
        # print('samp num electrons', torch.sum(samp['batch_atom_numbers'], dim=1))
        res = model(samp)

        dpm_samp = orbitals.calc_dipole_moment(samp, normalize_density=False)['dipole_moment']
        dpm_res = orbitals.calc_dipole_moment(res, normalize_density=False)['dipole_moment']

        dpm_err.append(utils.internal_to_debye(torch.norm(dpm_samp - dpm_res, dim=1).numpy(force=True)))


# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
        dens_err = torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1)
        print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
        print('true density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
        print('density error', dens_err)
        print('dpm error', dpm_err[i])
        density_err.append(dens_err.numpy(force=True))
        wA, atomic_charges, dipoles, volume_ratio = hirshfeld_analysis.hirshfeld_partitioning(samp['density'],
                                                                                            samp['atom_density_split'],
                                                                                            samp['batch_positions'], samp['batch_atom_numbers'],
                                                                                            samp['coords'], samp['coord_weights'],
                                                                                            to_bohr=True)
        pos = utils.angstrom_to_bohr(samp['batch_positions'][0].numpy(force=True))
        nums = samp['batch_atom_numbers'][0].numpy(force=True)
        print('v ratio true', volume_ratio[0])
        energy = MBDGeom(pos).mbd_energy_species(utils.numbers_to_symbols(nums), volume_ratio[0].numpy(force=True), 0.83)
        true_mbd_dict['atomic_charges'].append(atomic_charges.numpy(force=True))
        true_mbd_dict['dipoles'].append(dipoles.numpy(force=True))
        true_mbd_dict['volume_ratio'].append(volume_ratio.numpy(force=True))
        true_mbd_dict['mbd_energy'].append(utils.hartree_to_kcal(energy))
        print('MBD energy', utils.hartree_to_kcal(energy))
        if 'filename' in data[idx][0].keys():
            fname = data[idx][0]['filename']
            fname_dict[fname] = {}
            fname_dict[fname]['true_atomic_charges'] = atomic_charges.numpy(force=True)
            fname_dict[fname]['true_dipoles'] = dipoles.numpy(force=True)
            fname_dict[fname]['true_volume_ratio'] = volume_ratio.numpy(force=True)
            fname_dict[fname]['true_mbd_energy'] = utils.hartree_to_kcal(energy)

        if main_args.expansion_volumes:
            print('calculating volumes based on expansion')
            volume_ratio, eff_volumes = hirshfeld_analysis.volume_ratios_from_expansion(res, model.property_models['density'],
                                                                                         free_atom_volumes=volumes,
                                                                                         removed_free_atom=args.remove_atom_density)
            dipoles = orbitals.get_atomic_dipoles(res, model.property_models['density'], to_bohr=False)
            atomic_charges = orbitals.get_density_charges(res, removed_free_atom=args.remove_atom_density)
        else:
            print('calculating volumes based on hirshfeld partitioning')
            wA, atomic_charges, dipoles, volume_ratio = hirshfeld_analysis.hirshfeld_partitioning(res['density'],
                                                                                                samp['atom_density_split'],
                                                                                                samp['batch_positions'], samp['batch_atom_numbers'],
                                                                                                samp['coords'], samp['coord_weights'],
                                                                                                to_bohr=True)
        print('v ratio ml', volume_ratio[0])
        energy = MBDGeom(pos).mbd_energy_species(utils.numbers_to_symbols(nums), volume_ratio[0].numpy(force=True), 0.83)
        print('MBD ML energy', utils.hartree_to_kcal(energy))
        ml_mbd_dict['atomic_charges'].append(atomic_charges.numpy(force=True))
        ml_mbd_dict['dipoles'].append(dipoles.numpy(force=True))
        ml_mbd_dict['volume_ratio'].append(volume_ratio.numpy(force=True))
        ml_mbd_dict['mbd_energy'].append(utils.hartree_to_kcal(energy))
        allocated_memory = torch.cuda.memory_allocated()
        print(f"Memory allocated: {allocated_memory / (1024**2):.2f} MB")
        if 'filename' in data[idx][0].keys():
            fname = data[idx][0]['filename']
            fname_dict[fname]['ml_atomic_charges'] = atomic_charges.numpy(force=True)
            fname_dict[fname]['ml_dipoles'] = dipoles.numpy(force=True)
            fname_dict[fname]['ml_volume_ratio'] = volume_ratio.numpy(force=True)
            fname_dict[fname]['ml_mbd_energy'] = utils.hartree_to_kcal(energy)
            fname_dict[fname]['dens_err'] = density_err[-1]
            fname_dict[fname]['dpm_err'] = dpm_err[-1]

# %%
# np.save('results/true_mbd_dict.npy', true_mbd_dict, allow_pickle=True)
# np.save('results/ml_mbd_dict.npy', ml_mbd_dict, allow_pickle=True)
# if len(fname_dict.keys()) > 0:
#     np.save('results/mbd_fname_dict.npy', fname_dict, allow_pickle=True)

for key in ml_mbd_dict.keys():
    error = 0
    # print('key', key)
    for i in range(len(ml_mbd_dict[key])):
        if key == 'dipole':
            ml_val = np.array(ml_mbd_dict[key][i])
            true_val = np.array(true_mbd_dict[key][i])
            error += utils.au_to_debye(np.mean(np.norm(true_val - ml_val)))
            # if key == 'mbd_energy':
            #     print('ml val', ml_val)
            #     print('true val', true_val)
            #     print('error', error)
        else:
            ml_val = np.array(ml_mbd_dict[key][i])
            true_val = np.array(true_mbd_dict[key][i])
            error += np.mean(np.abs(true_val - ml_val))
            # if key == 'mbd_energy':
            #     print('ml val', ml_val)
            #     print('true val', true_val)
            #     print('error', error)
    print(key, 'MAE', error / len(ml_mbd_dict[key]))

# print('atomic charges mae', np.mean(np.abs(true_mbd_dict['atomic_charges'] - ml_mbd_dict['atomic_charges'])))
# print('dipoles mae', utils.internal_to_debye(np.mean(np.norm(true_mbd_dict['dipoles'] - ml_mbd_dict['dipoles'], axis=-1))))
# print('volume ratio mae', np.mean(np.abs(true_mbd_dict['volume_ratio'] - ml_mbd_dict['volume_ratio'])))
# print('mbd energy mae', np.mean(np.abs(true_mbd_dict['mbd_energy'] - ml_mbd_dict['mbd_energy'])))
print('density error', np.mean(density_err))
print('dpm error', np.mean(dpm_err))
# %%
np.save('results/true_mbd_dict_2.npy', true_mbd_dict, allow_pickle=True)
np.save('results/ml_mbd_dict_2.npy', ml_mbd_dict, allow_pickle=True)
if len(fname_dict.keys()) > 0:
    np.save('results/mbd_fname_dict_2.npy', fname_dict, allow_pickle=True)

