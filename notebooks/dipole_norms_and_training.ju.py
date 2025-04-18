# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..
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
from pyscf.dft import gen_grid, radi, numint
from pyscf.lib import param
from equiv_dens.training import utils as train_utils
from argparse import Namespace
from equiv_dens.training import model_loader
from pymbd.fortran import MBDGeom
from vdw import to_mbd
from equiv_dens.utils import hirshfeld_analysis, orbitals
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

hf.MUTE_CHKFILE = True
# %%
main_args = Namespace()

# main_args.args_file = "args/resorcinol_all_001.txt"
# main_args.args_file = "args/ethanethiol_all_006_test.txt"
# main_args.args_file = "args/ethanethiol_all_106_test.txt"
# main_args.args_file = "args/h2o_small_all_001.txt"
# main_args.args_file = "args/ethanethiol_df_coeffs_001_test.txt"
# main_args.args_file = "args/ethanethiol_all_001_coreless_test.txt"
main_args.args_file = "args/ethanethiol_all_010_coreless_mo_noninteracting.txt"
# main_args.args_file = "args/ethanethiol_all_004_coreless.txt"
# main_args.args_file = "args/qm7x250_dens_001_coreless.txt"
# main_args.args_file = "args/qm7x250_dens_001_coreless.txt"
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
main_args.save_file = 'ethanethiol_all_010_coreless_mo_noninteracting.txt'
# main_args.save_file = 'qm7x250_dens_001_coreless'
# main_args.save_file = 'ethanethiol_all_001_SH_even'
# main_args.save_file = 'ethanethiol_df_coeffs_001'
main_args.df_error = True
main_args.use_gpu = False
main_args.num_samples = 100
main_args.make_plots = True

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
args.integral_constraint = 'coeffs_in_coeffs_net'
# args.integral_constraint = None
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

required_properties = ['density', 'dipole_moment', 'mo_coeff', 'df_coeffs']

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file

# args.np_dataset_test = "datasets/qm7x_test_dft_augccpvdz_small_base.npy"
# args.dens_dataset_test = "datasets/qm7x_test_dft_augccpvdz_small.npy"
# args.np_dataset_test = "datasets/s66x8_pyscf_augccpvdz_base.npy"
# args.dens_dataset_test = "datasets/s66x8_pyscf_augccpvdz_calc.npy"

dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000000000000,
                           required_properties=required_properties,
                           center_positions=True,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           pyscf_grid=True,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
                           radii_adjust=args.radii_adjust,
                           calc_data=True,
                           atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy',
                           atom_dens_type='spline',
                           split_atom_dens=True,
                           density_grad=args.density_grad,
                           calc_basis_path=args.calc_basis_file,
                           # all_atom_numbers=np.array([1, 6, 7, 8, 16, 17]),
                           )

print('dataset length', len(dataset))
print('sample pos shape', dataset.get_properties([0])['positions'].shape)
print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
print('num samples', main_args.num_samples)
print('args use gpu', args.use_gpu)

# %%
# define DF dataset
df_losses = None
if main_args.df_error:
    # df_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
    #              'dpm_rmse': [], 'kl_loss': [],
    #              'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
    #              'coulomb': [], 'coulomb_int': [], 'mae_23': [], 'mae_43': [],
    #              'lda_23_mae': [], 'dpm_coord_rmse': [],
    #              'dpm_pos_coord_rmse': [], 'dpm_neg_coord_rmse': [],
    #              'dpm_int_rmse': []}

    dataset_df = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
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
                                  projected_density=True,
                                  radii_adjust=args.radii_adjust,
                                  # all_atom_numbers=np.array([1, 6, 7, 8, 16, 17]),
                                  )

# %%
test_indices = checkpoint['data_split_indices']['test']
print(test_indices)

# %%
# test model on a random sample
print('args integral constraint', args.integral_constraint)
model = model_loader.load_model(args, dataset)
for param in model.parameters():
    param.requires_grad = False
idx = [3]
samp = dataset.get_properties(idx)
if args.use_gpu:
    for key in samp.keys():
        if isinstance(samp[key], torch.Tensor):
            samp[key] = samp[key].cuda()

res = model(samp)


# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
print('sum atom numbers', torch.sum(samp['batch_atom_numbers'], dim=1))
print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
print('true density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
print('density error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1))
# %%
# test out different ways to calculate dipole moment
dens_int = orbitals.calculate_1e_intor_ml(res, dataset.orbital_basis_num, 'int1e_r')
print('dpm pred analytic', utils.au_to_debye(dens_int[0]))

dpm = orbitals.calc_dipole_moment_analytic(res, dataset.orbital_basis_num, 'ml_coeffs')['dipole_moment']
print('dpm pred analytic', utils.internal_to_debye(dpm))

dpm = orbitals.calc_dipole_moment(res)['dipole_moment']
print('dpm pred int', utils.internal_to_debye(dpm))

dpm = orbitals.calc_dipole_moment_analytic(res, 'augccpvdz', 'mo_coeffs')['dipole_moment']
print('dpm true analytic', utils.internal_to_debye(dpm))

dpm = orbitals.calc_dipole_moment(samp)['dipole_moment']
print('dpm true int', utils.internal_to_debye(dpm))

# %%
# get distribution of order 1 basis function magnitudes
def get_ord_1_mag_distribution(atoms):
    mag_per_type = {at.item(): [] for at in torch.unique(atoms['atom_numbers'])}
    for i, atom_sph in enumerate(atoms['spherical_coeffs']):
        for key in atom_sph:
            (z, L) = key
            if z == 0 or L != 1:
                continue
            width = atoms['radial_width'][i][key]
            scale = atoms['radial_scale'][i][key]
            sph_coeff = atoms['spherical_coeffs'][i][key]
            norms = 1 / orbitals.gto_norm(0, width) / orbitals.pyscf_gto_factor
            sph_norm = torch.norm(sph_coeff, dim=-2, keepdim=True)
            orb_norms = torch.sum(sph_norm * scale * norms, dim=-2).flatten()
            mag_per_type[z].append(orb_norms)
    for key in mag_per_type.keys():
        mag_per_type[key] = torch.cat(mag_per_type[key])
    return mag_per_type

print(get_ord_1_mag_distribution(res))
# %%
samp_df = dataset_df.get_properties(idx)

dpm = orbitals.calc_dipole_moment_analytic(samp_df, 'augccpvqzjkfit', 'df_coeffs')['dipole_moment']
print('df coeffs analytic', utils.internal_to_debye(dpm))

dpm = orbitals.calc_dipole_moment(samp_df)['dipole_moment']
print('df coeffs int', utils.internal_to_debye(dpm))
# %%
# calculate order one basis functions distribution for DF coeffs
samp_df_sph = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']}, dataset.orbital_basis_num,
                                             samp_df['batch_atom_numbers'], radial_basis=dataset.radial_coeffs)

samp_df.update(samp_df_sph)
dens_int = orbitals.calculate_1e_intor_ml(samp_df, dataset.orbital_basis_num, 'int1e_r')
print('dpm df analytic', utils.au_to_debye(dens_int[0]))

print(get_ord_1_mag_distribution(samp_df))
# %%
# # calculate order one basis functions distribution for ML-DF coeffs
#
# ml_df_coeffs, ml_df_auxmol = orbitals.ml_basis_to_df_coeffs(res, 'augccpvdz',
#                                            mo_coeff=torch.from_numpy(dataset.coeffs[3]['mo_coeff']),
#                                            mo_occ=torch.from_numpy(dataset.coeffs[3]['mo_occ']),)
#
#
# ml_df_sph = orbitals.vector_to_coeffs_dict({'spherical_coeffs': ml_df_coeffs}, dataset.orbital_basis_num,
#                                              samp_df['batch_atom_numbers'], radial_coeffs=False, radial_basis=None,
#                                            convert_to_equiv_dens=True)
#
# print('ml_df_sph', ml_df_sph.keys())
#
# ml_df_res = dict(res)
# ml_df_res.update(ml_df_sph)
# dens_int = orbitals.calculate_1e_intor_ml(ml_df_res, dataset.orbital_basis_num, 'int1e_r')
# print('dpm ml-df analytic', utils.au_to_debye(dens_int[0]))
# %%
def plot_L1_distributions(magnitude_dict, axes=None, label=None):
    # If no axes are provided, create a figure and subplots
    if axes is None:
        fig, axes = plt.subplots(len(magnitude_dict.keys()), 1, figsize=(8, len(magnitude_dict.keys()) * 4))
        axes_dict = {key: axes[i] for i, key in enumerate(magnitude_dict.keys())}
    else:
        # If axes are provided, continue using them
        axes_dict = axes

    # Loop over each atom type and its corresponding dipole magnitudes
    for key in magnitude_dict.keys():
        # Calculate the KDE for the dipole magnitudes
        kde = gaussian_kde(magnitude_dict[key])

        # Create a range of values over which to plot the KDE
        x_range = np.linspace(min(magnitude_dict[key]), max(magnitude_dict[key]), 1000)
        y_kde = kde(x_range)

        # Plot the KDE on the corresponding axis
        if label is None:
            label = f'Atom {key} KDE'
        axes_dict[key].plot(x_range, y_kde, label=label)
        axes_dict[key].set_title(f'Distribution for Atom Type {key}')
        axes_dict[key].set_xlabel('Dipole Magnitude')
        axes_dict[key].set_ylabel('Density')
        axes_dict[key].legend()
    
    # Return the axes dictionary so we can reuse the axes later
    return axes_dict

# %%
# plot distributions
res_mag = get_ord_1_mag_distribution(res)
print('norms ML', res_mag)
df_mag = get_ord_1_mag_distribution(samp_df)
print('norms DF', df_mag)


axes_dict = plot_L1_distributions(res_mag, label="ML")
axes_dict = plot_L1_distributions(df_mag, axes=axes_dict, label="DF")

plt.plot()
# %%
dpm_ml = orbitals.intor_dipole_moment_ml(res, dataset.orbital_basis_num, return_mat=True)
print('dpm_intmat.shape', dpm_intmat[0].shape)
print('dpm pred analytic', utils.internal_to_debye(dpm_ml['dipole_moment']))
dpm_df = orbitals.calc_dipole_moment_analytic(samp_df, 'augccpvqzjkfit', 'df_coeffs')['dipole_moment']
print('df coeffs analytic', utils.internal_to_debye(dpm))
dpm_ref = orbitals.calc_dipole_moment_analytic(samp, 'augccpvdz', 'mo_coeffs')['dipole_moment']
print(dpm_ref.shape)
print('dpm true analytic', utils.internal_to_debye(dpm_ref))
print('ml_dpm_error', utils.internal_to_debye(torch.norm(dpm_ref - dpm_ml['dipole_moment'])))
print('df_dpm_error', utils.internal_to_debye(torch.norm(dpm_ref - dpm_df)))
# %%
print('intmat shape', dpm_ml['dipole_intor'].shape)
print('df coeffs shape', dpm_ml['df_coeffs'].shape)
print(utils.internal_to_debye(torch.einsum('ikj, ij -> ik', dpm_ml['dipole_intor'], dpm_ml['df_coeffs'])))
# %%
# preparing L=1 coefficients for training to fit dipole.

# for param in model.parameters():
#     param.requires_grad = False

model = load_model(args, dataset)
model.train()
dpm_params = []
for name, param in model.named_parameters():
    if 'nonmixing' in name:
        print(name)
        if '.1.' in name or "coeff_1." in name:
            # param.requires_grad = True
            dpm_params.append(param)
            print('setting true')
        else:
            # param.requires_grad = False
            print('setting false')
# %%
dpm_optimizer = torch.optim.Adam(dpm_params, lr=1e-5, eps=args.epsilon,
                                           betas=(args.beta1, args.beta2), weight_decay=0.0, amsgrad=True)

dpm_ref = orbitals.calc_dipole_moment_analytic(samp, 'augccpvdz', 'mo_coeffs')['dipole_moment']
intor_mat = dpm_ml['dipole_intor']
for i in range(10):
    dpm_optimizer.zero_grad()
    res = model(samp)
    print(res['df_coeffs'].requires_grad)
    # print('res radial scale', res['radial_scale'][0][(1, 0)])
    # print('res radial width', res['radial_width'][0][(1, 0)])
    # print('res sph_coeffs', res['spherical_coeffs'][0][(1, 1)])
    dpm_int = -torch.einsum('ikj, ij -> ik', dpm_ml['dipole_intor'], res['df_coeffs'])
    print(dpm_int.requires_grad)
    loss = torch.norm(dpm_ref - dpm_int)
    print('ml_dpm_error', utils.internal_to_debye(torch.norm(dpm_ref - dpm_int)))
    print('ml_dens_error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights']) / torch.sum(res['batch_atom_numbers'], dim=1))
    loss.backward()
    dpm_optimizer.step()
# %%
# preparing L=1 coefficients for training to fit dipole.

# for param in model.parameters():
#     param.requires_grad = False

dens_params = []
for name, param in model.named_parameters():
    if 'nonmixing' in name:
        print(name)
        trunc_idx = name.find('nonmixing')
        trunc_name = name[trunc_idx:]
        if '.1.' in trunc_name or "coeff_1." in trunc_name:
            # param.requires_grad = True
            print('setting false')
        elif '.0.' in trunc_name or "coeff_0." in trunc_name:
            print('setting false')
        else:
            dens_params.append(param)
            # param.requires_grad = False
            print('setting true')
# %%
dens_optimizer = torch.optim.Adam(dens_params, lr=1e-5, eps=args.epsilon,
                                           betas=(args.beta1, args.beta2), weight_decay=0.0, amsgrad=True)

dpm_ref = orbitals.calc_dipole_moment_analytic(samp, 'augccpvdz', 'mo_coeffs')['dipole_moment']
intor_mat = dpm_ml['dipole_intor']
prev_df_coeffs = 0
for i in range(10):
    dpm_optimizer.zero_grad()
    res = model(samp)
    print(torch.sum(torch.abs(prev_df_coeffs - res['df_coeffs'])))
    prev_df_coeffs = res['df_coeffs']
    # print('res radial scale', res['radial_scale'][0][(1, 0)])
    # print('res radial width', res['radial_width'][0][(1, 0)])
    # print('res sph_coeffs', res['spherical_coeffs'][0][(1, 0)])
    # print('res sph_coeffs', res['spherical_coeffs'][0][(1, 1)])
    # print('res sph_coeffs', res['spherical_coeffs'][0][(1, 2)])
    # print('res sph_coeffs', res['spherical_coeffs'][0][(1, 3)])
    dpm_int = -torch.einsum('ikj, ij -> ik', dpm_ml['dipole_intor'], res['df_coeffs'])
    loss = torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'])
    print('ml_dpm_error', utils.internal_to_debye(torch.norm(dpm_ref - dpm_int)))
    print('ml_dens_error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights']) / torch.sum(res['batch_atom_numbers'], dim=1))
    loss.backward()
    dpm_optimizer.step()
