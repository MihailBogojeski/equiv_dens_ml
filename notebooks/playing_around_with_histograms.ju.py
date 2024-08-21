# %%
import os
from pyscf import gto
from pyscf.dft import numint
from pyscf.lib import param
from datetime import datetime

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

# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..

# %%
main_args = Namespace()

# main_args.args_file = "args/resorcinol_all_001.txt"
# main_args.args_file = "args/CO_dens_001.txt"
# main_args.args_file = "args/h2o_dens_001.txt"
main_args.args_file = "args/ethanethiol_all_006_test.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
# main_args.save_file = 'CO_dens_001.txt'
# main_args.save_file = 'h2o_dens_001.txt'
# main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
# main_args.save_file = 'resorcinol_all_005'
# main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
main_args.save_file = 'ethanethiol_all_006_test'
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

if args.restart is None:
    # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    model_code = generate_id()
    directory = os.path.join(args.save_dir, datetime.utcnow().strftime("%Y-%m-%d_") +
                             model_code)  # generate directory name
    # create directories
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # # write command line arguments to file (useful for reproducibility)
    # with open(os.path.join(directory, 'args.txt'), 'w') as f:
    #     for key in args.__dict__.keys():
    #         # special case for list input
    #         if isinstance(args.__dict__[key], list):
    #             for entry in args.__dict__[key]:
    #                 f.write('--' + key + '=' + str(entry) + "\n")
    #         else:
    #             f.write('--' + key + '=' + str(args.__dict__[key]) + "\n")
    checkpoint = None
    latest_checkpoint = 0
    step = 0
    restore = False
    data_split_indices = None
    # restarts run from latest checkpoint
else:
    # no restart directory specifie
    directory = args.restart
    # load latest checkpoint
    checkpoint_path = os.path.join(directory, 'checkpoints')  # checkpoint directory
    checkpoint = torch.load(os.path.join(
        checkpoint_path, 'latest_checkpoint.pth'), map_location='cpu')
    latest_checkpoint = checkpoint['step']
    model_code = checkpoint['ID']  # load ID
    step = checkpoint['step']
    for arg in vars(checkpoint['args']):
        if args.fix_arguments:
            if arg in hyperparam_args:
                # print('loading hyperparam arg', arg)
                setattr(args, arg, getattr(checkpoint['args'], arg))
        else:
            # print('loading all arg', arg)
            setattr(args, arg, getattr(checkpoint['args'], arg))
    restore = True
    data_split_indices = checkpoint['data_split_indices']

args.df_weight = 1.0

print('model code:', model_code)

# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
args.use_gpu = False

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

args.verbose = 0
args.use_gpu = False
print('args use gpu', args.use_gpu)
args.cube_grid = False
args.radii_adjust = True
args.expansion_constraint = None
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

required_properties = ['energy', 'forces', 'df_coeffs', 'density', 'dipole_moment']

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file

dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000000000000,
                           required_properties=required_properties,
                           center_positions=False,
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
                           )

print('dataset length', len(dataset))
print('sample pos shape', dataset.get_properties([0])['positions'].shape)
print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
print('num samples', main_args.num_samples)
print('args use gpu', args.use_gpu)

# %%
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
                                  center_positions=False,
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
                                  )

# %%
model = model_loader.load_model(args, dataset)
idx = 1
samp = dataset.get_properties([idx])
samp_df = dataset_df.get_properties([idx])

res = model(samp)

print('samp df_coeffs', samp_df['df_coeffs'].shape)
print('res df_coeffs', res['df_coeffs'].shape)

# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
print('density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
print('density integral', torch.sum(samp_df['density'] * samp_df['coord_weights'], dim=1))
# %%
test_indices = checkpoint['data_split_indices']['test']
print(test_indices)
# %%
center_of_mass = torch.sum(samp['batch_positions'] * samp['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
        / torch.sum(samp['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

distance_from_com = torch.norm(samp['coords'] - center_of_mass, dim=2)

density_mae = torch.sum(torch.abs(samp['density'] - res['density']) * samp['coord_weights']) / torch.sum(samp['atom_numbers'])
density_err = torch.abs(samp['density'] - res['density']) * samp['coord_weights']
res = orbitals.calc_dipole_moment(res)
dpm_err = torch.norm(samp['dipole_moment'] - res['dipole_moment'])
# dpm_point_err = density_errors.dipole_pointwise_int_loss(res['density'], samp['density'], samp['coords'], samp['coord_weights'])
print('density_mae', density_mae)
print('dpm norm error', dpm_err)
# %%
# ethanethiol
dpm_point_err = ((res['density'] - samp['density']) * samp['coord_weights']).unsqueeze(-1) * (samp['coords'] - center_of_mass)
dpm_point_norm = torch.norm((res['density'] * samp['coord_weights']).unsqueeze(-1) * (samp['coords'] - center_of_mass), dim=-1)
print(dpm_point_norm.shape)
print(torch.min(dpm_point_norm))
print(torch.max(dpm_point_norm))
# hist, b_edges = np.histogram(dpm_point_norm.flatten().detach().cpu().numpy(), bins=19)
# print(hist, b_edges)
plt.hist(dpm_point_norm.flatten().detach().cpu().numpy(), bins=99)
plt.show()

# %%
# H2O
dpm_point_err = ((res['density'] - samp['density']) * samp['coord_weights']).unsqueeze(-1) * (samp['coords'] - center_of_mass)
dpm_point_norm = torch.norm((res['density'] * samp['coord_weights']).unsqueeze(-1) * (samp['coords'] - center_of_mass), dim=-1)
print(dpm_point_norm.shape)
print(torch.min(dpm_point_norm))
print(torch.max(dpm_point_norm))
# hist, b_edges = np.histogram(dpm_point_norm.flatten().detach().cpu().numpy(), bins=19)
# print(hist, b_edges)
plt.hist(dpm_point_norm.flatten().detach().cpu().numpy(), bins=19)
plt.show()

# %%
# CO
dpm_point_err = ((res['density'] - samp['density']) * samp['coord_weights']).unsqueeze(-1) * (samp['coords'] - center_of_mass)
dpm_point_norm = torch.norm((res['density'] * samp['coord_weights']).unsqueeze(-1) * (samp['coords'] - center_of_mass), dim=-1)
print(dpm_point_norm.shape)
print(torch.min(dpm_point_norm))
print(torch.max(dpm_point_norm))
# hist, b_edges = np.histogram(dpm_point_norm.flatten().detach().cpu().numpy(), bins=19)
# print(hist, b_edges)
plt.hist(dpm_point_norm.flatten().detach().cpu().numpy(), bins=19)
plt.show()

# %%
hist, b_edges = np.histogram(dpm_point_norm.flatten().detach().cpu().numpy(), bins=191)

dpm_pos = torch.sum((res['batch_positions'] - center_of_mass) * res['batch_atom_numbers'].unsqueeze(-1), dim=1)
for ed in b_edges:
    idxs = dpm_point_norm <= ed
    dpm_neg = torch.sum((res['density'][idxs] * res['coord_weights'][idxs]).unsqueeze(-1) *
                        (res['coords'][idxs] - center_of_mass), dim=-2)
    dpm = dpm_pos - dpm_neg
    print('dpm err', torch.norm(dpm - samp['dipole_moment']))

dpm_neg = torch.sum((res['density'] * res['coord_weights']).unsqueeze(-1) *
                    (res['coords'] - center_of_mass), dim=-2)
dpm = dpm_pos - dpm_neg
print('dpm err', torch.norm(dpm - samp['dipole_moment']))
# %%
hist, b_edges = np.histogram(dpm_point_norm.flatten().detach().cpu().numpy(), bins=191)

dpm_pos = torch.sum((res['batch_positions'] - center_of_mass) * res['batch_atom_numbers'].unsqueeze(-1), dim=1)
for ed in np.flip(b_edges):
    idxs = dpm_point_norm >= ed
    dpm_neg = torch.sum((res['density'][idxs] * res['coord_weights'][idxs]).unsqueeze(-1) *
                        (res['coords'][idxs] - center_of_mass), dim=-2)
    dpm = dpm_pos - dpm_neg
    print('dpm err', torch.norm(dpm - samp['dipole_moment']))

dpm_neg = torch.sum((res['density'] * res['coord_weights']).unsqueeze(-1) *
                    (res['coords'] - center_of_mass), dim=-2)
dpm = dpm_pos - dpm_neg
print('dpm err', torch.norm(dpm - samp['dipole_moment']))
# %%
hist, b_edges = np.histogram(dpm_point_norm.flatten().detach().cpu().numpy(), bins=191)

dpm_pos = torch.sum((res['batch_positions'] - center_of_mass) * res['batch_atom_numbers'].unsqueeze(-1), dim=1)
for ed in np.flip(b_edges):
    idxs = dpm_point_norm >= ed
    dpm_neg = torch.sum((res['density'][idxs] * res['coord_weights'][idxs]).unsqueeze(-1) *
                        (res['coords'][idxs] - center_of_mass), dim=-2)
    dpm = dpm_pos - dpm_neg
    print('dpm err', torch.norm(dpm - samp['dipole_moment']))

dpm_neg = torch.sum((res['density'] * res['coord_weights']).unsqueeze(-1) *
                    (res['coords'] - center_of_mass), dim=-2)
dpm = dpm_pos - dpm_neg
print('dpm err', torch.norm(dpm - samp['dipole_moment']))
# %%
hist, b_edges = np.histogram(dpm_point_norm.flatten().detach().cpu().numpy(), bins=19)
mid = len(b_edges + 1) // 2

dpm_pos = torch.sum((res['batch_positions'] - center_of_mass) * res['batch_atom_numbers'].unsqueeze(-1), dim=1)
for i in range(mid):
    ed_0 = b_edges[mid - i - 1]
    ed_1 = b_edges[mid + i]
    print('idxs', idxs)
    idxs = np.logical_and(dpm_point_norm >= ed_0, dpm_point_norm <= ed_1)
    dpm_neg = torch.sum((res['density'][idxs] * res['coord_weights'][idxs]).unsqueeze(-1) *
                        (res['coords'][idxs] - center_of_mass), dim=-2)
    dpm = dpm_pos - dpm_neg
    print('dpm err', torch.norm(dpm - samp['dipole_moment']))

dpm_neg = torch.sum((res['density'] * res['coord_weights']).unsqueeze(-1) *
                    (res['coords'] - center_of_mass), dim=-2)
dpm = dpm_pos - dpm_neg
print('dpm err', torch.norm(dpm - samp['dipole_moment']))
