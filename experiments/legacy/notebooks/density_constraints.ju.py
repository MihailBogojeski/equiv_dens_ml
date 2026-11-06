# %%
import ase
from datetime import datetime
from equiv_dens.utils.misc import generate_id
import numpy as np
import pyscf
import time
import os
from pyscf.scf import hf
from pyscf import gto, df, lib
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
import equiv_dens.utils.orbitals as orbitals

from functools import partial
# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..

# %%
args, hyperparam_args = parse_command_line_arguments(arg_file='args/ethanethiol_all_006_test.txt')

print('type dtype', type(args.dtype))
args.fix_arguments = True
print('args np dir', args.np_dataset)

if args.restart is None:
    # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    model_code = generate_id()
    directory = os.path.join(args.save_dir, datetime.utcnow().strftime("%Y-%m-%d_") +
                             model_code)  # generate directory name
    # create directories
    if not os.path.exists(directory):
        os.makedirs(directory)
    # write command line arguments to file (useful for reproducibility)
    with open(os.path.join(directory, 'args.txt'), 'w') as f:
        for key in args.__dict__.keys():
            # special case for list input
            if isinstance(args.__dict__[key], list):
                for entry in args.__dict__[key]:
                    f.write('--' + key + '=' + str(entry) + "\n")
            else:
                f.write('--' + key + '=' + str(args.__dict__[key]) + "\n")
    restore = False
    # restarts run from latest checkpoint
else:
    # no restart directory specified
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
                print('loading hyperparam arg', arg)
                setattr(args, arg, getattr(checkpoint['args'], arg))
        else:
            print('loading all arg', arg)
            setattr(args, arg, getattr(checkpoint['args'], arg))
    restore = True
args.best_model_path = 'best_' + model_code + '.pth'
print('best_model_path', args.best_model_path)

print('model code:', model_code)

# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
args.use_gpu = False

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

args.verbose = 0
args.use_gpu = False
args.cube_grid = False
args.radii_adjust = True
args.expansion_constraint = 'sp'
args.integral_constraint = 'coeffs'
args.core_density_basis = 0.29
# args.core_density_basis = 0
if args.cube_grid:
    args.cube_origin = -0.25
    args.cube_extent = 0.5
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

required_properties = ['energy', 'forces', 'density', 'dipole_moment']
dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000000000000,
                           required_properties=required_properties,
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           pyscf_grid=args.pyscf_grid,
                           pyscf_rotate=rotate,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
                           )

dataset_df = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                              orbitals_path=args.orbitals_file,
                              density_n_samp=10000000000000000000000,
                              required_properties=required_properties,
                              center_positions=False,
                              radial_coeffs_file=args.radial_coeffs_file,
                              dtype=args.dtype,
                              grid_fn=grid_fn,
                              pyscf_grid=args.pyscf_grid,
                              pyscf_rotate=rotate,
                              sampling_fn=sampling_fn,
                              grid_extent=grid_extent,
                              grid_origin=grid_origin,
                              cutoff=args.cutoff,
                              df_loss_weights=args.df_loss_weights,
                              projected_density=True,
                              )
# %%
# load_model
model = load_model(args, dataset)

# %%
# no constraints
sample = dataset.get_properties([0])
res = model(sample)
print('code mae', torch.sum(torch.abs(sample['density'] - res['density']) * sample['coord_weights'])/torch.sum(sample['atom_numbers']))
print('code rmse', torch.sqrt(torch.sum((sample['density'] - res['density'])**2 * sample['coord_weights']))/(torch.sqrt(torch.sum(sample['density'] ** 2 * sample['coord_weights'], dim=1))))
print('dpm error', 4.8 * torch.mean(torch.abs(sample['dipole_moment'] - res['dipole_moment'])))
# %%
# core density
sample = dataset.get_properties([0])
res = model(sample)
print('code mae', torch.sum(torch.abs(sample['density'] - res['density']) * sample['coord_weights'])/torch.sum(sample['atom_numbers']))
print('code rmse', torch.sqrt(torch.sum((sample['density'] - res['density'])**2 * sample['coord_weights']))/(torch.sqrt(torch.sum(sample['density'] ** 2 * sample['coord_weights'], dim=1))))
print('dpm error', 4.8 * torch.mean(torch.abs(sample['dipole_moment'] - res['dipole_moment'])))

# %%
# integral constraints
sample = dataset.get_properties([0])
res = model(sample)
print('code mae', torch.sum(torch.abs(sample['density'] - res['density']) * sample['coord_weights'])/torch.sum(sample['atom_numbers']))
print('code rmse', torch.sqrt(torch.sum((sample['density'] - res['density'])**2 * sample['coord_weights']))/(torch.sqrt(torch.sum(sample['density'] ** 2 * sample['coord_weights'], dim=1))))
print('dpm error', 4.8 * torch.mean(torch.abs(sample['dipole_moment'] - res['dipole_moment'])))
# %%
# integral constraints and expansion constraints
sample = dataset.get_properties([0])
res = model(sample)
print('code mae', torch.sum(torch.abs(sample['density'] - res['density']) * sample['coord_weights'])/torch.sum(sample['atom_numbers']))
print('code rmse', torch.sqrt(torch.sum((sample['density'] - res['density'])**2 * sample['coord_weights']))/(torch.sqrt(torch.sum(sample['density'] ** 2 * sample['coord_weights'], dim=1))))
print('dpm error', 4.8 * torch.mean(torch.abs(sample['dipole_moment'] - res['dipole_moment'])))
# %%
# core density and constraints
sample = dataset.get_properties([0])
res = model(sample)
print('code mae', torch.sum(torch.abs(sample['density'] - res['density']) * sample['coord_weights'])/torch.sum(sample['atom_numbers']))
print('code rmse', torch.sqrt(torch.sum((sample['density'] - res['density'])**2 * sample['coord_weights']))/(torch.sqrt(torch.sum(sample['density'] ** 2 * sample['coord_weights'], dim=1))))
print('dpm error', 4.8 * torch.mean(torch.abs(sample['dipole_moment'] - res['dipole_moment'])))
print('density integral', torch.sum(res['density'] * sample['coord_weights']))
# %%
print('res density coeffs', res['spherical_coeffs'][0])
# sample = dataset.get_properties([0])
# res = model_old(sample)
# print('old code error', torch.sum(torch.abs(sample['density'] - res['density']) * sample['coord_weights'])/torch.sum(sample['atom_numbers']))
# %%
sample = dataset.get_properties([0])
res = {key: sample[key] for key in sample.keys()}
print('true density integral', torch.sum(sample['density'] * sample['coord_weights'], dim=1))
repres = model.density_repr_model(res)
dens_res = model.property_models['density'](repres)
print('ML density integral', torch.sum(dens_res['density'] * sample['coord_weights'], dim=1))

print('density error', float(torch.sum(torch.abs(sample['density'] - dens_res['density']) * sample['coord_weights'])/torch.sum(sample['atom_numbers'])))
sample_dpm = orbitals.calc_dipole_moment(sample, center_coordinates=True)['dipole_moment']
res_dpm = orbitals.calc_dipole_moment(dens_res, center_coordinates=True)['dipole_moment']
print('dpm error', 4.8 * torch.mean(torch.abs(sample_dpm - res_dpm)))
center_of_mass = torch.sum(sample['batch_positions'] * sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
    / torch.sum(sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)
# absolute dipole moment error at every grid point, RMSE
dpm_coord_error = torch.sqrt(torch.sum(torch.sum((torch.abs(sample['density'] - dens_res['density']) * sample['coord_weights']).unsqueeze(-1) *
                             (sample['coords'] - center_of_mass), dim=1)**2))
print('dpm coord rmse', torch.sum(dpm_coord_error))

r_dens = torch.clamp(dens_res['density'], min=0)
print('clamp density integral', torch.sum(r_dens * sample['coord_weights'], dim=1))
scale = float(1/torch.sum(r_dens * sample['coord_weights']) * torch.sum(sample['density'] * sample['coord_weights']))
print(f'scaling factor {scale:.8f}')
r_dens = r_dens / torch.sum(r_dens * sample['coord_weights']) * torch.sum(sample['atom_numbers'])
print('r_dens error', float(torch.sum(torch.abs(sample['density'] - r_dens) * sample['coord_weights'])/torch.sum(sample['atom_numbers'])))
dens_res['density'] = r_dens
res_dpm = orbitals.calc_dipole_moment(dens_res, center_coordinates=True)['dipole_moment']
print('dpm error', 4.8 * torch.mean(torch.abs(sample_dpm - res_dpm)))
# absolute dipole moment error at every grid point, RMSE
dpm_coord_error = torch.sqrt(torch.sum(torch.sum((torch.abs(sample['density'] - dens_res['density']) * sample['coord_weights']).unsqueeze(-1) *
                             (sample['coords'] - center_of_mass), dim=1)**2))
print('dpm coord rmse', torch.sum(dpm_coord_error))
# %%
print(model.density_repr_model[1].orbital_basis)

# %%
df_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
             'dpm_rmse': [],
             'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
             'mae_23': [], 'mae_43': [], 'lda_23_mae': [],
             'coulomb': []}

for i in range(min(len(dataset), main_args.num_samples)):
    sample = dataset.get_properties([i])
    sample_df = dataset_df.get_properties([i])
    r_dens = torch.clamp(sample_df['density'], min=0)
    r_dens = r_dens / torch.sum(r_dens * sample['coord_weights']) * torch.sum(sample['atom_numbers'])

    df_error = torch.sum(torch.abs(sample['density'] - r_dens) * sample['coord_weights'])/torch.sum(sample['atom_numbers'])
    df2_error = torch.sqrt(torch.sum((sample['density'] - r_dens)**2 * sample['coord_weights'])/torch.sum(sample['atom_numbers']))
    dpm_error = 4.8 * torch.mean(torch.abs(sample_df['dipole_moment'] - sample['dipole_moment']))
    dpm2_error = torch.sqrt(torch.mean(torch.sum(4.8 * (sample_df['dipole_moment'] - sample['dipole_moment'])**2)))
    dpm_mag_error = 4.8 * torch.mean(torch.abs(torch.linalg.norm(sample_df['dipole_moment']) - torch.linalg.norm(sample['dipole_moment'])))
    dpm_ang_error = (180 / torch.pi) * torch.acos(torch.mean(torch.sum(sample_df['dipole_moment']*sample['dipole_moment']) /
                                                  (torch.linalg.norm(sample_df['dipole_moment']) * torch.linalg.norm(sample['dipole_moment']))))

    df43_error = torch.sum(torch.abs(sample['density']**(4/3) -
                           r_dens**(4/3)) * sample['coord_weights'])
    df23_error = torch.sum(torch.abs(sample['density']**(2/3) -
                           r_dens**(2/3)) * sample['coord_weights'])
    lda_error = torch.abs(torch.sum((sample['density']**(4/3) -
                          r_dens**(4/3)) * sample['coord_weights']))
    lda23_error = torch.abs(torch.sum((sample['density']**(2/3) -
                            r_dens**(2/3)) * sample['coord_weights']))
    coul_error = density_errors._density_coulomb_loss(r_dens - sample['density'], sample['coords'], sample['coord_weights'])

    df_losses['dens_mae'].append(float(df_error))
    df_losses['dens_rmse'].append(float(df2_error))
    df_losses['dpm_mae'].append(float(dpm_error))
    df_losses['dpm_rmse'].append(float(dpm2_error))
    df_losses['dpm_mag'].append(float(dpm_mag_error))
    df_losses['dpm_ang'].append(float(dpm_ang_error))
    df_losses['lda_mae'].append(float(lda_error))
    df_losses['lda_23_mae'].append(float(lda23_error))
    df_losses['coulomb'].append(float(coul_error))
    df_losses['mae_43'].append(float(df43_error))
    df_losses['mae_23'].append(float(df23_error))
    # print('i', i, 'mae', df_error, 'rmse', df2_error, 'dpm', dpm_error, 'mag', dpm_mag_error, 'ang', dpm_ang_error, 'lda', lda_error)

print('DF losses')
for key in df_losses.keys():
    print(key)
    print(np.nanmean(df_losses[key]))

# %%
res_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
              'dpm_rmse': [],
              'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
              'coulomb': [], 'mae_23': [], 'mae_43': [],
              'lda_23_mae': []}
for i in range(min(len(dataset), main_args.num_samples)):
    sample = dataset.get_properties([i])
    # print('sample pdist', torch.cdist(sample['positions'], sample['positions'])[0, :3,:3])
    # print('res_pdist', torch.cdist(res_dataset['positions'][[i]], res_dataset['positions'][[i]])[0, :3,:3])
    mer = (i // 1000)
    sample = dataset.get_properties([i])
    r_dens = torch.clamp(res_dataset['density'][[i]], min=0)
    r_dens = r_dens / torch.sum(r_dens * sample['coord_weights']) * torch.sum(sample['atom_numbers'])

    df_error = torch.sum(torch.abs(sample['density'] - r_dens) * sample['coord_weights'])/torch.sum(sample['atom_numbers'])
    df2_error = torch.sqrt(torch.sum((sample['density'] - r_dens)**2 * sample['coord_weights'])/torch.sum(sample['atom_numbers']))
    dpm_error = 4.8 * torch.mean(torch.abs(res_dataset['dipole_moment'][[i]] - sample['dipole_moment']))
    dpm2_error = torch.sqrt(torch.mean(torch.sum((4.8*(res_dataset['dipole_moment'][[i]] - sample['dipole_moment']))**2)))
    dpm_mag_error = 4.8 * torch.mean(torch.abs(torch.linalg.norm(res_dataset['dipole_moment'][[i]]) - torch.linalg.norm(sample['dipole_moment'])))
    dpm_ang_error = (180 / torch.pi) * torch.acos(torch.mean(torch.sum(res_dataset['dipole_moment'][[i]]*sample['dipole_moment']) /
                                                  (torch.linalg.norm(res_dataset['dipole_moment'][[i]]) * torch.linalg.norm(sample['dipole_moment']))))

    df43_error = torch.sum(torch.abs(sample['density']**(4/3) -
                           r_dens**(4/3)) * sample['coord_weights'])
    df23_error = torch.sum(torch.abs(sample['density']**(2/3) -
                           r_dens**(2/3)) * sample['coord_weights'])
    lda_error = torch.abs(torch.sum((sample['density']**(4/3) -
                          r_dens**(4/3)) * sample['coord_weights']))
    lda23_error = torch.abs(torch.sum((sample['density']**(2/3) -
                            r_dens**(2/3)) * sample['coord_weights']))
    coul_error = density_errors._density_coulomb_loss(r_dens - sample['density'], sample['coords'], sample['coord_weights'])
    # print('i', i, 'mae', df_error, 'rmse', df2_error, 'dpm', dpm_error, 'mag', dpm_mag_error, 'ang', dpm_ang_error, 'lda', lda_error)
    res_losses['dens_mae'].append(float(df_error))
    res_losses['dens_rmse'].append(float(df2_error))
    res_losses['dpm_mae'].append(float(dpm_error))
    res_losses['dpm_rmse'].append(float(dpm2_error))
    res_losses['dpm_mag'].append(float(dpm_mag_error))
    res_losses['dpm_ang'].append(float(dpm_ang_error))
    res_losses['lda_mae'].append(float(lda_error))
    res_losses['lda_23_mae'].append(float(lda23_error))
    res_losses['coulomb'].append(float(coul_error))
    res_losses['mae_43'].append(float(df43_error))
    res_losses['mae_23'].append(float(df23_error))

print('Results losses')
for key in res_losses.keys():
    print(key)
    print(np.nanmean(res_losses[key]))

