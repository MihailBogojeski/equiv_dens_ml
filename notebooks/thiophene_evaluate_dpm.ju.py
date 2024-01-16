# %%
import ase
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

from functools import partial
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2

# %%
args, hyperparam_args = parse_command_line_arguments(arg_file='thiophene_poly_all_002_6mer_test.txt')

print('type dtype', type(args.dtype))
args.fix_arguments = True
print('args np dir', args.np_dataset)

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
args.expansion_constraint = None
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
mers_mae = [[] for _ in range(5)]
mers_dpm = [[] for _ in range(5)]
mers_dpm_mag = [[] for _ in range(5)]
mers_dpm_ang = [[] for _ in range(5)]
mers_lda = [[] for _ in range(5)]
for i in range(len(dataset)):
    mer = (i // 1000)
    sample = dataset.get_properties([i])
    sample_df = dataset_df.get_properties([i])
    df_error = torch.sum(torch.abs(sample['density'] - sample_df['density']) * sample['coord_weights'])/torch.sum(sample['atom_numbers'])
    df2_error = torch.sqrt(torch.sum((sample['density'] - sample_df['density'])**2 * sample['coord_weights'])/torch.sum(sample['atom_numbers']))
    dpm_error = 4.8 * torch.mean(torch.abs(sample_df['dipole_moment'] - sample['dipole_moment']))
    dpm_mag_error = 4.8 * torch.mean(torch.abs(torch.linalg.norm(sample_df['dipole_moment']) - torch.linalg.norm(sample['dipole_moment'])))
    dpm_ang_error = (180 / torch.pi) * torch.acos(torch.mean(torch.sum(sample_df['dipole_moment']*sample['dipole_moment']) /
                                                  (torch.linalg.norm(sample_df['dipole_moment']) * torch.linalg.norm(sample['dipole_moment']))))
    lda_error = utils.hartree_to_kcal(torch.abs(-(3 / 4) * (3 / torch.pi)**(1/3) *
                                                torch.sum((sample['density']**(4/3) -
                                                          torch.clamp(sample_df['density'], min=0)**(4/3)) *
                                                          sample['coord_weights'])))
    mers_mae[mer].append(float(df_error))
    mers_dpm[mer].append(float(dpm_error))
    mers_dpm_mag[mer].append(float(dpm_mag_error))
    mers_dpm_ang[mer].append(float(dpm_ang_error))
    mers_lda[mer].append(float(lda_error))
    
    print('i', i, 'mae', df_error, 'rmse', df2_error, 'dpm', dpm_error, 'mag', dpm_mag_error, 'ang', dpm_ang_error, 'lda', lda_error)
                                      
    if (i + 1) % 10 == 0:
        for j in range(len(mers_mae)):
            print('mer', j)
            print(np.mean(mers_mae[j]))
            print(np.mean(mers_dpm[j]))
            print(np.mean(mers_dpm_mag[j]))
            print(np.mean(mers_dpm_ang[j]))
            print(np.mean(mers_lda[j]))

for j in range(len(mers_mae)):
    print('mer', j)
    print(np.mean(mers_mae[j]))
    print(np.mean(mers_dpm[j]))
    print(np.mean(mers_dpm_mag[j]))
    print(np.mean(mers_dpm_ang[j]))
    print(np.mean(mers_lda[j]))

# mae = []
# dpm = []
# for i in range(len(dataset)):
#     sample = dataset.get_properties([i])
#     sample_df = dataset_df.get_properties([i])
#     df_error = torch.sum(torch.abs(sample['density'] - sample_df['density']) * sample['coord_weights'])/torch.sum(sample['atom_numbers'])
#     dpm_error = 4.8 * torch.mean(torch.abs(sample_df['dipole_moment'] - sample['dipole_moment']))
#     print('i', i, ', dpm error', dpm_error, 'positions')
#     mae.append(float(df_error))
#     dpm.append(float(dpm_error))
#
# print('mae', np.mean(mae))
# print('dpm', np.mean(dpm))

# %%
res_dataset = torch.load('datasets/thiophene_poly_all_002_3mer_test.pt', map_location='cpu')
mers_mae = [[] for _ in range(5)]
mers_dpm = [[] for _ in range(5)]
mers_dpm_mag = [[] for _ in range(5)]
mers_dpm_ang = [[] for _ in range(5)]
mers_lda = [[] for _ in range(5)]
for i in range(len(dataset)):
    sample = dataset.get_properties([i]) 
    #print('sample pdist', torch.cdist(sample['positions'], sample['positions'])[0, :3,:3])
    #print('res_pdist', torch.cdist(res_dataset['positions'][[i]], res_dataset['positions'][[i]])[0, :3,:3])
    mer = (i // 1000)
    sample = dataset.get_properties([i])
    r_dens = torch.clamp(res_dataset['density'][[i]], min=0)
    r_dens = r_dens / torch.sum(r_dens * sample['coord_weights']) * torch.sum(sample['atom_numbers'])
    #print('density_integral', torch.sum(sample['density'] * sample['coord_weights']))
    #print('resdensity_integral', torch.sum(res_dataset['density'][[i]] * sample['coord_weights']))
    #print('density sum', torch.sum(sample['density']))
    #print('resdensity sum', torch.sum(res_dataset['density'][[i]]))
    #print('samp_en', sample['energy'])
    #print('res en', res_dataset['energy'][i])
    
    #print('samp F', sample['forces'])
    #print('res F', res_dataset['forces'][i])
    df_error = torch.sum(torch.abs(sample['density'] - r_dens) * sample['coord_weights'])/torch.sum(sample['atom_numbers'])
    df2_error = torch.sqrt(torch.sum((sample['density'] - r_dens)**2 * sample['coord_weights'])/torch.sum(sample['atom_numbers']))
    dpm_error = 4.8 * torch.mean(torch.abs(res_dataset['dipole_moment'][[i]] - sample['dipole_moment']))
    dpm_mag_error = 4.8 * torch.mean(torch.abs(torch.linalg.norm(res_dataset['dipole_moment'][[i]]) - torch.linalg.norm(sample['dipole_moment'])))
    dpm_ang_error = (180 / torch.pi) * torch.acos(torch.mean(torch.sum(res_dataset['dipole_moment'][[i]]*sample['dipole_moment']) /
                                                  (torch.linalg.norm(res_dataset['dipole_moment'][[i]]) * torch.linalg.norm(sample['dipole_moment']))))
    lda_error = utils.hartree_to_kcal(torch.abs(-(3 / 4) * (3 / torch.pi)**(1/3) *
                                                torch.sum((sample['density']**(4/3) -
                                                          r_dens**(4/3)) *
                                                          sample['coord_weights'])))
    
    #print('i', i, 'mae', df_error, 'rmse', df2_error, 'dpm', dpm_error, 'mag', dpm_mag_error, 'ang', dpm_ang_error, 'lda', lda_error)
    mers_mae[mer].append(float(df_error))
    mers_dpm[mer].append(float(dpm_error))
    mers_dpm_mag[mer].append(float(dpm_mag_error))
    mers_dpm_ang[mer].append(float(dpm_ang_error))
    mers_lda[mer].append(float(lda_error))
    #if (i + 1) % 10 == 0:
    #    for j in range(len(mers_mae)):
    #        print('mer', j)
    #        print(np.mean(mers_mae[j]))
    #        print(np.mean(mers_dpm[j]))
    #        print(np.mean(mers_dpm_mag[j]))
    #        print(np.mean(mers_dpm_ang[j]))
    #        print(np.mean(mers_lda[j]))

for j in range(len(mers_mae)):
    print('mer', j)
    print(np.mean(mers_mae[j]))
    print(np.mean(mers_dpm[j]))
    print(np.mean(mers_dpm_mag[j]))
    print(np.mean(mers_dpm_ang[j]))
    print(np.mean(mers_lda[j]))

# %%
res_dataset = torch.load('datasets/thiophene_poly_all_002_6mer_test.pt', map_location='cpu')

idx = 123
sample = dataset.get_properties([idx])
sample_df = dataset_df.get_properties([idx])

# %%


# %%


# %%
res_dataset = torch.load('datasets/thiophene_poly_all_002_6mer_test.pt', map_location='cpu')
pred_mae = []
pred_rmse = []
pred_23 = []
pred_43 = []
pred_lda = []
pred_lda23 = []
df_mae = []
df_rmse = []
df_23 = []
df_lda = []
df_lda23 = []
df_43 = []
for i in range(5):
    #print('sample pdist', torch.cdist(sample['positions'], sample['positions'])[0, :3,:3])
    #print('res_pdist', torch.cdist(res_dataset['positions'][[i]], res_dataset['positions'][[i]])[0, :3,:3])
    print('i', i)
    sample = dataset.get_properties([i])
    r_dens = torch.clamp(res_dataset['density'][[i]], min=0)
    r_dens = r_dens / torch.sum(r_dens * sample['coord_weights']) * torch.sum(sample['atom_numbers'])
    #print('density_integral', torch.sum(sample['density'] * sample['coord_weights']))
    #print('resdensity_integral', torch.sum(res_dataset['density'][[i]] * sample['coord_weights']))
    #print('density sum', torch.sum(sample['density']))
    #print('resdensity sum', torch.sum(res_dataset['density'][[i]]))
    #print('samp_en', sample['energy'])
    #print('res en', res_dataset['energy'][i])
    
    #print('samp F', sample['forces'])
    #print('res F', res_dataset['forces'][i])
    df_error = torch.sum(torch.abs(sample['density'] - r_dens) * sample['coord_weights'])
    
    df2_error = torch.sqrt(torch.sum((sample['density'] - r_dens)**2 * sample['coord_weights'])) 
    
    df43_error = torch.sum(torch.abs(sample['density']**(4/3) -
                          r_dens**(4/3)) * sample['coord_weights'])
    
    df23_error = torch.sum(torch.abs(sample['density']**(2/3) -
                          r_dens**(2/3)) * sample['coord_weights'])
    
    lda_error = torch.abs(torch.sum((sample['density']**(4/3) -
                          r_dens**(4/3)) * sample['coord_weights']))
    lda23_error = torch.abs(torch.sum((sample['density']**(2/3) -
                          r_dens**(2/3)) * sample['coord_weights']))
    
    #print('i', i, 'mae', df_error, 'rmse', df2_error, 'dpm', dpm_error, 'mag', dpm_mag_error, 'ang', dpm_ang_error, 'lda', lda_error)
    pred_mae.append(float(df_error))
    pred_rmse.append(float(df2_error))
    pred_43.append(float(df43_error))
    pred_23.append(float(df23_error))
    pred_lda.append(float(lda_error))
    pred_lda23.append(float(lda23_error))
    
    sample_df = dataset_df.get_properties([i])
    df_error = torch.sum(torch.abs(sample['density'] - sample_df['density']) * sample['coord_weights'])
    df2_error = torch.sqrt(torch.sum((sample['density'] - sample_df['density'])**2 * sample['coord_weights']))
    df43_error = torch.sum(torch.abs(sample['density']**(4/3) -
                          torch.clamp(sample_df['density'], min=0)**(4/3)) * sample['coord_weights'])
    
    df23_error = torch.sum(torch.abs(sample['density']**(2/3) -
                          torch.clamp(sample_df['density'], min=0)**(2/3)) * sample['coord_weights'])
    lda_error = torch.abs(torch.sum((sample['density']**(4/3) -
                          torch.clamp(sample_df['density'], min=0)**(4/3)) *
                          sample['coord_weights']))
    lda23_error = torch.abs(torch.sum((sample['density']**(2/3) -
                            torch.clamp(sample_df['density'], min=0)**(2/3)) *
                            sample['coord_weights']))
    
    
    df_mae.append(float(df_error))
    df_rmse.append(float(df2_error))
    df_43.append(float(df43_error))
    df_23.append(float(df23_error))
    df_lda.append(float(lda_error))
    df_lda23.append(float(lda23_error))
    #if (i + 1) % 10 == 0:
    #    for j in range(len(mers_mae)):
    #        print('mer', j)
    #        print(np.mean(mers_mae[j]))
    #        print(np.mean(mers_dpm[j]))
    #        print(np.mean(mers_dpm_mag[j]))
    #        print(np.mean(mers_dpm_ang[j]))
    #        print(np.mean(mers_lda[j]))

print('ML dens errors')
print('MAE/IAE', np.nanmean(pred_mae))
print('RMSE', np.nanmean(pred_rmse))
print('pow(2/3)', np.nanmean(pred_23))
print('pow(4/3)', np.nanmean(pred_43))
print('LDA MAE', np.nanmean(pred_lda))
print('LDA (2/3) MAE', np.nanmean(pred_lda23))
print('DF dens errors')
print('MAE/IAE', np.nanmean(df_mae))
print('RMSE', np.nanmean(df_rmse))
print('pow(2/3)', np.nanmean(df_23))
print('pow(4/3)', np.nanmean(df_43))
print('LDA MAE', np.nanmean(df_lda))
print('LDA (2/3) MAE', np.nanmean(df_lda23))

# %%
from equiv_dens.training import density_errors
idx = 0
sample = dataset.get_properties([idx])
sample_df = dataset_df.get_properties([idx])

center_of_mass = torch.sum(sample['batch_positions'] * sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
                / torch.sum(sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

print(center_of_mass)
print(center_of_mass.shape)

print(sample['coords'].shape)
distance_from_com = torch.norm(sample['coords'] - center_of_mass, dim=2)

print(distance_from_com.shape)
plt.rcParams['text.usetex'] = True
dens_df_mae = torch.abs(sample['density'] - sample_df['density']) * sample['coord_weights']
dens_ml_mae = torch.abs(sample['density'] - res_dataset['density'][[idx]]) * sample['coord_weights']

dens_ml_LDA = torch.abs(density_errors.density_LDA_loss(res_dataset['density'][[idx]], sample['density'], sample['coord_weights']))
dens_df_LDA = torch.abs(density_errors.density_LDA_loss(sample_df['density'], sample['density'], sample['coord_weights']))

print('ml LDA MAE', dens_ml_LDA)
print('df LDA MAE', dens_df_LDA)

ml_max = torch.argmax(dens_ml_mae[0])
df_max = torch.argmax(dens_df_mae[0])

print('ml_mae', torch.sum(dens_ml_mae))
print('df_mae', torch.sum(dens_df_mae))

dens_ml_coul = torch.abs(density_errors.density_coulomb_loss(res_dataset['density'][[idx]] - sample['density'], sample['coords'], sample['coord_weights']))
dens_df_coul = torch.abs(density_errors.density_coulomb_loss(sample_df['density'] - sample['density'], sample['coords'], sample['coord_weights']))

print('ml coulomb', dens_ml_coul)
print('df coulomb', dens_df_coul)

print('ml mae [ml_max]', dens_ml_mae[:, ml_max])
print('df mae [ml_max]', dens_df_mae[:, ml_max])
print('scale [ml_max]', sample['coord_weights'][:,ml_max])


print('ml mae [df_max]', dens_ml_mae[:, df_max])
print('df mae [df_max]', dens_df_mae[:, df_max])
print('scale [df_max]', sample['coord_weights'][:, df_max])


print(sample['coords'].shape)
print(sample['positions'].shape)
distance_from_atoms = torch.norm(sample['coords'].unsqueeze(1) - sample['positions'].unsqueeze(2), dim=-1)

print(distance_from_atoms.shape)
min_distance_from_atoms = torch.min(distance_from_atoms, dim=1)[0]
print(min_distance_from_atoms.shape)

fig, axs = plt.subplots(2,1, figsize=(5,7))
axs[0].scatter(dens_df_mae.squeeze(), distance_from_com, label='DF', c='blue')
axs[0].scatter(dens_ml_mae.squeeze(), distance_from_com, label='ML', c='orange')

axs[1].scatter(dens_df_mae.squeeze(), min_distance_from_atoms, label='DF', c='blue')
axs[1].scatter(dens_ml_mae.squeeze(), min_distance_from_atoms, label='ML', c='orange')


axs[1].set_xlabel('absolute density error at gridpoint, scaled by volume of gridpoint')
axs[0].set_ylabel('distance from center of mass')
axs[1].set_ylabel('distance from closest atom')
fig.suptitle('Scaled absolute density error', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('figures/scaled_mae.png', dpi=300)

# %%
center_of_mass = torch.sum(sample['batch_positions'] * sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
                / torch.sum(sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

print(center_of_mass)
print(center_of_mass.shape)

print(sample['coords'].shape)
distance_from_com = torch.norm(sample['coords'] - center_of_mass, dim=2)

print(distance_from_com.shape)
plt.rcParams['text.usetex'] = True
dens_df_mse = (sample['density'] - sample_df['density'])**2 * sample['coord_weights']
dens_ml_mse = (sample['density'] - res_dataset['density'][[idx]])**2 * sample['coord_weights']

print('ml mse max', torch.max(dens_ml_mse))

print('df mse max', torch.max(dens_df_mse))

fig, axs = plt.subplots(2,1, figsize=(5, 7))

axs[0].scatter(dens_ml_mse.squeeze(), distance_from_com, label='ML', c='orange')
axs[0].scatter(dens_df_mse.squeeze(), distance_from_com, label='DF', c='blue')
axs[1].scatter(dens_ml_mse.squeeze(), min_distance_from_atoms, label='ML', c='orange')
axs[1].scatter(dens_df_mse.squeeze(), min_distance_from_atoms, label='DF', c='blue')
axs[1].set_xlabel('squared density error at gridpoint, scaled by volume of gridpoint')
axs[0].set_ylabel('distance from center of mass')
axs[1].set_ylabel('distance from closest atom')

fig.suptitle('Scaled square density error', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('figures/scaled_mse.png', dpi=300)

# %%
center_of_mass = torch.sum(sample['batch_positions'] * sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
                / torch.sum(sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

print(center_of_mass)
print(center_of_mass.shape)

print(sample['coords'].shape)
distance_from_com = torch.norm(sample['coords'] - center_of_mass, dim=2)

print(distance_from_com.shape)
plt.rcParams['text.usetex'] = True
dens_df_mae = torch.abs(sample['density'] - sample_df['density'])
dens_ml_mae = torch.abs(sample['density'] - res_dataset['density'][[idx]])

ml_max = torch.argmax(dens_ml_mae[0])
df_max = torch.argmax(dens_df_mae[0])

print('ml mae [ml_max]', dens_ml_mae[:, ml_max])
print('df mae [ml_max]', dens_df_mae[:, ml_max])
print('scale [ml_max]', sample['coord_weights'][:,ml_max])


print('ml mae [df_max]', dens_ml_mae[:, df_max])
print('df mae [df_max]', dens_df_mae[:, df_max])
print('scale [df_max]', sample['coord_weights'][:, df_max])

print(sample['coords'].shape)
print(sample['positions'].shape)
distance_from_atoms = torch.norm(sample['coords'].unsqueeze(1) - sample['positions'].unsqueeze(2), dim=-1)

print(distance_from_atoms.shape)
min_distance_from_atoms = torch.min(distance_from_atoms, dim=1)[0]
print(min_distance_from_atoms.shape)

fig, axs = plt.subplots(2,1, figsize=(5, 7))

axs[0].scatter(dens_ml_mae.squeeze(), distance_from_com, label='ML', c='orange')
axs[0].scatter(dens_df_mae.squeeze(), distance_from_com, label='DF', c='blue')

axs[1].scatter(dens_ml_mae.squeeze(), min_distance_from_atoms, label='ML', c='orange')
axs[1].scatter(dens_df_mae.squeeze(), min_distance_from_atoms, label='DF', c='blue')


axs[1].set_xlabel('absolute density error at gridpoint')
axs[0].set_ylabel('distance from center of mass')
axs[1].set_ylabel('distance from closest atom')

fig.suptitle('Unscaled absolute density error', fontsize=16)
plt.legend()
plt.tight_layout()

plt.savefig('figures/unscaled_mae.png', dpi=300)

# %%


