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

# %load_ext autoreload
# %autoreload 2
# %cd /home/mihail/Documents/workspace/equiv_dens/

# %%
main_args = Namespace()

# main_args.args_file = "args/resorcinol_all_001.txt"
main_args.args_file = "args/ethanethiol_all_006_test.txt"
# main_args.args_file = "args/ethanethiol_df_coeffs_001_test.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
main_args.save_file = 'ethanethiol_all_006'
# main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
# main_args.save_file = 'resorcinol_all_005'
# main_args.res_load_file = 'datasets/ethanethiol_df_coeffs_001_test.pt'
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
                           pyscf_rotate=rotate,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
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
                                  pyscf_rotate=rotate,
                                  sampling_fn=sampling_fn,
                                  grid_extent=grid_extent,
                                  grid_origin=grid_origin,
                                  cutoff=args.cutoff,
                                  df_loss_weights=args.df_loss_weights,
                                  projected_density=True,
                                  )

# %%
# res_dataset = torch.load(main_args.res_load_file, map_location='cpu')
# print(res_dataset.keys())
# %%
model = model_loader.load_model(args, dataset)
samp = dataset.get_properties([0])
samp_df = dataset_df.get_properties([0])

res = model(samp)

print('samp df_coeffs', samp_df['df_coeffs'].shape)
print('res df_coeffs', res['df_coeffs'].shape)

# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
df_coeffs_dict = {key: res[key] for key in res.keys()}

print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))

df_sph_coeffs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']}, dataset.orbital_basis_num, samp_df['atom_numbers'], radial_coeffs=False)

df_coeffs_dict['spherical_coeffs'] = df_sph_coeffs['spherical_coeffs']


df_dens = model.property_models['density'](df_coeffs_dict)
print('df density integral', torch.sum(df_dens['density'] * df_dens['coord_weights'], dim=1))
print('df integral', torch.sum(samp_df['density'] * samp_df['coord_weights'], dim=1))
print('dend integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
print('df pred vs df diff', torch.sum(torch.abs(df_dens['density'] - samp_df['density']) * samp_df['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
print('df pred vs dens diff', torch.sum(torch.abs(df_dens['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
print('df vs dens diff', torch.sum(torch.abs(samp_df['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))

# %%
print('main args use gpu', main_args.use_gpu)
print('args use gpu', args.use_gpu)
idx = 1

model = model_loader.load_model(args, dataset)
samp = dataset.get_properties([idx])
samp_df = dataset_df.get_properties([idx])

res = model(samp)

print('samp df_coeffs', samp_df['df_coeffs'].shape)
print('res df_coeffs', res['df_coeffs'].shape)

# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))

sph_coeffs_vec = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                                radial_coeffs=False, convert_to_pyscf=False)

dpm_err = 4.8 * torch.norm(samp_df['dipole_moment'] - samp['dipole_moment'])
print('dipole moment error', dpm_err)
# %%
print(dataset.orbital_basis_size)
#print(dataset.orbital_basis_num)
orbital_dict = orbitals.combine_orbital_basis(dataset.orbital_basis_num, 5)[0]
print('orbital_dict', orbital_dict)
dens_errors = []
coeff_errors = []
dpm_errors = []

masks_pos = {key: 0 for key in dataset.orbital_basis_size}
max_orbitals = orbital_dict[16]

atom = [(int(samp_df['batch_atom_numbers'][0, i].detach().cpu().numpy()),
        samp_df['batch_positions'][0, i].detach().cpu().numpy())
        for i in range(samp_df['batch_positions'].shape[1])] 
#print(atom)
df_coeffs_split = orbitals.split_df_coeffs(atom, sph_coeffs_vec['spherical_coeffs'].squeeze(), dataset.orbital_basis_size)
max_coeffs_len = max([len(coeffs[1]) for coeffs in df_coeffs_split])
print('max orbitals', max_orbitals)
for L in range(6):
    nc = 2 * L + 1
    max_coeffs = max_orbitals[L][1]
    for i in range(1, max_coeffs + 1):
        for z in masks_pos.keys():
            if L >= len(orbital_dict[z]) or i >= orbital_dict[z][L][1] + 1:
                continue
            masks_pos[z] += nc
            # print('L', L, 'max coeffs', max_coeffs, 'i', i, 'z', z, 'masks_pos', masks_pos[z])
        mask_coeffs = []
        for z, atom_coeffs in df_coeffs_split:
            atom_coeffs = torch.Tensor(atom_coeffs)
            mask = torch.zeros_like(atom_coeffs)
            mask[:masks_pos[z]] = 1
            # print('z', z, 'mask pos', masks_pos[z], 'mask', mask)
            mask_coeffs.append(atom_coeffs * mask)
        coeffs = torch.cat(mask_coeffs).unsqueeze(0)
        df_coeffs_dict = {key: res[key] for key in res.keys()}

        df_dens = model.property_models['density'](df_coeffs_dict)
        coeffs_dict = orbitals.vector_to_coeffs_dict({'spherical_coeffs': coeffs}, dataset.orbital_basis_num,
                                                     res['batch_atom_numbers'], radial_coeffs=False,
                                                     convert_to_equiv_dens=False)
        df_coeffs_dict['spherical_coeffs'] = coeffs_dict['spherical_coeffs']
        dens_coeff = model.property_models['density'](df_coeffs_dict)['density']
        dens_err = torch.sum(torch.abs(dens_coeff - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp_df['atom_numbers'])
        coeff_err = torch.mean(torch.abs(coeffs - samp_df['df_coeffs']))
        new_samp = {key: samp_df[key] for key in samp_df.keys()}
        new_samp['density'] = dens_coeff
        dpm = orbitals.calc_dipole_moment(new_samp)['dipole_moment']
        dpm_err = 4.8 * torch.norm(dpm - samp['dipole_moment'])
        dens_errors.append(float(dens_err))
        coeff_errors.append(float(coeff_err))
        dpm_errors.append(float(dpm_err))
        # print('len dpm errors', len(dpm_errors))
print('df dens errors', dens_errors)
# print('df coeff errors', coeff_errors)
print('df dpm errors', dpm_errors)

# %%
# normalize dens_errors from 0 to 1
dens_errors_norm = (dens_errors - np.min(dens_errors)) / (np.max(dens_errors) - np.min(dens_errors))
dpm_errors_norm = (dpm_errors - np.min(dpm_errors)) / (np.max(dpm_errors) - np.min(dpm_errors))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(dens_errors_norm, label='density error')
axs[0].plot(dpm_errors_norm, label='dpm error')
axs[0].set_xlabel('Number of coefficients included')
axs[0].set_ylabel('Normalized error')
axs[0].set_xticks(ticks=[14, 26, 36, 41, 44, 46], labels = ['L=0', '1', '2', '3', '4', '5'])
axs[1].plot(dens_errors_norm, label='density error')
axs[1].plot(dpm_errors_norm, label='dpm error')
axs[1].set_yscale('log')
axs[1].set_xlabel('Number of coefficients included')
axs[1].set_ylabel('Normalized error (log)')
axs[1].set_xticks(ticks=[14, 26, 36, 41, 44, 46], labels = ['L=0', '1', '2', '3', '4', '5'])
fig.text(0.5, -0.15,
         'These figures show how the errors of the machine learned density and dipole\n' + 
         'moment (normalized to a range of 0 to 1) change as more ML-basis functions are included,\n' +
         'starting from the ones with zero angular degree to the basis functions of highest \n' +
         'angular degree. We see that the dipole moment errors converge faster than the density errors,\n' +
         'already achieving minimal dipole moment errors, while the absolute density error is still\n'
         + 'larger than 4%. Here we see again that the higher spherical harmonic orders do not\n' +
         'further improve the dipole moment, however they are crucial for a good absolute density error.',
         ha='center', va='center', fontsize=12)
plt.tight_layout()
plt.legend()
plt.show()
# %%
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

orbital_dict = orbitals.combine_orbital_basis(dataset.orbital_basis_num, 5)[0]
print('orbital_dict', orbital_dict)
max_orbitals = orbital_dict[16]
model = model_loader.load_model(args, dataset)
all_dens_errors = []
all_dpm_errors = []
for idx in data_split_indices['test'][:100]:
    samp = dataset.get_properties([idx])
    samp_df = dataset_df.get_properties([idx])

    dpm_err = 4.8 * torch.norm(samp_df['dipole_moment'] - samp['dipole_moment'])
    print('dipole moment error', dpm_err)
    dens_errors = []
    coeff_errors = []
    dpm_errors = []

    masks_pos = {key: 0 for key in dataset.orbital_basis_size}

    atom = [(int(samp['batch_atom_numbers'][0, i].detach().cpu().numpy()),
            samp['batch_positions'][0, i].detach().cpu().numpy())
            for i in range(samp['batch_positions'].shape[1])]

    res = model(samp)
    print('res dens error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
    sph_coeffs_vec = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                                    radial_coeffs=False, convert_to_pyscf=False)

    df_coeffs_split = orbitals.split_df_coeffs(atom, sph_coeffs_vec['spherical_coeffs'].squeeze(), dataset.orbital_basis_size)
    max_coeffs_len = max([len(coeffs[1]) for coeffs in df_coeffs_split])
    for L in range(6):
        nc = 2 * L + 1
        max_coeffs = max_orbitals[L][1]
        for i in range(1, max_coeffs + 1):
            for z in masks_pos.keys():
                if L >= len(orbital_dict[z]) or i >= orbital_dict[z][L][1] + 1:
                    continue
                masks_pos[z] += nc
                # print('L', L, 'max coeffs', max_coeffs, 'i', i, 'z', z, 'masks_pos', masks_pos[z])
            mask_coeffs = []
            for z, atom_coeffs in df_coeffs_split:
                atom_coeffs = torch.Tensor(atom_coeffs)
                mask = torch.zeros_like(atom_coeffs)
                mask[:masks_pos[z]] = 1
                # print('z', z, 'mask pos', masks_pos[z], 'mask', mask)
                mask_coeffs.append(atom_coeffs * mask)
            coeffs = torch.cat(mask_coeffs).unsqueeze(0)
            df_coeffs_dict = {key: res[key] for key in res.keys()}
            coeffs_dict = orbitals.vector_to_coeffs_dict({'spherical_coeffs': coeffs}, dataset.orbital_basis_num,
                                                         res['batch_atom_numbers'], radial_coeffs=False,
                                                         convert_to_equiv_dens=False)
            df_coeffs_dict['spherical_coeffs'] = coeffs_dict['spherical_coeffs']
            dens_coeff = model.property_models['density'](df_coeffs_dict)['density']
            dens_err = torch.sum(torch.abs(dens_coeff - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers'])
            new_samp = {key: samp[key] for key in samp.keys()}
            new_samp['density'] = dens_coeff
            dpm = orbitals.calc_dipole_moment(new_samp)['dipole_moment']
            dpm_err = 4.8 * torch.norm(dpm - samp['dipole_moment'])
            dens_errors.append(float(dens_err))
            dpm_errors.append(float(dpm_err))
    all_dens_errors.append(dens_errors)
    all_dpm_errors.append(dpm_errors)

all_dens_errors = np.array(all_dens_errors)
all_dpm_errors = np.array(all_dpm_errors)
# %%
print(all_dpm_errors.shape)
print(all_dpm_errors.mean(axis=0))
print(np.mean(np.min(all_dpm_errors, axis=1)))
print(np.argmin(all_dpm_errors.mean(axis=0)))
print(np.min(all_dpm_errors.mean(axis=0)))

argmin_sample_test = np.argmin(all_dpm_errors, axis=1)
argmin_test = np.argmin(all_dpm_errors.mean(axis=0))
# %%
dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
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

dataset_df = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
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

orbital_dict = orbitals.combine_orbital_basis(dataset.orbital_basis_num, 5)[0]
print('orbital_dict', orbital_dict)
max_orbitals = orbital_dict[16]
model = model_loader.load_model(args, dataset)
all_dens_errors_valid = []
all_dpm_errors_valid = []
for idx in data_split_indices['valid']:
    samp = dataset.get_properties([idx])
    samp_df = dataset_df.get_properties([idx])

    dpm_err = 4.8 * torch.norm(samp_df['dipole_moment'] - samp['dipole_moment'])
    print('dipole moment error', dpm_err)
    dens_errors = []
    coeff_errors = []
    dpm_errors = []

    masks_pos = {key: 0 for key in dataset.orbital_basis_size}

    atom = [(int(samp['batch_atom_numbers'][0, i].detach().cpu().numpy()),
            samp['batch_positions'][0, i].detach().cpu().numpy())
            for i in range(samp['batch_positions'].shape[1])]

    res = model(samp)
    print('res dens error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
    sph_coeffs_vec = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                                    radial_coeffs=False, convert_to_pyscf=False)

    df_coeffs_split = orbitals.split_df_coeffs(atom, sph_coeffs_vec['spherical_coeffs'].squeeze(), dataset.orbital_basis_size)
    max_coeffs_len = max([len(coeffs[1]) for coeffs in df_coeffs_split])
    for L in range(6):
        nc = 2 * L + 1
        max_coeffs = max_orbitals[L][1]
        for i in range(1, max_coeffs + 1):
            for z in masks_pos.keys():
                if L >= len(orbital_dict[z]) or i >= orbital_dict[z][L][1] + 1:
                    continue
                masks_pos[z] += nc
                # print('L', L, 'max coeffs', max_coeffs, 'i', i, 'z', z, 'masks_pos', masks_pos[z])
            mask_coeffs = []
            for z, atom_coeffs in df_coeffs_split:
                atom_coeffs = torch.Tensor(atom_coeffs)
                mask = torch.zeros_like(atom_coeffs)
                mask[:masks_pos[z]] = 1
                # print('z', z, 'mask pos', masks_pos[z], 'mask', mask)
                mask_coeffs.append(atom_coeffs * mask)
            coeffs = torch.cat(mask_coeffs).unsqueeze(0)
            df_coeffs_dict = {key: res[key] for key in res.keys()}
            coeffs_dict = orbitals.vector_to_coeffs_dict({'spherical_coeffs': coeffs}, dataset.orbital_basis_num,
                                                         res['batch_atom_numbers'], radial_coeffs=False,
                                                         convert_to_equiv_dens=False)
            df_coeffs_dict['spherical_coeffs'] = coeffs_dict['spherical_coeffs']
            dens_coeff = model.property_models['density'](df_coeffs_dict)['density']
            dens_err = torch.sum(torch.abs(dens_coeff - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers'])
            new_samp = {key: samp[key] for key in samp.keys()}
            new_samp['density'] = dens_coeff
            dpm = orbitals.calc_dipole_moment(new_samp)['dipole_moment']
            dpm_err = 4.8 * torch.norm(dpm - samp['dipole_moment'])
            dens_errors.append(float(dens_err))
            dpm_errors.append(float(dpm_err))
    all_dens_errors_valid.append(dens_errors)
    all_dpm_errors_valid.append(dpm_errors)

all_dens_errors_valid = np.array(all_dens_errors_valid)
all_dpm_errors_valid = np.array(all_dpm_errors_valid)
# %%
print(all_dpm_errors_valid.shape)
print(all_dpm_errors_valid.mean(axis=0))
print(np.mean(np.min(all_dpm_errors_valid, axis=1)))
print(np.argmin(all_dpm_errors_valid.mean(axis=0)))
print(np.min(all_dpm_errors_valid.mean(axis=0)))

argmin_sample_valid = np.argmin(all_dpm_errors_valid, axis=1)
argmin_valid = np.argmin(all_dpm_errors_valid.mean(axis=0))

# %%
min_sample_valid = np.mean(np.min(all_dpm_errors_valid, axis=1))
min_all_valid = np.min(all_dpm_errors_valid.mean(axis=0))
min_sample_test = np.mean(np.min(all_dpm_errors, axis=1))
min_all_test = np.min(all_dpm_errors.mean(axis=0))
valid_loss = all_dpm_errors_valid.mean(axis=0)[-1]
test_loss = all_dpm_errors.mean(axis=0)[-1]
print('valid_loss', valid_loss, 'min_sample_valid', min_sample_valid, 'min_all_valid', min_all_valid)
print('base to all ratio', min_all_valid / valid_loss,
      'all to sample ratio', min_sample_valid / min_all_valid,
      'base to sample ratio', min_sample_valid / valid_loss)
print('test_loss', all_dpm_errors.mean(axis=0)[-1], 'min_sample_test', min_sample_test, 'min_all_test', min_all_test)
print('base to all ratio', min_all_test / test_loss,
      'all to sample ratio', min_sample_test / min_all_test,
      'base to sample ratio', min_sample_test / test_loss)
min_sample_valid_to_test = np.mean(all_dpm_errors[np.arange(all_dpm_errors.shape[0]), argmin_sample_valid])
min_all_valid_to_test = all_dpm_errors.mean(axis=0)[-1]
print('min_sample_test', min_sample_valid_to_test, 'min_all_test', min_all_valid_to_test, 'ratio', min_sample_valid_to_test / min_all_valid_to_test)
# %%
min_sample_valid = np.mean(np.min(all_dpm_errors_valid, axis=1))
min_all_valid = np.min(all_dpm_errors_valid.mean(axis=0))
min_sample_test = np.mean(np.min(all_dpm_errors, axis=1))
min_all_test = np.min(all_dpm_errors.mean(axis=0))
valid_loss = all_dpm_errors_valid.mean(axis=0)[-1]
test_loss = all_dpm_errors.mean(axis=0)[-1]
print('valid_loss', valid_loss, 'min_sample_valid', min_sample_valid, 'min_all_valid', min_all_valid)
print('base to all ratio', min_all_valid / valid_loss,
      'all to sample ratio', min_sample_valid / min_all_valid,
      'base to sample ratio', min_sample_valid / valid_loss)
print('test_loss', all_dpm_errors.mean(axis=0)[-1], 'min_sample_test', min_sample_test, 'min_all_test', min_all_test)
print('base to all ratio', min_all_test / test_loss,
      'all to sample ratio', min_sample_test / min_all_test,
      'base to sample ratio', min_sample_test / test_loss)
min_sample_valid_to_test = np.mean(all_dpm_errors[np.arange(all_dpm_errors.shape[0]), argmin_sample_valid])
min_all_valid_to_test = all_dpm_errors.mean(axis=0)[-1]
print('min_sample_test', min_sample_valid_to_test, 'min_all_test', min_all_valid_to_test, 'ratio', min_sample_valid_to_test / min_all_valid_to_test)
# %%
print(all_dpm_errors.shape)
print(all_dpm_errors.mean(axis=0))
print(np.argmin(all_dpm_errors.mean(axis=0)))
# %%
print(all_dpm_errors.shape)
print(all_dpm_errors.mean(axis=0))
print(np.argmin(all_dpm_errors.mean(axis=0)))
# %%
print(all_dpm_errors.shape)
print(all_dpm_errors.mean(axis=0))
print(np.argmin(all_dpm_errors.mean(axis=0)))
# %%
print(all_dpm_errors.shape)
print(all_dpm_errors.mean(axis=0))
print(np.argmin(all_dpm_errors.mean(axis=0)))
# %%
# normalize dens_errors from 0 to 1
dens_errors = all_dens_errors.mean(axis=0)
dpm_errors = all_dpm_errors.mean(axis=0)
dens_errors_norm = (dens_errors - np.min(dens_errors)) / (np.max(dens_errors) - np.min(dens_errors))
# dpm_errors_norm = (dpm_errors) / (np.max(dpm_errors))
dpm_errors_norm = (dpm_errors - np.min(dpm_errors)) / (np.max(dpm_errors) - np.min(dpm_errors))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(dens_errors_norm, label='density error')
axs[0].plot(dpm_errors_norm, label='dpm error')
axs[0].set_xlabel('Number of coefficients included')
axs[0].set_ylabel('Normalized error')
axs[0].set_xticks(ticks=[14, 26, 36, 41, 44, 46], labels = ['L=0', '1', '2', '3', '4', '5'])
axs[1].plot(dens_errors_norm, label='density error')
axs[1].plot(dpm_errors_norm, label='dpm error')
axs[1].set_yscale('log')
axs[1].set_xlabel('Number of coefficients included')
axs[1].set_ylabel('Normalized error (log)')
axs[1].set_xticks(ticks=[14, 26, 36, 41, 44, 46], labels = ['L=0', '1', '2', '3', '4', '5'])
fig.text(0.5, -0.15,
         'These figures show how the errors of the machine learned density and dipole\n' + 
         'moment (normalized to a range of 0 to 1) change as more ML-basis functions are included,\n' +
         'starting from the ones with zero angular degree to the basis functions of highest \n' +
         'angular degree. We see that the dipole moment errors converge faster than the density errors,\n' +
         'already achieving minimal dipole moment errors, while the absolute density error is still\n'
         + 'larger than 4%. Here we see again that the higher spherical harmonic orders do not\n' +
         'further improve the dipole moment, however they are crucial for a good absolute density error.',
         ha='center', va='center', fontsize=12)
plt.tight_layout()
plt.legend()
plt.show()

# %%
orbital_dict = orbitals.combine_orbital_basis(dataset.orbital_basis_num, 5)[0]
print('orbital_dict', orbital_dict)
max_orbitals = orbital_dict[16]
model = model_loader.load_model(args, dataset)
all_dens_errors = []
all_dpm_errors = []
for idx in data_split_indices['valid']:
    samp = dataset.get_properties([idx])
    samp_df = dataset_df.get_properties([idx])

    dpm_err = 4.8 * torch.norm(samp_df['dipole_moment'] - samp['dipole_moment'])
    print('dipole moment error', dpm_err)
    dens_errors = []
    coeff_errors = []
    dpm_errors = []

    masks_pos = {key: 0 for key in dataset.orbital_basis_size}

    atom = [(int(samp['batch_atom_numbers'][0, i].detach().cpu().numpy()),
            samp['batch_positions'][0, i].detach().cpu().numpy())
            for i in range(samp['batch_positions'].shape[1])]

    res = model(samp)
    print('res dens error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
    sph_coeffs_vec = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                                    radial_coeffs=False, convert_to_pyscf=False)

    df_coeffs_split = orbitals.split_df_coeffs(atom, sph_coeffs_vec['spherical_coeffs'].squeeze(), dataset.orbital_basis_size)
    max_coeffs_len = max([len(coeffs[1]) for coeffs in df_coeffs_split])
    for L in range(6):
        nc = 2 * L + 1
        max_coeffs = max_orbitals[L][1]
        for i in range(1, max_coeffs + 1):
            for z in masks_pos.keys():
                if L >= len(orbital_dict[z]) or i >= orbital_dict[z][L][1] + 1:
                    continue
                masks_pos[z] += nc
                # print('L', L, 'max coeffs', max_coeffs, 'i', i, 'z', z, 'masks_pos', masks_pos[z])
            mask_coeffs = []
            for z, atom_coeffs in df_coeffs_split:
                atom_coeffs = torch.Tensor(atom_coeffs)
                mask = torch.zeros_like(atom_coeffs)
                mask[:masks_pos[z]] = 1
                # print('z', z, 'mask pos', masks_pos[z], 'mask', mask)
                mask_coeffs.append(atom_coeffs * mask)
            coeffs = torch.cat(mask_coeffs).unsqueeze(0)
            df_coeffs_dict = {key: res[key] for key in res.keys()}
            coeffs_dict = orbitals.vector_to_coeffs_dict({'spherical_coeffs': coeffs}, dataset.orbital_basis_num,
                                                         res['batch_atom_numbers'], radial_coeffs=False,
                                                         convert_to_equiv_dens=False)
            df_coeffs_dict['spherical_coeffs'] = coeffs_dict['spherical_coeffs']
            dens_coeff = model.property_models['density'](df_coeffs_dict)['density']
            dens_err = torch.sum(torch.abs(dens_coeff - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers'])
            new_samp = {key: samp[key] for key in samp.keys()}
            new_samp['density'] = dens_coeff
            dpm = orbitals.calc_dipole_moment(new_samp)['dipole_moment']
            dpm_err = 4.8 * torch.norm(dpm - samp['dipole_moment'])
            dens_errors.append(float(dens_err))
            dpm_errors.append(float(dpm_err))
    all_dens_errors.append(dens_errors)
    all_dpm_errors.append(dpm_errors)

all_dens_errors = np.array(all_dens_errors)
all_dpm_errors = np.array(all_dpm_errors)
