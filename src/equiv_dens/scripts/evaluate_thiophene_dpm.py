import numpy as np
import os
from datetime import datetime

import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling, \
     CubicalGrid, spherical_grid, spherical_radial_sampling
import equiv_dens.utils.base as utils
from equiv_dens.utils.misc import generate_id

from functools import partial
import argparse
from equiv_dens.training import density_errors
import matplotlib.pyplot as plt


def make_plots(dens_ml, dens_df, sample_ref, save_name):
    center_of_mass = torch.sum(sample_ref['batch_positions'] * sample_ref['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
                    / torch.sum(sample_ref['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

    # print(center_of_mass)
    # print(center_of_mass.shape)
    #
    # print(sample_ref['coords'].shape)
    distance_from_com = torch.norm(sample_ref['coords'] - center_of_mass, dim=2)

    # print(distance_from_com.shape)
    plt.rcParams['text.usetex'] = True
    dens_df_mae = torch.abs(sample_ref['density'] - dens_df) * sample_ref['coord_weights']
    dens_ml_mae = torch.abs(sample_ref['density'] - dens_ml) * sample_ref['coord_weights']

    # dens_ml_LDA = torch.abs(density_errors.density_LDA_loss(dens_ml, sample_ref['density'], sample_ref['coord_weights']))
    # dens_df_LDA = torch.abs(density_errors.density_LDA_loss(dens_df, sample_ref['density'], sample_ref['coord_weights']))
    #
    # print('ml LDA MAE', dens_ml_LDA)
    # print('df LDA MAE', dens_df_LDA)
    #
    # ml_max = torch.argmax(dens_ml_mae[0])
    # df_max = torch.argmax(dens_df_mae[0])
    #
    # print('ml_mae', torch.sum(dens_ml_mae))
    # print('df_mae', torch.sum(dens_df_mae))
    #
    # dens_ml_coul = torch.abs(density_errors._density_coulomb_loss(dens_ml - sample_ref['density'], sample_ref['coords'], sample_ref['coord_weights']))
    # dens_df_coul = torch.abs(density_errors._density_coulomb_loss(dens_df - sample_ref['density'], sample_ref['coords'], sample_ref['coord_weights']))
    #
    # print('ml coulomb', dens_ml_coul)
    # print('df coulomb', dens_df_coul)

    # print('ml mae [ml_max]', dens_ml_mae[:, ml_max])
    # print('df mae [ml_max]', dens_df_mae[:, ml_max])
    # print('scale [ml_max]', sample_ref['coord_weights'][:, ml_max])
    #
    # print('ml mae [df_max]', dens_ml_mae[:, df_max])
    # print('df mae [df_max]', dens_df_mae[:, df_max])
    # print('scale [df_max]', sample_ref['coord_weights'][:, df_max])

    # print(sample_ref['coords'].shape)
    # print(sample_ref['positions'].shape)
    distance_from_atoms = torch.norm(sample_ref['coords'].unsqueeze(1) - sample_ref['positions'].unsqueeze(2), dim=-1)

    # print(distance_from_atoms.shape)
    min_distance_from_atoms = torch.min(distance_from_atoms, dim=1)[0]
    # print(min_distance_from_atoms.shape)

    fig, axs = plt.subplots(3, 2, figsize=(7, 7))
    # print('axs', axs.shape)

    axs[1, 0].scatter(dens_df_mae.squeeze(), distance_from_com, label='DF', c='blue')
    axs[1, 0].scatter(dens_ml_mae.squeeze(), distance_from_com, label='ML', c='orange')

    axs[1, 1].scatter(dens_df_mae.squeeze(), min_distance_from_atoms, label='DF', c='blue')
    axs[1, 1].scatter(dens_ml_mae.squeeze(), min_distance_from_atoms, label='ML', c='orange')

    axs[1, 1].set_xlabel('absolute density error at gridpoint,\n scaled by volume of gridpoint')
    axs[1, 0].set_xlabel('absolute density error at gridpoint,\n scaled by volume of gridpoint')
    axs[1, 0].set_ylabel('distance from center of mass')
    axs[1, 1].set_ylabel('distance from closest atom')
    # axs[1, 0].set_title('ML scaled MAE')
    # axs[1, 1].set_title('DF scaled MAE')
    # fig.suptitle('Scaled absolute density error', fontsize=16)

    dens_df_mse = (sample_ref['density'] - dens_df)**2 * sample_ref['coord_weights']
    dens_ml_mse = (sample_ref['density'] - dens_ml)**2 * sample_ref['coord_weights']

    # print('ml mse max', torch.max(dens_ml_mse))

    # print('df mse max', torch.max(dens_df_mse))

    axs[2, 0].scatter(dens_ml_mse.squeeze(), distance_from_com, label='ML', c='orange')
    axs[2, 0].scatter(dens_df_mse.squeeze(), distance_from_com, label='DF', c='blue')

    axs[2, 1].scatter(dens_ml_mse.squeeze(), min_distance_from_atoms, label='ML', c='orange')
    axs[2, 1].scatter(dens_df_mse.squeeze(), min_distance_from_atoms, label='DF', c='blue')

    axs[2, 1].set_xlabel('squared density error at gridpoint,\n scaled by volume of gridpoint')
    axs[2, 0].set_xlabel('squared density error at gridpoint,\n scaled by volume of gridpoint')
    axs[2, 0].set_ylabel('distance from center of mass')
    axs[2, 1].set_ylabel('distance from closest atom')
    # axs[2, 0].set_title('ML scaled MSE')
    # axs[2, 1].set_title('DF scaled MSE')

    dens_df_mae = torch.abs(sample_ref['density'] - dens_df)
    dens_ml_mae = torch.abs(sample_ref['density'] - dens_ml)

    axs[0, 0].scatter(dens_ml_mae.squeeze(), distance_from_com, label='ML', c='orange')
    axs[0, 0].scatter(dens_df_mae.squeeze(), distance_from_com, label='DF', c='blue')

    axs[0, 1].scatter(dens_ml_mae.squeeze(), min_distance_from_atoms, label='ML', c='orange')
    axs[0, 1].scatter(dens_df_mae.squeeze(), min_distance_from_atoms, label='DF', c='blue')

    axs[0, 1].set_xlabel('absolute density error at gridpoint')
    axs[0, 0].set_xlabel('absolute density error at gridpoint')
    axs[0, 0].set_ylabel('distance from center of mass')
    axs[0, 1].set_ylabel('distance from closest atom')
    # axs[0, 0].set_title('ML unscaled MAE')
    # axs[0, 1].set_title('DF unscaled MAE')

    plt.legend()
    plt.tight_layout()

    # plt.show()
    plt.savefig('figures/' + save_name + '_density_errors.png', dpi=300)


parser = argparse.ArgumentParser()
parser.add_argument('args_file', type=str)
parser.add_argument('ref_np_load_file', type=str)
parser.add_argument('ref_dens_load_file', type=str)
parser.add_argument('res_load_file', type=str)
parser.add_argument('save_file', type=str)
parser.add_argument('--df_error', action='store_true', default=False)
parser.add_argument('--use_gpu', action='store_true', default=False)
parser.add_argument('--num_samples', type=int, default=-1)
parser.add_argument('--make_plots', action='store_true', default=False)

main_args = parser.parse_args()

args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)

# print('type dtype', type(args.dtype))
args.fix_arguments = True
# print('args np dir', args.np_dataset)

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
                # print('loading hyperparam arg', arg)
                setattr(args, arg, getattr(checkpoint['args'], arg))
        else:
            # print('loading all arg', arg)
            setattr(args, arg, getattr(checkpoint['args'], arg))
    restore = True


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
args.np_dataset_test = main_args.ref_np_load_file
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
if main_args.df_error:
    df_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
                 'dpm_rmse': [],
                 'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
                 'mae_23': [], 'mae_43': [], 'lda_23_mae': [],
                 'coulomb': []}

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

    np.save('datasets/' + main_args.save_file + '_df_losses.npy', df_losses, allow_pickle=True)
    print('DF losses')
    for key in df_losses.keys():
        print(key)
        print(np.nanmean(df_losses[key]))

res_dataset = torch.load(main_args.res_load_file, map_location='cpu')
print('res dataset length', len(res_dataset))
print('res sample pos shape', res_dataset['positions'].shape)
print('res sample dens shape', res_dataset['density'].shape)
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

    # if (i + 1) % 10 == 0:
    #    for j in range(len(mers_mae)):
    #        print('mer', j)
    #        print(np.mean(mers_mae[j]))
    #        print(np.mean(mers_dpm[j]))
    #        print(np.mean(mers_dpm_mag[j]))
    #        print(np.mean(mers_dpm_ang[j]))
    #        print(np.mean(mers_lda[j]))

np.save('datasets/' + main_args.save_file + '.npy', res_losses, allow_pickle=True)
print('Results losses')
for key in res_losses.keys():
    print(key)
    print(np.nanmean(res_losses[key]))
print('true dens max', torch.max(sample['density']))
print('DF dens max', torch.max(sample_df['density']))
print('ML dens max', torch.max(r_dens))
if main_args.make_plots and main_args.df_error:
    make_plots(r_dens, sample_df['density'], sample, main_args.save_file)
