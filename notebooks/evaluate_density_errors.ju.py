# %% import numpy as np
import os
from datetime import datetime

import torch
from pyscf.dft import numint
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

# %load_ext autoreload
# %autoreload 2
# %cd /home/mihail/Documents/workspace/equiv_dens/

# %%
def calc_density_errors(density, atoms, error_dict):
    r_dens = torch.clamp(density, min=0)
    r_dens = r_dens / torch.sum(r_dens * atoms['coord_weights']) * torch.sum(atoms['atom_numbers'])

    tmp_atoms = {key: atoms[key] for key in atoms}
    tmp_atoms['density'] = r_dens
    dpm = orbitals.calc_dipole_moment(tmp_atoms, center_coordinates=True)['dipole_moment']

    df_error = torch.sum(torch.abs(atoms['density'] - r_dens) * atoms['coord_weights'])/torch.sum(atoms['atom_numbers'])
    df2_error = torch.sqrt(torch.sum((atoms['density'] - r_dens)**2 * atoms['coord_weights'])/torch.sum(atoms['atom_numbers']))

    dpm_error = 4.8 * torch.mean(torch.abs(dpm - atoms['dipole_moment']))
    dpm2_error = 4.8 * torch.norm(dpm - atoms['dipole_moment'])

    dpm_mag_error = 4.8 * torch.mean(torch.abs(torch.linalg.norm(dpm) - torch.linalg.norm(atoms['dipole_moment'])))
    dpm_ang_error = (180 / torch.pi) * torch.acos(torch.mean(torch.sum(dpm*atoms['dipole_moment']) /
                                                  (torch.linalg.norm(dpm) * torch.linalg.norm(atoms['dipole_moment']))))

    df43_error = torch.sum(torch.abs(atoms['density']**(4/3) -
                           r_dens**(4/3)) * atoms['coord_weights'])
    df23_error = torch.sum(torch.abs(atoms['density']**(2/3) -
                           r_dens**(2/3)) * atoms['coord_weights'])
    lda_error = torch.abs(torch.sum((atoms['density']**(4/3) -
                          r_dens**(4/3)) * atoms['coord_weights']))
    lda23_error = torch.abs(torch.sum((atoms['density']**(2/3) -
                            r_dens**(2/3)) * atoms['coord_weights']))
    coul_error = density_errors._density_coulomb_loss(r_dens - atoms['density'], atoms['coords'], atoms['coord_weights'])

    coul_int_error = torch.abs(density_errors.density_hartree_loss(r_dens, atoms['density'], atoms['coords'], atoms['coord_weights']))

    center_of_mass = torch.sum(atoms['batch_positions'] * atoms['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
        / torch.sum(atoms['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)
    # absolute dipole moment error at every grid point, RMSE
    dpm_coord_error = torch.sqrt(torch.sum(torch.sum((torch.abs(atoms['density'] - r_dens) * atoms['coord_weights']).unsqueeze(-1) *
                                 (atoms['coords'] - center_of_mass), dim=1)**2))
    # positive dipole moment error at every grid point, RMSE
    dpm_pos_coord_error = torch.sqrt(torch.sum(torch.sum((((atoms['density'] - r_dens) * ((atoms['density'] - r_dens) > 0)) * atoms['coord_weights']).unsqueeze(-1) *
                                     (atoms['coords'] - center_of_mass), dim=1)**2))
    # negative dipole moment error at every grid point, RMSE
    dpm_neg_coord_error = torch.sqrt(torch.sum(torch.sum((((atoms['density'] - r_dens) * ((atoms['density'] - r_dens) < 0)) * atoms['coord_weights']).unsqueeze(-1) *
                                     (atoms['coords'] - center_of_mass), dim=1)**2))
    # unsigned dipole moment error at every grid point, RMSE
    # dpm_int_error = torch.sqrt(torch.sum(torch.sum(((atoms['density'] - r_dens) * atoms['coord_weights']).unsqueeze(-1) *
    #                            (atoms['coords'] - center_of_mass), dim=1)**2))
    # dpm_int_error = torch.sum(((atoms['density'] - r_dens) * atoms['coord_weights']) *
    #                           torch.norm(atoms['coords'] - center_of_mass), dim=2)
    dpm_int_error = torch.norm(torch.sum(((atoms['density'] - r_dens) * atoms['coord_weights']).unsqueeze(-1) *
                               (atoms['coords'] - center_of_mass), dim=1))

    kl_error = density_errors.density_KL_loss(r_dens, atoms['density'], atoms['atom_numbers'], atoms['coord_weights'])

    error_dict['dens_mae'].append(float(df_error))
    error_dict['dens_rmse'].append(float(df2_error))
    error_dict['dpm_mae'].append(float(dpm_error))
    error_dict['dpm_rmse'].append(float(dpm2_error))
    error_dict['dpm_mag'].append(float(dpm_mag_error))
    error_dict['dpm_ang'].append(float(dpm_ang_error))
    error_dict['lda_mae'].append(float(lda_error))
    error_dict['lda_23_mae'].append(float(lda23_error))
    error_dict['coulomb'].append(float(coul_error))
    error_dict['coulomb_int'].append(float(coul_int_error))
    error_dict['mae_43'].append(float(df43_error))
    error_dict['mae_23'].append(float(df23_error))
    error_dict['dpm_coord_rmse'].append(float(dpm_coord_error))
    error_dict['dpm_pos_coord_rmse'].append(float(dpm_pos_coord_error))
    error_dict['dpm_neg_coord_rmse'].append(float(dpm_neg_coord_error))
    error_dict['dpm_int_rmse'].append(float(dpm_int_error))
    error_dict['kl_loss'].append(float(kl_error))
    # print('i', i, 'mae', df_error, 'rmse', df2_error, 'dpm', dpm_error, 'mag', dpm_mag_error, 'ang', dpm_ang_error, 'lda', lda_error)


def calc_density_grad_errors(density, atoms, error_dict):
    dens_true = torch.permute(torch.cat([atoms['density'].unsqueeze(-1), atoms['density_grad']], dim=-1), (2, 1, 0)).squeeze()
    ni = numint.NumInt()
    exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', dens_true.numpy(force=True).astype(np.double), deriv=1, xctype='GGA')[0])
    dens_calc = dens_true[0] * atoms['coord_weights']
    # print('exchange correlation', exc_eff_pbe.shape)
    exc_pbe = torch.sum(exc_eff_pbe * dens_calc)
    exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', density.numpy(force=True).astype(np.double), deriv=1, xctype='GGA')[0])
    dens_calc = density[0] * atoms['coord_weights']
    # print('exchange correlation', exc_eff_pbe.shape)
    exc_sum_pbe = torch.sum(exc_eff_pbe * dens_calc)
    error_dict['exc_mae'].append(float(torch.abs(exc_sum_pbe - exc_pbe)))

    vw_pred = torch.nansum(torch.norm(density[1:], dim=0)**2 / density * atoms['coord_weights'].squeeze())
    vw_true = torch.nansum(torch.norm(dens_true[1:], dim=0)**2 / dens_true * atoms['coord_weights'].squeeze())
    error_dict['vw_mae'].append(float(torch.abs(vw_pred - vw_true)))
    error_dict['grad_norm_err'].append(float(torch.sum(torch.norm(density[1:] - dens_true[1:], dim=0) * atoms['coord_weigts'].squeeze())
                                             / torch.sum(atoms['atom_numbers'])))

# %%
def make_plots(dens_ml, dens_df, sample_ref, save_name):
    dens_ml = torch.clamp(dens_ml, min=0)
    dens_ml = dens_ml / torch.sum(dens_ml * sample_ref['coord_weights']) * torch.sum(sample_ref['atom_numbers'])

    dens_df = torch.clamp(dens_df, min=0)
    dens_df = dens_df / torch.sum(dens_df * sample_ref['coord_weights']) * torch.sum(sample_ref['atom_numbers'])

    center_of_mass = torch.sum(sample_ref['batch_positions'] * sample_ref['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
                    / torch.sum(sample_ref['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

    distance_from_com = torch.norm(sample_ref['coords'] - center_of_mass, dim=2)

    plt.rcParams['text.usetex'] = True

    dens_df_mae = torch.abs(dens_df - sample_ref['density']) * sample_ref['coord_weights']
    dens_ml_mae = torch.abs(dens_ml - sample_ref['density']) * sample_ref['coord_weights']
    # calculate difference between DF and ML errors
    dens_mae_diff = dens_df_mae - dens_ml_mae
    distance_from_atoms = torch.norm(sample_ref['coords'].unsqueeze(1) - sample_ref['positions'].unsqueeze(2), dim=-1)

    min_distance_from_atoms = torch.min(distance_from_atoms, dim=1)[0]

    fig, axs = plt.subplots(5, 4, figsize=(12, 14))

    axs[1, 0].scatter(dens_df_mae.squeeze(), distance_from_com, label='DF', c='blue')
    axs[1, 0].scatter(dens_ml_mae.squeeze(), distance_from_com, label='ML', c='orange')

    axs[1, 1].scatter(dens_df_mae.squeeze(), min_distance_from_atoms, label='DF', c='blue')
    axs[1, 1].scatter(dens_ml_mae.squeeze(), min_distance_from_atoms, label='ML', c='orange')

    axs[1, 0].set_xlabel('absolute density error at gridpoint,\n scaled by volume of gridpoint')
    axs[1, 1].set_xlabel('absolute density error at gridpoint,\n scaled by volume of gridpoint')
    axs[1, 0].set_ylabel('distance from center of mass')
    axs[1, 1].set_ylabel('distance from closest atom')

    diff_gz = dens_mae_diff >= 0
    diff_lz = dens_mae_diff < 0

    axs[1, 2].scatter(dens_mae_diff[diff_gz].squeeze(), distance_from_com[diff_gz], label='DF_err>ML_err')
    axs[1, 2].scatter(dens_mae_diff[diff_lz].squeeze(), distance_from_com[diff_lz], label='ML_err>DF_err')

    axs[1, 3].scatter(dens_mae_diff[diff_gz].squeeze(), min_distance_from_atoms[diff_gz], label='DF_err>ML_err')
    axs[1, 3].scatter(dens_mae_diff[diff_lz].squeeze(), min_distance_from_atoms[diff_lz], label='ML_err>DF_err')

    axs[1, 2].set_xlabel('DF_MAE - ML_MAE at gridpoint,\n scaled by volume of gridpoint')
    axs[1, 3].set_xlabel('DF_MAE - ML_MAE at gridpoint,\n scaled by volume of gridpoint')
    axs[1, 2].set_ylabel('distance from center of mass')
    axs[1, 3].set_ylabel('distance from closest atom')

    dens_df_mse = (dens_df - sample_ref['density'])**2 * sample_ref['coord_weights']
    dens_ml_mse = (dens_ml - sample_ref['density'])**2 * sample_ref['coord_weights']
    dens_mse_diff = dens_df_mse - dens_ml_mse

    axs[2, 0].scatter(dens_ml_mse.squeeze(), distance_from_com, label='ML', c='orange')
    axs[2, 0].scatter(dens_df_mse.squeeze(), distance_from_com, label='DF', c='blue')

    axs[2, 1].scatter(dens_ml_mse.squeeze(), min_distance_from_atoms, label='ML', c='orange')
    axs[2, 1].scatter(dens_df_mse.squeeze(), min_distance_from_atoms, label='DF', c='blue')

    axs[2, 1].set_xlabel('squared density error at gridpoint,\n scaled by volume of gridpoint')
    axs[2, 0].set_xlabel('squared density error at gridpoint,\n scaled by volume of gridpoint')
    axs[2, 0].set_ylabel('distance from center of mass')
    axs[2, 1].set_ylabel('distance from closest atom')

    diff_gz = dens_mse_diff >= 0
    diff_lz = dens_mse_diff < 0

    axs[2, 2].scatter(dens_mse_diff[diff_gz].squeeze(), distance_from_com[diff_gz], label='DF_err>ML_err')
    axs[2, 2].scatter(dens_mse_diff[diff_lz].squeeze(), distance_from_com[diff_lz], label='ML_err>DF_err')

    axs[2, 3].scatter(dens_mse_diff[diff_gz].squeeze(), min_distance_from_atoms[diff_gz], label='DF_err>ML_err')
    axs[2, 3].scatter(dens_mse_diff[diff_lz].squeeze(), min_distance_from_atoms[diff_lz], label='ML_err>DF_err')

    axs[2, 2].set_xlabel('DF_MSE - ML_MSE at gridpoint,\n scaled by volume of gridpoint')
    axs[2, 3].set_xlabel('DF_MSE - ML_MSE at gridpoint,\n scaled by volume of gridpoint')
    axs[2, 2].set_ylabel('distance from center of mass')
    axs[2, 3].set_ylabel('distance from closest atom')

    dens_df_mae = torch.abs(dens_df - sample_ref['density'])
    dens_ml_mae = torch.abs(dens_ml - sample_ref['density'])
    dens_mae_diff = dens_df_mae - dens_ml_mae

    axs[0, 0].scatter(dens_ml_mae.squeeze(), distance_from_com, label='ML', c='orange')
    axs[0, 0].scatter(dens_df_mae.squeeze(), distance_from_com, label='DF', c='blue')

    axs[0, 1].scatter(dens_ml_mae.squeeze(), min_distance_from_atoms, label='ML', c='orange')
    axs[0, 1].scatter(dens_df_mae.squeeze(), min_distance_from_atoms, label='DF', c='blue')

    axs[0, 1].set_xlabel('absolute density error at gridpoint')
    axs[0, 0].set_xlabel('absolute density error at gridpoint')
    axs[0, 0].set_ylabel('distance from center of mass')
    axs[0, 1].set_ylabel('distance from closest atom')

    diff_gz = dens_mae_diff >= 0
    diff_lz = dens_mae_diff < 0

    axs[0, 2].scatter(dens_mae_diff[diff_gz].squeeze(), distance_from_com[diff_gz], label='DF_err>ML_err')
    axs[0, 2].scatter(dens_mae_diff[diff_lz].squeeze(), distance_from_com[diff_lz], label='ML_err>DF_err')

    axs[0, 3].scatter(dens_mae_diff[diff_gz].squeeze(), min_distance_from_atoms[diff_gz], label='DF_err>ML_err')
    axs[0, 3].scatter(dens_mae_diff[diff_lz].squeeze(), min_distance_from_atoms[diff_lz], label='ML_err>DF_err')

    axs[0, 2].set_xlabel('DF_MAE - ML_MAE at gridpoint,')
    axs[0, 3].set_xlabel('DF_MAE - ML_MAE at gridpoint,')
    axs[0, 2].set_ylabel('distance from center of mass')
    axs[0, 3].set_ylabel('distance from closest atom')
    dpm_df_mae = torch.sum((dens_df - sample_ref['density']).unsqueeze(-1) * sample_ref['coords'], dim=2) * sample_ref['coord_weights']
    dpm_ml_mae = torch.sum((dens_ml - sample_ref['density']).unsqueeze(-1) * sample_ref['coords'], dim=2) * sample_ref['coord_weights']
    # calculate difference between DF and ML errors
    dpm_mae_diff = dpm_df_mae - dpm_ml_mae

    axs[3, 0].scatter(dpm_df_mae.squeeze(), distance_from_com, label='DF', c='blue')
    axs[3, 0].scatter(dpm_ml_mae.squeeze(), distance_from_com, label='ML', c='orange')

    axs[3, 1].scatter(dpm_df_mae.squeeze(), min_distance_from_atoms, label='DF', c='blue')
    axs[3, 1].scatter(dpm_ml_mae.squeeze(), min_distance_from_atoms, label='ML', c='orange')

    axs[3, 1].set_xlabel('absolute dipole moment error at gridpoint,\n scaled by volume of gridpoint')
    axs[3, 0].set_xlabel('absolute dipole moment error at gridpoint,\n scaled by volume of gridpoint')
    axs[3, 0].set_ylabel('distance from center of mass')
    axs[3, 1].set_ylabel('distance from closest atom')

    diff_gz = dpm_mae_diff >= 0
    diff_lz = dpm_mae_diff < 0

    axs[3, 2].scatter(dpm_mae_diff[diff_gz].squeeze(), distance_from_com[diff_gz], label='DF_err>ML_err')
    axs[3, 2].scatter(dpm_mae_diff[diff_lz].squeeze(), distance_from_com[diff_lz], label='ML_err>DF_err')

    axs[3, 3].scatter(dpm_mae_diff[diff_gz].squeeze(), min_distance_from_atoms[diff_gz], label='DF_err>ML_err')
    axs[3, 3].scatter(dpm_mae_diff[diff_lz].squeeze(), min_distance_from_atoms[diff_lz], label='ML_err>DF_err')

    axs[3, 2].set_xlabel('DF_DPM_MAE - ML_DPM_MAE at gridpoint,\n scaled by volume of gridpoint')
    axs[3, 3].set_xlabel('DF_DPM_MAE - ML_DPM_MAE at gridpoint,\n scaled by volume of gridpoint')
    axs[3, 2].set_ylabel('distance from center of mass')
    axs[3, 3].set_ylabel('distance from closest atom')

    dens_df_err = (dens_df - sample_ref['density']) * sample_ref['coord_weights']
    dens_ml_err = (dens_ml - sample_ref['density']) * sample_ref['coord_weights']
    # calculate difference between DF and ML errors
    dens_err_diff = torch.abs(dens_df_err) - torch.abs(dens_ml_err)

    axs[4, 0].scatter(dens_df_err.squeeze(), distance_from_com, label='DF', c='blue')
    axs[4, 0].scatter(dens_ml_err.squeeze(), distance_from_com, label='ML', c='orange')

    axs[4, 1].scatter(dens_df_err.squeeze(), min_distance_from_atoms, label='DF', c='blue')
    axs[4, 1].scatter(dens_ml_err.squeeze(), min_distance_from_atoms, label='ML', c='orange')

    axs[4, 0].set_xlabel('density error at gridpoint,\n scaled by volume of gridpoint')
    axs[4, 1].set_xlabel('density error at gridpoint,\n scaled by volume of gridpoint')
    axs[4, 0].set_ylabel('distance from center of mass')
    axs[4, 1].set_ylabel('distance from closest atom')

    diff_gz = dens_err_diff >= 0
    diff_lz = dens_err_diff < 0

    axs[4, 2].scatter(dens_err_diff[diff_gz].squeeze(), distance_from_com[diff_gz], label='DF_err>ML_err')
    axs[4, 2].scatter(dens_err_diff[diff_lz].squeeze(), distance_from_com[diff_lz], label='ML_err>DF_err')

    axs[4, 3].scatter(dens_err_diff[diff_gz].squeeze(), min_distance_from_atoms[diff_gz], label='DF_err>ML_err')
    axs[4, 3].scatter(dens_err_diff[diff_lz].squeeze(), min_distance_from_atoms[diff_lz], label='ML_err>DF_err')

    axs[4, 2].set_xlabel('DF_ERR - ML_ERR at gridpoint,\n scaled by volume of gridpoint')
    axs[4, 3].set_xlabel('DF_ERR - ML_ERR at gridpoint,\n scaled by volume of gridpoint')
    axs[4, 2].set_ylabel('distance from center of mass')
    axs[4, 3].set_ylabel('distance from closest atom')

    plt.legend()
    plt.tight_layout()

    plt.show()
    # plt.savefig('figures/' + save_name + '_density_errors_ext.png', dpi=300)
# %%
def make_scatter_plots(res_density, dataset_df, dataset, save_name):
    dens_df_mae = []
    dens_ml_mae = []
    dpm_df_rmse = []
    dpm_ml_rmse = []
    lda_df_mae = []
    lda_ml_mae = []
    dpm_coord_df_rmse = []
    dpm_coord_ml_rmse = []
    for i in range(len(dataset)):
        print('i', i)
        sample_ref = dataset.get_properties([i])
        dens_df = dataset_df.get_properties([i])['density']
        dens_ml = res_density[[i]]
        print('dens df shape', dens_df.shape)
        print('dens ml shape', dens_ml.shape)
        dens_ml = torch.clamp(dens_ml, min=0)
        dens_ml = dens_ml / torch.sum(dens_ml * sample_ref['coord_weights']) * torch.sum(sample_ref['atom_numbers'])

        dens_df = torch.clamp(dens_df, min=0)
        dens_df = dens_df / torch.sum(dens_df * sample_ref['coord_weights']) * torch.sum(sample_ref['atom_numbers'])

        center_of_mass = torch.sum(sample_ref['batch_positions'] * sample_ref['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
                        / torch.sum(sample_ref['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

        distance_from_com = torch.norm(sample_ref['coords'] - center_of_mass, dim=2)

        distance_from_atoms = torch.norm(sample_ref['coords'].unsqueeze(1) - sample_ref['positions'].unsqueeze(2), dim=-1)
        min_distance_from_atoms = torch.min(distance_from_atoms, dim=1)[0]

        tmp_atoms = {key: sample_ref[key] for key in sample_ref}
        tmp_atoms['density'] = dens_ml
        dpm_ml = orbitals.calc_dipole_moment(tmp_atoms, center_coordinates=True)['dipole_moment']
        tmp_atoms = {key: sample_ref[key] for key in sample_ref}
        tmp_atoms['density'] = dens_df
        dpm_df = orbitals.calc_dipole_moment(tmp_atoms, center_coordinates=True)['dipole_moment']

        dens_df_mae.append(torch.sum(torch.abs(sample_ref['density'] - dens_df) * sample_ref['coord_weights'])/torch.sum(sample_ref['atom_numbers']))
        dens_ml_mae.append(torch.sum(torch.abs(sample_ref['density'] - dens_ml) * sample_ref['coord_weights'])/torch.sum(sample_ref['atom_numbers']))

        dpm_df_rmse.append(4.8 * torch.norm(dpm_df - sample_ref['dipole_moment']))
        dpm_ml_rmse.append(4.8 * torch.norm(dpm_ml - sample_ref['dipole_moment']))

        lda_df_mae.append(torch.abs(torch.sum((sample_ref['density']**(4/3) -
                          dens_df**(4/3)) * sample_ref['coord_weights'])))
        lda_ml_mae.append(torch.abs(torch.sum((sample_ref['density']**(4/3) -
                          dens_ml**(4/3)) * sample_ref['coord_weights'])))

        # absolute dipole moment error at every grid point, RMSE
        dpm_coord_df_rmse.append(torch.sum(torch.norm((sample_ref['density'] - dens_df).unsqueeze(-1) *
                                 (sample_ref['coords'] - center_of_mass), dim=2) * sample_ref['coord_weights']))
        dpm_coord_ml_rmse.append(torch.sum(torch.norm((sample_ref['density'] - dens_ml).unsqueeze(-1) *
                                 (sample_ref['coords'] - center_of_mass), dim=2) * sample_ref['coord_weights']))

    plt.rcParams['text.usetex'] = True

    fig, axs = plt.subplots(1, 4, figsize=(10, 2))

    # absolute density errors scaled
    axs[0].scatter(dens_df_mae, dens_ml_mae, s=1)

    axs[1].scatter(dpm_df_rmse, dpm_ml_rmse, s=1)

    axs[2].scatter(dpm_coord_df_rmse, dpm_coord_ml_rmse, s=1)

    axs[3].scatter(lda_df_mae, lda_ml_mae, s=1)

    axs[0].set_title('absolute density error')
    axs[1].set_xlabel('dipole moment error')
    axs[2].set_xlabel('dipole moment absolute coordinate error')
    axs[3].set_xlabel('dipole moment integral coordinate errorc')

    axs[0].set_xlabel('density fitting error')
    axs[1].set_xlabel('density fitting error')
    axs[2].set_xlabel('density fitting error')
    axs[3].set_xlabel('density fitting error')
    axs[0].set_ylabel('machine learning error')

    plt.legend()
    plt.tight_layout()

    # plt.show()
    plt.savefig('figures/' + save_name + '_density_errors_scatter.png', dpi=300)
    # plt.savefig('figures/' + save_name + '_density_errors_ext.png', dpi=300)
# %%
main_args = Namespace()

main_args.args_file = "args/resorcinol_all_001.txt"
# main_args.args_file = "args/ethanethiol_all_006_test.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
# main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
# main_args.save_file = 'ethanethiol_all_006'
main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
main_args.save_file = 'resorcinol_all_005'
main_args.df_error = True
main_args.use_gpu = True
main_args.num_samples = 10
main_args.make_plots = True

df_losses = None

# %%
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
                           atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit.npy',
                           atom_dens_type='spline',
                           split_atom_dens=False,
                           density_grad=True,
                           )
print('dataset length', len(dataset))
print('sample pos shape', dataset.get_properties([0])['positions'].shape)
print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
# %%
print('num samples', main_args.num_samples)
df_losses = None
if main_args.df_error:
    df_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
                 'dpm_rmse': [], 'kl_loss': [],
                 'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
                 'coulomb': [], 'coulomb_int': [], 'mae_23': [], 'mae_43': [],
                 'lda_23_mae': [], 'dpm_coord_rmse': [],
                 'dpm_pos_coord_rmse': [], 'dpm_neg_coord_rmse': [],
                 'dpm_int_rmse': []}

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
                                  atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit.npy',
                                  atom_dens_type='spline',
                                  split_atom_dens=False,
                                  density_grad=True,
                                  )
    for i in range(min(len(dataset), main_args.num_samples)):
        sample = dataset.get_properties([i])
        sample_df = dataset_df.get_properties([i])
        calc_density_errors(sample_df['density'], sample, df_losses)
        # print('i', i, 'mae', df_error, 'rmse', df2_error, 'dpm', dpm_error, 'mag', dpm_mag_error, 'ang', dpm_ang_error, 'lda', lda_error)

    np.save('datasets/' + main_args.save_file + '_df_losses.npy', df_losses, allow_pickle=True)
    print('DF losses')
    for key in df_losses.keys():
        print(key)
        print(np.nanmean(df_losses[key]))
# %%
res_dataset = torch.load(main_args.res_load_file, map_location='cpu')
print('res dataset length', len(res_dataset))
print('res sample pos shape', res_dataset['positions'].shape)
print('res sample dens shape', res_dataset['density'].shape)
res_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
              'dpm_rmse': [], 'kl_loss': [],
              'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
              'coulomb': [], 'coulomb_int': [], 'mae_23': [], 'mae_43': [],
              'lda_23_mae': [], 'dpm_coord_rmse': [],
              'dpm_pos_coord_rmse': [], 'dpm_neg_coord_rmse': [],
              'dpm_int_rmse': []}
for i in range(min(len(dataset), main_args.num_samples)):
    sample = dataset.get_properties([i])
    # print('sample pdist', torch.cdist(sample['positions'], sample['positions'])[0, :3,:3])
    # print('res_pdist', torch.cdist(res_dataset['positions'][[i]], res_dataset['positions'][[i]])[0, :3,:3])
    sample = dataset.get_properties([i])
    calc_density_errors(res_dataset['density'][[i]], sample, res_losses)

np.save('datasets/' + main_args.save_file + '.npy', res_losses, allow_pickle=True)
print('Results losses')
for key in res_losses.keys():
    print(key)
    print(np.nanmean(res_losses[key]), np.nanmean(df_losses[key]) if df_losses is not None else "")
print('true dens max', torch.max(sample['density']))
if main_args.df_error:
    print('DF dens max', torch.max(sample_df['density']))
print('ML dens max', torch.max(res_dataset['density'][[i]]))
print('dpm pos neg diff', np.nanmean(res_losses['dpm_pos_coord_rmse']) - np.nanmean(res_losses['dpm_neg_coord_rmse']),
      np.nanmean(df_losses['dpm_pos_coord_rmse']) - np.nanmean(df_losses['dpm_neg_coord_rmse']))
# %%
res_dataset = torch.load(main_args.res_load_file, map_location='cpu')
print('res dataset length', len(res_dataset))
print('res sample pos shape', res_dataset['positions'].shape)
print('res sample dens shape', res_dataset['density'].shape)
res_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
              'dpm_rmse': [], 'kl_loss': [],
              'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
              'coulomb': [], 'coulomb_int': [], 'mae_23': [], 'mae_43': [],
              'lda_23_mae': [], 'dpm_coord_rmse': [],
              'dpm_pos_coord_rmse': [], 'dpm_neg_coord_rmse': [],
              'dpm_int_rmse': []}
for i in range(min(len(dataset), main_args.num_samples)):
    sample = dataset.get_properties([i])
    # print('sample pdist', torch.cdist(sample['positions'], sample['positions'])[0, :3,:3])
    # print('res_pdist', torch.cdist(res_dataset['positions'][[i]], res_dataset['positions'][[i]])[0, :3,:3])
    sample = dataset.get_properties([i])
    calc_density_errors(res_dataset['density'][[i]], sample, res_losses)

np.save('datasets/' + main_args.save_file + '.npy', res_losses, allow_pickle=True)
print('Results losses')
for key in res_losses.keys():
    print(key)
    print(np.nanmean(res_losses[key]), np.nanmean(df_losses[key]) if df_losses is not None else "")
print('true dens max', torch.max(sample['density']))
if main_args.df_error:
    print('DF dens max', torch.max(sample_df['density']))
print('ML dens max', torch.max(res_dataset['density'][[i]]))
print('dpm pos neg diff', np.nanmean(res_losses['dpm_pos_coord_rmse']) - np.nanmean(res_losses['dpm_neg_coord_rmse']),
      np.nanmean(df_losses['dpm_pos_coord_rmse']) - np.nanmean(df_losses['dpm_neg_coord_rmse']))

# %%
res_dataset = torch.load(main_args.res_load_file, map_location='cpu')
print('res dataset length', len(res_dataset))
print('res sample pos shape', res_dataset['positions'].shape)
print('res sample dens shape', res_dataset['density'].shape)
res_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
              'dpm_rmse': [], 'kl_loss': [],
              'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
              'coulomb': [], 'coulomb_int': [], 'mae_23': [], 'mae_43': [],
              'lda_23_mae': [], 'dpm_coord_rmse': [],
              'dpm_pos_coord_rmse': [], 'dpm_neg_coord_rmse': [],
              'dpm_int_rmse': []}
for i in range(min(len(dataset), main_args.num_samples)):
    sample = dataset.get_properties([i])
    # print('sample pdist', torch.cdist(sample['positions'], sample['positions'])[0, :3,:3])
    # print('res_pdist', torch.cdist(res_dataset['positions'][[i]], res_dataset['positions'][[i]])[0, :3,:3])
    sample = dataset.get_properties([i])
    calc_density_errors(res_dataset['density'][[i]], sample, res_losses)

np.save('datasets/' + main_args.save_file + '.npy', res_losses, allow_pickle=True)
print('Results losses')
for key in res_losses.keys():
    print(key)
    print(np.nanmean(res_losses[key]), np.nanmean(df_losses[key]) if df_losses is not None else "")
print('true dens max', torch.max(sample['density']))
if main_args.df_error:
    print('DF dens max', torch.max(sample_df['density']))
print('ML dens max', torch.max(res_dataset['density'][[i]]))
print('dpm pos neg diff', np.nanmean(res_losses['dpm_pos_coord_rmse']) - np.nanmean(res_losses['dpm_neg_coord_rmse']),
      np.nanmean(df_losses['dpm_pos_coord_rmse']) - np.nanmean(df_losses['dpm_neg_coord_rmse']))

# %%
if main_args.make_plots and main_args.df_error:
    # make_plots(res_dataset['density'][[i]], sample_df['density'], sample, main_args.save_file)
    make_scatter_plots(res_dataset['density'], dataset_df, dataset, main_args.save_file)
# %%

# %%

args.spherical_grid_level = 4
grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
sampling_fn = partial(spherical_radial_sampling, rotate=False)
grid_origin = 0
grid_extent = None
rotate = False

dataset_new = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
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
# %%
sample = dataset.get_properties([0])
sample_new = dataset_new.get_properties([0])

dpm_error = 4.8 * torch.norm(sample_new['dipole_moment'] - sample['dipole_moment'])
# center_of_mass = torch.sum(sample['batch_positions'] * sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
#     / torch.sum(sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)
# dpm_coord_error = torch.sqrt(torch.sum(torch.sum((torch.abs(sample['density'] - sample_new['density']) * sample['coord_weights']).unsqueeze(-1) *
#                              (sample['coords'] - center_of_mass), dim=1)**2))
# dpm_int_error = torch.norm(torch.sum(((sample['density'] - sample_new['density']) * sample['coord_weights']).unsqueeze(-1) *
#                            (sample['coords'] - center_of_mass), dim=1))

center_of_mass = torch.sum(sample['batch_positions'] * sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
    / torch.sum(sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

sample_copy = {key: sample[key] for key in sample.keys()}
dpm_new = orbitals.calc_dipole_moment(sample_new, positive_density=False, normalize_density=False)['dipole_moment']
dpm_old = orbitals.calc_dipole_moment(sample_copy, positive_density=False, normalize_density=False)['dipole_moment']
dpm_old2 = orbitals.calc_dipole_moment(sample_copy)['dipole_moment']

dpm2_error = 4.8 * torch.norm(dpm_old - sample['dipole_moment'])
dpm3_error = 4.8 * torch.norm(dpm_old2 - sample['dipole_moment'])

dpm_nonorm_error = 4.8 * torch.norm(dpm_old - dpm_new)
dpm_norm_error = 4.8 * torch.norm(dpm_old2 - dpm_new)

print('dpm error', float(dpm_error))
print('dpm old-new error', float(dpm2_error))
print('dpm old2-new error', float(dpm3_error))
print('dpm all error', float(dpm_nonorm_error))
print('dpm all error', float(dpm_norm_error))
print('dpm error diff', float(torch.abs(dpm2_error - dpm_nonorm_error)))

density = res_dataset['density'][[0]]
pred_dpm = res_dataset['dipole_moment'][[0]]

r_dens = torch.clamp(density, min=0)
r_dens = r_dens / torch.sum(r_dens * sample['coord_weights']) * torch.sum(sample['atom_numbers'])

r_dens_atoms = {key: sample[key] for key in sample.keys()}
r_dens_atoms['density'] = r_dens
r_dens_dpm = orbitals.calc_dipole_moment(r_dens_atoms)['dipole_moment']
r_dens_atoms['density'] = density
pred_nonorm_dpm = orbitals.calc_dipole_moment(r_dens_atoms, normalize_density=False, positive_density=False)['dipole_moment']
pred_calc_dpm = orbitals.calc_dipole_moment(r_dens_atoms)['dipole_moment']

dpm_rdens_error = 4.8 * torch.norm(r_dens_dpm - sample['dipole_moment'])
dpm_nonorm_error = 4.8 * torch.norm(pred_nonorm_dpm - sample['dipole_moment'])
dpm_calc_error = 4.8 * torch.norm(pred_calc_dpm - sample['dipole_moment'])
dpm_base_error = 4.8 * torch.norm(res_dataset['dipole_moment'][[0]] - sample['dipole_moment'])

print('rdens dpm error', float(dpm_rdens_error))
print('nonorm pred dens error', float(dpm_nonorm_error))
print('dpm base error', float(dpm_base_error))
print('calc pred dens error', float(dpm_calc_error))
print('base vs pred_calc error', float(torch.abs(dpm_base_error - dpm_calc_error)))
