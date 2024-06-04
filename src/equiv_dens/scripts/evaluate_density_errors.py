from pyscf import gto
import argparse

import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from pyscf.dft import numint
from equiv_dens.data.density_dataset import AtomsDensityData
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from pyscf.lib import param
from equiv_dens.training.model_loader import load_model

from equiv_dens.training import density_errors
import matplotlib.pyplot as plt
import numpy as np
import equiv_dens.training.utils as train_utils


def str2bool(s):
    if s.lower() in ('true', 't', '1'):
        return True
    elif s.lower() in ('false', 'f', '0'):
        return False
    else:
        return s


# %%
def calc_density_errors(density, atoms, error_dict):
    r_dens = torch.clamp(density, min=0)
    r_dens = r_dens / torch.sum(r_dens * atoms['coord_weights']) * torch.sum(atoms['atom_numbers'])

    tmp_atoms = {key: atoms[key] for key in atoms}
    tmp_atoms['density'] = r_dens
    dpm = orbitals.calc_dipole_moment(tmp_atoms, center_coordinates=True)['dipole_moment']

    df_error = torch.sum(torch.abs(atoms['density'] - r_dens) * atoms['coord_weights']) \
        / torch.sum(atoms['atom_numbers'])
    df2_error = torch.sqrt(torch.sum((atoms['density'] - r_dens)**2 * atoms['coord_weights'])
                           / torch.sum(atoms['atom_numbers']))

    dpm_error = 4.8 * torch.mean(torch.abs(dpm - atoms['dipole_moment']))
    dpm2_error = 4.8 * torch.norm(dpm - atoms['dipole_moment'])

    dpm_mag_error = 4.8 * torch.mean(torch.abs(torch.linalg.norm(dpm) - torch.linalg.norm(atoms['dipole_moment'])))
    dpm_ang_error = (180 / torch.pi) * torch.acos(torch.mean(torch.sum(dpm * atoms['dipole_moment']) /
                                                  (torch.linalg.norm(dpm) * torch.linalg.norm(atoms['dipole_moment']))))

    df43_error = torch.sum(torch.abs(atoms['density']**(4 / 3) -
                           r_dens**(4 / 3)) * atoms['coord_weights'])
    df23_error = torch.sum(torch.abs(atoms['density']**(2 / 3) -
                           r_dens**(2 / 3)) * atoms['coord_weights'])
    lda_error = torch.abs(torch.sum((atoms['density']**(4 / 3) -
                          r_dens**(4 / 3)) * atoms['coord_weights']))
    lda23_error = torch.abs(torch.sum((atoms['density']**(2 / 3) -
                            r_dens**(2 / 3)) * atoms['coord_weights']))
    coul_error = density_errors._density_coulomb_loss(r_dens - atoms['density'], atoms['coords'], atoms['coord_weights'])

    coul_int_error = torch.abs(density_errors.density_hartree_loss(r_dens, atoms['density'], atoms['coords'], atoms['coord_weights']))

    center_of_mass = torch.sum(atoms['batch_positions'] * atoms['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
        / torch.sum(atoms['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)
    # absolute dipole moment error at every grid point, RMSE
    dpm_coord_error = torch.sum(torch.norm((atoms['density'] - r_dens).unsqueeze(-1) *
                                (atoms['coords'] - center_of_mass), dim=2) * atoms['coord_weights'])
    # positive dipole moment error at every grid point, RMSE
    dpm_pos_coord_error = torch.sqrt(torch.sum(torch.sum((((atoms['density'] - r_dens)
                                                           * ((atoms['density'] - r_dens) > 0))
                                                          * atoms['coord_weights']).unsqueeze(-1)
                                                         * (atoms['coords'] - center_of_mass), dim=1)**2))
    # negative dipole moment error at every grid point, RMSE
    dpm_neg_coord_error = torch.sqrt(torch.sum(torch.sum((((atoms['density'] - r_dens)
                                                           * ((atoms['density'] - r_dens) < 0))
                                                          * atoms['coord_weights']).unsqueeze(-1)
                                                         * (atoms['coords'] - center_of_mass), dim=1)**2))
    # unsigned dipole moment error at every grid point, RMSE
    dpm_int_error = torch.norm(torch.sum(((atoms['density'] - r_dens) * atoms['coord_weights']).unsqueeze(-1) *
                               (atoms['coords'] - center_of_mass), dim=1))
    dpm_norm_error = torch.abs(torch.sum(((atoms['density'] - r_dens) * atoms['coord_weights']) *
                               torch.norm(atoms['coords'] - center_of_mass, dim=-1), dim=1))

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
    error_dict['dpm_norm_error'].append(float(dpm_norm_error))
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
    error_dict['exc_mae'].append(float(utils.hartree_to_kcal(torch.abs(exc_sum_pbe - exc_pbe))))

    vw_pred = torch.nansum(torch.norm(density[1:], dim=0)**2 / density[0] * atoms['coord_weights'].squeeze()) / 8
    vw_true = torch.nansum((torch.norm(dens_true[1:], dim=0)**2 / dens_true[0]) * atoms['coord_weights'].squeeze()) / 8
    error_dict['vw_mae'].append(float(torch.abs(vw_pred - vw_true)))
    error_dict['grad_norm_err'].append(float(torch.sum(torch.norm(density[1:] - dens_true[1:], dim=0) * atoms['coord_weights'].squeeze())
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

    _, axs = plt.subplots(5, 4, figsize=(10, 12))

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
    dpm_df_mae = torch.norm((dens_df - sample_ref['density']).unsqueeze(-1) * sample_ref['coords'], dim=2) * sample_ref['coord_weights']
    dpm_ml_mae = torch.norm((dens_ml - sample_ref['density']).unsqueeze(-1) * sample_ref['coords'], dim=2) * sample_ref['coord_weights']
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

    # plt.show()
    plt.savefig('figures/' + save_name + '_density_errors_all.png', dpi=300)


def make_compact_plots(dens_ml, dens_df, sample_ref, save_name):
    dens_ml = torch.clamp(dens_ml, min=0)
    dens_ml = dens_ml / torch.sum(dens_ml * sample_ref['coord_weights']) * torch.sum(sample_ref['atom_numbers'])

    dens_df = torch.clamp(dens_df, min=0)
    dens_df = dens_df / torch.sum(dens_df * sample_ref['coord_weights']) * torch.sum(sample_ref['atom_numbers'])

    center_of_mass = torch.sum(sample_ref['batch_positions'] * sample_ref['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
        / torch.sum(sample_ref['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

    distance_from_com = torch.norm(sample_ref['coords'] - center_of_mass, dim=2)

    # distance_from_atoms = torch.norm(sample_ref['coords'].unsqueeze(1) - sample_ref['positions'].unsqueeze(2), dim=-1)
    # min_distance_from_atoms = torch.min(distance_from_atoms, dim=1)[0]

    plt.rcParams['text.usetex'] = True

    _, axs = plt.subplots(1, 4, figsize=(10, 2))

    # absolute density errors unscaled
    dens_df_err = (dens_df - sample_ref['density'])
    dens_ml_err = (dens_ml - sample_ref['density'])

    # absolute density errors scaled
    dens_df_sc_err = (dens_df - sample_ref['density']) * sample_ref['coord_weights']
    dens_ml_sc_err = (dens_ml - sample_ref['density']) * sample_ref['coord_weights']

    # calculate difference between DF and ML errors
    dpm_df_err = (dens_df - sample_ref['density']) * torch.norm(sample_ref['coords'], dim=2) * sample_ref['coord_weights']
    dpm_ml_err = (dens_ml - sample_ref['density']) * torch.norm(sample_ref['coords'], dim=2) * sample_ref['coord_weights']

    sort_by_dist = torch.argsort(distance_from_com)

    dist_splits = torch.split(sort_by_dist, (sort_by_dist.shape[1] // 2) + 1, dim=1)

    avg_dists = []
    dist_ml_errors = []
    dist_df_errors = []
    for split in dist_splits:
        split = split.flatten()
        avg_dists.append(torch.mean(distance_from_com[:, split]))
        # dist_df_errors.append(torch.sum(dpm_df_err[:, split]))
        # dist_ml_errors.append(torch.sum(dpm_ml_err[:, split]))
        dist_df_errors.append(torch.norm(torch.sum(((sample_ref['density'][:, split] - dens_df[:, split]) *
                                                    sample_ref['coord_weights'][:, split]).unsqueeze(-1) *
                                                   (sample_ref['coords'][:, split, :] - center_of_mass), dim=1)))
        dist_ml_errors.append(torch.norm(torch.sum(((sample_ref['density'][:, split] - dens_ml[:, split]) *
                                                    sample_ref['coord_weights'][:, split]).unsqueeze(-1) *
                                                   (sample_ref['coords'][:, split, :] - center_of_mass), dim=1)))
    avg_dists = np.array(avg_dists)
    dist_df_errors = np.array(dist_df_errors)
    dist_ml_errors = np.array(dist_ml_errors)

    print('avg_dists', avg_dists)
    print('dist_df_errors', dist_df_errors)
    print('dist_ml_errors', dist_ml_errors)
    axs[0].scatter(dens_df_sc_err.squeeze(), distance_from_com, label='DF', c='blue', s=1)
    axs[0].scatter(dens_ml_sc_err.squeeze(), distance_from_com, label='ML', c='orange', s=1)

    axs[1].scatter(dens_df_err.squeeze(), distance_from_com, label='DF', c='blue', s=1)
    axs[1].scatter(dens_ml_err.squeeze(), distance_from_com, label='ML', c='orange', s=1)

    axs[2].scatter(dpm_df_err.squeeze(), distance_from_com, label='DF', c='blue', s=1)
    axs[2].scatter(dpm_ml_err.squeeze(), distance_from_com, label='ML', c='orange', s=1)

    axs[3].barh(avg_dists - 0.15, dist_df_errors, height=0.3, label='DF')
    axs[3].barh(avg_dists + 0.15, dist_ml_errors, height=0.3, label='ML')

    axs[0].set_xlabel('$\\rho$ MAE at gridpoint')
    axs[1].set_xlabel('$\\rho$ MAE at gridpoint,\n scaled by volume of gridpoint')
    axs[2].set_xlabel('$\\mu$ error at gridpoint,\n scaled by volume of gridpoint')
    axs[3].set_xlabel('$\\mu$ norm error,\n segmented by distance from center')
    axs[0].set_ylabel('distance from CoM')

    plt.legend()
    plt.tight_layout()

    # plt.show()
    plt.savefig('figures/' + save_name + '_density_errors_compact.png', dpi=300)


def make_scatter_plots(res_density, dataset_df, dataset, save_name):
    dens_df_mae = []
    dens_ml_mae = []
    dpm_df_rmse = []
    dpm_ml_rmse = []
    lda_df_mae = []
    lda_ml_mae = []
    dpm_coord_df_rmse = []
    dpm_coord_ml_rmse = []
    for i in range(len(res_density)):
        print('i', i)
        sample_ref = dataset.get_properties([i])
        dens_df = dataset_df.get_properties([i])['density']
        dens_ml = res_density[[i]]
        dens_ml = torch.clamp(dens_ml, min=0)
        dens_ml = dens_ml / torch.sum(dens_ml * sample_ref['coord_weights']) * torch.sum(sample_ref['atom_numbers'])

        dens_df = torch.clamp(dens_df, min=0)
        dens_df = dens_df / torch.sum(dens_df * sample_ref['coord_weights']) * torch.sum(sample_ref['atom_numbers'])

        center_of_mass = torch.sum(sample_ref['batch_positions'] * sample_ref['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
            / torch.sum(sample_ref['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

        # distance_from_com = torch.norm(sample_ref['coords'] - center_of_mass, dim=2)

        # distance_from_atoms = torch.norm(sample_ref['coords'].unsqueeze(1) - sample_ref['positions'].unsqueeze(2), dim=-1)
        # min_distance_from_atoms = torch.min(distance_from_atoms, dim=1)[0]

        tmp_atoms = {key: sample_ref[key] for key in sample_ref}
        tmp_atoms['density'] = dens_ml
        dpm_ml = orbitals.calc_dipole_moment(tmp_atoms, center_coordinates=True)['dipole_moment']
        tmp_atoms = {key: sample_ref[key] for key in sample_ref}
        tmp_atoms['density'] = dens_df
        dpm_df = orbitals.calc_dipole_moment(tmp_atoms, center_coordinates=True)['dipole_moment']

        dens_df_mae.append(torch.sum(torch.abs(sample_ref['density'] - dens_df) * sample_ref['coord_weights'])
                           / torch.sum(sample_ref['atom_numbers']))
        dens_ml_mae.append(torch.sum(torch.abs(sample_ref['density'] - dens_ml) * sample_ref['coord_weights'])
                           / torch.sum(sample_ref['atom_numbers']))

        dpm_df_rmse.append(4.8 * torch.norm(dpm_df - sample_ref['dipole_moment']))
        dpm_ml_rmse.append(4.8 * torch.norm(dpm_ml - sample_ref['dipole_moment']))

        lda_df_mae.append(torch.abs(torch.sum((sample_ref['density']**(4 / 3) -
                          dens_df**(4 / 3)) * sample_ref['coord_weights'])))
        lda_ml_mae.append(torch.abs(torch.sum((sample_ref['density']**(4 / 3) -
                          dens_ml**(4 / 3)) * sample_ref['coord_weights'])))

        # absolute dipole moment error at every grid point, RMSE
        dpm_coord_df_rmse.append(torch.sum(torch.norm((sample_ref['density'] - dens_df).unsqueeze(-1) *
                                 (sample_ref['coords'] - center_of_mass), dim=2) * sample_ref['coord_weights']))
        dpm_coord_ml_rmse.append(torch.sum(torch.norm((sample_ref['density'] - dens_ml).unsqueeze(-1) *
                                 (sample_ref['coords'] - center_of_mass), dim=2) * sample_ref['coord_weights']))

    plt.rcParams['text.usetex'] = True

    _, axs = plt.subplots(1, 4, figsize=(10, 2))

    # absolute density errors scaled
    axs[0].scatter(dens_df_mae, dens_ml_mae, s=1)

    axs[1].scatter(dpm_df_rmse, dpm_ml_rmse, s=1)

    axs[2].scatter(dpm_coord_df_rmse, dpm_coord_ml_rmse, s=1)

    axs[3].scatter(lda_df_mae, lda_ml_mae, s=1)

    axs[0].set_title('absolute density error')
    axs[1].set_title('dipole moment error')
    axs[2].set_title('dipole moment absolute coordinate error')
    axs[3].set_title('LDA integral error')

    axs[0].set_xlabel('density fitting error')
    axs[1].set_xlabel('density fitting error')
    axs[2].set_xlabel('density fitting error')
    axs[3].set_xlabel('density fitting error')
    axs[0].set_ylabel('machine learning error')

    plt.legend()
    plt.tight_layout()

    # plt.show()
    plt.savefig('figures/' + save_name + '_density_errors_scatter.png', dpi=300)


parser = argparse.ArgumentParser()
parser.add_argument('args_file', type=str)
parser.add_argument('ref_np_load_file', type=str)
parser.add_argument('ref_dens_load_file', type=str)
parser.add_argument('res_load_file', type=str)
parser.add_argument('save_file', type=str)
parser.add_argument('--ml_error', type=str2bool, default=True)
parser.add_argument('--df_error', action='store_true', default=False)
parser.add_argument('--free_atom_error', action='store_true', default=False)
parser.add_argument('--density_grad_error', action='store_true', default=False)
parser.add_argument('--use_gpu', action='store_true', default=False)
parser.add_argument('--num_samples', type=int, default=-1)
parser.add_argument('--make_plots', action='store_true', default=False)

main_args = parser.parse_args()

df_losses = None

# %%
args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)

args, hyperparam_args, test_vars = train_utils.init_training_vars(args, hyperparam_args)
checkpoint = test_vars['checkpoint']
args_dict = vars(args)

print('model code:', test_vars['model_code'])

# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
args.use_gpu = False

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

grid_vars = train_utils.init_grid_vars(args, test=True)
print('grid vars', grid_vars)

required_properties = ['energy', 'forces', 'density', 'dipole_moment', 'df_coeffs']

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file


dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000,
                           required_properties=required_properties,
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           L0_coeffs_file=args.L0_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_vars['grid_fn'],
                           pyscf_grid=args.pyscf_grid,
                           pyscf_rotate=grid_vars['rotate'],
                           sampling_fn=grid_vars['sampling_fn'],
                           grid_extent=grid_vars['grid_extent'],
                           grid_origin=grid_vars['grid_origin'],
                           verbose=args.verbose,
                           timing=args.timing,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           atom_dens_path=args.atom_dens_path,
                           atom_dens_type=args.atom_dens_type,
                           projected_density=False,
                           density_grad=main_args.density_grad_error,
                           )


if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
if main_args.density_grad_error:
    grad_str = '_grad'
else:
    grad_str = ''
# %%
print('num samples', main_args.num_samples)
df_losses = None
if main_args.df_error:

    dataset_df = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                                  orbitals_path=args.orbitals_file,
                                  density_n_samp=10000000000000,
                                  required_properties=required_properties,
                                  center_positions=False,
                                  radial_coeffs_file=args.radial_coeffs_file,
                                  L0_coeffs_file=args.L0_coeffs_file,
                                  dtype=args.dtype,
                                  grid_fn=grid_vars['grid_fn'],
                                  pyscf_grid=args.pyscf_grid,
                                  pyscf_rotate=grid_vars['rotate'],
                                  sampling_fn=grid_vars['sampling_fn'],
                                  grid_extent=grid_vars['grid_extent'],
                                  grid_origin=grid_vars['grid_origin'],
                                  verbose=args.verbose,
                                  timing=args.timing,
                                  cutoff=args.cutoff,
                                  df_loss_weights=args.df_loss_weights,
                                  atom_dens_path=args.atom_dens_path,
                                  atom_dens_type=args.atom_dens_type,
                                  projected_density=True,
                                  density_grad=main_args.density_grad_error,
                                  )
    if not main_args.density_grad_error:
        df_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
                     'dpm_rmse': [], 'kl_loss': [],
                     'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
                     'coulomb': [], 'coulomb_int': [], 'mae_23': [], 'mae_43': [],
                     'lda_23_mae': [], 'dpm_coord_rmse': [],
                     'dpm_pos_coord_rmse': [], 'dpm_neg_coord_rmse': [],
                     'dpm_int_rmse': [], 'dpm_norm_error': []}
        for i in range(min(len(dataset), main_args.num_samples)):
            sample = dataset.get_properties([i])
            sample_df = dataset_df.get_properties([i])
            print('density integral', torch.sum(sample_df['density'] * sample['coord_weights']))
            calc_density_errors(sample_df['density'], sample, df_losses)
    else:
        df_losses = {'exc_mae': [], 'vw_mae': [], 'grad_norm_err': []}
        # print('i', i, 'mae', df_error, 'rmse', df2_error, 'dpm', dpm_error, 'mag', dpm_mag_error, 'ang', dpm_ang_error, 'lda', lda_error)
        for i in range(min(len(dataset), main_args.num_samples)):
            atoms = dataset.get_properties([i])
            samp_df = dataset_df.get_properties([i])
            basis = 'augccpvdz'
            auxbasis = 'augccpvqzjkfit'

            pos = atoms['batch_positions'][0]
            atom_types = atoms['batch_atom_numbers'][0]
            atom = []
            for j in range(len(atom_types)):
                atom.append((atom_types[j].numpy(force=True), pos[j, :].numpy(force=True)))
            auxmol = gto.M(atom=atom, basis=auxbasis)
            auxmol.build()
            df_basis = samp_df['df_coeffs'].squeeze()
            coords = samp_df['coords'] / param.BOHR
            ao = numint.eval_ao(auxmol, coords[0], deriv=1)
            rho_pbe = np.einsum('ijk,k->ij', ao, df_basis.numpy(force=True))
            dens_grad_df = torch.from_numpy(rho_pbe)
            calc_density_grad_errors(dens_grad_df, atoms, df_losses)
    np.save('datasets/' + main_args.save_file + '_df' + grad_str + '_losses.npy', df_losses, allow_pickle=True)
    print('DF losses')
    for key in df_losses.keys():
        print(key)
        print(np.nanmean(df_losses[key]))


if main_args.free_atom_error:
    if not main_args.density_grad_error:
        free_atom_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
                            'dpm_rmse': [], 'kl_loss': [],
                            'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
                            'coulomb': [], 'coulomb_int': [], 'mae_23': [], 'mae_43': [],
                            'lda_23_mae': [], 'dpm_coord_rmse': [],
                            'dpm_pos_coord_rmse': [], 'dpm_neg_coord_rmse': [],
                            'dpm_int_rmse': [], 'dpm_norm_error': []}

        for i in range(min(len(dataset), main_args.num_samples)):
            sample = dataset.get_properties([i])
            print('density integral', torch.sum(sample['atom_density'] * sample['coord_weights']))
            calc_density_errors(sample['atom_density'], sample, free_atom_losses)
            # print('i', i, 'mae', df_error, 'rmse', df2_error, 'dpm', dpm_error, 'mag', dpm_mag_error, 'ang', dpm_ang_error, 'lda', lda_error)
    else:
        free_atom_losses = {'exc_mae': [], 'vw_mae': [], 'grad_norm_err': []}
        for i in range(min(len(dataset), main_args.num_samples)):
            sample = dataset.get_properties([i])
            print('density integral', torch.sum(sample['atom_density'] * sample['coord_weights']))
            dens_grad_atom = sample['atom_density_grad'].squeeze()
            dens_grad_atom = dens_grad_atom.t()
            dens_grad_atom = torch.cat([sample['atom_density'], dens_grad_atom], dim=0)
            print('dens_grad atom shape', dens_grad_atom.shape)
            calc_density_grad_errors(dens_grad_atom, sample, free_atom_losses)

    np.save('datasets/' + main_args.save_file + '_free_atom' + grad_str + '_losses.npy', free_atom_losses, allow_pickle=True)
    print('Free atom losses')
    for key in free_atom_losses.keys():
        print(key)
        print(np.nanmean(free_atom_losses[key]))

# %%
if main_args.ml_error:
    res_dataset = torch.load(main_args.res_load_file, map_location='cpu')
    if not main_args.density_grad_error:
        print('res dataset length', len(res_dataset))
        print('res sample pos shape', res_dataset['positions'].shape)
        print('res sample dens shape', res_dataset['density'].shape)
        res_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
                      'dpm_rmse': [], 'kl_loss': [],
                      'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
                      'coulomb': [], 'coulomb_int': [], 'mae_23': [], 'mae_43': [],
                      'lda_23_mae': [], 'dpm_coord_rmse': [],
                      'dpm_pos_coord_rmse': [], 'dpm_neg_coord_rmse': [],
                      'dpm_int_rmse': [], 'dpm_norm_error': []}
        for i in range(min(len(dataset), main_args.num_samples)):
            # print('sample pdist', torch.cdist(sample['positions'], sample['positions'])[0, :3,:3])
            # print('res_pdist', torch.cdist(res_dataset['positions'][[i]], res_dataset['positions'][[i]])[0, :3,:3])
            sample = dataset.get_properties([i])
            # print('sample positions', sample['positions'])
            # print('res dataset_positions', res_dataset['positions'][[i]])
            # print('density integral', torch.sum(res_dataset['density'][[i]] * sample['coord_weights']))
            calc_density_errors(res_dataset['density'][[i]], sample, res_losses)
    else:
        res_losses = {'exc_mae': [], 'vw_mae': [], 'grad_norm_err': []}
        if 'density_grad' in res_dataset.keys():
            for i in range(min(len(dataset), main_args.num_samples)):
                sample = dataset.get_properties([i])
                dens_ml = torch.cat([res_dataset['density'][[i]], res_dataset['density_grad'][i].t()], dim=0)
                calc_density_grad_errors(res_dataset['density_grad'][[i]], sample, res_losses)
        else:
            for i in range(min(len(dataset), main_args.num_samples)):
                sample = dataset.get_properties([i])
                model = load_model(args, dataset)
                res = model(sample)
                dens_ml = torch.cat([res['density'], res['density_grad'][0].t()], dim=0)
                calc_density_grad_errors(dens_ml, sample, res_losses)

    np.save('datasets/' + main_args.save_file + grad_str + '.npy', res_losses, allow_pickle=True)
    print('Results losses')
    for key in res_losses.keys():
        print(key)
        print(np.nanmean(res_losses[key]), np.nanmean(df_losses[key]) if df_losses is not None else "")
    print('true dens max', torch.max(sample['density']))
    print('ML dens max', torch.max(res_dataset['density'][[i]]))
    if main_args.df_error:
        print('DF dens max', torch.max(sample_df['density']))
        print('dpm pos neg diff', np.nanmean(res_losses['dpm_pos_coord_rmse']) - np.nanmean(res_losses['dpm_neg_coord_rmse']),
              np.nanmean(df_losses['dpm_pos_coord_rmse']) - np.nanmean(df_losses['dpm_neg_coord_rmse']))

# %%
if main_args.make_plots and main_args.df_error:
    # make_plots(res_dataset['density'][[i]], sample_df['density'], sample, main_args.save_file)
    make_compact_plots(res_dataset['density'][[i]], sample_df['density'], sample, main_args.save_file)
    # make_scatter_plots(res_dataset['density'], dataset_df, dataset, main_args.save_file)
