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
from equiv_dens.utils.hirshfeld_analysis import hirshfeld_partitioning

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

# main_args.args_file = "args/resorcinol_all_005.txt"
# main_args.args_file = "args/CO_dens_001.txt"
# main_args.args_file = "args/h2o_dens_002.txt"
main_args.args_file = "args/ethanethiol_all_006_test.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
# main_args.save_file = 'CO_dens_001.txt'
# main_args.save_file = 'h2o_dens_002.txt'
# main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
# main_args.save_file = 'resorcinol_all_005'
main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
main_args.save_file = 'ethanethiol_all_006_test'
# main_args.save_file = 'resorcinol_all_005'
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

args.np_dataset_test = "datasets/ethanethiol_md_traj_loc_65000_dft_augccpvdz.npy"
args.dens_dataset_test = "datasets/ethanethiol_md_traj_loc_65000_dft_augccpvdz_df_augccpvqzjkfit.npy"

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
                           atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit.npy',
                           atom_dens_type='spline',
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
                                  )

# %%
test_indices = checkpoint['data_split_indices']['test']
print(test_indices)
# %%
model = model_loader.load_model(args, dataset)
idx = 3
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
print('density error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
print('density df error', torch.sum(torch.abs(samp_df['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
# %%

def get_atomwise_metrics(atoms, expansion_model):
    charges = orbitals.get_density_charges(atoms)
    dipoles = orbitals.get_atomic_dipoles(atoms, expansion_model)
    return charges, dipoles


# %%
charges_ml, dipoles_ml = get_atomwise_metrics(res, model.property_models['density'])
print('charges ml', charges_ml)
print('dipoles ml', dipoles_ml)
df_sph_coeffs = {'spherical_coeffs': samp_df['df_coeffs']}
dens = orbitals.sample_projected_density(samp, torch.tensor(samp_df['df_coeffs']).unsqueeze(0),
                                         auxbasis='augccpvqzjkfit')
print('df dens diff to samp', torch.sum(torch.abs(samp['density'] - dens) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
df_coeffs = orbitals.vector_to_coeffs_dict(df_sph_coeffs, dataset.orbital_basis_num, samp_df['batch_atom_numbers'], radial_coeffs=True,
                                           radial_basis=dataset.radial_coeffs)
samp_df.update(df_coeffs)

res_df = model.property_models['density'](samp_df)
print('df_density integral', torch.sum(res_df['density'] * samp_df['coord_weights'], dim=1))
print('df_density error', torch.sum(torch.abs(res_df['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
charges_df, dipoles_df = get_atomwise_metrics(res_df, model.property_models['density'])
print('charges df', charges_df)
print('dipoles df', dipoles_df)

basis = 'augccpvdz'
auxbasis = 'augccpvqzjkfit'
ml_df_coeffs, auxmol_ext = orbitals.ml_basis_to_df_coeffs(res, basis, auxbasis,
                                                          mo_coeff=samp['mo_coeff'][0],
                                                          mo_occ=samp['mo_occ'][0])
ml_coeffs = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, samp['batch_atom_numbers'], radial_coeffs=False)

dens = orbitals.sample_projected_density(samp, ml_coeffs['spherical_coeffs'],
                                         auxbasis, auxmol=auxmol_ext,)

print('ml dens diff to samp', torch.sum(torch.abs(samp['density'] - dens) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
print('ml dens integral', torch.sum(dens * res['coord_weights'], dim=1))
ml_df_coeffs = torch.from_numpy(ml_df_coeffs).unsqueeze(0)
ml_df_sph = orbitals.vector_to_coeffs_dict({'spherical_coeffs': ml_df_coeffs}, dataset.orbital_basis_num, samp['batch_atom_numbers'],
                                           radial_coeffs=False)

res_ml_df = {key: res[key] for key in res}
res_ml_df['spherical_coeffs'] = ml_df_sph['spherical_coeffs']
res_ml_df = model.property_models['density'](res_ml_df)

print('ml_df_density integral', torch.sum(res_ml_df['density'] * res_ml_df['coord_weights'], dim=1))
print('ml_df_density error', torch.sum(torch.abs(res_ml_df['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
charges_ml_df, dipoles_ml_df = get_atomwise_metrics(res_ml_df, model.property_models['density'])
print('charges ml_df', charges_ml_df)
print('dipoles ml_df', dipoles_ml_df)
# %%
model = model_loader.load_model(args, dataset)
idx = 4
samp = dataset.get_properties([idx])
samp_df = dataset_df.get_properties([idx])

res = model(samp)

df_sph_coeffs = {'spherical_coeffs': samp_df['df_coeffs']}
dens = orbitals.sample_projected_density(samp, torch.tensor(samp_df['df_coeffs']).unsqueeze(0),
                                         auxbasis='augccpvqzjkfit')
print('dens diff to samp', torch.sum(torch.abs(samp['density'] - dens) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
df_coeffs = orbitals.vector_to_coeffs_dict(df_sph_coeffs, dataset.orbital_basis_num, samp_df['batch_atom_numbers'], radial_coeffs=True,
                                           radial_basis=dataset.radial_coeffs)
samp_df.update(df_coeffs)

res_df = model.property_models['density'](samp_df)
print('df_density integral', torch.sum(res_df['density'] * samp_df['coord_weights'], dim=1))
print('df_density error', torch.sum(torch.abs(res_df['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))

basis = 'augccpvdz'
auxbasis = 'augccpvqzjkfit'
ml_df_coeffs, auxmol_ext = orbitals.ml_basis_to_df_coeffs(res_df, basis, auxbasis,
                                                          mo_coeff=samp['mo_coeff'][0],
                                                          mo_occ=samp['mo_occ'][0])

dens = orbitals.sample_projected_density(samp, torch.tensor(ml_df_coeffs).unsqueeze(0),
                                         auxbasis, auxmol=auxmol_ext,)
print('dens diff to samp', torch.sum(torch.abs(samp['density'] - dens) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))



# %%
dens = orbitals.sample_density(samp, dataset.coeffs[idx]['mo_coeff'], dataset.coeffs[idx]['mo_occ'], basis=basis)
print('dens diff to samp', torch.sum(torch.abs(samp['density'] - dens) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
ml_df_coeffs, auxmol_ext = orbitals.ml_basis_to_df_coeffs(res, basis, auxbasis,
                                                          mo_coeff=samp['mo_coeff'][0],
                                                          mo_occ=samp['mo_occ'][0])
dens = orbitals.sample_projected_density(samp, torch.tensor(ml_df_coeffs).unsqueeze(0),
                                         auxbasis, auxmol=auxmol_ext,)
print('dens diff to samp', torch.sum(torch.abs(samp['density'] - dens) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
# %%
# atoms = np.load('md_logs/2022-12-23_pJuMyc8e/simulation_-ethanethiol_cluster_all_001_compressed_0.npy', allow_pickle=True).item()
# print(atoms['positions'].shape)
# print(atoms['atom_numbers'].shape)
# idx = -1
# atoms['positions'] = atoms['positions'][[idx]]
# atoms = utils.model_input_from_atoms(atoms, args.use_gpu, density_expansion=True, pyscf_grid=args.pyscf_grid,
#                                      grid_spec=dataset.grid_spec, grid_sampling_fn=dataset.sampling_fn,
#                                      cutoff=args.cutoff)
# print(atoms['positions'].shape)
# print(atoms['atom_numbers'].shape)
# atoms_samp = {'positions': samp['positions'], 'atom_numbers': samp['atom_numbers']}
# atoms_samp = utils.model_input_from_atoms(atoms_samp, args.use_gpu, density_expansion=True, pyscf_grid=args.pyscf_grid,
#                                           grid_spec=dataset.grid_spec, grid_sampling_fn=dataset.sampling_fn,
#                                           cutoff=args.cutoff)
# %%
# res  = model(atoms)
# print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
# res_samp = model(atoms_samp)
# print('res_samp density integral', torch.sum(res_samp['density'] * res_samp['coord_weights'], dim=1))
# %%
# %%
ml_dpm_errors = {'dens_mae': [], 'dpm_norm': [],
                 'dpm_point_abs': [], 'dpm_point_norm': [],
                 'dpm_r_norm': [], 'dpm_ang': [], 'dpm_mag': [],
                 'atom_charges': [], 'atom_dpm': [],
                 'dist_SH': [], 'dist_HC': [], 'hirshfeld_charges': [],
                 }
df_dpm_errors = {'dens_mae': [], 'dpm_norm': [],
                 'dpm_point_abs': [], 'dpm_point_norm': [],
                 'dpm_r_norm': [], 'dpm_ang': [], 'dpm_mag': [],
                 'atom_charges': [], 'atom_dpm': [],
                 'dist_SH': [], 'dist_HC': [], 'hirshfeld_charges': [],
                 }
ml_df_dpm_errors = {'dens_mae': [], 'dpm_norm': [],
                    'dpm_point_abs': [], 'dpm_point_norm': [],
                    'dpm_r_norm': [], 'dpm_ang': [], 'dpm_mag': [],
                    'atom_charges': [], 'atom_dpm': [],
                    'dist_SH': [], 'dist_HC': [], 'hirshfeld_charges': [],
                    }
true_hirshfeld_charges = []

for i in range(100):
    print('i', i)
    samp = dataset.get_properties([i])
    samp_df = dataset_df.get_properties([i])

    center_of_mass = torch.sum(samp['batch_positions'] * samp['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
        / torch.sum(samp['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

    atom_dens = dataset.sample_atom_density(samp['batch_positions'], samp['batch_atom_numbers'], samp['coords'],
                                            individual_dens=True)
    _, elec_charges = hirshfeld_partitioning(samp['density'], atom_dens, samp['batch_atom_numbers'],
                                              samp['coords'], samp['coord_weights'])
    true_hirshfeld_charges.append(elec_charges)
    res = model(samp)
    distance_from_com = torch.norm(samp['coords'] - center_of_mass, dim=2)

    density_mae = torch.sum(torch.abs(samp['density'] - res['density']) * samp['coord_weights']) / torch.sum(samp['atom_numbers'])
    # density_err = torch.abs(samp['density'] - res['density']) * samp['coord_weights']
    res = orbitals.calc_dipole_moment(res)

    _, elec_charges_ml = hirshfeld_partitioning(res['density'], atom_dens, samp['batch_atom_numbers'],
                                              samp['coords'], samp['coord_weights'])
    dpm_err = torch.norm(samp['dipole_moment'] - res['dipole_moment'])
    # dpm_point_err = ((res['density'] - samp['density']) * samp['coord_weights']).unsqueeze(-1) * (samp['coords'] - center_of_mass)
    dpm_point_abs_err = density_errors.dipole_pointwise_abs_loss(res['density'], samp['density'], samp['coords'], samp['coord_weights'])
    dpm_point_norm_err = density_errors.dipole_pointwise_int_loss(res['density'], samp['density'], samp['coords'], samp['coord_weights'])
    dpm_point_err_r_norm = torch.abs(torch.sum(((res['density'] - samp['density']) * samp['coord_weights']) * torch.norm(samp['coords'] - center_of_mass, dim=-1), dim=1))
    dpm_mag_err = torch.abs(torch.norm(samp['dipole_moment']) - torch.norm(res['dipole_moment']))
    dpm_ang_error = (180 / torch.pi) * torch.acos(torch.mean(torch.sum(res['dipole_moment']*samp['dipole_moment']) /
                                                  (torch.norm(res['dipole_moment']) * torch.norm(samp['dipole_moment']))))
    sh_dist = torch.norm(samp['batch_positions'][:, -1] - samp['batch_positions'][:, 5], dim=-1)
    hc_dist = torch.norm(samp['batch_positions'][:, 5] - samp['batch_positions'][:, 6], dim=-1)
    atom_charges, atom_dpm = \
        get_atomwise_metrics(res, model.property_models['density'])

    ml_dpm_errors['dens_mae'].append(density_mae.detach().cpu())
    ml_dpm_errors['dpm_norm'].append(dpm_err.detach().cpu())
    ml_dpm_errors['dpm_point_abs'].append(dpm_point_abs_err.detach().cpu())
    ml_dpm_errors['dpm_point_norm'].append(dpm_point_norm_err.detach().cpu())
    ml_dpm_errors['dpm_r_norm'].append(dpm_point_err_r_norm.detach().cpu())
    ml_dpm_errors['dpm_ang'].append(dpm_ang_error.detach().cpu())
    ml_dpm_errors['dpm_mag'].append(dpm_mag_err.detach().cpu())
    ml_dpm_errors['atom_charges'].append(atom_charges.detach().cpu())
    ml_dpm_errors['atom_dpm'].append(atom_dpm.detach().cpu())
    ml_dpm_errors['dist_SH'].append(sh_dist.detach().cpu())
    ml_dpm_errors['dist_HC'].append(hc_dist.detach().cpu())
    ml_dpm_errors['hirshfeld_charges'].append(elec_charges_ml.detach().cpu())
    # print('density_mae', density_mae)
    # print('dpm norm error', dpm_err)
    # print('dpm pointwise error', dpm_point_err.sum())
    # print('dpm point abs error', dpm_point_abs_err)
    # print('dpm point norm error', dpm_point_norm_err)
    # print('dpm point r norm error', dpm_point_err_r_norm)
    # print('')

    _, elec_charges_df = hirshfeld_partitioning(samp_df['density'], atom_dens, samp['batch_atom_numbers'],
                                              samp['coords'], samp['coord_weights'])
    density_df_mae = torch.sum(torch.abs(samp['density'] - samp_df['density']) * samp['coord_weights']) / torch.sum(samp['atom_numbers'])
    # density_df_err = torch.abs(samp['density'] - samp_df['density']) * samp['coord_weights']
    dpm_df_err = torch.norm(samp['dipole_moment'] - samp_df['dipole_moment'])
    # dpm_df_point_err = ((samp_df['density'] - samp['density']) * samp['coord_weights']).unsqueeze(-1) * (samp['coords'] - center_of_mass)
    dpm_df_point_abs_err = density_errors.dipole_pointwise_abs_loss(samp_df['density'], samp['density'], samp['coords'], samp['coord_weights'])
    dpm_df_point_err_r_norm = torch.abs(torch.sum(((samp_df['density'] - samp['density']) * samp['coord_weights']) * torch.norm(samp['coords'] - center_of_mass, dim=-1), dim=1))
    dpm_df_point_norm_err = density_errors.dipole_pointwise_int_loss(samp_df['density'], samp['density'], samp['coords'], samp['coord_weights'])
    dpm_df_mag_err = torch.abs(torch.norm(samp['dipole_moment']) - torch.norm(samp_df['dipole_moment']))
    dpm_df_ang_error = (180 / torch.pi) * torch.acos(torch.mean(torch.sum(samp_df['dipole_moment']*samp['dipole_moment']) /
                                                     (torch.norm(samp_df['dipole_moment']) * torch.norm(samp['dipole_moment']))))
    df_sph_coeffs = {'spherical_coeffs': samp_df['df_coeffs']}
    df_coeffs = orbitals.vector_to_coeffs_dict(df_sph_coeffs, dataset.orbital_basis_num, samp_df['batch_atom_numbers'], radial_coeffs=True,
                                               radial_basis=dataset.radial_coeffs)
    samp_df.update(df_coeffs)
    res_df = model.property_models['density'](samp_df)
    atom_charges, atom_dpm = \
        get_atomwise_metrics(res_df, model.property_models['density'])

    df_dpm_errors['dens_mae'].append(density_df_mae.detach().cpu())
    df_dpm_errors['dpm_norm'].append(dpm_df_err.detach().cpu())
    df_dpm_errors['dpm_point_abs'].append(dpm_df_point_abs_err.detach().cpu())
    df_dpm_errors['dpm_point_norm'].append(dpm_df_point_norm_err.detach().cpu())
    df_dpm_errors['dpm_r_norm'].append(dpm_df_point_err_r_norm.detach().cpu())
    df_dpm_errors['dpm_ang'].append(dpm_df_ang_error.detach().cpu())
    df_dpm_errors['dpm_mag'].append(dpm_df_mag_err.detach().cpu())
    df_dpm_errors['atom_charges'].append(atom_charges.detach().cpu())
    df_dpm_errors['atom_dpm'].append(atom_dpm.detach().cpu())
    df_dpm_errors['dist_SH'].append(sh_dist.detach().cpu())
    df_dpm_errors['dist_HC'].append(hc_dist.detach().cpu())
    df_dpm_errors['hirshfeld_charges'].append(elec_charges_df.detach().cpu())

    # print('density_df_mae', density_df_mae)
    # print('dpm df norm error', dpm_df_err)
    # print('dpm df pointwise error', dpm_df_point_err.sum())
    # print('dpm df point abs err', dpm_df_point_abs_err)
    # print('dpm df point norm error', dpm_df_point_norm_err)
    # print('dpm df point r norm error', dpm_df_point_err_r_norm)
    # print('')

    basis = 'augccpvdz'
    auxbasis = 'augccpvqzjkfit'
    ml_df_coeffs, auxmol_ext = orbitals.ml_basis_to_df_coeffs(res, basis, auxbasis,
                                                              mo_coeff=samp['mo_coeff'][0],
                                                              mo_occ=samp['mo_occ'][0])
    dens = orbitals.sample_projected_density(samp, torch.tensor(ml_df_coeffs).unsqueeze(0),
                                             auxbasis, auxmol=auxmol_ext,)
    # print(samp['atom_numbers'])
    # print('auxmol ext basis', auxmol_ext._bas)
    # print('auxmol ext env shape', auxmol_ext._env.shape)
    # #
    _, elec_charges_ml_df = hirshfeld_partitioning(dens, atom_dens, samp['batch_atom_numbers'],
                                              samp['coords'], samp['coord_weights'])
    samp2 = dataset.get_properties([i])
    dpm = orbitals.calc_dipole_moment(samp2, density=dens, normalize_density=False, positive_density=False)['dipole_moment']
    # print('dpm', dpm)
    # print('samp dpm', samp['dipole_moment'])
    density_ml_df_mae = torch.sum(torch.abs(samp['density'] - dens) * samp['coord_weights']) / torch.sum(samp['atom_numbers'])
    # density_ml_df_err = torch.abs(samp['density'] - dens) * samp['coord_weights']
    dpm_ml_df_err = torch.norm(samp['dipole_moment'] - dpm)
    # dpm_ml_df_point_err = ((dens - samp['density']) * samp['coord_weights']).unsqueeze(-1) * (samp['coords'] - center_of_mass)
    dpm_ml_df_point_abs_err = density_errors.dipole_pointwise_abs_loss(dens, samp['density'], samp['coords'], samp['coord_weights'])
    dpm_ml_df_point_err_r_norm = torch.abs(torch.sum(((dens - samp['density']) * samp['coord_weights']) * torch.norm(samp['coords'] - center_of_mass, dim=-1), dim=1))
    dpm_ml_df_point_norm_err = density_errors.dipole_pointwise_int_loss(dens, samp['density'], samp['coords'], samp['coord_weights'])
    dpm_ml_df_mag_err = torch.abs(torch.norm(samp['dipole_moment']) - torch.norm(dpm))
    dpm_ml_df_ang_error = (180 / torch.pi) * torch.acos(torch.mean(torch.sum(dpm*samp['dipole_moment']) /
                                                        (torch.norm(dpm) * torch.norm(samp['dipole_moment']))))

    ml_df_coeffs, auxmol_ext = orbitals.ml_basis_to_df_coeffs(res, basis, auxbasis,
                                                              mo_coeff=samp['mo_coeff'][0],
                                                              mo_occ=samp['mo_occ'][0])
    ml_df_coeffs = torch.from_numpy(ml_df_coeffs).unsqueeze(0)
    ml_df_sph = orbitals.vector_to_coeffs_dict({'spherical_coeffs': ml_df_coeffs}, dataset.orbital_basis_num, samp['batch_atom_numbers'],
                                               radial_coeffs=False)

    res_ml_df = {key: res[key] for key in res}
    res_ml_df['spherical_coeffs'] = ml_df_sph['spherical_coeffs']
    res_ml_df = model.property_models['density'](res_ml_df)
    atom_charges, atom_dpm = \
        get_atomwise_metrics(res_ml_df, model.property_models['density'])

    ml_df_dpm_errors['dens_mae'].append(density_ml_df_mae.detach().cpu())
    ml_df_dpm_errors['dpm_norm'].append(dpm_ml_df_err.detach().cpu())
    ml_df_dpm_errors['dpm_point_abs'].append(dpm_ml_df_point_abs_err.detach().cpu())
    ml_df_dpm_errors['dpm_point_norm'].append(dpm_ml_df_point_norm_err.detach().cpu())
    ml_df_dpm_errors['dpm_r_norm'].append(dpm_ml_df_point_err_r_norm.detach().cpu())
    ml_df_dpm_errors['dpm_ang'].append(dpm_ml_df_ang_error.detach().cpu())
    ml_df_dpm_errors['dpm_mag'].append(dpm_ml_df_mag_err.detach().cpu())
    ml_df_dpm_errors['atom_charges'].append(atom_charges.detach().cpu())
    ml_df_dpm_errors['atom_dpm'].append(atom_dpm.detach().cpu())
    ml_df_dpm_errors['dist_SH'].append(sh_dist.detach().cpu())
    ml_df_dpm_errors['dist_HC'].append(hc_dist.detach().cpu())
    ml_df_dpm_errors['hirshfeld_charges'].append(elec_charges_ml_df.detach().cpu())
    # print('density_ml_df_mae', density_ml_df_mae)
    # print('dpm ml df norm error', dpm_ml_df_err)
    # print('dpm ml df pointwise error', dpm_ml_df_point_err.sum())
    # print('dpm ml df point abs err', dpm_ml_df_point_abs_err)
    # print('dpm ml df point norm error', dpm_ml_df_point_norm_err)
    # print('dpm ml df point r norm error', dpm_ml_df_point_err_r_norm)
    # print('')

print('ml_dpm_errors', ml_dpm_errors)
print('df_dpm_errors', df_dpm_errors)
print('ml_df_dpm_errors', ml_df_dpm_errors)
np.save('results/ethanethiol_ml_dpm_errors_loc65.npy', ml_dpm_errors, allow_pickle=True)
np.save('results/ethanethiol_df_dpm_errors_loc65.npy', df_dpm_errors, allow_pickle=True)
np.save('results/ethanethiol_ml_df_dpm_errors_loc65.npy', ml_df_dpm_errors, allow_pickle=True)
# %%
hirshfeld_charges = [charges.detach().cpu().numpy() for charges in true_hirshfeld_charges]
np.save('results/ethanethiol_hirshfeld_charges_loc65.npy', hirshfeld_charges, allow_pickle=True)
# %%
ml_dpm_errors = np.load('results/ethanethiol_ml_dpm_errors_loc65.npy', allow_pickle=True).item()
df_dpm_errors = np.load('results/ethanethiol_df_dpm_errors_loc65.npy', allow_pickle=True).item()
ml_df_dpm_errors = np.load('results/ethanethiol_ml_df_dpm_errors_loc65.npy', allow_pickle=True).item()
# %%
print('ml_dpm_errors atom dpm', ml_dpm_errors['atom_dpm'])
print('df_dpm_errors atom dpm', df_dpm_errors['atom_dpm'])
print('ML_df_dpm_errors atom dpm', ml_df_dpm_errors['atom_dpm'])
# %%
for key in ml_dpm_errors.keys():
    print('key', key)
    print('len key', len(ml_dpm_errors[key]))
    for i in range(len(ml_dpm_errors[key])):
        factor = 1
        if key == 'atom_charges':
            continue
        if 'dpm' in key and 'ang' not in key:
            factor = 4.8

        ml_dpm_errors[key][i] = factor * ml_dpm_errors[key][i]
        df_dpm_errors[key][i] = factor * df_dpm_errors[key][i]
        ml_df_dpm_errors[key][i] = factor * ml_df_dpm_errors[key][i]
        if torch.numel(ml_df_dpm_errors[key][i]) == 1:
            ml_dpm_errors[key][i] = float(ml_dpm_errors[key][i])
            df_dpm_errors[key][i] = float(df_dpm_errors[key][i])
            ml_df_dpm_errors[key][i] = float(ml_df_dpm_errors[key][i])
# %%
fig, axs = plt.subplots(6, 1, figsize=(10, 15))
keys = ['dens_mae', 'dpm_norm', 'dpm_r_norm', 'dpm_ang', 'dpm_mag', 'dist_HC']
labels = ['density APE (%)', 'dipole error (Debye)', 'dipole ||r|| error (Debye)', 'dipole ang error (Deg)', 'dipole magnitude error (Debye)', f'S-H distance (Ang)']
for i, key in enumerate(keys):
    # print(key)
    # print(np.array(ml_dpm_errors[key]))
    # print(len(ml_dpm_errors[key]))
    axs[i].plot(np.arange(len(ml_dpm_errors[key]))/2, ml_dpm_errors[key], label='ML')
    axs[i].plot(np.arange(len(ml_dpm_errors[key]))/2, df_dpm_errors[key], label='DF')
    axs[i].plot(np.arange(len(ml_dpm_errors[key]))/2, ml_df_dpm_errors[key], label='ML-DF')
    axs[i].set_ylabel(labels[i])
    if i == 0:
        axs[i].set_ylim(0.0005, 0.0015)
axs[i].set_xlabel('time (fs)')
fig.text(0.5, 0.05,
         'These figures show how the errors of the machine learned density and dipole\n' +
         'moment measured in different ways change in a cutout of an MD trajectory of ethanethiol.',
         ha='center', va='center', fontsize=12)
plt.legend()
plt.savefig('figures/ethanethiol_md_dpm_errors_loc65k.png', dpi=300)
plt.show()
# %%
from matplotlib.lines import Line2D

fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axes = {1: 0, 6: 1, 16: 2}
print(ml_dpm_errors['atom_charges'][0].shape)
atom_types = samp['batch_atom_numbers'][0].numpy()
print('atom_types shape', atom_types.shape)
cmap = plt.get_cmap("tab10")
for i in range(ml_dpm_errors['atom_charges'][0].shape[1]):
    axs_idx = axes[atom_types[i]]
    charge_i_ml = [ml_dpm_errors['atom_charges'][j][0, i] for j in range(len(ml_dpm_errors['atom_charges']))]
    charge_i_df = [df_dpm_errors['atom_charges'][j][0, i] for j in range(len(df_dpm_errors['atom_charges']))]
    charge_i_ml_df = [ml_df_dpm_errors['atom_charges'][j][0, i] for j in range(len(ml_df_dpm_errors['atom_charges']))]
    hirshfeld = [hirshfeld_charges[j][0, i] for j in range(len(hirshfeld_charges))]
    axs[axs_idx].plot(np.arange(len(ml_dpm_errors['atom_charges']))/2, charge_i_ml, label="ML", color=cmap(0))
    # axs[axs_idx].plot(np.arange(len(df_dpm_errors['atom_charges']))/2, charge_i_df, label="DF", color=cmap(1))
    # axs[axs_idx].plot(np.arange(len(ml_df_dpm_errors['atom_charges']))/2, charge_i_ml_df, label="ML-DF", color=cmap(2))
    axs[axs_idx].plot(np.arange(len(ml_df_dpm_errors['atom_charges']))/2, hirshfeld, label="hirshfeld", color=cmap(3))
    axs[axs_idx].set_ylabel('charges of {} atoms'.format(utils.numbers_to_symbols([atom_types[i]])[0]))
fig.text(0.5, 0.05,
         'These figures show how the atomic charges based on the machine learned density coeffs\n' +
         'change in a cutout of an MD trajectory of ethanethiol.',
         ha='center', va='center', fontsize=12)
axs[-1].plot(np.arange(len(ml_dpm_errors['dist_SH']))/2, ml_dpm_errors['dist_SH'], label='ML')
axs[-1].set_ylabel('S-H distance (Ang)')
axs[-1].set_xlabel('time (fs)')
custom_lines = [Line2D([0], [0], color=cmap(0), lw=2),
                Line2D([0], [0], color=cmap(1), lw=2),
                Line2D([0], [0], color=cmap(2), lw=2)
                ]

plt.legend(custom_lines, ['ML', 'DF', 'ML-DF'])
# plt.savefig('figures/ethanethiol_md_charges_loc65k.png', dpi=300)
plt.show()

# %%
from matplotlib.lines import Line2D

fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axes = {1: 0, 6: 1, 16: 2}
atom_types = samp['batch_atom_numbers'][0].numpy()
print('atom_types shape', atom_types.shape)
cmap = plt.get_cmap("tab10")
for i in range(ml_dpm_errors['atom_dpm'][0].shape[1]):
    print(atom_types[i])
    axs_idx = axes[atom_types[i]]
    dpm_i_ml = [torch.norm(ml_dpm_errors['atom_dpm'][j][0, i]) for j in range(len(ml_dpm_errors['atom_dpm']))]
    dpm_i_df = [torch.norm(df_dpm_errors['atom_dpm'][j][0, i]) for j in range(len(df_dpm_errors['atom_dpm']))]
    dpm_i_ml_df = [torch.norm(ml_df_dpm_errors['atom_dpm'][j][0, i]) for j in range(len(ml_df_dpm_errors['atom_dpm']))]
    print('dpm ml', ml_df_dpm_errors['atom_dpm'][0][0, i])
    print('dpm norm ml', torch.norm(ml_df_dpm_errors['atom_dpm'][0][0, i]))
    axs[axs_idx].plot(np.arange(len(ml_dpm_errors['atom_dpm']))/2, dpm_i_ml, label="ML", color=cmap(0))
    axs[axs_idx].plot(np.arange(len(df_dpm_errors['atom_dpm']))/2, dpm_i_df, label="DF", color=cmap(1))
    axs[axs_idx].plot(np.arange(len(ml_df_dpm_errors['atom_dpm']))/2, dpm_i_ml_df, label="ML-DF", color=cmap(2))
    axs[axs_idx].set_ylabel('dpm magnitude of\n {} atoms (Debye)'.format(utils.numbers_to_symbols([atom_types[i]])[0]))
fig.text(0.5, 0.05,
         'These figures show how the atomic dipole moments based on the machine learned density coeffs\n' +
         'change in a cutout of an MD trajectory of ethanethiol.',
         ha='center', va='center', fontsize=12)
axs[-1].plot(np.arange(len(ml_dpm_errors['dist_SH']))/2, ml_dpm_errors['dist_SH'], label='ML')
axs[-1].set_ylabel('S-H distance (Ang)')
axs[-1].set_xlabel('time (fs)')
custom_lines = [Line2D([0], [0], color=cmap(0), lw=2),
                Line2D([0], [0], color=cmap(1), lw=2),
                Line2D([0], [0], color=cmap(2), lw=2)
                ]

plt.legend(custom_lines, ['ML', 'DF', 'ML-DF'])
plt.savefig('figures/ethanethiol_md_atom_dpm_loc65k.png', dpi=300)
plt.show()
# %%
def cosine_similarity(x, y):
    dpm_ang_error = (180 / torch.pi) * torch.acos(torch.mean(torch.sum(x * y) /
                                                             (torch.linalg.norm(x) * torch.linalg.norm(y))))
    return dpm_ang_error
# %%
from matplotlib.lines import Line2D

fig, axs = plt.subplots(7, 1, figsize=(10, 15))
axes = {1: 0, 6: 1, 16: 2}
atom_types = samp['batch_atom_numbers'][0].numpy()
print('atom_types shape', atom_types.shape)
mean_dpm_ml = np.mean([ml_dpm_errors['atom_dpm'][i].detach().cpu().numpy()
                      for i in range(len(ml_dpm_errors['atom_dpm']))], axis=0)
mean_dpm_ml = torch.from_numpy(mean_dpm_ml).squeeze(0)

mean_dpm_df = np.mean([df_dpm_errors['atom_dpm'][i].detach().cpu().numpy()
                      for i in range(len(df_dpm_errors['atom_dpm']))], axis=0)
mean_dpm_df = torch.from_numpy(mean_dpm_df).squeeze(0)

mean_dpm_ml_df = np.mean([ml_df_dpm_errors['atom_dpm'][i].detach().cpu().numpy()
                          for i in range(len(ml_df_dpm_errors['atom_dpm']))], axis=0)
mean_dpm_ml_df = torch.from_numpy(mean_dpm_ml_df).squeeze(0)
cmap = plt.get_cmap("tab10")
for i in range(ml_dpm_errors['atom_dpm'][0].shape[1]):
    print(atom_types[i])
    axs_idx = axes[atom_types[i]]
    dpm_i_ml = [torch.norm(ml_dpm_errors['atom_dpm'][j][0, i]) - torch.norm(mean_dpm_ml[i])
                for j in range(len(ml_dpm_errors['atom_dpm']))]
    dpm_i_df = [torch.norm(df_dpm_errors['atom_dpm'][j][0, i]) - torch.norm(mean_dpm_df[i])
                for j in range(len(df_dpm_errors['atom_dpm']))]
    dpm_i_ml_df = [torch.norm(ml_df_dpm_errors['atom_dpm'][j][0, i]) - torch.norm(mean_dpm_ml_df[i])
                   for j in range(len(ml_df_dpm_errors['atom_dpm']))]

    print('dpm ml', ml_df_dpm_errors['atom_dpm'][0][0, i])
    print('dpm norm ml', torch.norm(ml_df_dpm_errors['atom_dpm'][0][0, i]))
    axs[axs_idx].plot(np.arange(len(ml_dpm_errors['atom_dpm']))/2, dpm_i_ml, label="ML", color=cmap(0))
    axs[axs_idx].plot(np.arange(len(df_dpm_errors['atom_dpm']))/2, dpm_i_df, label="DF", color=cmap(1))
    axs[axs_idx].plot(np.arange(len(ml_df_dpm_errors['atom_dpm']))/2, dpm_i_ml_df, label="ML-DF", color=cmap(2))
    axs[axs_idx].set_ylabel('dpm magnitude of \n{} atoms (Debye)'.format(utils.numbers_to_symbols([atom_types[i]])[0]))

for i in range(ml_dpm_errors['atom_dpm'][0].shape[1]):
    print(atom_types[i])
    axs_idx = axes[atom_types[i]] + 3
    ang_i_ml = [cosine_similarity(ml_dpm_errors['atom_dpm'][j][0, i], mean_dpm_ml[i])
                for j in range(len(ml_dpm_errors['atom_dpm']))]
    ang_i_df = [cosine_similarity(df_dpm_errors['atom_dpm'][j][0, i], mean_dpm_df[i])
                for j in range(len(df_dpm_errors['atom_dpm']))]
    ang_i_ml_df = [cosine_similarity(ml_df_dpm_errors['atom_dpm'][j][0, i], mean_dpm_ml_df[i])
                   for j in range(len(ml_df_dpm_errors['atom_dpm']))]
    axs[axs_idx].plot(np.arange(len(ml_dpm_errors['atom_dpm']))/2, ang_i_ml, label="ML", color=cmap(0))
    axs[axs_idx].plot(np.arange(len(df_dpm_errors['atom_dpm']))/2, ang_i_df, label="DF", color=cmap(1))
    axs[axs_idx].plot(np.arange(len(ml_df_dpm_errors['atom_dpm']))/2, ang_i_ml_df, label="ML-DF", color=cmap(2))
    axs[axs_idx].set_ylabel('dpm angle of \n{} atoms (deg)'.format(utils.numbers_to_symbols([atom_types[i]])[0]))
fig.text(0.5, 0.05,
         'These figures show how the atomic dipole moments based on the machine learned density coeffs\n' +
         'change in a cutout of an MD trajectory of ethanethiol,\n' +
         'wirth respect to the mean dipole moment of the trajectory.',
         ha='center', va='center', fontsize=12)
axs[-1].plot(np.arange(len(ml_dpm_errors['dist_SH']))/2, ml_dpm_errors['dist_SH'], label='ML')
axs[-1].set_ylabel('S-H distance (Ang)')
axs[-1].set_xlabel('time (fs)')
custom_lines = [Line2D([0], [0], color=cmap(0), lw=2),
                Line2D([0], [0], color=cmap(1), lw=2),
                Line2D([0], [0], color=cmap(2), lw=2)
                ]

plt.legend(custom_lines, ['ML', 'DF', 'ML-DF'])
plt.savefig('figures/ethanethiol_md_atom_dpm_dev_loc65k.png', dpi=300)
plt.show()
# %%
for key in keys:
    print('ML', key, np.nanmean(ml_dpm_errors[key]))
    print('DF', key, np.nanmean(df_dpm_errors[key]))
    print('ML-DF', key, np.nanmean(ml_df_dpm_errors[key]))
    print('')
# %%
for i in range(len(ml_dpm_errors['dpm_norm'])):
    print('i', i, 'ml err', ml_dpm_errors['dpm_norm'][i], 'df err', df_dpm_errors['dpm_norm'][i], 'ml-df err', ml_df_dpm_errors['dpm_norm'][i])
# %%
def calculate_dihedral_angle(positions):
    """
    Calculate the dihedral angle defined by four atoms.

    Parameters:
    positions (torch.Tensor): A tensor of shape (#atoms=4, 3) containing the positions of the atoms.

    Returns:
    float: The dihedral angle in degrees.
    """
    # Vectors between atoms
    v12 = positions[1] - positions[0]
    v23 = positions[2] - positions[1]
    v34 = positions[3] - positions[2]

    # Normal vectors to planes
    n123 = torch.cross(v12, v23)
    n234 = torch.cross(v23, v34)

    # Normalize the normal vectors
    n123_normalized = n123 / torch.norm(n123)
    n234_normalized = n234 / torch.norm(n234)

    # Calculate the angle between the normals
    cosine_angle = torch.dot(n123_normalized, n234_normalized)
    angle = torch.acos(cosine_angle)

    # Convert to degrees
    angle_degrees = angle * (180.0 / torch.pi)

    # Ensure the angle is between 0 and 180
    angle_degrees = torch.clamp(angle_degrees, min=0, max=180)

    return angle_degrees.item()
# %%
# Example usage
positions = torch.tensor([
    [1.0, 0.0, 0.0],  # Atom 1
    [0.0, 1.0, 0.0],  # Atom 2
    [0.0, 0.0, 1.0],  # Atom 3
    [1.0, 1.0, 1.0]   # Atom 4
])

dihedral_angle = calculate_dihedral_angle(positions)
print(f"Dihedral Angle: {dihedral_angle} degrees")
# %%
dihedrals = [] 
for i in range(100):
    print('i', i)
    samp = dataset.get_properties([i])
    pos = samp['positions'][0, -4:]
    pos = pos[[2, 1 ,3, 0], :]
    print('dists 0-1', torch.norm(pos[0] - pos[1]))
    print('dists 1-2', torch.norm(pos[1] - pos[2]))
    print('dists 2-3', torch.norm(pos[2] - pos[3]))
    dihedral_angle = calculate_dihedral_angle(pos)
    print('dihedral_angle', dihedral_angle)
    dihedrals.append(dihedral_angle)
# %%
fig, axs = plt.subplots(7, 1, figsize=(10, 15))
keys = ['dens_mae', 'dpm_norm', 'dpm_r_norm', 'dpm_ang', 'dpm_mag', 'dist_SH']
labels = ['density APE (%)', 'dipole error (Debye)', 'dipole ||r|| error (Debye)', 'dipole ang error (Deg)', 'dipole magnitude error (Debye)', f'S-H distance (Ang)']
for i, key in enumerate(keys):
    # print(key)
    # print(np.array(ml_dpm_errors[key]))
    # print(len(ml_dpm_errors[key]))
    axs[i].plot(np.arange(len(ml_dpm_errors[key]))/2, ml_dpm_errors[key], label='ML')
    axs[i].plot(np.arange(len(ml_dpm_errors[key]))/2, df_dpm_errors[key], label='DF')
    axs[i].plot(np.arange(len(ml_dpm_errors[key]))/2, ml_df_dpm_errors[key], label='ML-DF')
    axs[i].set_ylabel(labels[i])
    if i == 0:
        axs[i].set_ylim(0.0005, 0.0015)

axs[-1].plot(np.arange(len(ml_dpm_errors[key]))/2, dihedrals, label='ML')
axs[-1].set_xlabel('time (fs)')
fig.text(0.5, 0.05,
         'These figures show how the errors of the machine learned density and dipole\n' +
         'moment measured in different ways change in a cutout of an MD trajectory of ethanethiol.',
         ha='center', va='center', fontsize=12)
plt.legend()
plt.savefig('figures/ethanethiol_md_dpm_errors_loc65k.png', dpi=300)
plt.show()
# %%
from matplotlib.lines import Line2D

fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axes = {1: 0, 6: 1, 16: 2}
print(ml_dpm_errors['hirshfeld_charges'][0].shape)
atom_types = samp['batch_atom_numbers'][0].numpy()
print('atom_types shape', atom_types.shape)
cmap = plt.get_cmap("tab10")
for i in range(ml_dpm_errors['hirshfeld_charges'][0].shape[1]):
    axs_idx = axes[atom_types[i]]
    charge_i_ml = [ml_dpm_errors['hirshfeld_charges'][j][0, i] for j in range(len(ml_dpm_errors['hirshfeld_charges']))]
    charge_i_df = [df_dpm_errors['hirshfeld_charges'][j][0, i] for j in range(len(df_dpm_errors['hirshfeld_charges']))]
    charge_i_ml_df = [ml_df_dpm_errors['hirshfeld_charges'][j][0, i] for j in range(len(ml_df_dpm_errors['hirshfeld_charges']))]
    axs[axs_idx].plot(np.arange(len(ml_dpm_errors['hirshfeld_charges']))/2, charge_i_ml, label="ML", color=cmap(0))
    axs[axs_idx].plot(np.arange(len(df_dpm_errors['hirshfeld_charges']))/2, charge_i_df, label="DF", color=cmap(1))
    axs[axs_idx].plot(np.arange(len(ml_df_dpm_errors['hirshfeld_charges']))/2, charge_i_ml_df, label="ML-DF", color=cmap(2))
    axs[axs_idx].set_ylabel('charges of {} atoms'.format(utils.numbers_to_symbols([atom_types[i]])[0]))
fig.text(0.5, 0.05,
         'These figures show how the atomic charges based on the machine learned density coeffs\n' +
         'change in a cutout of an MD trajectory of ethanethiol.',
         ha='center', va='center', fontsize=12)
axs[-1].plot(np.arange(len(ml_dpm_errors['dist_SH']))/2, ml_dpm_errors['dist_SH'], label='ML')
axs[-1].set_ylabel('S-H distance (Ang)')
axs[-1].set_xlabel('time (fs)')
custom_lines = [Line2D([0], [0], color=cmap(0), lw=2),
                Line2D([0], [0], color=cmap(1), lw=2),
                Line2D([0], [0], color=cmap(2), lw=2)
                ]

plt.legend(custom_lines, ['ML', 'DF', 'ML-DF'])
plt.savefig('figures/ethanethiol_md_hirshfeld_charges_loc65k.png', dpi=300)
plt.show()
