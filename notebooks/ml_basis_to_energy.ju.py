# %%
# %load_ext autoreload
# %autoreload 2
# %cd /home/mihail/Documents/workspace/equiv_dens/

import os
from pyscf.dft import numint
from pyscf.lib import param
from datetime import datetime, timezone
from pyscf import gto, dft
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling, \
    spherical_grid, spherical_radial_sampling
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from equiv_dens.utils.misc import generate_id

from functools import partial
from argparse import Namespace
import numpy as np
from equiv_dens.training import model_loader
import time

# %%
main_args = Namespace()

# main_args.args_file = "args/resorcinol_all_001.txt"
# main_args.args_file = "args/ethanethiol_all_006_test.txt"
# main_args.args_file = "args/ethanethiol_all_106_test.txt"
# main_args.args_file = "args/h2o_small_all_001.txt"
# main_args.args_file = "args/ethanethiol_df_coeffs_001_test.txt"
main_args.args_file = "args/ethanethiol_all_001_coreless_test.txt"
# main_args.args_file = "args/ethanethiol_all_004_coreless.txt"
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
main_args.save_file = 'ethanethiol_all_001_coreless'
# main_args.save_file = 'ethanethiol_all_004_coreless'
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

if args.restart is None:
    # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    model_code = generate_id()
    directory = os.path.join(args.save_dir, datetime.now(timezone.UTC).strftime("%Y-%m-%d_") +
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
args.integral_constraint = False
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

args.np_dataset_test = "datasets/ethanethiol_md_traj_every1000_dft_augccpvdz.npy"
args.dens_dataset_test = "datasets/ethanethiol_md_traj_every1000_dft_augccpvdz_df_augccpvqzjkfit.npy"

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
                           split_atom_dens=True,
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
model = model_loader.load_model(args, dataset)
idx = [3]
samp = dataset.get_properties(idx)
samp_df = dataset_df.get_properties(idx)

res = model(samp)

print('samp df_coeffs', samp_df['df_coeffs'].shape)
print('res df_coeffs', res['df_coeffs'].shape)

# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
print('density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
print('density integral', torch.sum(samp_df['density'] * samp_df['coord_weights'], dim=1))
print('density error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1)
                       / torch.sum(samp['batch_atom_numbers'], dim=1))
print('density df error', torch.sum(torch.abs(samp_df['density'] - samp['density']) * samp['coord_weights'], dim=1)
                          / torch.sum(samp['batch_atom_numbers'], dim=1))

# %%
# Calculate LDA energy
basis = 'augccpvdz'
auxbasis = 'augccpvqzjkfit'

start = time.time()
pos = samp['batch_positions'][0]
atom_types = samp['batch_atom_numbers'][0]
print('pos shape', pos.shape)
print('atom nums shape', atom_types.shape)
atom = []
for j in range(len(atom_types)):
    atom.append((atom_types[j].numpy(force=True), pos[j, :].numpy(force=True)))
print(atom)

mol_lda = gto.M(atom=atom, basis=basis)
# print(mol.pack())
mf_lda = dft.RKS(mol_lda)
mf_lda.chkfile = False
mf_lda.xc = 'lda'
# mf.max_cycle = 1000
mf_lda.kernel()
print('time elapsed', time.time() - start)
print('total energy regular', mf_lda.e_tot)
dm = mf_lda.make_rdm1()
m_kin = mol_lda.intor('int1e_kin')
m_nuc = mol_lda.intor('int1e_nuc')
h1e = mf_lda.get_hcore()
veff = mf_lda.get_veff()

etot_lda = mf_lda.e_tot
ekin_lda = np.einsum('ij,ji', dm, m_kin)
enuc_lda = np.einsum('ij,ji', dm, m_nuc)
ecoul_lda = veff.ecoul
exc_lda = veff.exc
ecoul_nuc_lda = mf_lda.energy_nuc()

print('total energy', etot_lda)
print('kinetic energy', ekin_lda)
print('nuclear energy', enuc_lda)
print('coulomb repulsion energy', ecoul_lda)
print('exc energy', exc_lda)
print('coulomb attraction', ecoul_nuc_lda)
print('energy_sum', ekin_lda + enuc_lda + ecoul_lda + exc_lda + ecoul_nuc_lda)

# %%
coords_scale = (samp['coords'] - samp['pos_shift']) / param.BOHR
mol = dataset.mols[idx[0]]
ao = numint.eval_ao(mol, coords_scale[0], deriv=0)
rho_lda = numint.eval_rho2(mol, ao, mo_coeff=mf_lda.mo_coeff,
                           mo_occ=mf_lda.mo_occ, xctype='LDA')
dens_lda = torch.from_numpy(rho_lda)
dens_lda_base = dens_lda.clone() 
# Evaluate energy for sample density

ni = numint.NumInt()
exc_eff_lda = torch.from_numpy(ni.eval_xc_eff('lda', rho_lda, deriv=0)[0])
print('exchange correlation', exc_eff_lda.shape)
exc_sum_lda = torch.sum(dens_lda * exc_eff_lda * samp['coord_weights'])
print('exc sub lda', exc_sum_lda)
print('exc_lda', exc_lda)
print('exc diff', np.abs(exc_sum_lda - exc_lda))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_lda - exc_lda)))

# %%
ni = numint.NumInt()
exc_eff_lda = torch.from_numpy(ni.eval_xc_eff('lda', samp['density'].numpy(force=True).astype(np.double)[0],
                                              deriv=0)[0])
print('dens integral', torch.sum(samp['density'] * samp['coord_weights']))
print('dens diff', torch.sum(torch.abs(samp['density'] - dens_lda_base.unsqueeze(0)) * samp['coord_weights'])
      / torch.sum(samp['batch_atom_numbers'], dim=-1))
print('exchange correlation', exc_eff_lda.shape)
exc_sum_lda = torch.sum(samp['density'] * exc_eff_lda * samp['coord_weights'])
print('exc sub lda', exc_sum_lda)
print('exc_lda', exc_lda)
print('exc diff', np.abs(exc_sum_lda - exc_lda))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_lda - exc_lda)))

# %%
# Evaluate LDA energy for ML-DF coeffs
df_bases_lda, auxmols_lda = orbitals.ml_basis_to_df_coeffs(res, basis, [mf_lda.mo_coeff], [mf_lda.mo_occ])
coords = samp['coords'] / param.BOHR
auxmol_lda = auxmols_lda[0]
ao = numint.eval_ao(auxmol_lda, coords[0])
print('ao.shape', ao.shape)
rho_lda = np.einsum('ij,j->i', ao, df_bases_lda[0])
dens_lda = torch.from_numpy(rho_lda)
print('dens integral', torch.sum(dens_lda * samp['coord_weights']))
print('dens diff', torch.sum(torch.abs(samp['density'] - dens_lda.unsqueeze(0)) * samp['coord_weights'])
      / torch.sum(samp['batch_atom_numbers'], dim=-1))
print('dens diff', torch.sum(torch.abs(dens_lda.unsqueeze(0) - dens_lda_base.unsqueeze(0)) * samp['coord_weights'])
      / torch.sum(samp['batch_atom_numbers'], dim=-1))

ni = numint.NumInt()

exc_eff_lda = torch.from_numpy(ni.eval_xc_eff('lda', rho_lda, deriv=0)[0])
print('exchange correlation', exc_eff_lda.shape)
exc_sum_lda = torch.sum(dens_lda * exc_eff_lda * samp['coord_weights'])
print('exc energy', exc_sum_lda)
print('exc diff', np.abs(exc_sum_lda - exc_lda))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_lda - exc_lda)))

# %%
# Evaluate LDA energy for DF coeffs
auxmol = gto.M(atom=atom, basis=auxbasis)
auxmol.build()
ml_coeffs_lda = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']}, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=True, convert_to_equiv_dens=True, radial_basis=dataset.radial_coeffs)
res_df_lda = {key: res[key] for key in res}
res_df_lda['spherical_coeffs'] = ml_coeffs_lda['spherical_coeffs']
res_df_lda['radial_width'] = ml_coeffs_lda['radial_width']
res_df_lda['radial_scale'] = ml_coeffs_lda['radial_scale']
df_bases_lda, auxmols_lda = orbitals.ml_basis_to_df_coeffs(res_df_lda, basis, [mf_lda.mo_coeff], [mf_lda.mo_occ])
coords = samp['coords'] / param.BOHR
auxmol_lda = auxmols_lda[0]
ao = numint.eval_ao(auxmol_lda, coords[0])
print('ao.shape', ao.shape)
rho_lda = np.einsum('ij,j->i', ao, df_bases_lda[0])
dens_lda = torch.from_numpy(rho_lda)
print('dens integral', torch.sum(dens_lda * samp['coord_weights']))

ni = numint.NumInt()

exc_eff_lda = torch.from_numpy(ni.eval_xc_eff('lda', rho_lda, deriv=0)[0])
print('exchange correlation', exc_eff_lda.shape)
exc_sum_lda = torch.sum(dens_lda * exc_eff_lda * samp['coord_weights'])
print('exc energy', exc_sum_lda)
print('exc diff', np.abs(exc_sum_lda - exc_lda))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_lda - exc_lda)))

# %%
# Evaluate LDA energy for ML coeffs
auxmol_lda = orbitals.ml_basis_to_auxmol(res)
df_coeffs_lda = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                               radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
coords = samp['coords'] / param.BOHR
ao = numint.eval_ao(auxmol_lda, coords[0], deriv=0)
print('ao.shape', ao.shape)
rho_lda = np.einsum('ij,j->i', ao, df_coeffs_lda)
dens_lda = torch.from_numpy(rho_lda)
print('dens integral', torch.sum(dens_lda * samp['coord_weights']))

ni = numint.NumInt()

exc_eff_lda = torch.from_numpy(ni.eval_xc_eff('lda', rho_lda, deriv=0)[0])
print('exchange correlation', exc_eff_lda.shape)
exc_sum_lda = torch.sum(dens_lda * exc_eff_lda * samp['coord_weights'])
print('exc energy', exc_sum_lda)
print('exc diff', np.abs(exc_sum_lda - exc_lda))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_lda - exc_lda)))

# %%
# Calculate energy for PBE functional
basis = 'augccpvdz'
auxbasis = 'augccpvqzjkfit'

start = time.time()
pos = samp['batch_positions'][0]
atom_types = samp['batch_atom_numbers'][0]
print('pos shape', pos.shape)
print('atom nums shape', atom_types.shape)
atom = []
for j in range(len(atom_types)):
    atom.append((atom_types[j].numpy(force=True), pos[j, :].numpy(force=True)))
print(atom)

mol = gto.M(atom=atom, basis=basis)
print('mol pack', mol.pack())
# print(mol.pack())
mf_pbe = dft.RKS(mol)
mf_pbe.chkfile = False
mf_pbe.xc = 'pbe'
# mf.max_cycle = 1000
mf_pbe.kernel()
print('time elapsed', time.time() - start)
print('total energy regular', mf_pbe.e_tot)
dm = mf_pbe.make_rdm1()
m_kin = mol.intor('int1e_kin')
m_nuc = mol.intor('int1e_nuc')
h1e_pbe = mf_pbe.get_hcore()
veff_pbe = mf_pbe.get_veff()
exc_pbe = veff_pbe.exc
ecoul_pbe = veff_pbe.ecoul
ecoul_nuc_pbe = mf_pbe.energy_nuc()

etot_pbe = mf_pbe.e_tot
ekin_pbe = np.einsum('ij,ji', dm, m_kin)
enuc_pbe = np.einsum('ij,ji', dm, m_nuc)

print('total energy', etot_pbe)
print('kinetic energy', ekin_pbe)
print('nuclear energy', enuc_pbe)
print('coulomb repulsion energy', ecoul_pbe)
print('exc energy', exc_pbe)
print('coulomb attraction', ecoul_nuc_pbe)
print('energy_sum', ekin_pbe + enuc_pbe + ecoul_pbe + exc_pbe + ecoul_nuc_pbe)
# %%
# Evaluate PBE energy for density from MO coeffs from calculation
coords_scale = (samp['coords'] - samp['pos_shift']) / param.BOHR
mol = dataset.mols[idx[0]]
mol.build()
ao = numint.eval_ao(mol, coords_scale[0], deriv=1)
rho_pbe = numint.eval_rho2(mol, ao, mo_coeff=mf_pbe.mo_coeff,
                           mo_occ=mf_pbe.mo_occ, xctype='GGA')
dens_pbe = torch.from_numpy(rho_pbe)
dens_pbe_base = dens_pbe.clone()
# Evaluate energy for sample density
print('dens integral', torch.sum(dens_pbe_base[[0]] * samp['coord_weights']))

ni = numint.NumInt()
exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', rho_pbe, deriv=1, xctype='GGA')[0])
dens_calc = dens_pbe_base[0] * samp['coord_weights']
print('exchange correlation', exc_eff_pbe.shape)
exc_sum_pbe = torch.sum(exc_eff_pbe * dens_calc)
print('exc_pbe', exc_pbe)
print('exc diff', np.abs(exc_sum_pbe - exc_pbe))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_pbe - exc_pbe)))
# %%
# Evaluate PBE energy for density from dataset
coords_scale = (samp['coords'] - samp['pos_shift']) / param.BOHR
coeffs = dataset.coeffs[idx[0]]
mol = dataset.mols[idx[0]]
mol.build()
ao = numint.eval_ao(mol, coords_scale[0], deriv=1)
rho_pbe = numint.eval_rho2(mol, ao, mo_coeff=coeffs['mo_coeff'],
                           mo_occ=coeffs['mo_occ'], xctype='GGA')
dens_pbe = torch.from_numpy(rho_pbe)
print('dens integral', torch.sum(dens_pbe[[0]] * samp['coord_weights']))
print('dens samp integral', torch.sum(samp['density'] * samp['coord_weights']))
print('dens diff', torch.sum(torch.abs(samp['density'] - dens_pbe_base[[0]]) * samp['coord_weights'])
      / torch.sum(samp['batch_atom_numbers'], dim=-1))
print('dens diff', torch.sum(torch.abs(dens_pbe[[0]] - dens_pbe_base[[0]]) * samp['coord_weights'])
      / torch.sum(samp['batch_atom_numbers'], dim=-1))
exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', rho_pbe, deriv=1, xctype='GGA')[0])
dens_calc = dens_pbe[0] * samp['coord_weights']
print('exchange correlation', exc_eff_pbe.shape)
exc_sum_pbe = torch.sum(exc_eff_pbe * dens_calc)
print('exc_pbe', exc_pbe)
print('exc diff', np.abs(exc_sum_pbe - exc_pbe))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_pbe - exc_pbe)))
# %%
# Evaluate PBE energy for ML-DF coeffs
df_bases_pbe, auxmols_pbe = orbitals.ml_basis_to_df_coeffs(res, basis, [mf_pbe.mo_coeff], [mf_pbe.mo_occ])
coords = samp['coords'] / param.BOHR
auxmol_pbe = auxmols_pbe[0]
ao = numint.eval_ao(auxmol_pbe, coords[0], deriv=1)
print('ao.shape', ao.shape)
rho_pbe = np.einsum('ijk,k->ij', ao, df_bases_pbe[0])
dens_pbe = torch.from_numpy(rho_pbe)
print('dens integral', torch.sum(dens_pbe[[0]] * samp['coord_weights']))

ni = numint.NumInt()

exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', rho_pbe, deriv=1, xctype='GGA')[0])
dens_calc = dens_pbe[0] * samp['coord_weights']
print('exchange correlation', exc_eff_pbe.shape)
exc_sum_pbe = torch.sum(exc_eff_pbe * dens_calc)
print('exc energy', exc_sum_pbe)
print('exc diff', np.abs(exc_sum_pbe - exc_pbe))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_pbe - exc_pbe)))

# %%
# Evaluate PBE energy for DF coeffs
auxmol = gto.M(atom=atom, basis=auxbasis)
auxmol.build()
ml_coeffs_pbe = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']}, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=True, convert_to_equiv_dens=True, radial_basis=dataset.radial_coeffs)
res_df_pbe = {key: res[key] for key in res}
res_df_pbe['spherical_coeffs'] = ml_coeffs_pbe['spherical_coeffs']
res_df_pbe['radial_width'] = ml_coeffs_pbe['radial_width']
res_df_pbe['radial_scale'] = ml_coeffs_pbe['radial_scale']
df_bases_pbe, auxmols_pbe = orbitals.ml_basis_to_df_coeffs(res_df_pbe, basis, [mf_pbe.mo_coeff], [mf_pbe.mo_occ])
coords = samp['coords'] / param.BOHR
auxmol_pbe = auxmols_pbe[0]
ao = numint.eval_ao(auxmol_pbe, coords[0], deriv=1)
print('ao.shape', ao.shape)
rho_pbe = np.einsum('ijk,k->ij', ao, df_bases_pbe[0])
dens_pbe = torch.from_numpy(rho_pbe)
print('dens integral', torch.sum(dens_pbe[[0]] * samp['coord_weights']))

ni = numint.NumInt()

exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', rho_pbe, deriv=1, xctype='GGA')[0])
dens_calc = dens_pbe[0] * samp['coord_weights']
print('exchange correlation', exc_eff_pbe.shape)
exc_sum_pbe = torch.sum(exc_eff_pbe * dens_calc)
print('exc energy', exc_sum_pbe)
print('exc diff', np.abs(exc_sum_pbe - exc_pbe))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_pbe - exc_pbe)))

# %%
# Evaluate PBE energy for ML coeffs
auxmol_pbe = orbitals.ml_basis_to_auxmol(res)
df_coeffs_pbe = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
coords = samp['coords'] / param.BOHR
ao = numint.eval_ao(auxmol_pbe, coords[0], deriv=1)
print('ao.shape', ao.shape)
rho_pbe = np.einsum('ijk,k->ij', ao, df_coeffs_pbe)
dens_pbe = torch.from_numpy(rho_pbe)
print('dens integral', torch.sum(dens_pbe * samp['coord_weights']))

ni = numint.NumInt()

exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', rho_pbe, deriv=1, xctype='GGA')[0])
dens_calc = dens_pbe[0] * samp['coord_weights']
print('exchange correlation', exc_eff_pbe.shape)
exc_sum_pbe = torch.sum(exc_eff_pbe * dens_calc)
print('exc energy', exc_sum_pbe)
print('exc diff', np.abs(exc_sum_pbe - exc_pbe))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_pbe - exc_pbe)))

# %%
# Evaluate PBE energy for ML coeffs coreless
auxmol_pbe = orbitals.ml_basis_to_auxmol(res)
df_coeffs_pbe = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
coords = samp['coords'] / param.BOHR
ao = numint.eval_ao(auxmol_pbe, coords[0], deriv=1)
print('ao.shape', ao.shape)
rho_pbe = np.einsum('ijk,k->ij', ao, df_coeffs_pbe)
dens_pbe = torch.from_numpy(rho_pbe)
print('dens integral', torch.sum(dens_pbe * samp['coord_weights']))

rho_atoms = 0
atom_dens_dict = dataset.atom_dens
for i in range(res['batch_positions'].shape[1]):
    anum = int(torch.max(res['batch_atom_numbers'][:, i]))
    mo_coeffs = atom_dens_dict[anum]
    coeffs = [{'mo_coeff': mo_coeffs['mo_coeff'],
                'mo_occ': mo_coeffs['mo_occ']}] * coords.shape[0]
    atom = utils.npy_to_pyscf(res['batch_positions'][:, [i]].numpy(force=True),
                              res['batch_atom_numbers'][:, [i]].numpy(force=True),
                              basis)
    mol = atom[0]
    # print('mo coeff shape', coeffs[0]['mo_coeff'].shape)
    ao = numint.eval_ao(mol, coords[0], deriv=1)
    # print('ao shape', ao.shape)
    rho_at = numint.eval_rho2(mol, ao, mo_coeff=coeffs[0]['mo_coeff'], mo_occ=coeffs[0]['mo_occ'],
                              xctype='GGA')
    # print('rho at shape', rho_at.shape)
    rho_atoms += rho_at
    print('rho_atoms shape', rho_atoms.shape)

dens_pbe += torch.from_numpy(rho_atoms)
print('dens integral', torch.sum(dens_pbe[[0]] * samp['coord_weights']))

ni = numint.NumInt()

exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', rho_pbe, deriv=1, xctype='GGA')[0])
dens_calc = dens_pbe[0] * samp['coord_weights']
print('exchange correlation', exc_eff_pbe.shape)
exc_sum_pbe = torch.sum(exc_eff_pbe * dens_calc)
print('exc energy', exc_sum_pbe)
print('exc diff', np.abs(exc_sum_pbe - exc_pbe))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_pbe - exc_pbe)))

# %%
# Evaluate PBE energy for ML coeffs coreless, from spline
auxmol_pbe = orbitals.ml_basis_to_auxmol(res)
df_coeffs_pbe = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                               radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
coords_scale = samp['coords'] / param.BOHR
ao = numint.eval_ao(auxmol_pbe, coords_scale[0], deriv=1)
# print('ao.shape', ao.shape)
rho_pbe = np.einsum('ijk,k->ij', ao, df_coeffs_pbe)
dens_pbe = torch.from_numpy(rho_pbe)
print('dens integral', torch.sum(dens_pbe[[0]] * samp['coord_weights']))

rho_atoms = 0
atom_dens_dict = dataset.atom_dens
for i in range(res['batch_positions'].shape[1]):
    anum = int(torch.max(res['batch_atom_numbers'][:, i]))
    anum_nz = anum != 0
    coords_at = samp['coords'] - res['batch_positions'][:, i]
    spline_basis = dataset.atom_dens[anum]['spline_interp']
    x_in = torch.norm(coords_at[0], dim=-1) / param.BOHR
    rho_at = torch.from_numpy(spline_basis(np.log(x_in)))
    deriv = spline_basis.derivative()
    # dspline(ln(norm(r))) / dr = dspline / dln(norm(r)) * dln(norm(r)) / dnorm(r) * dnorm(r) / dr
    spline_deriv = torch.from_numpy(deriv(np.log(x_in)))
    rho_deriv = spline_deriv.unsqueeze(-1) * (1 / x_in).unsqueeze(-1) * (coords_at[0] / x_in.unsqueeze(-1)) / param.BOHR
    rho_deriv = torch.permute(rho_deriv, (1, 0))

    rho_at[rho_at < 0] = 0
    rho_at = torch.cat([rho_at.unsqueeze(0), rho_deriv], dim=0)
    #
    print('rho_at integral', torch.sum(rho_at[[0]] * samp['coord_weights']))
    # print('rho at shape', rho_at.shape)
    rho_atoms += rho_at
print('dens integral', torch.sum(dens_pbe[[0]] * samp['coord_weights']))
dens_pbe += rho_atoms
print('dens pbe shape', dens_pbe.shape)
print('rho atoms integral', torch.sum(rho_atoms[[0]] * samp['coord_weights']))
print('dens integral', torch.sum(dens_pbe[[0]] * samp['coord_weights']))
rho_pbe = dens_pbe.numpy(force=True)
ni = numint.NumInt()

exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', rho_pbe, deriv=1, xctype='GGA')[0])
dens_calc = dens_pbe[0] * samp['coord_weights']
print('exchange correlation', exc_eff_pbe.shape)
exc_sum_pbe = torch.sum(exc_eff_pbe * dens_calc)
print('exc energy', exc_sum_pbe)
print('exc diff', np.abs(exc_sum_pbe - exc_pbe))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_pbe - exc_pbe)))

# %%
# Compare analytical derivatives between mo coeffs and spline
rho_atoms = 0
atom_dens_dict = dataset.atom_dens
coords_scale = samp['coords'] / param.BOHR
for i in range(res['batch_positions'].shape[1]):
    anum = int(torch.max(res['batch_atom_numbers'][:, i]))
    mo_coeffs = atom_dens_dict[anum]
    coeffs = [{'mo_coeff': mo_coeffs['mo_coeff'],
               'mo_occ': mo_coeffs['mo_occ']}] * coords.shape[0]
    atom = utils.npy_to_pyscf(res['batch_positions'][:, [i]].numpy(force=True),
                              res['batch_atom_numbers'][:, [i]].numpy(force=True),
                              basis)
    mol = atom[0]
    # print('mo coeff shape', coeffs[0]['mo_coeff'].shape)
    ao = numint.eval_ao(mol, coords_scale[0], deriv=1)
    # print('ao shape', ao.shape)
    rho_mo = numint.eval_rho2(mol, ao, mo_coeff=coeffs[0]['mo_coeff'], mo_occ=coeffs[0]['mo_occ'],
                              xctype='GGA')
    rho_mo = torch.from_numpy(rho_mo)
    # print('rho at shape', rho_at.shape)
    # rho_atoms += rho_at
    # print('rho_atoms shape', rho_atoms.shape)

    deltar = float(0.00001)
    coords_at = samp['coords'] - res['batch_positions'][:, i]
    coords_delta = []
    for i in range(3):
        deltac = torch.zeros((1, 1, 3))
        deltac[0, 0, i] = deltar
        coords_delta.append(coords_at - deltac / 2)
        coords_delta.append(coords_at + deltac / 2)

    coords_delta = torch.cat(coords_delta, dim=0)
    spline_basis = atom_dens_dict[anum]['spline_interp']
    x_in = torch.norm(coords_at[0], dim=-1) / param.BOHR
    x_in_delta = torch.norm(coords_delta, dim=-1) / param.BOHR
    rho_sp_delta = torch.from_numpy(spline_basis(np.log(x_in_delta)))
    # print('rho sp delta', rho_sp_delta.shape)
    rho_sp_num_delta = torch.zeros_like(coords_at)[0]
    # print('rho sp num delta', rho_sp_num_delta.shape)
    rho_sp_num_delta[:, 0] = (rho_sp_delta[1, :] - rho_sp_delta[0, :]) / deltar
    rho_sp_num_delta[:, 1] = (rho_sp_delta[3, :] - rho_sp_delta[2, :]) / deltar
    rho_sp_num_delta[:, 2] = (rho_sp_delta[5, :] - rho_sp_delta[4, :]) / deltar
    rho_sp = torch.from_numpy(spline_basis(np.log(x_in)))
    deriv = spline_basis.derivative()
    # dspline(ln(norm(r))) / dr = dspline / dln(norm(r)/bohr) * dln(norm(r)/bohr) / dnorm(r)/bohr * dnorm(r)/bohr / dr
    spline_deriv = torch.from_numpy(deriv(np.log(x_in))).to(torch.float64)
    delta = float(0.0035)
    plus_delta = torch.from_numpy(spline_basis(np.log(x_in) + delta / 2)).to(torch.float64)
    minus_delta = torch.from_numpy(spline_basis(np.log(x_in) - delta / 2)).to(torch.float64)
    spline_num_deriv = (plus_delta - minus_delta) / delta
    # print('spline deriv', spline_deriv)
    # print('spline num deriv', spline_num_deriv)
    print('delta diff', torch.sum(torch.abs(spline_deriv - spline_num_deriv) * samp['coord_weights']))
    rho_deriv = spline_deriv.unsqueeze(-1) * (coords_at[0] / (x_in**2).unsqueeze(-1)) / param.BOHR
    # rho_deriv *= 1/0.28
    rho_deriv_rat = torch.zeros_like(rho_deriv)
    print('rho_mo deriv', rho_mo[1:])
    print('rho_deriv', rho_deriv)
    print('rho_sp_num_delta', rho_sp_num_delta)
    rho_deriv_rat[rho_sp_num_delta != 0] = rho_deriv[rho_sp_num_delta != 0] / rho_sp_num_delta[rho_sp_num_delta != 0]
    print('rho_deriv ratio', rho_deriv_rat)
    print('rho_deriv diff', torch.sum(torch.abs(rho_deriv - rho_sp_num_delta) * samp['coord_weights'].squeeze().unsqueeze(-1)))
    rho_deriv = torch.permute(rho_deriv, (1, 0))

    rho_sp[rho_sp < 0] = 0
    rho_sp = torch.cat([rho_sp.unsqueeze(0), rho_deriv], dim=0)
    #
    print('rho_sp integral', torch.sum(rho_sp[[0]] * samp['coord_weights']))
    print('rho_mo integral', torch.sum(rho_mo[[0]] * samp['coord_weights']))
    print('rho diff', torch.sum(torch.abs(rho_sp[[0]] - rho_mo[[0]]) * samp['coord_weights']))

    print('rho_sp deriv integral', torch.sum(rho_sp[1:] * samp['coord_weights']))
    print('rho_mo deriv integral', torch.sum(rho_mo[1:] * samp['coord_weights']))
    print('rho deriv diff', torch.sum(torch.abs(rho_sp[1:] - rho_mo[1:]) * samp['coord_weights']))
    # print('rho at shape', rho_at.shape)
    # rho_atoms += rho_at
# %%
# Load dataset with option for density gradients
args.spherical_grid_level = 1
args.density_grad = False
grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
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
                           split_atom_dens=True,
                           density_grad=True,
                           )

print('dataset length', len(dataset))
print('sample pos shape', dataset.get_properties([0])['positions'].shape)
print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
print('num samples', main_args.num_samples)
print('args use gpu', args.use_gpu)

# %%
model = model_loader.load_model(args, dataset)
model.eval()
idx = [3]
samp = dataset.get_properties(idx)
samp_df = dataset_df.get_properties(idx)

res = model(samp)

print('samp df_coeffs', samp_df['df_coeffs'].shape)
print('res df_coeffs', res['df_coeffs'].shape)

# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
print('density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
print('density integral', torch.sum(samp_df['density'] * samp_df['coord_weights'], dim=1))
print('density error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1)
                       / torch.sum(samp['batch_atom_numbers'], dim=1))
print('density df error', torch.sum(torch.abs(samp_df['density'] - samp['density']) * samp['coord_weights'], dim=1)
                          / torch.sum(samp['batch_atom_numbers'], dim=1))
print('res density grad', res['density_grad'].shape)
# %%
# Evaluate PBE energy for density with derivatives directly from data loader
ni = numint.NumInt()
samp = dataset.get_properties(idx)
coords_scale = (samp['coords'] - samp['pos_shift']) / param.BOHR
print('dens integral', torch.sum(samp['density'] * samp['coord_weights']))
dens_pbe = torch.permute(torch.cat([samp['density'].unsqueeze(-1), samp['density_grad']], dim=-1), (2, 1, 0)).squeeze()
print('dens_deriv integral', torch.sum(torch.abs(dens_pbe[1:] * samp['coord_weights'])))
print(dens_pbe.shape)
exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', dens_pbe.numpy(force=True).astype(np.double), deriv=1, xctype='GGA')[0])
dens_calc = dens_pbe[0] * samp['coord_weights']
# print('exchange correlation', exc_eff_pbe.shape)
exc_pbe = torch.sum(exc_eff_pbe * dens_calc)
# print('exc_pbe', exc_pbe)
# print('exc diff', np.abs(exc_sum_pbe - exc_pbe))
# print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_pbe - exc_pbe)))
# %%
# Evaluate PBE energy for ML coeffs coreless, from spline from dataset
auxmol_pbe = orbitals.ml_basis_to_auxmol(res)
df_coeffs_pbe = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                               radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
coords_scale = samp['coords'] / param.BOHR
ao = numint.eval_ao(auxmol_pbe, coords_scale[0], deriv=1)
# print('ao.shape', ao.shape)
rho_pbe = np.einsum('ijk,k->ij', ao, df_coeffs_pbe)
dens_pbe = torch.from_numpy(rho_pbe)
print('dens integral', torch.sum(dens_pbe[[0]] * samp['coord_weights']))
print('dens_deriv integral', torch.sum(torch.abs(dens_pbe[1:] * samp['coord_weights'])))
print('dens pbd 1: shape', dens_pbe[1:].shape)
print('atom density grad shape', samp['atom_density_grad'][0].shape)
print('atom dens_deriv integral', torch.sum(torch.abs(samp['atom_density_grad'][0].t() * samp['coord_weights'])))
print('dens_pbe shape', dens_pbe.shape)
print('atom dens dataset shape', samp['atom_density'].shape)
print('atom dens grad dataset shape', samp['atom_density_grad'].shape)
dens_at = torch.cat([samp['atom_density'], samp['atom_density_grad'].squeeze().t()], dim=0)
print('dens at shape', dens_at.shape)

dens_pbe += dens_at
print('dens pbe shape', dens_pbe.shape)
print('rho atoms integral', torch.sum(dens_at[[0]] * samp['coord_weights']))
print('dens integral', torch.sum(dens_pbe[[0]] * samp['coord_weights']))
rho_pbe = dens_pbe.numpy(force=True)
ni = numint.NumInt()

exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', rho_pbe, deriv=1, xctype='GGA')[0])
dens_calc = dens_pbe[0] * samp['coord_weights']
print('exchange correlation', exc_eff_pbe.shape)
exc_sum_pbe = torch.sum(exc_eff_pbe * dens_calc)
print('exc energy', exc_sum_pbe)
print('exc diff', np.abs(exc_sum_pbe - exc_pbe))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_pbe - exc_pbe)))
# %%
# Evaluate PBE energy for ML coeffs coreless, from spline from dataset
coords_scale = (samp['coords'] - samp['pos_shift']) / param.BOHR
res = model(samp)
dens_pbe = torch.permute(torch.cat([res['density'].unsqueeze(-1), res['density_grad']], dim=-1), (2, 1, 0)).squeeze()
print(dens_pbe.shape)
print('dens integral', torch.sum(dens_pbe[[0]] * samp['coord_weights']))
print('dens_deriv integral', torch.sum(torch.abs(dens_pbe[1:] * samp['coord_weights'])))
exc_eff_pbe = torch.from_numpy(ni.eval_xc_eff('pbe', dens_pbe.numpy(force=True).astype(np.double), deriv=1, xctype='GGA')[0])
dens_calc = dens_pbe[0] * samp['coord_weights']
# print('exchange correlation', exc_eff_pbe.shape)
exc_sum_pbe = torch.sum(exc_eff_pbe * dens_calc)
print('exc energy', exc_sum_pbe)
print('exc diff', np.abs(exc_sum_pbe - exc_pbe))
print('exc diff kcal/mol', utils.hartree_to_kcal(np.abs(exc_sum_pbe - exc_pbe)))
