import os
from pyscf.dft import numint
from pyscf.lib import param
from datetime import datetime
from pyscf import gto, dft, df, lib
from pyscf.scf import hf

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
import scipy
from equiv_dens.training import model_loader
import time
# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..
# %%
main_args = Namespace()

# main_args.args_file = "args/resorcinol_all_001.txt"
# main_args.args_file = "args/ethanethiol_all_006_test.txt"
main_args.args_file = "args/ethanethiol_all_106_test.txt"
# main_args.args_file = "args/h2o_small_all_001.txt"
# main_args.args_file = "args/ethanethiol_df_coeffs_001_test.txt"
# main_args.args_file = "args/ethanethiol_all_001_coreless_test.txt"
# main_args.args_file = "args/ethanethiol_all_004_coreless.txt"
# main_args.args_file = "args/ethanethiol_all_001_SH_even.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
# main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
# main_args.res_load_file = 'datasets/ethanethiol_all_001_coreless_test_results.npy'
# main_args.res_load_file = None
# main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
# main_args.save_file = 'ethanethiol_all_006'
main_args.save_file = 'ethanethiol_all_106'
# main_args.save_file = 'h2o_small_all_001'
# main_args.save_file = 'resorcinol_all_005'
# main_args.save_file = 'ethanethiol_all_001_coreless'
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
print('density error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1))
print('density df error', torch.sum(torch.abs(samp_df['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1))
# %%
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
# print(mol.pack())
mf = dft.RKS(mol)
mf.chkfile = False
mf.xc = 'pbe'
# mf.max_cycle = 1000
mf.kernel()
print('time elapsed', time.time() - start)
print('total energy regular', mf.e_tot)
# %%
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
# print(mol.pack())
mf = dft.RKS(mol).density_fit(auxbasis=auxbasis)
mf.chkfile = False
mf.xc = 'pbe'
# mf.max_cycle = 1000
mf.kernel()
print('time elapsed', time.time() - start)
print('total energy df', mf.e_tot)
# %%
print(mf.with_df.auxmol._basis)
auxmol = gto.M(atom=atom, basis=auxbasis)
auxmol.build()
auxmol = orbitals.ml_basis_to_pyscf_env(res, auxmol)
print(auxmol._basis)
# %%
atom_types = []

num_dict = {}

for an in samp['atom_numbers'][0]:
    an = an.item()
    symb = utils.numbers_to_symbols([an])[0]
    if an in num_dict:
        atom_types.append(symb + str(num_dict[an]))
        num_dict[an] += 1
    else:
        atom_types.append(symb + str(0))
        num_dict[an] = 1
print('num_dict', num_dict)
print('atom_types', atom_types)
# %%
samp_df = dataset_df.get_properties(idx)

df_coeffs = samp_df['df_coeffs']
df_coeffs_dict = orbitals.vector_to_coeffs_dict({'spherical_coeffs': df_coeffs},
                                                dataset.orbital_basis_num,
                                                samp['batch_atom_numbers'],
                                                convert_to_equiv_dens=False,
                                                radial_basis=dataset.radial_coeffs)
print(df_coeffs)
# print(df_coeffs_dict)

basis_dict = {}

for i in range(len(df_coeffs_dict['radial_scale'])):
    at_symb = atom_types[i]
    for key in df_coeffs_dict['radial_scale'][i].keys():
        z, L = key
        radial_widths = df_coeffs_dict['radial_width'][i][key].squeeze()
        radial_scales = df_coeffs_dict['radial_scale'][i][key].squeeze()
        # print('z, L', z, L)
        # print('radial widths shape', radial_widths.shape)
        g_norm = orbitals.gto_norm_pyscf(L, radial_widths)
        radial_scales = radial_scales / g_norm
        if at_symb in basis_dict:
            for j in range(radial_widths.shape[-1]):
                basis_dict[at_symb].append([[L, [radial_widths[j].item(), radial_scales[j].item()]]])
        if at_symb not in basis_dict:
            basis_dict[at_symb] = [[[L, [radial_widths[j].item(), radial_scales[j].item()]]
                                   for j in range(radial_widths.shape[-1])]]
print(basis_dict)
# %%
atom_new = []
for j in range(len(atom_types)):
    atom_new.append((atom_types[j], pos[j, :].numpy(force=True)))
print('atom', atom_new)
auxmol_2 = gto.M(atom=atom_new, basis=basis_dict)
auxmol_2.build()
# auxmol = orbitals.ml_basis_to_pyscf_env(res, auxmol)
print(auxmol_2._basis)
# %%
mol = gto.M(atom=atom, basis=basis)
mf = dft.RKS(mol)
mf.chkfile = False
mf.xc = 'pbe'
mf.kernel()
dm1 = hf.make_rdm1(mf.mo_coeff, mf.mo_occ)
auxmol = gto.M(atom=atom, basis=auxbasis)

# Define the auxiliary fitting basis for 3-center integrals. Use the function
# make_auxmol to construct the auxiliary Mole object (auxmol) which will be
# used to generate integrals.

# ints_3c is the 3-center integral tensor (ij|P), where i and j are the
# indices of AO basis and P is the auxiliary basis
ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
ints_2c2e = auxmol.intor('int2c2e')

nao = mol.nao
naux = auxmol.nao

df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
df_coef = df_coef.reshape(naux, nao, nao)
# print('df_coeff shape', df_coef.shape)
# print('atoms', auxmol_ext._atm)
df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)
print('df basis', df_basis)
# %%
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_basis], scale_coords=True, projected=True)
print('density diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
# %%
# Define the auxiliary fitting basis for 3-center integrals. Use the function
# make_auxmol to construct the auxiliary Mole object (auxmol) which will be
# used to generate integrals.

# ints_3c is the 3-center integral tensor (ij|P), where i and j are the
# indices of AO basis and P is the auxiliary basis
ints_3c2e = df.incore.aux_e2(mol, auxmol_2, intor='int3c2e')
ints_2c2e = auxmol_2.intor('int2c2e')

nao = mol.nao
naux = auxmol_2.nao

df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
df_coef = df_coef.reshape(naux, nao, nao)
# print('df_coeff shape', df_coef.shape)
# print('atoms', auxmol_ext._atm)
df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)
print('df basis', df_basis)
# %%
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_basis], scale_coords=True, projected=True)
print('density diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
# %%
df_bases, auxmol_exts = orbitals.ml_basis_to_df_coeffs(res, basis, auxbasis, [mf.mo_coeff], [mf.mo_occ])
dens = orbitals.sample_density_base(auxmol_exts, samp_df['coords'], df_bases, scale_coords=True, projected=True)
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
# %%
df_bases, auxmol_exts = orbitals.ml_basis_to_df_coeffs(res, basis, auxbasis, [mf.mo_coeff], [mf.mo_occ])
dens = orbitals.sample_density_base(auxmol_exts, samp_df['coords'], df_bases, scale_coords=True, projected=True)
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density res diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
# %%
auxmol = orbitals.ml_basis_to_auxmol(res)
df_coeffs = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_coeffs], scale_coords=True, projected=True)
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
# %%
idx = [3]
samp = dataset.get_properties(idx)
samp_df = dataset_df.get_properties(idx)

res = model(samp)
print('density df diff', torch.sum(torch.abs(res['density'] - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(res['density'] - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density res diff', torch.sum(torch.abs(res['density'] - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
# %%
auxmol = gto.M(atom=atom, basis=auxbasis)
auxmol.build()
ml_coeffs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']}, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=True, convert_to_equiv_dens=True, radial_basis=dataset.radial_coeffs)
df_coeffs = orbitals.coeffs_dict_to_vector(ml_coeffs, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_coeffs], scale_coords=True, projected=True)
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

res_df = {key: res[key] for key in res}
res_df['spherical_coeffs'] = ml_coeffs['spherical_coeffs']
res_df['radial_width'] = ml_coeffs['radial_width']
res_df['radial_scale'] = ml_coeffs['radial_scale']
dens_df = model.property_models['density'](res_df)['density']
print('density df diff', torch.sum(torch.abs(dens_df - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens_df - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens_df - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

dens_ml = model.property_models['density'](res)['density']
print('density df diff', torch.sum(torch.abs(dens_ml - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens_ml - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens_ml - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

auxmol = orbitals.ml_basis_to_auxmol(res)
df_coeffs = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_coeffs], scale_coords=True, projected=True)
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

df_bases, auxmol_exts = orbitals.ml_basis_to_df_coeffs(res, basis, auxbasis, [mf.mo_coeff], [mf.mo_occ])
dens = orbitals.sample_density_base(auxmol_exts, samp_df['coords'], df_bases, scale_coords=True, projected=True)
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density res diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

auxmol = orbitals.ml_basis_to_auxmol(res_df)
df_coeffs = orbitals.coeffs_dict_to_vector(res_df, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_coeffs], scale_coords=True, projected=True)
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
# %%
idx = [3]
samp = dataset.get_properties(idx)
samp_df = dataset_df.get_properties(idx)

res = model(samp)
print('density integral', torch.sum(res['density'] * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(res['density'] - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(res['density'] - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density res diff', torch.sum(torch.abs(res['density'] - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

ml_coeffs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']}, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=True, convert_to_equiv_dens=True, radial_basis=dataset.radial_coeffs)
res_df = {key: res[key] for key in res}
res_df['spherical_coeffs'] = ml_coeffs['spherical_coeffs']
res_df['radial_width'] = ml_coeffs['radial_width']
res_df['radial_scale'] = ml_coeffs['radial_scale']

dens_df = model.property_models['density'](res_df)['density']
print('density integral', torch.sum(dens_df * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(dens_df - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens_df - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens_df - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

auxmol = orbitals.ml_basis_to_auxmol(res_df)
df_coeffs = orbitals.coeffs_dict_to_vector(res_df, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_coeffs], scale_coords=True, projected=True)
print('density integral', torch.sum(dens * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

auxmol = orbitals.ml_basis_to_auxmol(res)
df_coeffs = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_coeffs], scale_coords=True, projected=True)
print('density integral', torch.sum(dens * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
# %%
idx = [3]
samp = dataset.get_properties(idx)
samp_df = dataset_df.get_properties(idx)

res = model(samp)
print('density integral', torch.sum(res['density'] * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(res['density'] - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(res['density'] - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density res diff', torch.sum(torch.abs(res['density'] - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

ml_coeffs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']}, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=True, convert_to_equiv_dens=True, radial_basis=dataset.radial_coeffs)
res_df = {key: res[key] for key in res}
res_df['spherical_coeffs'] = ml_coeffs['spherical_coeffs']
res_df['radial_width'] = ml_coeffs['radial_width']
res_df['radial_scale'] = ml_coeffs['radial_scale']

dens_df = model.property_models['density'](res_df)['density']
print('density integral', torch.sum(dens_df * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(dens_df - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens_df - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens_df - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

auxmol = gto.M(atom=atom, basis=auxbasis)
auxmol = orbitals.ml_basis_to_pyscf_env(res_df, auxmol)
df_coeffs = orbitals.coeffs_dict_to_vector(res_df, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_coeffs], scale_coords=True, projected=True)
print('density integral', torch.sum(dens * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

auxmol = gto.M(atom=atom, basis=auxbasis)
auxmol = orbitals.ml_basis_to_pyscf_env(res, auxmol)
df_coeffs = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_coeffs], scale_coords=True, projected=True)
print('density integral', torch.sum(dens * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
# %%
idx = [3]
samp = dataset.get_properties(idx)
samp_df = dataset_df.get_properties(idx)

res = model(samp)
print('density integral', torch.sum(res['density'] * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(res['density'] - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(res['density'] - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density res diff', torch.sum(torch.abs(res['density'] - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

ml_coeffs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']}, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=True, convert_to_equiv_dens=True, radial_basis=dataset.radial_coeffs)
res_df = {key: res[key] for key in res}
res_df['spherical_coeffs'] = ml_coeffs['spherical_coeffs']
res_df['radial_width'] = ml_coeffs['radial_width']
res_df['radial_scale'] = ml_coeffs['radial_scale']

dens_df = model.property_models['density'](res_df)['density']
print('density integral', torch.sum(dens_df * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(dens_df - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens_df - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens_df - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

auxmol = orbitals.ml_basis_to_auxmol(res_df)
# print('radial_widths', res_df['radial_width'])
# print()
# print('radial_scales', res_df['radial_scale'])
# print()
# print('auxmol basis', auxmol.basis)
# print()
# print('auxmol env', auxmol._env)
# print()
df_coeffs = orbitals.coeffs_dict_to_vector(res_df, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_coeffs], scale_coords=True, projected=True)
print('density integral', torch.sum(dens * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))

auxmol = orbitals.ml_basis_to_auxmol(res)
# print('radial_widths', res['radial_width'])
# print()
# print('radial_scales', res['radial_scale'])
# print()
# print('auxmol basis', auxmol.basis)
# print()
# print('auxmol env', auxmol._env)
# print()
df_coeffs = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                           radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
dens = orbitals.sample_density_base([auxmol], samp_df['coords'], [df_coeffs], scale_coords=True, projected=True)
print('density integral', torch.sum(dens * samp_df['coord_weights'], dim=-1))
print('density df diff', torch.sum(torch.abs(dens - samp_df['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - samp['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
print('density diff', torch.sum(torch.abs(dens - res['density']) * samp_df['coord_weights'], dim=-1)/torch.sum(samp_df['atom_numbers'], dim=-1))
