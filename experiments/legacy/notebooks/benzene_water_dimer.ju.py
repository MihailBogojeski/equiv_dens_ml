# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..

# %%
import ase
import numpy as np
import pyscf
import time
import os
from pyscf.scf import hf
from pyscf import gto, dft, df, lib
from pyscf.gto import mole
import scipy

import torch
from pyscf.dft import gen_grid, radi, numint
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.errors import ErrorDict
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
     CubicalGrid, spherical_grid, spherical_radial_sampling
from equiv_dens.training.model_loader import load_model
import equiv_dens.utils.base as utils

from functools import partial

import ase.io
from pyscf.lib import param
from equiv_dens.training import utils as train_utils
from argparse import Namespace
from equiv_dens.training import model_loader
from equiv_dens.utils import orbitals
import copy

hf.MUTE_CHKFILE = True

# %%
# load dimer data
data = dict(np.load('datasets/water_benzene_results.npz', allow_pickle=True))
water_data = {'positions': data['water_positions'], 'atom_numbers': data['water_Z']}
benzene_data = {'positions': data['benzene_positions'], 'atom_numbers': data['benzene_Z']}

# %%
# calculate interatomic distances to confirm unit
water_centers = utils.center_of_mass(torch.tensor(water_data['positions']), torch.tensor(water_data['atom_numbers']))
benzene_centers = utils.center_of_mass(torch.tensor(benzene_data['positions']), torch.tensor(benzene_data['atom_numbers']))

dists = torch.norm(water_centers - benzene_centers, dim=-1)

# %%
# combine water and benzene positions
combined_pos = np.concatenate([water_data['positions'], benzene_data['positions']], axis=1)
combined_Z = np.concatenate([water_data['atom_numbers'], benzene_data['atom_numbers']], axis=1)
combined_data = {'positions': combined_pos, 'atom_numbers': combined_Z}

# %%
# calculate dft energies and estimate SAPT0 energies using pyscf
calc_sapt = []
coul_els = []
coul_en_components = []

for i in range(len(data['elst'])):
    curr_en_comps = {}
    # water molecule
    atom_types = water_data['atom_numbers'][i]
    pos = water_data['positions'][i]
    atom = []
    for j in range(len(atom_types)):
        atom.append((atom_types[j], pos[j, :]))
    w_mol = gto.M(atom=atom, basis='augccpvdz')
    # benzene molecule
    atom_types = benzene_data['atom_numbers'][i]
    pos = benzene_data['positions'][i]
    atom = []
    for j in range(len(atom_types)):
        atom.append((atom_types[j], pos[j, :]))
    b_mol = gto.M(atom=atom, basis='augccpvdz')
    # combined molecule
    atom_types = combined_data['atom_numbers'][i]
    pos = combined_data['positions'][i]
    atom = []
    for j in range(len(atom_types)):
        atom.append((atom_types[j], pos[j, :]))
    c_mol = gto.M(atom=atom, basis='augccpvdz')

    w_mf = dft.RKS(w_mol)
    w_mf.chkfile = False
    w_mf.xc = 'pbe'
    w_mf.kernel()
    w_coul = w_mf.get_veff().ecoul

    curr_en_comps['water_coul'] = w_coul
    curr_en_comps['water_mol'] = w_mol.pack()
    curr_en_comps['water_mo_coeffs'] = w_mf.mo_coeff
    curr_en_comps['water_mo_occ'] = w_mf.mo_occ

    auxbasis = 'augccpvqzjkfit'
    w_dm = w_mf.make_rdm1()
    w_auxmol = df.addons.make_auxmol(w_mol, auxbasis)

    ints_3c2e = df.incore.aux_e2(w_mol, w_auxmol, intor='int3c2e')
    ints_2c2e = w_auxmol.intor('int2c2e')

    nao = w_mol.nao
    naux = w_auxmol.nao
    w_df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
    w_df_coef = w_df_coef.reshape(naux, nao, nao)
    if w_dm.ndim > 2:
        w_df_basis = []
        for j in range(w_dm.shape[0]):
            w_df_basis.append(lib.einsum('Pij,ij->P', w_df_coef, w_dm[j]))
        w_df_basis = np.stack(w_df_basis, axis=0)
        print(w_df_basis.shape)

    else:
        w_df_basis = lib.einsum('Pij,ij->P', w_df_coef, w_dm)

    curr_en_comps['water_df_basis'] = w_df_basis
    b_mf = dft.RKS(b_mol)
    b_mf.chkfile = False
    b_mf.xc = 'pbe'
    b_mf.kernel()
    b_coul = b_mf.get_veff().ecoul
    print('kernel coul water', w_coul, 'coul benzene', b_coul)

    curr_en_comps['benzene_coul'] = b_coul
    curr_en_comps['benzene_mol'] = b_mol.pack()
    curr_en_comps['benzene_mo_coeffs'] = b_mf.mo_coeff
    curr_en_comps['benzene_mo_occ'] = b_mf.mo_occ

    b_dm = b_mf.make_rdm1()
    b_auxmol = df.addons.make_auxmol(b_mol, auxbasis)

    ints_3c2e = df.incore.aux_e2(b_mol, b_auxmol, intor='int3c2e')
    ints_2c2e = b_auxmol.intor('int2c2e')

    nao = b_mol.nao
    naux = b_auxmol.nao
    b_df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
    b_df_coef = b_df_coef.reshape(naux, nao, nao)
    if b_dm.ndim > 2:
        b_df_basis = []
        for j in range(b_dm.shape[0]):
            b_df_basis.append(lib.einsum('Pij,ij->P', b_df_coef, b_dm[j]))
        b_df_basis = np.stack(b_df_basis, axis=0)
        print(b_df_basis.shape)

    else:
        b_df_basis = lib.einsum('Pij,ij->P', b_df_coef, b_dm)

    curr_en_comps['benzene_df_basis'] = b_df_basis

    # c_mf = dft.RKS(c_mol)
    # c_mf.chkfile = False
    # c_mf.xc = 'pbe'
    # c_mf.kernel()
    # c_coul = c_mf.get_veff().ecoul

    # print('Coulomb energies, water:', w_coul, 'benzene:', b_coul, 'combined:', c_coul)
    # print('SAPT0', c_coul - w_coul - b_coul)
    m_nuc = c_mol.intor('int1e_nuc')

    c_dm = np.zeros((w_dm.shape[0] + b_dm.shape[0], w_dm.shape[1] + b_dm.shape[1]))
    c_dm[:41, :41] = w_dm
    c_dm[41:, 41:] = b_dm

    curr_en_comps['water_dm'] = w_dm
    curr_en_comps['benzene_dm'] = b_dm
    curr_en_comps['combined_dm'] = c_dm

    vj, _ = hf.get_jk(c_mol, c_dm)
    coul_en_c = np.einsum('ij,ji->', vj, c_dm).real * .5
    curr_en_comps['nonint_coul'] = coul_en_c

    c_df_basis = np.concatenate([w_df_basis, b_df_basis], axis=0)

    curr_en_comps['nonint_df_basis'] = c_df_basis
    auxmol = df.addons.make_auxmol(c_mol, auxbasis)
    #
    #

    # coulomb_mat = auxmol.intor('int2c2e')
    # coulomb_en = 0.5 * np.einsum('i,ij,j->', c_df_basis, coulomb_mat, c_df_basis)
    # coulomb_mat_b = b_auxmol.intor('int2c2e')
    # coulomb_mat_w = w_auxmol.intor('int2c2e')
    # coulomb_en_b = 0.5 * np.einsum('i,ij,j->', b_df_basis, coulomb_mat_b, b_df_basis)
    # coulomb_en_w = 0.5 * np.einsum('i,ij,j->', w_df_basis, coulomb_mat_w, w_df_basis)
    # print('coulomb en df', coulomb_en)
    # curr_en_comps['nonint_coul'] = coulomb_en
    # curr_en_comps['water_coul_df'] = coulomb_en_w
    # curr_en_comps['benzene_coul_df'] = coulomb_en_b
    # coulomb_en_els = coulomb_en - w_coul - b_coul

    coulomb_en_els = coul_en_c - w_coul - b_coul
    coul_els.append(coulomb_en_els)
    curr_en_comps['nonint_coul_els'] = coulomb_en_els
    print('coulomb en els', coulomb_en_els)
    # print('coulomb_en', coulomb_en)
    # mat_mask = np.ones_like(coulomb_mat)
    # mat_mask[:294, :294] = 0
    # mat_mask[294:, 294:] = 0
    # coulomb_mat_els = coulomb_mat * mat_mask
    # coulomb_en_els = 0.5 * np.einsum('i,ij,j->', c_df_basis, coulomb_mat_els, c_df_basis)
    # print('coulomb_en els', coulomb_en_els)
    # print('coulomb_en diff', coulomb_en - w_coul - b_coul)
    # print('positions', c_mol.atom_coords(unit='Angstrom'))
    b_Z_w_Z = c_mol.energy_nuc() - w_mf.energy_nuc() - b_mf.energy_nuc()
    print('nuc en', b_Z_w_Z)
    curr_en_comps['bZwZ_nuc_en'] = b_Z_w_Z

    c_e_nuc = np.einsum('ij,ji', c_dm, m_nuc)

    w_m_nuc = w_mol.intor('int1e_nuc')
    w_e_nuc = np.einsum('ij,ji', w_dm, w_m_nuc)
    b_m_nuc = b_mol.intor('int1e_nuc')
    b_e_nuc = np.einsum('ij,ji', b_dm, b_m_nuc)
    c_e_off = c_e_nuc - w_e_nuc - b_e_nuc
    print('nuc el en', c_e_off)
    print('w el en', w_e_nuc)
    print('b el en', b_e_nuc)
    curr_en_comps['nonint_nuc_el'] = c_e_off
    curr_en_comps['w_nuc_el'] = w_e_nuc
    curr_en_comps['b_nuc_el'] = b_e_nuc

    c2 = coulomb_en_els + b_Z_w_Z + c_e_off

    curr_en_comps['nonint_sapt0'] = c2

    print('SAPT0 estimate', utils.hartree_to_kcal(c2))
    print('data electrostatic', data['elst'][i])
    calc_sapt.append(utils.hartree_to_kcal(c2))
    coul_en_components.append(curr_en_comps)
    # auxbasis = 'augccpvqzjkfit'
    # dm1 = c_mf.make_rdm1(c_mf.mo_coeff, c_mf.mo_occ)
    # auxmol = df.addons.make_auxmol(c_mol, auxbasis)
    #
    # ints_3c2e = df.incore.aux_e2(c_mol, auxmol, intor='int3c2e')
    # ints_2c2e = auxmol.intor('int2c2e')
    # print('ints3c2e shape', ints_3c2e.shape)
    # print('ints2c2e shape', ints_2c2e.shape)
    #
    # nao = c_mol.nao
    # naux = auxmol.nao
    # df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
    # df_coef = df_coef.reshape(naux, nao, nao)
    # if dm1.ndim > 2:
    #     df_basis = []
    #     for j in range(dm1.shape[0]):
    #         df_basis.append(lib.einsum('Pij,ij->P', df_coef, dm1[j]))
    #     df_basis = np.stack(df_basis, axis=0)
    #     print(df_basis.shape)
    #
    # else:
    #     df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)
    #
    # print('df_basis shape', df_basis.shape)
    #
    # coulomb_mat = auxmol.intor('int2c2e')
    # coulomb_en = 0.5 * np.einsum('i,ij,j->', df_basis, coulomb_mat, df_basis)
    # print('coulomb_en', coulomb_en)
    # mat_mask = np.ones_like(coulomb_mat)
    # mat_mask[:294, :294] = 0
    # mat_mask[294:, 294:] = 0
    # coulomb_mat_els = coulomb_mat * mat_mask
    # coulomb_en_els = 0.5 * np.einsum('i,ij,j->', df_basis, coulomb_mat_els, df_basis)
    # print('coulomb_en els', coulomb_en_els)
# converged SCF energy = -76.3590310209066
# converged SCF energy = -231.961320422493
# SAPT0 estimate -10.101265052434567
# data electrostatic -9.505
# converged SCF energy = -76.3590310208921
# converged SCF energy = -231.961320422492
# SAPT0 estimate -5.544706255674774
# data electrostatic -4.87
# converged SCF energy = -76.3590310563804
# converged SCF energy = -231.961320422491
# SAPT0 estimate -3.9670776143403086
# data electrostatic -2.857
# converged SCF energy = -76.3590310390325
# converged SCF energy = -231.961320422491
# SAPT0 estimate -11.059839700088027
# data electrostatic -10.785
# converged SCF energy = -76.3590310563743
# converged SCF energy = -231.961320422492
# SAPT0 estimate -2.4129082827917583
# data electrostatic -1.808
# converged SCF energy = -76.3590310390346
# converged SCF energy = -231.961320422493
# SAPT0 estimate -2.515088776587494
# data electrostatic -1.966
# converged SCF energy = -76.3590310390323
# converged SCF energy = -231.961320422492
# SAPT0 estimate -3.011988970379576
# data electrostatic -2.146
# converged SCF energy = -76.3590310209067
# converged SCF energy = -231.961320422491
# SAPT0 estimate -3.599502564118458
# data electrostatic -2.415
# converged SCF energy = -76.359030927724
# converged SCF energy = -231.961314673628
# SAPT0 estimate -4.363095368885455
# data electrostatic -2.815
# converged SCF energy = -76.3590310208922
# converged SCF energy = -231.961320422492
# SAPT0 estimate -5.544709262566857
# data electrostatic -4.87
# %%
# print('calc vs ref SAPT0 error', np.mean(np.abs(np.array(calc_sapt) - data['elst'])))
# for key in coul_en_components[0].keys():
#     print(type(coul_en_components[0][key]))
# np.save('results/calc_en_comps.npy', coul_en_components, allow_pickle=True)
# %%
coul_en_components = np.load('results/calc_en_comps.npy', allow_pickle=True)
print(coul_en_components[0].keys())
# %%
c_comp = coul_en_components[0]
w_mol = gto.Mole(**c_comp['water_mol'])
w_auxmol = gto.M(atom=w_mol.atom, basis='augccpvqzjkfit')
w_df_nuc_el = orbitals.calculate_1e_intor(w_auxmol, 'int1e_nuc', torch.tensor(c_comp['water_df_basis']), 'df_coeffs') * 0.5
w_df_coul = orbitals.calculate_int2c2e(w_auxmol, torch.tensor(c_comp['water_df_basis']))

print('water nuc_el DF vs MO diff', utils.hartree_to_kcal(w_df_nuc_el - c_comp['w_nuc_el']))
print('water coul DF vs MO diff', utils.hartree_to_kcal(w_df_coul - c_comp['water_coul']))

b_mol = gto.Mole(**c_comp['benzene_mol'])
b_auxmol = gto.M(atom=b_mol.atom, basis='augccpvqzjkfit')
b_df_nuc_el = orbitals.calculate_1e_intor(b_auxmol, 'int1e_nuc', torch.tensor(c_comp['benzene_df_basis']), 'df_coeffs') * 0.5
b_df_coul = orbitals.calculate_int2c2e(b_auxmol, torch.tensor(c_comp['benzene_df_basis']))

print('benzene nuc_el DF vs MO diff', utils.hartree_to_kcal(b_df_nuc_el - c_comp['b_nuc_el']))
print('benzene coul DF vs MO diff', utils.hartree_to_kcal(b_df_coul - c_comp['benzene_coul']))

atom_types = combined_data['atom_numbers'][0]
pos = combined_data['positions'][0]
atom = []
for j in range(len(atom_types)):
    atom.append((atom_types[j], pos[j, :]))
mol = gto.M(atom=atom, basis='augccpvdz')
auxmol = gto.M(atom=atom, basis='augccpvqzjkfit')

df_nuc_el = orbitals.calculate_1e_intor(auxmol, 'int1e_nuc', torch.tensor(c_comp['nonint_df_basis']), 'df_coeffs') * 0.5
df_coul = orbitals.calculate_int2c2e(auxmol, torch.tensor(c_comp['nonint_df_basis']))
df_nuc_el = df_nuc_el - w_df_nuc_el - b_df_nuc_el

print('combined nuc_el DF vs MO diff', utils.hartree_to_kcal(df_nuc_el - c_comp['nonint_nuc_el']))
print('combined coul DF vs MO diff', utils.hartree_to_kcal(df_coul - c_comp['nonint_coul']))

# todo calculate nuclear energy of DF coeffs
# calculate nuclear energy of atom denisty df coeffs
# compare to nuclear energy of atom density both from MO and DF


# %%
# basic arguments for model loading
main_args = Namespace()

main_args.args_file = "args/qm7x250_dens_001_coreless.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
main_args.save_file = 'qm7x250_dens_001_coreless'
main_args.use_gpu = False

# %%
# load arguments and dataset
args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)

args.fix_arguments = True

args, hyperparam_args, train_vars = train_utils.init_training_vars(args, hyperparam_args)
checkpoint = train_vars['checkpoint']

# determine whether GPU is used for training

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

args.verbose = 0
args.use_gpu = main_args.use_gpu
print('args use gpu', args.use_gpu)
args.expansion_constraint = None
args.integral_constraint = 'coeffs_in_coeff_net'
# args.integral_constraint = None
args.ignore_missing_keywords = True

required_properties = ['density', 'dipole_moment']

args.spherical_grid_level = 1
args.cube_grid = False
grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
sampling_fn = partial(spherical_radial_sampling, rotate=False)
grid_origin = 0
grid_extent = None
args.radii_adjust = True
# grid_vars = train_utils.init_grid_vars(args)
rotate = False

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file

# args.np_dataset_test = "datasets/qm7x_test_dft_augccpvdz_small_base.npy"
# args.dens_dataset_test = "datasets/qm7x_test_dft_augccpvdz_small.npy"
args.np_dataset_test = "datasets/s66x8_pyscf_augccpvdz_base.npy"
args.dens_dataset_test = "datasets/s66x8_pyscf_augccpvdz_calc.npy"

dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000000000000,
                           required_properties=required_properties,
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           pyscf_grid=True,
                           pyscf_rotate=rotate,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
                           calc_data=True,
                           radii_adjust=args.radii_adjust,
                           atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy',
                           atom_dens_type='mo_coeffs',
                           split_atom_dens=True,
                           density_grad=args.density_grad,
                           calc_basis_path='datasets/augccpvdz_orbital_basis.npy',
                           all_atom_numbers=np.array([1, 6, 7, 8, 16, 17]),
                           all_atom_coeffs=True,
                           )

print('dataset length', len(dataset))
samp = dataset.get_properties([0])
print('sample pos shape', samp['positions'].shape)
print('sample dens shape', samp['density'].shape)
print('sample dens integral', torch.sum(samp['density'][0] * samp['coord_weights'][0]))
print('args use gpu', args.use_gpu)

# %%
# evaluate model and test density integral
model = model_loader.load_model(args, dataset)
for param in model.parameters():
    param.requires_grad = False
idx = [0, 3]
samp = dataset.get_properties(idx)
if args.use_gpu:
    for key in samp.keys():
        if isinstance(samp[key], torch.Tensor):
            samp[key] = samp[key].cuda()

res = model(samp)

print('sum atom numbers', torch.sum(samp['batch_atom_numbers'], dim=1))
print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
print('true density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
print('density error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1))

# %%
# create model input from  benzene water data
res_arr = []
for i in range(7):
    max_idx = min((i + 1) * 10, len(data['elst']))
    c_input = {key: combined_data[key][i * 10:max_idx] for key in combined_data.keys()}
    w_input = {key: water_data[key][i * 10:max_idx] for key in water_data.keys()}
    b_input = {key: benzene_data[key][i * 10:max_idx] for key in benzene_data.keys()}
    samp = orbitals.model_input_from_atoms(c_input,
                                           density_expansion=True,
                                           pyscf_grid=True,
                                           skip_compress=True,
                                           grid_spec=dataset.grid_spec,
                                           grid_sampling_fn=dataset.grid_fn,
                                           cutoff=args.cutoff,
                                           dtype=torch.float32,
                                           atom_dens_type="df_coeffs",
                                           free_atom_densities=dataset.atom_dens,
                                           split_atom_densities=False,
                                           basis=None,
                                           all_atom_coeffs=True,
                                           )
    samp_w = orbitals.model_input_from_atoms(w_input,
                                             density_expansion=True,
                                             pyscf_grid=True,
                                             skip_compress=True,
                                             grid_spec=dataset.grid_spec,
                                             grid_sampling_fn=dataset.grid_fn,
                                             cutoff=args.cutoff,
                                             dtype=torch.float32,
                                             atom_dens_type="df_coeffs",
                                             free_atom_densities=dataset.atom_dens,
                                             split_atom_densities=False,
                                             basis=None,
                                             all_atom_coeffs=True,
                                             coord_params={'coords': samp['coords'],
                                                           'coord_weights': samp['coord_weights']},
                                             )
    samp_b = orbitals.model_input_from_atoms(b_input,
                                             density_expansion=True,
                                             pyscf_grid=True,
                                             skip_compress=True,
                                             grid_spec=dataset.grid_spec,
                                             grid_sampling_fn=dataset.grid_fn,
                                             cutoff=args.cutoff,
                                             dtype=torch.float32,
                                             atom_dens_type="df_coeffs",
                                             free_atom_densities=dataset.atom_dens,
                                             split_atom_densities=False,
                                             basis=None,
                                             all_atom_coeffs=True,
                                             coord_params={'coords': samp['coords'],
                                                           'coord_weights': samp['coord_weights']},
                                             )

    print('args integral constraint', args.integral_constraint)
    samp_b['coords'] = samp['coords'].clone()
    samp_b['coord_weights'] = samp['coord_weights'].clone()
    samp_w['coords'] = samp['coords'].clone()
    samp_b['coord_weights'] = samp['coord_weights'].clone()
    # print('samp w pos', samp_w['batch_positions'])
    # print('samp w pos', samp_w['batch_atom_numbers'])
    # print('w input', w_input)
    # print('samp b pos', samp_b['batch_positions'])
    # print('samp c pos', samp['batch_positions'])
    # print('c input', c_input)
    # print('samp b coords', samp_b['coords'])
    if args.use_gpu:
        for key in samp.keys():
            if isinstance(samp[key], torch.Tensor):
                samp[key] = samp[key].cuda()
                samp_w[key] = samp_w[key].cuda()
                samp_b[key] = samp_b[key].cuda()

    res_w = model(samp_w)
    res_b = model(samp_b)

    res = copy.deepcopy(samp)
    res['spherical_coeffs'] = res_w['spherical_coeffs'] + res_b['spherical_coeffs']
    res['radial_scale'] = res_w['radial_scale'] + res_b['radial_scale']
    res['radial_width'] = res_w['radial_width'] + res_b['radial_width']
    res = model.property_models['density'](res)
    res['density'] += res['atom_density']

    print('sum atom numbers', torch.sum(samp['batch_atom_numbers'], dim=1))
    print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
    print('res_w density integral', torch.sum(res_w['density'] * res_w['coord_weights'], dim=1))
    print('res_b density integral', torch.sum(res_b['density'] * res_b['coord_weights'], dim=1))
    print('res density diff to sum', torch.sum(torch.abs((res_b['density'] + res_w['density']) - res['density']) * res['coord_weights'][0], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1))

    print(res['batch_atom_numbers'])
    ress = {'samp_w': samp_w, 'samp_b': samp_b, 'samp': samp,
            'res_w': res_w, 'res_b': res_b, 'res': res}
    res_arr.append(ress)

# %%
# evaluate density based on sample free atom density MO coeffs
samp = res_arr[0]['samp']
mol = utils.npy_to_pyscf(samp['batch_positions'].numpy(),
                         samp['batch_atom_numbers'].numpy(),
                         basis=samp['atom_mo_coeffs_basis'], build=True, skip_zero=False)
print('atom_mo_coeffs_basis', samp['atom_mo_coeffs_basis'])

coords = samp['coords'][0]
coords = coords * utils.to_bohr
ao = numint.eval_ao(mol[0], coords, deriv=0)
print('H mo coeffs', dataset.atom_dens[1]['mo_coeff'])
print('ao shape', ao.shape)
print('mo coeff shape', samp['atom_mo_coeffs'][0].shape)
print('mo occ shape', samp['atom_mo_coeffs_occ'][0].shape)
print('mo coeff diag', np.diag(samp['atom_mo_coeffs'][0]))
print('mo coeff occ', samp['atom_mo_coeffs_occ'][0])
print('mo_coeffs', samp['atom_mo_coeffs'][0])
rho = numint.eval_rho2(mol[0], ao, mo_coeff=samp['atom_mo_coeffs'][0].numpy(), mo_occ=samp['atom_mo_coeffs_occ'][0].numpy())  
rho = torch.tensor(rho).to(samp['atom_density'])
print('rho.shape', rho.shape)
print('rho density integral', torch.sum(rho * samp['coord_weights'][0]))
print('atom density integral', torch.sum(samp['atom_density'][0] * samp['coord_weights'][0]))
print('mol charge', torch.sum(samp['batch_atom_numbers'], dim=-1))
print('density error', torch.sum(torch.abs(rho - samp['atom_density'][0]) * samp['coord_weights'][0]) / torch.sum(samp['batch_atom_numbers'][0]))

# %%
auxmol = utils.npy_to_pyscf(samp['batch_positions'].numpy(),
                            samp['batch_atom_numbers'].numpy(),
                            basis=samp['atom_df_coeffs_basis'], build=True, skip_zero=False)
# evaluate density based on sample free atom density DF coeffs
print('atom_df_coeffs_basis', samp['atom_df_coeffs_basis'])

# print(orbitals.calculate_1e_intor(auxmol[0], 'int1e_ovlp', samp['atom_df_coeffs'][[0], samp['atom_df_coeffs'][0] != 0], 'df_coeffs'))
print(orbitals.calculate_1e_intor(auxmol[0], 'int1e_ovlp', samp['atom_df_coeffs'][0], 'df_coeffs'))

coords = samp['coords'][0]
coords = coords * utils.to_bohr
ao = numint.eval_ao(auxmol[0], coords, deriv=0)
# rho = np.einsum('ij,j->i', ao, samp['atom_df_coeffs'][[0], samp['atom_df_coeffs'][0] != 0])
rho = np.einsum('ij,j->i', ao, samp['atom_df_coeffs'][0])
rho = torch.tensor(rho).to(samp['atom_density'])
print('rho.shape', rho.shape)
print('df density integral', torch.sum(rho * samp['coord_weights'][0]))
print('atom density integral', torch.sum(samp['atom_density'][0] * samp['coord_weights'][0]))
print('density error', torch.sum(torch.abs(rho - samp['atom_density'][0]) * samp['coord_weights'][0]) / torch.sum(samp['batch_atom_numbers'][0]))

# %%
# calculate coulomb integral for free atom DF coeffs + ML-DF coeffs
# join DF coeffs from free atom and ML basis
df_coeffs_ml = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                              radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

auxmol = utils.npy_to_pyscf(samp['batch_positions'].numpy(),
                            samp['batch_atom_numbers'].numpy(),
                            basis=samp['atom_df_coeffs_basis'], build=True, skip_zero=False)
mol = utils.npy_to_pyscf(samp['batch_positions'].numpy(),
                         samp['batch_atom_numbers'].numpy(),
                         basis=samp['atom_mo_coeffs_basis'], build=True, skip_zero=False)
df_coeffs_ml_b = orbitals.coeffs_dict_to_vector(res_b, dataset.orbital_basis_num, res_b['batch_atom_numbers'],
                                                radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

auxmol_b = utils.npy_to_pyscf(samp_b['batch_positions'].numpy(),
                              samp_b['batch_atom_numbers'].numpy(),
                              basis=samp_b['atom_df_coeffs_basis'], build=True, skip_zero=False)
mol_b = utils.npy_to_pyscf(samp_b['batch_positions'].numpy(),
                           samp_b['batch_atom_numbers'].numpy(),
                           basis=samp_b['atom_mo_coeffs_basis'], build=True, skip_zero=False)

df_coeffs_ml_w = orbitals.coeffs_dict_to_vector(res_w, dataset.orbital_basis_num, res_w['batch_atom_numbers'],
                                                radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

auxmol_w = utils.npy_to_pyscf(samp_w['batch_positions'].numpy(),
                              samp_w['batch_atom_numbers'].numpy(),
                              basis=samp_w['atom_df_coeffs_basis'], build=True, skip_zero=False)
mol_w = utils.npy_to_pyscf(samp_w['batch_positions'].numpy(),
                           samp_w['batch_atom_numbers'].numpy(),
                           basis=samp_w['atom_mo_coeffs_basis'], build=True, skip_zero=False)
coulomb_en_elss = []
pred_sapt0 = []
for i in range(10):

    auxmol_ml = orbitals.ml_basis_to_auxmol(res, i, skip_zero=False)
    auxmol_ml_b = orbitals.ml_basis_to_auxmol(res_b, i, skip_zero=False)
    auxmol_ml_w = orbitals.ml_basis_to_auxmol(res_w, i, skip_zero=False)

    w_dm = hf.make_rdm1(mo_coeff=samp_w['atom_mo_coeffs'][i], mo_occ=samp_w['atom_mo_coeffs_occ'][i])
    b_dm = hf.make_rdm1(mo_coeff=samp_b['atom_mo_coeffs'][i], mo_occ=samp_b['atom_mo_coeffs_occ'][i])
    c_dm = hf.make_rdm1(mo_coeff=samp['atom_mo_coeffs'][i], mo_occ=samp['atom_mo_coeffs_occ'][i])

    # print('auxmol_ml basis', auxmol_ml._basis)
    # print('auxmol_ml atoms', auxmol_ml.atom)
    # print('auxmol basis', auxmol[0]._basis)
    # print('auxmol atoms', auxmol[0].atom)
    # print('df_coeffs_ml', df_coeffs_ml[0, -142:])
    # print('df_coeffs_atom', samp['atom_df_coeffs'][0, -20:])
    # print('datasets orbitals bass size', dataset.orbital_basis_size)
    auxmol_new, joined_coeffs = orbitals.join_free_atom_and_ml_basis(auxmol_ml, auxmol[i],
                                                                     df_coeffs_ml[i],
                                                                     samp['atom_df_coeffs'][i])
    auxmol_new_b, joined_coeffs_b = orbitals.join_free_atom_and_ml_basis(auxmol_ml_b, auxmol_b[i],
                                                                         df_coeffs_ml_b[i],
                                                                         samp_b['atom_df_coeffs'][i])
    auxmol_new_w, joined_coeffs_w = orbitals.join_free_atom_and_ml_basis(auxmol_ml_w, auxmol_w[i],
                                                                         df_coeffs_ml_w[i],
                                                                         samp_w['atom_df_coeffs'][i])

    # calculate integral of joined density analytically for DF basis
    print('analytic density integral', orbitals.calculate_1e_intor(auxmol_new, 'int1e_ovlp', joined_coeffs, 'df_coeffs'))

    # coords = samp['coords'][i]
    # coords = coords * utils.to_bohr
    # ao = numint.eval_ao(auxmol_new, coords, deriv=0)
    # # rho = np.einsum('ij,j->i', ao, samp['atom_df_coeffs'][[0], samp['atom_df_coeffs'][0] != 0])
    # rho = np.einsum('ij,j->i', ao, joined_coeffs)
    # rho = torch.tensor(rho).to(samp['atom_density'])

    # calculate coulomb of joined density analytically for DF basis
    # print('analytic ML coulomb dimer', )
    coulomb_en_ml = orbitals.calculate_int2c2e(auxmol_ml, df_coeffs_ml[i])
    coulomb_en_ml_w = orbitals.calculate_int2c2e(auxmol_ml_w, df_coeffs_ml_w[i])
    coulomb_en_ml_b = orbitals.calculate_int2c2e(auxmol_ml_b, df_coeffs_ml_b[i])

    vj, _ = hf.get_jk(mol[i], c_dm)
    coulomb_en_atom = np.einsum('ij,ji->', vj, c_dm).real * .5
    vj_w, _ = hf.get_jk(mol_w[i], w_dm)
    coulomb_en_atom_w = np.einsum('ij,ji->', vj_w, w_dm).real * .5
    vj_b, _ = hf.get_jk(mol_b[i], b_dm)
    coulomb_en_atom_b = np.einsum('ij,ji->', vj_b, b_dm).real * .5

    coulomb_en_mix = orbitals.calculate_int2c2e(auxmol_ml, df_coeffs_ml[i],
                                                auxmol[i], samp['atom_df_coeffs'][i])
    coulomb_en_mix_w = orbitals.calculate_int2c2e(auxmol_ml_w, df_coeffs_ml_w[i],
                                                  auxmol_w[i], samp_w['atom_df_coeffs'][i])
    coulomb_en_mix_b = orbitals.calculate_int2c2e(auxmol_ml_b, df_coeffs_ml_b[i],
                                                  auxmol_b[i], samp_b['atom_df_coeffs'][i])

    coulomb_en_sum = coulomb_en_ml + coulomb_en_atom + 2 * coulomb_en_mix
    coulomb_en_sum_w = coulomb_en_ml_w + coulomb_en_atom_w + 2 * coulomb_en_mix_w
    coulomb_en_sum_b = coulomb_en_ml_b + coulomb_en_atom_b + 2 * coulomb_en_mix_b

    coulomb_en_sum_els = coulomb_en_sum - coulomb_en_sum_w - coulomb_en_sum_b
    # print('coulomb_en_sum_els', coulomb_en_sum_els)

    coulomb_en = orbitals.calculate_int2c2e(auxmol_new, joined_coeffs)
    coul_w = orbitals.calculate_int2c2e(auxmol_new_w, joined_coeffs_w)
    coul_b = orbitals.calculate_int2c2e(auxmol_new_b, joined_coeffs_b)
    coulomb_en_els = coulomb_en_sum - coul_w - coul_b
    # print('coulomb_en_els', coulomb_en_els)

    b_Z_w_Z = auxmol_new.energy_nuc() - auxmol_new_w.energy_nuc() - auxmol_new_b.energy_nuc()

    w_m_nuc = mol_w[i].intor('int1e_nuc')
    w_e_nuc_mo = np.einsum('ij,ji', w_dm, w_m_nuc)
    b_m_nuc = mol_b[i].intor('int1e_nuc')
    b_e_nuc_mo = np.einsum('ij,ji', b_dm, b_m_nuc)
    c_m_nuc = mol[i].intor('int1e_nuc')
    c_e_nuc_mo = np.einsum('ij,ji', c_dm, c_m_nuc)

    c_e_nuc_df = orbitals.calculate_1e_intor(auxmol_ml, 'int1e_nuc', df_coeffs_ml[i],
                                             coeffs_type='df_coeffs') * 0.5
    c_e_nuc = c_e_nuc_df + c_e_nuc_mo
    b_e_nuc_df = orbitals.calculate_1e_intor(auxmol_ml_b, 'int1e_nuc', df_coeffs_ml_b[i],
                                             coeffs_type='df_coeffs') * 0.5
    b_e_nuc = b_e_nuc_df + b_e_nuc_mo
    w_e_nuc_df = orbitals.calculate_1e_intor(auxmol_ml_w, 'int1e_nuc', df_coeffs_ml_w[i],
                                             coeffs_type='df_coeffs') * 0.5
    w_e_nuc = w_e_nuc_df + w_e_nuc_mo
    c_e_off = c_e_nuc - w_e_nuc - b_e_nuc

    # print('w e nuc mo', w_e_nuc_mo)
    # print('b e nuc mo', b_e_nuc_mo)
    # print('c_e_off', c_e_off)
    # print('w e nuc', w_e_nuc)
    # print('b e nuc', b_e_nuc)
    # print('true c_e_off', coul_en_components[i]['nonint_nuc_el'])
    # print('true w_e_nuc', coul_en_components[i]['w_nuc_el'])
    # print('true b_e_nuc', coul_en_components[i]['b_nuc_el'])
    # print('water nuc el diff', utils.hartree_to_kcal(np.abs(w_e_nuc - coul_en_components[i]['w_nuc_el'])))
    # print('benzene nuc el diff', utils.hartree_to_kcal(np.abs(b_e_nuc - coul_en_components[i]['b_nuc_el'])))
    print('combined nuc el diff', utils.hartree_to_kcal(c_e_off - coul_en_components[i]['nonint_nuc_el']))
    print('combined coul diff', utils.hartree_to_kcal(coulomb_en_sum_els - coul_en_components[i]['nonint_coul_els']))

    c2 = coulomb_en_sum_els + b_Z_w_Z + c_e_off
    coulomb_en_elss.append(coulomb_en_sum_els)
    print('sapt0 energy', utils.hartree_to_kcal(c2))
    pred_sapt0.append(utils.hartree_to_kcal(c2))
    # print('analytic free atom coulomb', orbitals.calculate_int2c2e(auxmol[i], samp['atom_df_coeffs'][i]))

# %%
els_ref = np.array(coul_els)
els_ml = np.array(coulomb_en_elss)
print(utils.hartree_to_kcal((np.mean(np.abs(els_ml - els_ref)))))

# %% 
sapt_ref = np.array(calc_sapt)
sapt_ml = np.array(pred_sapt0)
print(np.mean(np.abs(sapt_ml - sapt_ref)))

# %%
# calculate coulomb integral for free atom DF coeffs + ML-DF coeffs
# join DF coeffs from free atom and ML basis
coulomb_en_elss = []
pred_sapt0 = []
# for j in range(len(res_arr)):
for j in range(1):
    res = res_arr[j]['res']
    samp = res_arr[j]['samp']
    res_w = res_arr[j]['res_w']
    res_b = res_arr[j]['res_b']
    samp_w = res_arr[j]['samp_w']
    samp_b = res_arr[j]['samp_b']

    df_coeffs_ml = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                                  radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

    auxmol = utils.npy_to_pyscf(samp['batch_positions'].numpy(),
                                samp['batch_atom_numbers'].numpy(),
                                basis=samp['atom_df_coeffs_basis'], build=True, skip_zero=False)
    mol = utils.npy_to_pyscf(samp['batch_positions'].numpy(),
                             samp['batch_atom_numbers'].numpy(),
                             basis=samp['atom_mo_coeffs_basis'], build=True, skip_zero=False)
    df_coeffs_ml_b = orbitals.coeffs_dict_to_vector(res_b, dataset.orbital_basis_num, res_b['batch_atom_numbers'],
                                                    radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

    auxmol_b = utils.npy_to_pyscf(samp_b['batch_positions'].numpy(),
                                  samp_b['batch_atom_numbers'].numpy(),
                                  basis=samp_b['atom_df_coeffs_basis'], build=True, skip_zero=False)
    mol_b = utils.npy_to_pyscf(samp_b['batch_positions'].numpy(),
                               samp_b['batch_atom_numbers'].numpy(),
                               basis=samp_b['atom_mo_coeffs_basis'], build=True, skip_zero=False)

    df_coeffs_ml_w = orbitals.coeffs_dict_to_vector(res_w, dataset.orbital_basis_num, res_w['batch_atom_numbers'],
                                                    radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

    auxmol_w = utils.npy_to_pyscf(samp_w['batch_positions'].numpy(),
                                  samp_w['batch_atom_numbers'].numpy(),
                                  basis=samp_w['atom_df_coeffs_basis'], build=True, skip_zero=False)
    mol_w = utils.npy_to_pyscf(samp_w['batch_positions'].numpy(),
                               samp_w['batch_atom_numbers'].numpy(),
                               basis=samp_w['atom_mo_coeffs_basis'], build=True, skip_zero=False)
    for i in range(len(auxmol)):
        auxmol_ml = orbitals.ml_basis_to_auxmol(res, i, skip_zero=False)
        auxmol_ml_b = orbitals.ml_basis_to_auxmol(res_b, i, skip_zero=False)
        auxmol_ml_w = orbitals.ml_basis_to_auxmol(res_w, i, skip_zero=False)

        w_dm = hf.make_rdm1(mo_coeff=samp_w['atom_mo_coeffs'][i], mo_occ=samp_w['atom_mo_coeffs_occ'][i])
        b_dm = hf.make_rdm1(mo_coeff=samp_b['atom_mo_coeffs'][i], mo_occ=samp_b['atom_mo_coeffs_occ'][i])
        c_dm = hf.make_rdm1(mo_coeff=samp['atom_mo_coeffs'][i], mo_occ=samp['atom_mo_coeffs_occ'][i])

        # print('auxmol_ml basis', auxmol_ml._basis)
        # print('auxmol_ml atoms', auxmol_ml.atom)
        # print('auxmol basis', auxmol[0]._basis)
        # print('auxmol atoms', auxmol[0].atom)
        # print('df_coeffs_ml', df_coeffs_ml[0, -142:])
        # print('df_coeffs_atom', samp['atom_df_coeffs'][0, -20:])
        # print('datasets orbitals bass size', dataset.orbital_basis_size)
        # auxmol_new, joined_coeffs = orbitals.join_free_atom_and_ml_basis(auxmol_ml, auxmol[i],
        #                                                                  df_coeffs_ml[i],
        #                                                                  samp['atom_df_coeffs'][i])
        # auxmol_new_b, joined_coeffs_b = orbitals.join_free_atom_and_ml_basis(auxmol_ml_b, auxmol_b[i],
        #                                                                      df_coeffs_ml_b[i],
        #                                                                      samp_b['atom_df_coeffs'][i])
        # auxmol_new_w, joined_coeffs_w = orbitals.join_free_atom_and_ml_basis(auxmol_ml_w, auxmol_w[i],
        #                                                                      df_coeffs_ml_w[i],
        #                                                                      samp_w['atom_df_coeffs'][i])

        # # calculate integral of joined density analytically for DF basis
        # print('analytic density integral', orbitals.calculate_1e_intor(auxmol_new, 'int1e_ovlp', joined_coeffs, 'df_coeffs'))

        # coords = samp['coords'][i]
        # coords = coords * utils.to_bohr
        # ao = numint.eval_ao(auxmol_new, coords, deriv=0)
        # # rho = np.einsum('ij,j->i', ao, samp['atom_df_coeffs'][[0], samp['atom_df_coeffs'][0] != 0])
        # rho = np.einsum('ij,j->i', ao, joined_coeffs)
        # rho = torch.tensor(rho).to(samp['atom_density'])

        # calculate coulomb of joined density analytically for DF basis
        # print('analytic ML coulomb dimer', )
        coulomb_en_ml = orbitals.calculate_int2c2e(auxmol_ml, df_coeffs_ml[i])
        coulomb_en_ml_w = orbitals.calculate_int2c2e(auxmol_ml_w, df_coeffs_ml_w[i])
        coulomb_en_ml_b = orbitals.calculate_int2c2e(auxmol_ml_b, df_coeffs_ml_b[i])
        # print('coulomb en ml c', coulomb_en_ml)
        # print('coulomb en ml w', coulomb_en_ml_w)

        vj, _ = hf.get_jk(mol[i], c_dm)
        coulomb_en_atom = np.einsum('ij,ji->', vj, c_dm).real * .5
        vj_w, _ = hf.get_jk(mol_w[i], w_dm)
        coulomb_en_atom_w = np.einsum('ij,ji->', vj_w, w_dm).real * .5
        vj_b, _ = hf.get_jk(mol_b[i], b_dm)
        coulomb_en_atom_b = np.einsum('ij,ji->', vj_b, b_dm).real * .5
        # print('coulomb en atom c', coulomb_en_atom)
        # print('coulomb en atom w', coulomb_en_atom_w)

        coulomb_en_mix = orbitals.calculate_int2c2e(auxmol_ml, df_coeffs_ml[i],
                                                    auxmol[i], samp['atom_df_coeffs'][i])
        coulomb_en_mix_w = orbitals.calculate_int2c2e(auxmol_ml_w, df_coeffs_ml_w[i],
                                                      auxmol_w[i], samp_w['atom_df_coeffs'][i])
        coulomb_en_mix_b = orbitals.calculate_int2c2e(auxmol_ml_b, df_coeffs_ml_b[i],
                                                      auxmol_b[i], samp_b['atom_df_coeffs'][i])

        # print('coulomb en mix c', coulomb_en_mix)
        # print('coulomb en mix w', coulomb_en_mix_w)

        coulomb_en_sum = coulomb_en_ml + coulomb_en_atom + 2 * coulomb_en_mix
        coulomb_en_sum_w = coulomb_en_ml_w + coulomb_en_atom_w + 2 * coulomb_en_mix_w
        coulomb_en_sum_b = coulomb_en_ml_b + coulomb_en_atom_b + 2 * coulomb_en_mix_b

        coulomb_en_sum_els = coulomb_en_sum - coulomb_en_sum_w - coulomb_en_sum_b
        # print('coulomb_en_sum_els', coulomb_en_sum_els)

        # coulomb_en = orbitals.calculate_int2c2e(auxmol_new, joined_coeffs)
        # coul_w = orbitals.calculate_int2c2e(auxmol_new_w, joined_coeffs_w)
        # coul_b = orbitals.calculate_int2c2e(auxmol_new_b, joined_coeffs_b)
        # coulomb_en_els = coulomb_en - coul_w - coul_b
        print('coul en w diff', utils.hartree_to_kcal(coulomb_en_sum_w - coul_w))
        print('coul en b diff', utils.hartree_to_kcal(coulomb_en_sum_b - coul_b))
        # print('coulomb_en_els', coulomb_en_els)

        b_Z_w_Z = auxmol_ml.energy_nuc() - auxmol_ml_w.energy_nuc() - auxmol_ml_b.energy_nuc()

        w_m_nuc = mol_w[i].intor('int1e_nuc')
        w_e_nuc_mo = np.einsum('ij,ji', w_dm, w_m_nuc)
        b_m_nuc = mol_b[i].intor('int1e_nuc')
        b_e_nuc_mo = np.einsum('ij,ji', b_dm, b_m_nuc)
        c_m_nuc = mol[i].intor('int1e_nuc')
        c_e_nuc_mo = np.einsum('ij,ji', c_dm, c_m_nuc)

        c_e_nuc_df = orbitals.calculate_1e_intor(auxmol_ml, 'int1e_nuc', df_coeffs_ml[i],
                                                 coeffs_type='df_coeffs') * 0.5
        c_e_nuc = c_e_nuc_df + c_e_nuc_mo
        b_e_nuc_df = orbitals.calculate_1e_intor(auxmol_ml_b, 'int1e_nuc', df_coeffs_ml_b[i],
                                                 coeffs_type='df_coeffs') * 0.5
        b_e_nuc = b_e_nuc_df + b_e_nuc_mo
        w_e_nuc_df = orbitals.calculate_1e_intor(auxmol_ml_w, 'int1e_nuc', df_coeffs_ml_w[i],
                                                 coeffs_type='df_coeffs') * 0.5
        w_e_nuc = w_e_nuc_df + w_e_nuc_mo
        c_e_off = c_e_nuc - w_e_nuc - b_e_nuc

        # print('w e nuc mo', w_e_nuc_mo)
        # print('b e nuc mo', b_e_nuc_mo)
        # print('c_e_off', c_e_off)
        # print('w e nuc', w_e_nuc)
        # print('b e nuc', b_e_nuc)
        # print('true c_e_off', coul_en_components[i]['nonint_nuc_el'])
        # print('true w_e_nuc', coul_en_components[i]['w_nuc_el'])
        # print('true b_e_nuc', coul_en_components[i]['b_nuc_el'])
        print('water nuc el diff', utils.hartree_to_kcal(np.abs(w_e_nuc - coul_en_components[i]['w_nuc_el'])))
        print('benzene nuc el diff', utils.hartree_to_kcal(np.abs(b_e_nuc - coul_en_components[i]['b_nuc_el'])))
        print('combined coul diff', utils.hartree_to_kcal(coulomb_en_sum_els - coul_en_components[i]['nonint_coul_els']))
        print('combined nuc el diff', utils.hartree_to_kcal(c_e_off - coul_en_components[i]['nonint_nuc_el']))

        c2 = coulomb_en_sum_els + b_Z_w_Z + c_e_off
        coulomb_en_elss.append(coulomb_en_sum_els)
        print('sapt0 energy', utils.hartree_to_kcal(c2))
        pred_sapt0.append(utils.hartree_to_kcal(c2))
        # print('analytic free atom coulomb', orbitals.calculate_int2c2e(auxmol[i], samp['atom_df_coeffs'][i]))

# %%
els_ref = np.array(coul_els)
els_ml = np.array(coulomb_en_elss)
print(utils.hartree_to_kcal((np.mean(np.abs(els_ml - els_ref)))))

# %% 
sapt_ref = np.array(calc_sapt)
sapt_ml = np.array(pred_sapt0)
print(np.mean(np.abs(sapt_ml - sapt_ref)))
# %%
# %%
relevant_keys = ['atom_mo_coeffs', 'atom_mo_coeffs_occ', 'atom_df_coeffs',
                 'atom_df_coeffs_occ', 'batch_atom_numbers', 'batch_positions']
all_res_comb = {'ml_df_coeffs_basis': [], 'atom_mo_coeffs_basis': [], 'atom_df_coeffs_basis': []}
for res_dict in res_arr:
    batch_size = res_dict['res']['batch_positions'].shape[0]
    for key in relevant_keys:
        if key not in all_res_comb:
            all_res_comb[key] = res_dict['res'][key]
        else:
            all_res_comb[key] = torch.cat((all_res_comb[key], res_dict['res'][key]), dim=0)

    all_res_comb['atom_mo_coeffs_basis'].extend([res_dict['res']['atom_mo_coeffs_basis']] * batch_size)
    all_res_comb['atom_df_coeffs_basis'].extend([res_dict['res']['atom_df_coeffs_basis']] * batch_size)

    anum = torch.max(res_dict['res']["batch_atom_numbers"], dim=0)[0]
    atom_types, _ = orbitals.create_ghost_atom_types(anum)

    for idx in range(batch_size):
        ml_basis = orbitals.ml_basis_to_pyscf_basis(res_dict['res'], atom_types, idx)
        all_res_comb['ml_df_coeffs_basis'].append(ml_basis)

    ml_df_coeffs = orbitals.coeffs_dict_to_vector(res_dict['res'], dataset.orbital_basis_num, res_dict['res']['batch_atom_numbers'],
                                                  radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

    if 'ml_df_coeffs' not in all_res_comb.keys():
        all_res_comb['ml_df_coeffs'] = ml_df_coeffs
    else:
        all_res_comb['ml_df_coeffs'] = torch.cat((all_res_comb['ml_df_coeffs'], ml_df_coeffs), dim=0)

for key in all_res_comb.keys():
    if isinstance(all_res_comb[key], torch.Tensor):
        print(key, all_res_comb[key].shape)
    else:
        print(key)
np.save('results/benzene_water_dimer_ml_results.npy', all_res_comb, allow_pickle=True)

# %%
relevant_keys = ['atom_mo_coeffs', 'atom_mo_coeffs_occ', 'atom_df_coeffs',
                 'atom_df_coeffs_occ', 'batch_atom_numbers', 'batch_positions']
all_res_comb = {'ml_df_coeffs_basis': [], 'atom_mo_coeffs_basis': [], 'atom_df_coeffs_basis': []}
for res_dict in res_arr:
    batch_size = res_dict['res_w']['batch_positions'].shape[0]
    for key in relevant_keys:
        if key not in all_res_comb:
            all_res_comb[key] = res_dict['res_w'][key]
        else:
            all_res_comb[key] = torch.cat((all_res_comb[key], res_dict['res_w'][key]), dim=0)

    all_res_comb['atom_mo_coeffs_basis'].extend([res_dict['res_w']['atom_mo_coeffs_basis']] * batch_size)
    all_res_comb['atom_df_coeffs_basis'].extend([res_dict['res_w']['atom_df_coeffs_basis']] * batch_size)

    anum = torch.max(res_dict['res_w']["batch_atom_numbers"], dim=0)[0]
    atom_types, _ = orbitals.create_ghost_atom_types(anum)

    for idx in range(batch_size):
        ml_basis = orbitals.ml_basis_to_pyscf_basis(res_dict['res_w'], atom_types, idx)
        all_res_comb['ml_df_coeffs_basis'].append(ml_basis)

    ml_df_coeffs = orbitals.coeffs_dict_to_vector(res_dict['res_w'], dataset.orbital_basis_num, res_dict['res_w']['batch_atom_numbers'],
                                                  radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

    if 'ml_df_coeffs' not in all_res_comb.keys():
        all_res_comb['ml_df_coeffs'] = ml_df_coeffs
    else:
        all_res_comb['ml_df_coeffs'] = torch.cat((all_res_comb['ml_df_coeffs'], ml_df_coeffs), dim=0)
for key in all_res_comb.keys():
    if isinstance(all_res_comb[key], torch.Tensor):
        print(key, all_res_comb[key].shape)
    else:
        print(key)
np.save('results/water_dimer_ml_results.npy', all_res_comb, allow_pickle=True)

# %%
relevant_keys = ['atom_mo_coeffs', 'atom_mo_coeffs_occ', 'atom_df_coeffs',
                 'atom_df_coeffs_occ', 'batch_atom_numbers', 'batch_positions']
all_res_comb = {'ml_df_coeffs_basis': [], 'atom_mo_coeffs_basis': [], 'atom_df_coeffs_basis': []}
for res_dict in res_arr:
    batch_size = res_dict['res_b']['batch_positions'].shape[0]
    for key in relevant_keys:
        if key not in all_res_comb:
            all_res_comb[key] = res_dict['res_b'][key]
        else:
            all_res_comb[key] = torch.cat((all_res_comb[key], res_dict['res_b'][key]), dim=0)

    all_res_comb['atom_mo_coeffs_basis'].extend([res_dict['res_b']['atom_mo_coeffs_basis']] * batch_size)
    all_res_comb['atom_df_coeffs_basis'].extend([res_dict['res_b']['atom_df_coeffs_basis']] * batch_size)

    anum = torch.max(res_dict['res_b']["batch_atom_numbers"], dim=0)[0]
    atom_types, _ = orbitals.create_ghost_atom_types(anum)

    for idx in range(batch_size):
        ml_basis = orbitals.ml_basis_to_pyscf_basis(res_dict['res_b'], atom_types, idx)
        all_res_comb['ml_df_coeffs_basis'].append(ml_basis)

    ml_df_coeffs = orbitals.coeffs_dict_to_vector(res_dict['res_b'], dataset.orbital_basis_num, res_dict['res_b']['batch_atom_numbers'],
                                                  radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

    if 'ml_df_coeffs' not in all_res_comb.keys():
        all_res_comb['ml_df_coeffs'] = ml_df_coeffs
    else:
        all_res_comb['ml_df_coeffs'] = torch.cat((all_res_comb['ml_df_coeffs'], ml_df_coeffs), dim=0)

for key in all_res_comb.keys():
    if isinstance(all_res_comb[key], torch.Tensor):
        print(key, all_res_comb[key].shape)
    else:
        print(key)
np.save('results/benzene_dimer_ml_results.npy', all_res_comb, allow_pickle=True)
# %%
