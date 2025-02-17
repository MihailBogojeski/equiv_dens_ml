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
import pickle

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

import ase.io
import dftd4.pyscf as d4disp
from pyscf.dft import gen_grid, radi, numint
from pyscf.lib import param
from equiv_dens.training import utils as train_utils
from argparse import Namespace
from equiv_dens.training import model_loader
from pymbd.fortran import MBDGeom
from vdw import to_mbd
from equiv_dens.utils import hirshfeld_analysis, orbitals
from equiv_dens.utils import sapt0

hf.MUTE_CHKFILE = True

# %%
# # load dimer data
# data = dict(np.load('datasets/water_benzene_results.npz', allow_pickle=True))
# water_data = {'positions': data['water_positions'], 'atom_numbers': data['water_Z']}
# benzene_data = {'positions': data['benzene_positions'], 'atom_numbers': data['benzene_Z']}
#
# # %%
# # calculate interatomic distances to confirm unit
# water_centers = utils.center_of_mass(torch.tensor(water_data['positions']), torch.tensor(water_data['atom_numbers']))
# benzene_centers = utils.center_of_mass(torch.tensor(benzene_data['positions']), torch.tensor(benzene_data['atom_numbers']))
#
# dists = torch.norm(water_centers - benzene_centers, dim=-1)
# # %%
# calc_results = np.load('results/calc_en_comps.npy', allow_pickle=True)
# print(calc_results)
# # %%
# w_mols = utils.npy_to_pyscf(water_data['positions'], water_data['atom_numbers'], basis='augccpvdz')
# b_mols = utils.npy_to_pyscf(benzene_data['positions'], benzene_data['atom_numbers'], basis='augccpvdz')
# dm_1 = [calc_results[idx]['water_dm'] for idx in range(3)]
# dm_2 = [calc_results[idx]['benzene_dm'] for idx in range(3)]
# sapt0.calculate_sapt0_ref(w_mols[:3], b_mols[:3], dm_1, dm_2)
# %%
data = np.load('datasets/s66x8_pyscf_augccpvdz_calc.npy', allow_pickle=True)

print('len data', len(data))
print('data 0', data[0])
print('data 0', data[50])
print('data 0', data[250])
print('data 0', data[500])
# %%
pos = [np.array([d[0]['atom'][i][1] for i in range(len(d[0]['atom']))]) for d in data]
print([p.shape for p in pos])
print(data[-25][0])
# %%
# find unique molecules
# start with extracting filesnames and finding filenames with same molecule twice
filenames = [d[0]['filename'][7:-7] for d in data]
print(filenames)
# %%
atom = data[-25][0]['atom'][:12]
mol = gto.M(atom=atom, basis='ccpvdz')
mf = dft.RKS(mol)
mf.chkfile = False
mf.xc = 'pbe'
mf.kernel()
# %%
print(mf.dip_moment())
print(utils.angstrom_to_bohr(mf.dip_moment() / 4.8))
# %%
# print(filenames)
mol_names = {}
for i, fname in enumerate(filenames):
    if len(fname) % 2 == 0:
        half = int(len(fname) / 2)
        f1 = fname[:half]
        f2 = fname[half:]
        half_pos = int(pos[i].shape[0] / 2)
        if f1 == f2:
            mol_names[f1] = half_pos
mol_names['Benzene'] = 12
print(mol_names)
converged = False
while not converged:
    new_mol_names = {}
    for i, fname in enumerate(filenames):
        for mol_name in mol_names:
            if mol_name == fname[:len(mol_name)]:
                pos_size = pos[i][mol_names[mol_name]:].shape[0]
                new_mol_names[fname[len(mol_name):]] = pos_size

    new_mol_names.update(mol_names)
    print('new mol names', new_mol_names)
    print(len(new_mol_names.keys()))
    if mol_names == new_mol_names:
        converged = True
    else:
        print('not converged yet')
    mol_names = new_mol_names
print('mol names', mol_names)
# %%
# see if name fragments cover all dimers
covered = []
for i, fname in enumerate(filenames):
    covered1 = False
    covered2 = False
    print('fname', fname, pos[i].shape[0])
    for name in mol_names.keys():
        name_len = len(name)
        if name == fname[:name_len]:
            covered1 = True
            print('name1', name, mol_names[name])
            break

    for name in mol_names.keys():
        if name == fname[name_len:]:
            covered2 = True
            print('name2', name, mol_names[name])
            break
    covered.append(covered1 and covered2)
    print()

print(all(covered))
# %%
print(filenames.index('BenzeneBenzenepipi'))
print(filenames.index('BenzeneBenzeneTS'))
print(data[184][0])
print(data[368])
# %%
# basic arguments for model loading
main_args = Namespace()

main_args.args_file = "args/qm7x250_dens_001_coreless.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
main_args.save_file = 'qm7x250_dens_001_coreless'
main_args.use_gpu = True 

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

args.np_dataset_test = "datasets/qm7x_test_dft_augccpvdz_small_base.npy"
args.dens_dataset_test = "datasets/qm7x_test_dft_augccpvdz_small.npy"
# args.np_dataset_test = "datasets/s66x8_pyscf_augccpvdz_base.npy"
# args.dens_dataset_test = "datasets/s66x8_pyscf_augccpvdz_calc.npy"

dataset_grid = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
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

print('dataset length', len(dataset_grid))
samp = dataset_grid.get_properties([0])
print('sample pos shape', samp['positions'].shape)
print('sample dens shape', samp['density'].shape)
print('sample dens integral', torch.sum(samp['density'][0] * samp['coord_weights'][0]))
print('args use gpu', args.use_gpu)

model_grid = model_loader.load_model(args, dataset_grid)
# %%
for param in model_grid.parameters():
    param.requires_grad = False
idx = [2, 4]
samp = dataset_grid.get_properties(idx)
print('loaded sample')
if args.use_gpu:
    for key in samp.keys():
        if isinstance(samp[key], torch.Tensor):
            samp[key] = samp[key].cuda()

res = model_grid(samp)
print('allocated gpu memory', torch.cuda.memory_allocated() / 1024**2)

print('sum atom numbers', torch.sum(samp['batch_atom_numbers'], dim=1))
print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
print('true density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
print('density error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1))
res = None
# %%
with open('./datasets/mol_ase_atoms.pickle', 'rb') as f:
   all_molecules =  pickle.load(f)
with open('./datasets/h_rats_qchem_pbe_adz.pickle', 'rb') as f:
    hrats = pickle.load(f)

for k,v in hrats.items():
    print("mol", k)
    print("atoms", all_molecules[k].get_chemical_symbols())
    print("hrat", v)
# %%

print(all_molecules.keys())
positions = []
atom_numbers = []
calc_key = []

for key in all_molecules.keys():
    positions.append(all_molecules[key].get_positions())
    atom_numbers.append(all_molecules[key].get_atomic_numbers())
    calc_key.append(key)

atom_numbers, mols = utils.compress_batch_atoms(atom_numbers, {'positions': positions})

mols['atom_numbers'] = atom_numbers
mols['calc_key'] = calc_key

print('num mols', len(mols['calc_key']))
# %%
res_arr = sapt0.evaluate_ml_monomer(mols, model_grid, dataset_grid.orbital_basis_num, use_gpu=True,
                                    atom_dens_dict=dataset_grid.atom_dens, cutoff=args.cutoff,
                                    grid_spec=dataset_grid.grid_spec, grid_fn=dataset_grid.sampling_fn,
                                    density_expansion=True, batch_size=1, collate=False,
                                    split_atom_densities=True)
# %%
mbd_res = {}
for i in range(len(res_arr)):
    res = res_arr[i]
    wA, atomic_charges, dipoles, volume_ratio, r3_vol, r3_vol_free = hirshfeld_analysis.hirshfeld_partitioning(res['density'],
                                                                                          res['atom_density_split'],
                                                                                          res['batch_positions'], res['batch_atom_numbers'],
                                                                                          res['coords'], res['coord_weights'],
                                                                                          to_bohr=True)
    k, v = list(hrats.items())[i]
    print('atom_numbers', res['batch_atom_numbers'][res['batch_atom_numbers'] == 8], 'volume_ratio', volume_ratio[res['batch_atom_numbers'] == 8]*1.025)
    print("mol", k)
    at_O = np.array(all_molecules[k].get_chemical_symbols()) == 'O'
    print("hrat", v[at_O])
    mbd_res[mols['calc_key'][i]] = {
        'atom_numbers': res['batch_atom_numbers'].numpy(force=True),
        'positions': res['batch_positions'].numpy(force=True),
        'volume_ratios': volume_ratio.numpy(force=True),
        'free_atom_volumes': r3_vol_free.numpy(force=True),
        'atom_mo_coeffs': res['atom_mo_coeffs'].numpy(force=True),
        'atom_mo_occ': res['atom_mo_coeffs_occ'].numpy(force=True),
        'ml_df_coeffs': res['df_coeffs'].numpy(force=True),
        'atom_mo_coeffs_basis': res['atom_mo_coeffs_basis'],
        'ml_df_coeffs_basis': res['ml_df_coeffs_basis'],
    }

print(mbd_res)
res_arr = None
# %%
# pickle dump results
with open('./datasets/mbd_res.pickle', 'wb') as f:
    pickle.dump(mbd_res, f)
# %%
args.density_weight = 0.0
args.dpm_intor = True

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

model = model_loader.load_model(args, dataset)
# %%
for param in model.parameters():
    param.requires_grad = False
idx = [2, 4]
samp = dataset.get_properties(idx)
print('loaded sample')
if args.use_gpu:
    for key in samp.keys():
        if isinstance(samp[key], torch.Tensor):
            samp[key] = samp[key].cuda()

res = model(samp)
print('allocated gpu memory', torch.cuda.memory_allocated() / 1024**2)

print('sum atom numbers', torch.sum(samp['batch_atom_numbers'], dim=1))
print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
print('true density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
print('density error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1))
res = None
# %%
with open('./datasets/nenci_monomer_ase.pickle', 'rb') as f:
    monomer1, monomer2 = pickle.load( f)
with open('./datasets/calc_dict.pickle', 'rb') as f:
    calculations = pickle.load(f)
# %%
print('calculations', calculations['001'])
print('monomer1 keys', monomer1.keys())
print('monomer1 keys len', len(monomer1.keys()))
monomer1.pop('024_Benzene-Benzene_1.00.xyz', None)
monomer2.pop('024_Benzene-Benzene_1.00.xyz', None)
# %%
pos_1 = []
pos_2 = []
anum_1 = []
anum_2 = []
calc_keys = []


for i, key in enumerate(monomer1.keys()):
        pos_1.append(monomer1[key].get_positions())
        anum_1.append(monomer1[key].get_atomic_numbers())
        pos_2.append(monomer2[key].get_positions())
        anum_2.append(monomer2[key].get_atomic_numbers())
        calc_keys.append(key)
        if (i + 1) % 63 == 0:
            pos_1.append(monomer1[key].get_positions())
            anum_1.append(monomer1[key].get_atomic_numbers())
            pos_2.append(monomer2[key].get_positions())
            anum_2.append(monomer2[key].get_atomic_numbers())
            calc_keys.append(key + '_dup')

atom_numbers_1, mols1 = utils.compress_batch_atoms(anum_1, {'positions': pos_1})
mols1['atom_numbers'] = atom_numbers_1
mols1['calc_key'] = calc_keys

atom_numbers_2, mols2 = utils.compress_batch_atoms(anum_2, {'positions': pos_2})
mols2['atom_numbers'] = atom_numbers_2
mols2['calc_key'] = calc_keys

print('mols1 positions shape', mols1['positions'].shape)
print('mols1 atom numbers shape', mols1['atom_numbers'].shape)
print('mols1 keys', len(mols1['calc_key']))
print('mols2 positions shape', mols2['positions'].shape)
print('mols2 atom numbers shape', mols2['atom_numbers'].shape)
print('mols2 keys', len(mols2['calc_key']))

# %%
batch_size = 4
res1, res2, res12 = sapt0.evaluate_ml_dimer(mols1, mols2,
                                            model, dataset.orbital_basis_num, use_gpu=True,
                                            atom_dens_dict=dataset.atom_dens, cutoff=args.cutoff,
                                            batch_size=batch_size, collate=False)
# %%
torch.save(res1, 'datasets/dimer_res_monomer1.pth')
torch.save(res2, 'datasets/dimer_res_monomer2.pth')
torch.save(res12, 'datasets/dimer_res_dimer.pth')
# %%
res1 = torch.load('datasets/dimer_res_monomer1.pth')
res2 = torch.load('datasets/dimer_res_monomer2.pth')
res12 = torch.load('datasets/dimer_res_dimer.pth')
# %%
# convert dimer results to numpy format and batch by dimer type
dimer1_res = {}
dimer2_res = {}
dimer12_res = {}
batch_size = 4

batches_per_dimer_type = 64 / batch_size

relevant_batch_keys = ['atom_mo_coeffs', 'atom_mo_coeffs_occ', 'atom_df_coeffs',
                       'atom_df_coeffs_occ', 'batch_atom_numbers', 'batch_positions']
relevant_fix_keys = ['ml_df_coeffs_basis', 'atom_mo_coeffs_basis', 'atom_df_coeffs_basis']

for i in range(int(mols1['atom_numbers'].shape[0] / 64)):
    calc_idxs = np.arange(i * 64, (i + 1) * 64)
    print('calc idxs', calc_idxs)
    dimer_key = mols1['calc_key'][calc_idxs[0]][:3]
    calc_keys = [mols1['calc_key'][idx] for idx in calc_idxs]


    batch_idxs = np.arange(i * batches_per_dimer_type, (i + 1) * batches_per_dimer_type).astype(int)
    print('batch_idxs', batch_idxs)

    m1_batch_arr = orbitals.collate_ml_outs([res1[idx] for idx in batch_idxs], relevant_batch_keys,
                                            relevant_fix_keys, dataset.orbital_basis_num)
    m2_batch_arr = orbitals.collate_ml_outs([res2[idx] for idx in batch_idxs], relevant_batch_keys,
                                            relevant_fix_keys, dataset.orbital_basis_num)
    m12_batch_arr = orbitals.collate_ml_outs([res12[idx] for idx in batch_idxs], relevant_batch_keys,
                                             relevant_fix_keys, dataset.orbital_basis_num)

    for key in m1_batch_arr.keys():
        if isinstance(m1_batch_arr[key], torch.Tensor):
            m1_batch_arr[key] = m1_batch_arr[key].numpy(force=True)
            m2_batch_arr[key] = m2_batch_arr[key].numpy(force=True)
            m12_batch_arr[key] = m12_batch_arr[key].numpy(force=True)

    for key in m1_batch_arr.keys():
        if isinstance(m1_batch_arr[key], np.ndarray):
            print('m1', key, m1_batch_arr[key].shape)
            print('m2', key, m2_batch_arr[key].shape)
            print('m12', key, m12_batch_arr[key].shape)
        else:
            print('m1', key, len(m1_batch_arr[key]))
            print('m2', key, len(m2_batch_arr[key]))
            print('m12', key, len(m12_batch_arr[key]))

    m1_batch_arr['calc_keys'] = calc_keys
    m2_batch_arr['calc_keys'] = calc_keys
    m12_batch_arr['calc_keys'] = calc_keys

    dimer1_res[dimer_key] = m1_batch_arr
    dimer2_res[dimer_key] = m2_batch_arr
    dimer12_res[dimer_key] = m12_batch_arr
# %%
# pickle dump results
with open('./datasets/monomer1_density_res.pickle', 'wb') as f:
    pickle.dump(dimer1_res, f)
with open('./datasets/monomer2_density_res.pickle', 'wb') as f:
    pickle.dump(dimer2_res, f)
with open('./datasets/dimer_density_res.pickle', 'wb') as f:
    pickle.dump(dimer12_res, f)
# %%
# batch dimer results by dimer type
dimer1_res = {}
dimer2_res = {}
dimer12_res = {}
batch_size = 4

batches_per_dimer_type = 64 / batch_size

relevant_batch_keys = ['atom_mo_coeffs', 'atom_mo_coeffs_occ', 'atom_df_coeffs',
                       'atom_df_coeffs_occ', 'batch_atom_numbers', 'batch_positions']
relevant_fix_keys = ['ml_df_coeffs_basis', 'atom_mo_coeffs_basis', 'atom_df_coeffs_basis']

for i in range(int(mols1['atom_numbers'].shape[0] / 64)):
    calc_idxs = np.arange(i * 64, (i + 1) * 64)
    print('calc idxs', calc_idxs)
    dimer_key = mols1['calc_key'][calc_idxs[0]][:3]
    calc_keys = [mols1['calc_key'][idx] for idx in calc_idxs]


    batch_idxs = np.arange(i * batches_per_dimer_type, (i + 1) * batches_per_dimer_type).astype(int)
    print('batch_idxs', batch_idxs)

    m1_batch_arr = orbitals.collate_ml_outs([res1[idx] for idx in batch_idxs], relevant_batch_keys,
                                            relevant_fix_keys, dataset.orbital_basis_num)
    m2_batch_arr = orbitals.collate_ml_outs([res2[idx] for idx in batch_idxs], relevant_batch_keys,
                                            relevant_fix_keys, dataset.orbital_basis_num)
    m12_batch_arr = orbitals.collate_ml_outs([res12[idx] for idx in batch_idxs], relevant_batch_keys,
                                             relevant_fix_keys, dataset.orbital_basis_num)

    m1_batch_arr['calc_keys'] = calc_keys
    m2_batch_arr['calc_keys'] = calc_keys
    m12_batch_arr['calc_keys'] = calc_keys

    dimer1_res[dimer_key] = m1_batch_arr
    dimer2_res[dimer_key] = m2_batch_arr
    dimer12_res[dimer_key] = m12_batch_arr

        # monomer1_batch[mols1['calc_key'][i]] = {
        #     'atom_numbers': res1[i]['batch_atom_numbers'].numpy(force=True),
        #     'positions': res1[i]['batch_positions'].numpy(force=True),
        #     'atom_mo_coeffs': res1[i]['atom_mo_coeffs'].numpy(force=True),
        #     'atom_mo_occ': res1[i]['atom_mo_coeffs_occ'].numpy(force=True),
        #     'ml_df_coeffs': res1[i]['df_coeffs'].numpy(force=True),
        #     'atom_mo_coeffs_basis': res1[i]['atom_mo_coeffs_basis'],
        #     'ml_df_coeffs_basis': res1[i]['ml_df_coeffs_basis'],
        # }
        #
        # dimer2_res[mols2['calc_key'][i]] = {
        #     'atom_numbers': res2[i]['batch_atom_numbers'].numpy(force=True),
        #     'positions': res2[i]['batch_positions'].numpy(force=True),
        #     'atom_mo_coeffs': res2[i]['atom_mo_coeffs'].numpy(force=True),
        #     'atom_mo_occ': res2[i]['atom_mo_coeffs_occ'].numpy(force=True),
        #     'ml_df_coeffs': res2[i]['df_coeffs'].numpy(force=True),
        #     'atom_mo_coeffs_basis': res2[i]['atom_mo_coeffs_basis'],
        #     'ml_df_coeffs_basis': res2[i]['ml_df_coeffs_basis'],
        # }
        #
        # dimer12_res[mols1['calc_key'][i]] = {
        #     'atom_numbers': res12[i]['batch_atom_numbers'].numpy(force=True),
        #     'positions': res12[i]['batch_positions'].numpy(force=True),
        #     'atom_mo_coeffs': res12[i]['atom_mo_coeffs'].numpy(force=True),
        #     'atom_mo_occ': res12[i]['atom_mo_coeffs_occ'].numpy(force=True),
        #     'ml_df_coeffs': res12[i]['df_coeffs'].numpy(force=True),
        #     'atom_mo_coeffs_basis': res12[i]['atom_mo_coeffs_basis'],
        #     'ml_df_coeffs_basis': res12[i]['ml_df_coeffs_basis'],
        # }
# %%
torch.save(dimer1_res, 'datasets/monomer1_density_res.pth')
torch.save(dimer2_res, 'datasets/monomer2_density_res.pth')
torch.save(dimer12_res, 'datasets/dimer_density_res.pth')
# %%
print(dimer1_res['063']['calc_keys'])
# %%
# pickle dump results
with open('./datasets/monomer1_density_res.pickle', 'rb') as f:
    dimer1_res = pickle.load(f)
with open('./datasets/monomer2_density_res.pickle', 'rb') as f:
    dimer2_res = pickle.load(f)
with open('./datasets/dimer_density_res.pickle', 'rb') as f:
    dimer12_res = pickle.load(f)
# %%
dimer1_res = torch.load('datasets/monomer1_density_res.pth')
dimer2_res = torch.load('datasets/monomer2_density_res.pth')
dimer12_res = torch.load('datasets/dimer_density_res.pth')
# %%
sapt0_res = {} 
for key in dimer1_res.keys():
    res1 = dimer1_res[key]
    res2 = dimer2_res[key]
    res12 = dimer12_res[key]
    print(res1.keys())
    print(res2.keys())
    print(res12.keys())
    sapt0_elst, efield_1_at_2, efield_2_at_1, dens_ovlp = sapt0.calculate_sapt0_ml(res1, res2,
                                                                                   res12, dataset.orbital_basis_num,
                                                                                   precalc_basis=True)
    sapt0_res[key] = {
        'sapt0_elst': sapt0_elst,
        'efield_at_monomer1': efield_2_at_1,
        'efield_at_monomer2': efield_1_at_2,
        'dens_ovlp': dens_ovlp
    }

# %%
with open('./datasets/sapt0_ml_res.pickle', 'wb') as f:
    pickle.dump(sapt0_res, f)
# %%
with open('./datasets/interaction_energies_adz.pickle', 'rb') as f:
    interaction_energies, mbd_energies = pickle.load(f)
with open('./datasets/sapt_adz.pickle', 'rb') as f:
    sapt_for_calc = pickle.load(f)
with open('./datasets/efields_adz.pickle', 'rb') as f:
    efield_at_monomer1, efield_at_monomer2 = pickle.load(f)
# %%
for i in range(i):
    key = calculations['001'][i]
    print('key', key)
    print(f"interatction energy between the dimer {sapt_for_calc[key]} kcal/mol using pbe with a aug-ccpvdz basis") 
    print('E field from monomer 1 at monomer 2', efield_at_monomer2[key])
    print('E field from monomer 2 at monomer 1', efield_at_monomer1[key])
