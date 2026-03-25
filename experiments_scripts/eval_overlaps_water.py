import pickle
import sys
from pyscf import gto, df
from pyscf.dft import numint
from pyscf.scf import hf
import numpy as np
from argparse import Namespace
from functools import partial
import torch
from copy import copy, deepcopy

hf.MUTE_CHKFILE = True

from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
     CubicalGrid, spherical_grid, spherical_radial_sampling
from equiv_dens.training.model_loader import load_model
import equiv_dens.utils.base as utils
from equiv_dens.training import utils as train_utils
from equiv_dens.training import model_loader
from equiv_dens.utils import orbitals

# basic arguments for model loading
main_args = Namespace()

file = sys.argv[1]
from_calc_data = bool(int(sys.argv[2]))
print('from calc data', from_calc_data)

main_args.args_file = file
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
main_args.save_file = 'qm7x250_dens_001_coreless'
main_args.use_gpu = False
# %%
# #load arguments and dataset
args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)

args.fix_arguments = True

args, hyperparam_args, train_vars = train_utils.init_training_vars(args, hyperparam_args)
checkpoint = train_vars['checkpoint']

# #determine whether GPU is used for training

# #load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

args.verbose = 0
args.use_gpu = main_args.use_gpu
args.expansion_constraint = None
args.integral_constraint = None
# #args.integral_constraint = None
args.ignore_missing_keywords = True

required_properties = ['density', 'dipole_moment']

args.spherical_grid_level = 1
args.cube_grid = False
grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
sampling_fn = partial(spherical_radial_sampling, rotate=False)
grid_origin = 0
grid_extent = None
args.radii_adjust = True
# #grid_vars = train_utils.init_grid_vars(args)
rotate = False

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file

# #args.np_dataset_test = "datasets/qm7x_test_dft_augccpvdz_small_base.npy"
# #args.dens_dataset_test = "datasets/qm7x_test_dft_augccpvdz_small.npy"
args.np_dataset_test = "/home/ml-dft/equiv_dens/datasets/s66x8_pyscf_augccpvdz_base.npy"
args.dens_dataset_test = "/home/ml-dft/equiv_dens/datasets/s66x8_pyscf_augccpvdz_calc.npy"

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
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
                           atom_dens_path=args.atom_dens_path,
                           atom_dens_type='mo_coeffs',
                           split_atom_dens=True,
                           density_grad=args.density_grad,
                           calc_basis_path='/home/ml-dft/equiv_dens/datasets/augccpvdz_orbital_basis.npy',
                           all_atom_coeffs=True,
                           dens_sqrt=args.dens_sqrt,
                           valence_dens=args.valence_dens,
                           full_valence=args.full_valence,
                           )

dataset_test = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
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
                           atom_dens_path=args.atom_dens_path,
                           atom_dens_type='mo_coeffs',
                           split_atom_dens=True,
                           density_grad=args.density_grad,
                           calc_basis_path='/home/ml-dft/equiv_dens/datasets/augccpvdz_orbital_basis.npy',
                           all_atom_coeffs=True,
                           dens_sqrt=args.dens_sqrt,
                           valence_dens=args.valence_dens,
                           full_valence=args.full_valence,
                           )

samp = dataset.get_properties([0])
model = model_loader.load_model(args, dataset)
for param in model.parameters():
    param.requires_grad = False

# %%
# #evaluate model and test density integral


# %% [markdown]
"""
Each calculation name is a key for the dicts monomer1 and monomer2. monomer1[key] is an ase.Atoms object of just the first monomer in each calc and monomer2[key] is the corresponding Atoms object for monomer2.
"""

# %%
with open('ref_ovlps_for_mihail/calc_dict_fixed.pickle', 'rb') as f:
    calculations = pickle.load(f)


if not from_calc_data:
    idx = list(range(8))
    with open('/home/ml-dft/equiv_dens/datasets/nenci_monomer_ase.pickle', 'rb') as f:
        mono1, mono2 = pickle.load(f)
    pos1 = []
    z1 = []
    pos2 = []
    z2 = []
    for i in idx:
        key = calculations['001'][i]
        print(key)
        print(mono1[key])
        print(mono2[key])

        atoms1 = mono1[key]
        atoms2 = mono2[key]

        p1 = mono1[key].get_positions()
        a1 = mono1[key].get_atomic_numbers()
        p2 = mono2[key].get_positions()
        a2 = mono2[key].get_atomic_numbers()
        pos1.append(p1)
        pos2.append(p2)
        z1.append(a1)
        z2.append(a2)

    pos1 = np.stack(pos1, axis=0)
    pos2 = np.stack(pos2, axis=0)
    z1 = np.stack(z1, axis=0)
    z2 = np.stack(z2, axis=0)
    input1 = {'atom_numbers': z1, 'positions': pos1}
    input2 = {'atom_numbers': z2, 'positions': pos2}
    coord_params = None
else:
    idx = [0, 1, 2, 3, 4]
    samp = dataset_test.get_properties(idx)
    if args.use_gpu:
        for key in samp.keys():
            if isinstance(samp[key], torch.Tensor):
                samp[key] = samp[key].cuda()

    def split_dimer_idxs(atom_numbers):
        uniques, counts = torch.unique(atom_numbers, return_counts=True)
        idxs1 = []
        idxs2 = []
        idx = 0
        for i in range(len(uniques)):
            half_count = counts[i]//2
            idxs1.extend(list(range(idx, idx + half_count)))
            idxs2.extend(list(range(idx + half_count, idx + counts[i])))
            idx += counts[i]
        return torch.LongTensor(idxs1), torch.LongTensor(idxs2)

    idxs1, idxs2 = split_dimer_idxs(samp['batch_atom_numbers'][0])

    input1 = {'atom_numbers': samp['batch_atom_numbers'][:, idxs1], 'positions': samp['batch_positions'][:, idxs1]}
    input2 = {'atom_numbers': samp['batch_atom_numbers'][:, idxs2], 'positions': samp['batch_positions'][:, idxs2]}
    coord_params = {'coords': samp['coords'], 'coord_weights': samp['coord_weights']}

input12 = {key: np.concatenate([input1[key], input2[key]], axis=1) for key in input1.keys()}

samp12 = orbitals.model_input_from_atoms(input12,
                                         density_expansion=True,
                                         skip_compress=True,
                                         grid_spec=dataset.grid_spec,
                                         cutoff=args.cutoff,
                                         dtype=torch.float32,
                                         atom_dens_type="mo_coeffs",
                                         free_atom_densities=dataset.atom_dens,
                                         split_atom_densities=False,
                                         basis=None,
                                         all_atom_coeffs=False,
                                         coord_params=coord_params,
                                         )
samp1 = orbitals.model_input_from_atoms(input1,
                                         density_expansion=False,
                                         skip_compress=True,
                                         grid_spec=None,
                                         cutoff=args.cutoff,
                                         dtype=torch.float32,
                                         atom_dens_type="mo_coeffs",
                                         free_atom_densities=dataset.atom_dens,
                                         split_atom_densities=False,
                                         basis=None,
                                         all_atom_coeffs=False,
                                         coord_params=None,
                                         )
samp2 = orbitals.model_input_from_atoms(input2,
                                        density_expansion=False,
                                        skip_compress=True,
                                        grid_spec=None,
                                        cutoff=args.cutoff,
                                        dtype=torch.float32,
                                        atom_dens_type="mo_coeffs",
                                        free_atom_densities=dataset.atom_dens,
                                        split_atom_densities=False,
                                        basis=None,
                                        all_atom_coeffs=False,
                                        coord_params=None,
                                        )

samp1['coords'] = samp12['coords']
samp1['coord_weights'] = samp12['coord_weights']
samp2['coords'] = samp12['coords']
samp2['coord_weights'] = samp12['coord_weights']
# res1 = model.density_repr_model(samp1)
# res2 = model.density_repr_model(samp2)
res1 = model(samp1)
res2 = model(samp2)
dens_comb = res1['density'] + res2['density']
print('sum atom numbers', torch.sum(samp12['batch_atom_numbers'], dim=1))
print('res density integral', torch.sum(dens_comb * res1['coord_weights'], dim=1))
if from_calc_data:
    print('true density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
    print('density error', torch.sum(torch.abs(dens_comb - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1))

df_coeffs_ml1 = orbitals.coeffs_dict_to_vector(res1, dataset.orbital_basis_num, res1['batch_atom_numbers'],
                                                radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

df_coeffs_ml2 = orbitals.coeffs_dict_to_vector(res2, dataset.orbital_basis_num, res2['batch_atom_numbers'],
                                                radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

print('intor res1', orbitals.calculate_1e_intor_ml(res1, dataset.orbital_basis_num, 'int1e_ovlp'))
print('intor res2', orbitals.calculate_1e_intor_ml(res2, dataset.orbital_basis_num, 'int1e_ovlp'))


input12 = {key: np.concatenate([input1[key], input2[key]], axis=1) for key in input1.keys()}

res12 = samp12
res12['spherical_coeffs'] = res1['spherical_coeffs'] + res2['spherical_coeffs']
res12['radial_scale'] = res1['radial_scale'] + res2['radial_scale']
res12['radial_width'] = res1['radial_width'] + res2['radial_width']

print('intor res12', orbitals.calculate_1e_intor_ml(res12, dataset.orbital_basis_num, 'int1e_ovlp'))

with open('ref_ovlps_for_mihail/c01_all_ovlps.pickle', 'rb') as f:
    ovlps = pickle.load(f)

if from_calc_data:
    data_idx = [2, 3, 4, 5, 6]
else:
    data_idx = idx
keylist = list(ovlps.keys())
ovlps = {keylist[i]:ovlps[keylist[i]] for i in data_idx}
ovlps_calc = [ovlps[key]['S_rho_3all'] for key in ovlps.keys()]


# overlap integral between ML densities
ovlps_ml = []
for i in idx:
    auxmol_ml1 = orbitals.ml_basis_to_auxmol(res1, i, skip_zero=False)
    auxmol_ml2 = orbitals.ml_basis_to_auxmol(res2, i, skip_zero=False)
    ovlp_ml = gto.mole.intor_cross('int1e_ovlp', auxmol_ml1, auxmol_ml2)
    ovlp_int_ml = np.einsum('i, ij, j -> ', df_coeffs_ml1[i], ovlp_ml, df_coeffs_ml2[i])
    print('ovlp ML int', ovlp_int_ml)
    ovlps_ml.append(ovlp_int_ml)

print('calc ovlps', ovlps_calc)
print('ML ovlps', ovlps_ml)
print('ovlp_diff', np.mean(np.abs(np.array(ovlps_calc) - np.array(ovlps_ml))))

