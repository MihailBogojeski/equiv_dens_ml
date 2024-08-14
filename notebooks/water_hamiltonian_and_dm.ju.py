# %%
import numpy as np
import time
import os
import ase
from pyscf import gto, dft, df, lib
from pyscf.scf import hf
import scipy
from equiv_dens.utils import base as utils
from equiv_dens.training import utils as train_utils
from argparse import Namespace
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.utils import orbitals
from equiv_dens.data.custom_samplers import set_up_data_loader
import torch
hf.MUTE_CHKFILE = True
# %%
# %cd ..
# %load_ext autoreload
# %autoreload 2
# %%
##
#data = np.load('datasets/h2o_dynamic_centered.npy', allow_pickle=True).item()
#basis = 'augccpvdz'
#set_type = 'test'
#save_path = 'datasets/h2o_small_' + set_type + '_dft_augccpvdz_df_hm_dm_oe_calc.npy'
#
#results = np.load(save_path, allow_pickle=True)
#for res in results:
#    calc = res[1]
#    mo_coeff = calc['mo_coeff']
#    mo_occ = calc['mo_occ']
#    mo_en = calc['mo_energies']
#    mol = gto.M(atom=res[0]['atom'], basis=basis)
#    s1e = mol.intor('int1e_ovlp')
#    ks = calc['hamiltonian_matrix'] 
#    mf = dft.RKS(mol)
#    moe_calc, mo_calc = mf.eig(ks, s1e)
#    print('energy', calc['energy'])


# %%
set_type = 'test'
save_path = 'datasets/h2o_small_' + set_type + '_dft_augccpvdz_df_hm_dm_oe_calc.npy'

results = np.load(save_path, allow_pickle=True)
print('results', results[0][1]['mo_coeff'].shape)
args_file = "args/h2o_small_all_001_hm_dm.txt"
args, hyperparam_args = parse_command_line_arguments(arg_file=args_file)
args, hyperparam_args, train_vars = train_utils.init_training_vars(args, hyperparam_args)
args.center_energy = False
checkpoint = train_vars['checkpoint']
args_dict = vars(args)
grid_vars = train_utils.init_grid_vars(args)
required_properties = train_utils.get_required_properties_from_args(args)
print('required_properties', required_properties)
required_properties.extend(['hamiltonian_matrix', 'mo_coeff', 'mo_energies'])

print('data split indices', train_vars['data_split_indices'])
dataset, train_dataset, valid_dataset, test_dataset, valid_cube_dataset, data_split_indices =\
    train_utils.prepare_datasets(args, required_properties,
                                    grid_vars, train_vars['data_split_indices'])

print(len(dataset))
print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))
# %%
test_data_loader = set_up_data_loader(test_dataset, 1,
                                           electron_num_batching=False,
                                           use_gpu=True, shuffle=False)
test_iter = iter(test_data_loader)
samp = next(test_iter)

print('samp energy', samp['energy'])
print('res energy', results[0][1]['energy'])
print('samp df coeffs', samp['df_coeffs'])
print('results', results[0][1]['df_coeff'])

# %%
print('dataset orbital basis', dataset.orbital_basis)
print('dataset orbital basis num', dataset.orbital_basis_num)
print('dataset orbital basis size', dataset.orbital_basis_size)
# %%
print('dataset orbital basis', dataset.orbital_basis)
print('dataset orbital basis num', dataset.orbital_basis_num)
print('dataset orbital basis size', dataset.orbital_basis_size)
# %%

ham = results[0][1]['hamiltonian_matrix']
mo_coeff = results[0][1]['mo_coeff']
atom = results[0][0]['atom']
basis = results[0][0]['basis']
print('basis', basis)
print('dataset basis size', dataset.orbital_basis_size)
print('df coeffs size', results[0][1]['df_coeff'].shape)
mol = gto.M(atom=atom, basis=basis)
ao_basis, _ = orbitals.get_basis_from_mol(mol)
atom_numbers = samp['atom_numbers']
print('ao basis', ao_basis)
all_atom_numbers = np.unique(atom_numbers.flatten())
all_atom_numbers = all_atom_numbers[all_atom_numbers > 0]
orbital_basis_size = {}
for key in ao_basis.keys():
    z = utils.symbols_to_numbers([key])[0]
    if z in all_atom_numbers:
        orbital_basis_size[z] = 0
        for orb in ao_basis[key]:
            orbital_basis_size[z] += ((2 * orb[2]) + 1)
print('orbital_basis_size', orbital_basis_size)
print('mo_coeff size', mo_coeff.shape)
print('hm size', results[0][1]['hamiltonian_matrix'].shape)
mo_coeff_split = orbitals.split_df_coeffs(atom, mo_coeff, orbital_basis_size)
hm_split = orbitals.split_ao_matrix(atom, ham, orbital_basis_size)
mo_nums, mo_comp = utils.compress_batch_atoms(samp['batch_atom_numbers'].numpy(force=True),
                               {'mo_coeff': [mo_coeff_split], 'hamiltonian_matrix': [hm_split]},
                               orbital_basis_size)

print('hm_split', hm_split)
atom2 = [atom[1], atom[2], atom[0]]
mo_comp_split = orbitals.split_ao_matrix(atom2, mo_comp['hamiltonian_matrix'][0], orbital_basis_size)
print()
print()
print()
print()
print('mo_comp_split', mo_comp_split)

# %%
df_coeffs = samp['df_coeffs']

orbs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': df_coeffs}, dataset.orbital_basis_num,
                                      samp['atom_numbers'], False, True)

df_coeffs_back = orbitals.coeffs_dict_to_vector(orbs, dataset.orbital_basis_num,
                                                samp['atom_numbers'], False, False, True)['spherical_coeffs']
df_coeffs_noconv = orbitals.coeffs_dict_to_vector(orbs, dataset.orbital_basis_num,
                                                samp['atom_numbers'], False, False, False)['spherical_coeffs']


print('df coeffs', df_coeffs)
print('df coeffs back', df_coeffs_back)
print('df coeffs diff', torch.mean(torch.abs(df_coeffs - df_coeffs_back)))

# %%
from equiv_dens.utils import orbital_conversions

df_coeffs_conv_back = orbital_conversions.convert_ao(df_coeffs_noconv, samp['atom_numbers'], to_internal=False)
df_coeffs_ml = orbital_conversions.convert_ao(df_coeffs, samp['atom_numbers'], to_internal=True)
# print('df coeffs diff', torch.mean(torch.abs(df_coeffs_conv_back - df_coeffs_back)))
print('df_coeffs', df_coeffs[0, :20])
print('df_coeffs conv back', df_coeffs_conv_back[0, :20])
print('df_coeffs noconv', df_coeffs_noconv[0, :20])
print('df coeffs diff', torch.mean(torch.abs(df_coeffs_conv_back - df_coeffs)))
print('df coeffs diff', torch.mean(torch.abs(df_coeffs_noconv - df_coeffs_ml)))

# %%
mo_coeffs = samp['mo_coeff']

orbs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': mo_coeffs}, dataset.calc_basis_num,
                                      samp['atom_numbers'], False, True)

mo_coeffs_back = orbitals.coeffs_dict_to_vector(orbs, dataset.calc_basis_num,
                                                samp['atom_numbers'], False, False, True)['spherical_coeffs']
mo_coeffs_noconv = orbitals.coeffs_dict_to_vector(orbs, dataset.calc_basis_num,
                                                samp['atom_numbers'], False, False, False)['spherical_coeffs']


print('mo coeffs', mo_coeffs)
print('mo coeffs back', mo_coeffs_back)
print('mo coeffs diff', torch.mean(torch.abs(mo_coeffs - mo_coeffs_back)))

# %%
from equiv_dens.utils import orbital_conversions

mo_coeffs_conv_back = orbital_conversions.convert_ao(
    mo_coeffs_noconv, samp['atom_numbers'],
    to_internal=False, convention='pyscf_augccpvdz')
mo_coeffs_ml = orbital_conversions.convert_ao(
    mo_coeffs, samp['atom_numbers'],
    to_internal=True, convention='pyscf_augccpvdz')
# print('df coeffs diff', torch.mean(torch.abs(mo_coeffs_conv_back - mo_coeffs_back)))
# print('mo_coeffs', mo_coeffs[0, :20])
# print('mo_coeffs conv back', mo_coeffs_conv_back[0, :20])
# print('mo_coeffs noconv', mo_coeffs_noconv[0, :20])
print('mo coeffs diff', torch.mean(torch.abs(mo_coeffs_conv_back - mo_coeffs)))
print('mo coeffs diff', torch.mean(torch.abs(mo_coeffs_noconv - mo_coeffs_ml)))

# %%
ham = results[0][1]['hamiltonian_matrix']
mo_coeff = results[0][1]['mo_coeff']
atom = results[0][0]['atom']
basis = results[0][0]['basis']
moe = results[0][1]['mo_energies']
print('basis', basis)
mol = gto.M(atom=atom, basis=basis)
orbital_basis_size = dataset.calc_basis_size
print('orbital_basis_size', orbital_basis_size)
print('mo_coeff size', mo_coeff.shape)
print('hm size', results[0][1]['hamiltonian_matrix'].shape)
s1e = mol.intor('int1e_ovlp_sph')
mf = dft.RKS(mol)
moe_calc, mo_calc1 = mf.eig(ham, s1e)
print('moe calc', moe_calc)
print('moe res', moe)
# %%
mo_coeffs = samp['mo_coeff']
ham = samp['hamiltonian_matrix']
atom2 = [atom[1], atom[2], atom[0]]
mol = gto.M(atom=atom2, basis=basis)
moe = samp['mo_energies']
s1e = mol.intor('int1e_ovlp_sph')
mf = dft.RKS(mol)
moe_calc, mo_calc2 = mf.eig(ham[0], s1e)
print('moe calc', moe_calc)
print('moe res', moe)
# %%
s1e = mol.intor('int1e_ovlp_sph')
ham = orbital_conversions.convert_ao_matrix(
    ham, samp['atom_numbers'],
    to_internal=True, convention='pyscf_augccpvdz',
)

mo_coeffs_conv = orbital_conversions.convert_ao(
        mo_coeffs, samp['atom_numbers'],
        to_internal=True, convention='pyscf_augccpvdz',
)
s1e  = orbital_conversions.convert_ao_matrix(
    torch.from_numpy(s1e).unsqueeze(0), samp['atom_numbers'],
    to_internal=True, convention='pyscf_augccpvdz',
)

moe_eig, mo_coeffs_eig = hf.eig(ham[0], s1e[0])

print('moe eig', moe_eig)
print('moe res', moe)

print('mo_coeffs res', results[0][1]['mo_coeff'][:, 0])
print('mo coeffs calc', mo_calc1[:, 0])
print('mo_coeffs samp', mo_coeffs[0, :, 0])
print('mo_coeffs samp calc', mo_calc2[:, 0])
print('mo coeffs conv', mo_coeffs_conv[0, :, 0])
print('mo_coeffs eig', mo_coeffs_eig[:, 0])

# mo_coeff_split = orbitals.split_df_coeffs(atom, mo_coeff, orbital_basis_size)
# hm_split = orbitals.split_ao_matrix(atom, ham, orbital_basis_size)
# mo_nums, mo_comp = utils.compress_batch_atoms(samp['batch_atom_numbers'].numpy(force=True),
#                                {'mo_coeff': [mo_coeff_split], 'hamiltonian_matrix': [hm_split]},
#                                orbital_basis_size)
#
# print('hm_split', hm_split)
# atom2 = [atom[1], atom[2], atom[0]]
# mo_comp_split = orbitals.split_ao_matrix(atom2, mo_comp['hamiltonian_matrix'][0], orbital_basis_size)
# print()
# print()
# print()
# print()
# print('mo_comp_split', mo_comp_split)
