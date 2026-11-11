# %%
import ase
import numpy as np
import pyscf
import time
import os
from pyscf.scf import hf
from pyscf import gto, df, lib
from pyscf.gto import mole
import scipy

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

# %load_ext autoreload
# %autoreload 2

# %%
thio_1mer_combo = np.load('datasets/thiophene1mer_combo-1000_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_1mer_rand = np.load('datasets/thiophene1mer_rand-1000_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_1mer_rand = np.load('datasets/thiophene1mer_rand-1000_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_2mer_500 = np.load('datasets/thiophene2mer_Bidx-500_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_2mer_500_rest = np.load('datasets/thiophene2mer_bidx-500_rest_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_2mer_rand = np.load('datasets/thiophene2mer_rand-1000_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_3mer_250 = np.load('datasets/thiophene3mer_Cidx-250_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_3mer_250_rest = np.load('datasets/thiophene3mer_cidx-250_rest_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_3mer_500 = np.load('datasets/thiophene3mer_bidx-500_rest_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_3mer_rand_a = np.load('datasets/thiophene3mer_rand-250a_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_3mer_rand_b = np.load('datasets/thiophene3mer_rand-250b_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_3mer_rand_c = np.load('datasets/thiophene3mer_rand-250c_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio_3mer_rand_d = np.load('datasets/thiophene3mer_rand-250d_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
# print('thio_1mer_combo', thio_1mer_combo[1])
# print('thio_1mer_rand', thio_1mer_rand[1])
# print('thio_2mer_500[1]', thio_2mer_500[1])
# print('thio_2mer_500_rest[1]', thio_2mer_500_rest[1])
# print('thio_2mer_rand[1]', thio_2mer_rand[1])
# print('thio_3mer_250[1]', thio_3mer_250[1])
# print('thio_3mer_250_rest[1]', thio_3mer_250_rest[1])
# print('thio_3mer_500[1]', thio_3mer_500[1])
# print('thio_3mer_rand[1]', thio_3mer_rand[1])

print('len thio_1mer_combo', len(thio_1mer_combo))
print('len thio_1mer_rand', len(thio_1mer_rand))
print('len thio_2mer_500[1]', len(thio_2mer_500))
print('len thio_2mer_500_rest[1]', len(thio_2mer_500_rest))
print('len thio_2mer_rand[1]', len(thio_2mer_rand))
print('len thio_3mer_250[1]', len(thio_3mer_250))
print('len thio_3mer_250_rest[1]', len(thio_3mer_250_rest))
print('len thio_3mer_500[1]', len(thio_3mer_500))
print('len thio_3mer_rand_a[1]', len(thio_3mer_rand_a))
# thio1mer_combo = utils.calc_dict_to_npy(thio_1mer_combo, compress_atoms=True)
# thio1mer_rand = utils.calc_dict_to_npy(thio_1mer_rand, compress_atoms=True)
# thio2mer_500 = utils.calc_dict_to_npy(thio_2mer_500, compress_atoms=True)
# thio2mer_500_rest = utils.calc_dict_to_npy(thio_2mer_500_rest, compress_atoms=True)
# thio2mer_rand = utils.calc_dict_to_npy(thio_2mer_rand, compress_atoms=True)
# thio3mer_250 = utils.calc_dict_to_npy(thio_3mer_250, compress_atoms=True)
# thio3mer_250_rest = utils.calc_dict_to_npy(thio_3mer_250_rest, compress_atoms=True)
# thio3mer_500 = utils.calc_dict_to_npy(thio_3mer_500, compress_atoms=True)
# thio3mer_rand = utils.calc_dict_to_npy(thio_3mer_rand, compress_atoms=True)

# %%
thio_12mer_train = thio_1mer_combo + thio_2mer_500 + thio_2mer_500_rest
thio_123mer_train = thio_1mer_combo + thio_2mer_500 + thio_2mer_500_rest + thio_3mer_250 + thio_3mer_250_rest + thio_3mer_500
thio_1mer_test = thio_1mer_rand
thio_2mer_test = thio_2mer_rand
thio_3mer_test = thio_3mer_rand_a + thio_3mer_rand_b + thio_3mer_rand_c + thio_3mer_rand_d
thio_12mer_test = thio_1mer_rand + thio_2mer_rand
thio_123mer_test = thio_1mer_rand + thio_2mer_rand + thio_3mer_test
# %%

thio12_train = utils.calc_dict_to_npy(thio_12mer_train, compress_atoms=True)
thio123_train = utils.calc_dict_to_npy(thio_123mer_train, compress_atoms=True)
thio1_test = utils.calc_dict_to_npy(thio_1mer_test, compress_atoms=True)
thio2_test = utils.calc_dict_to_npy(thio_2mer_test, compress_atoms=True)
thio3_test = utils.calc_dict_to_npy(thio_3mer_test, compress_atoms=True)
thio12_test = utils.calc_dict_to_npy(thio_12mer_test, compress_atoms=True)
thio123_test = utils.calc_dict_to_npy(thio_123mer_test, compress_atoms=True)

print('thio12_train pos', thio12_train['positions'].shape)
print('thio12_train pos', thio12_train['positions'][999:1001])
print('thio12_train energy', thio12_train['energy'][999:1001])
print('thio12_train forces', thio12_train['forces'][999:1001])
print('thio123_train pos', thio123_train['positions'].shape)
print('thio123_test pos', thio123_test['positions'].shape)

# %%
np.save('datasets/thiophene12mer_train.npy', thio12_train)
np.save('datasets/thiophene123mer_train.npy', thio123_train)
np.save('datasets/thiophene1mer_test.npy', thio1_test)
np.save('datasets/thiophene2mer_test.npy', thio2_test)
np.save('datasets/thiophene3mer_test.npy', thio3_test)
np.save('datasets/thiophene12mer_test.npy', thio12_test)
np.save('datasets/thiophene123mer_test.npy', thio123_test)

np.save('datasets/thiophene12mer_train_pyscf_augccpvdz.npy', thio_12mer_train)
np.save('datasets/thiophene123mer_train_pyscf_augccpvdz.npy', thio_123mer_train)
np.save('datasets/thiophene1mer_test_pyscf_augccpvdz.npy', thio_1mer_test)
np.save('datasets/thiophene2mer_test_pyscf_augccpvdz.npy', thio_2mer_test)
np.save('datasets/thiophene3mer_test_pyscf_augccpvdz.npy', thio_3mer_test)
np.save('datasets/thiophene12mer_test_pyscf_augccpvdz.npy', thio_12mer_test)
np.save('datasets/thiophene123mer_test_pyscf_augccpvdz.npy', thio_123mer_test)

# %%
# extract thiophene 4 mer geometries from training dataset
thio4mer = os.listdir("datasets")
thio4mer = [x for x in thio4mer if 'thiophene4mer' in x and 'test' not in x]

thio4mer_test = sorted([x for x in thio4mer if 'rand' in x])
thio4mer_train = sorted([x for x in thio4mer if 'rand' not in x])
print('thio4mer_test', thio4mer_test)
print('thio4mer_train', thio4mer_train)
print('thio4mer_test len', len(thio4mer_test))
print('thio4mer_train len', len(thio4mer_train))
print('thio4mer_train len', len(thio4mer_train))

# %%
thio_4mer_train = []
for i in range(len(thio4mer_train)):
    thio_4mer_train += np.load('datasets/' + thio4mer_train[i], allow_pickle=True).tolist()

print(len(thio_4mer_train))

thio4_train = utils.calc_dict_to_npy(thio_4mer_train, compress_atoms=True)

np.save('datasets/thiophene4mer_train_pyscf_augccpvdz.npy', thio_4mer_train)
np.save('datasets/thiophene4mer_train.npy', thio4_train)
# %%
thio_4mer_test = []
for i in range(len(thio4mer_test)):
    thio_4mer_test += np.load('datasets/' + thio4mer_test[i], allow_pickle=True).tolist()

print(len(thio_4mer_test))

thio4_test = utils.calc_dict_to_npy(thio_4mer_test, compress_atoms=True)

np.save('datasets/thiophene4mer_test_pyscf_augccpvdz.npy', thio_4mer_test)
np.save('datasets/thiophene4mer_test.npy', thio4_test)
# %%
# extract thiophene 6 mer geometries from training dataset
thio6mer = os.listdir("datasets")
thio6mer = [x for x in thio6mer if 'thiophene6mer' in x and 'test' not in x and 'npy' in x]

thio6mer_test = sorted([x for x in thio6mer if 'r1000' in x])
thio6mer_train = sorted([x for x in thio6mer if 'r1000' not in x])
print('thio6mer_test', thio6mer_test)
print('thio6mer_train', thio6mer_train)
print('thio6mer_test len', len(thio6mer_test))
print('thio6mer_train len', len(thio6mer_train))

# %%
thio_6mer_train = []
for i in range(len(thio6mer_train)):
    thio_6mer_train += np.load('datasets/' + thio6mer_train[i], allow_pickle=True).tolist()

print(len(thio_6mer_train))

thio6_train = utils.calc_dict_to_npy(thio_6mer_train, compress_atoms=True)

np.save('datasets/thiophene6mer_train_pyscf_augccpvdz.npy', thio_6mer_train)
np.save('datasets/thiophene6mer_train.npy', thio6_train)
# %%
thio_6mer_test = []
for i in range(len(thio6mer_test)):
    thio_6mer_test += np.load('datasets/' + thio6mer_test[i], allow_pickle=True).tolist()

print(len(thio_6mer_test))

thio6_test = utils.calc_dict_to_npy(thio_6mer_test, compress_atoms=True)

np.save('datasets/thiophene6mer_test_pyscf_augccpvdz.npy', thio_6mer_test)
np.save('datasets/thiophene6mer_test.npy', thio6_test)

# %%
thio_123mer_train = np.load('datasets/thiophene123mer_train_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio4mer_train = np.load('datasets/thiophene4mer_train_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio6mer_train = np.load('datasets/thiophene6mer_train_pyscf_augccpvdz.npy', allow_pickle=True).tolist()

thio_train_all = thio_123mer_train + thio4mer_train + thio6mer_train

thio_train_all_npy = utils.calc_dict_to_npy(thio_train_all, compress_atoms=True)
print(len(thio_train_all))
print(thio_train_all_npy['positions'].shape)

np.save('datasets/thiophene_all_train_pyscf_augccpvdz.npy', thio_train_all)
np.save('datasets/thiophene_all_train.npy', thio_train_all_npy)

# %%
print(len(thio_123mer_train[0][0]['atom']))
print(len(thio_123mer_train[1000][0]['atom']))
print(len(thio_123mer_train[2000][0]['atom']))
print(len(thio4mer_train[0][0]['atom']))
print(len(thio6mer_train[0][0]['atom']))

# %%
thio_123mer_test = np.load('datasets/thiophene123mer_test_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio4mer_test = np.load('datasets/thiophene4mer_test_pyscf_augccpvdz.npy', allow_pickle=True).tolist()
thio6mer_test = np.load('datasets/thiophene6mer_test_pyscf_augccpvdz.npy', allow_pickle=True).tolist()

thio_test_all = thio_123mer_test + thio4mer_test + thio6mer_test
thio_test_all_npy = utils.calc_dict_to_npy(thio_test_all, compress_atoms=True)

print(len(thio_test_all))
print(thio_test_all_npy['positions'].shape)

np.save('datasets/thiophene_all_test_pyscf_augccpvdz.npy', thio_test_all)
np.save('datasets/thiophene_all_test.npy', thio_test_all_npy)
# %%
print(np.sum(thio_test_all_npy['atom_numbers'], axis=1))
