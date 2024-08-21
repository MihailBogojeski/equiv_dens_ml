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

#!/usr/bin/env python3
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.errors import ErrorDict
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.utils.grids import cubical_grid, cubical_sampling, \
    spherical_grid, spherical_radial_sampling
from equiv_dens.training.model_loader import load_model
import equiv_dens.utils.base as utils

from functools import partial

# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..

# %%
data = np.load('datasets/ethanethiol_pyscf_augccpvdz_dft_test.npy', allow_pickle=True)
basis_sets = ['augccpvqzjkfit', 'augccpvdzjkfit', 'augccpvqz', 'def2-universal-jfit']
calc_sets = {basis: [] for basis in basis_sets}
for i in range(3):
    print(i)
    calc = data[i][1]
    atoms = data[i][0]
    mol = mole.unpack(data[i][0])
    mol.build()
    print(calc.keys())
    mo_coeff = calc['mo_coeff']
    mo_occ = calc['mo_occ']
    dm1 = hf.make_rdm1(mo_coeff, mo_occ)
    print('density matrix', dm1.shape)

    # Define the auxiliary fitting basis for 3-center integrals. Use the function
    # make_auxmol to construct the auxiliary Mole object (auxmol) which will be
    # used to generate integrals.
    for basis in basis_sets:
        copy_calc = {key: calc[key] for key in calc}
        auxbasis = basis
        auxmol = df.addons.make_auxmol(mol, auxbasis)

        # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
        # indices of AO basis and P is the auxiliary basis
        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
        ints_2c2e = auxmol.intor('int2c2e')

        nao = mol.nao
        naux = auxmol.nao

        # Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
        df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        print('df_coeff shape', df_coef.shape)
        df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

        copy_calc['df_coeff'] = df_basis
        copy_calc['auxbasis'] = auxbasis
        calc_sets[basis].append((atoms, copy_calc))
    # np.save('datasets/ethanethiol_pyscf_augccpvdz_dft_test_df_augccpvqzjkfit.npy', data, allow_pickle=True)

npy_dat = utils.calc_dict_to_npy(data[:3], convert_forces=False, compress_atoms=False)
np.save('datasets/ethanethiol_dft_test_small.npy', npy_dat, allow_pickle=True)
for basis in basis_sets:
    np.save('datasets/ethanethiol_pyscf_augccpvdz_dft_test_df_' + basis + '_small.npy', calc_sets[basis], allow_pickle=True)
# %%
args, hyperparam_args = parse_command_line_arguments(arg_file='args/ethanethiol_dens_001_mae.txt')

print('type dtype', type(args.dtype))
args.fix_arguments = True
print('args np dir', args.np_dataset)
# no restart directory specified
if args.restart is None:
    # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    # create directories
    checkpoint = None
    latest_checkpoint = 0
    step = 0
    restore = False
    data_split_indices = None
    # restarts run from latest checkpoint
else:
    directory = args.restart  # load directory name
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
                print('loading hyperparam arg', arg)
                setattr(args, arg, getattr(checkpoint['args'], arg))
        else:
            print('loading all arg', arg)
            setattr(args, arg, getattr(checkpoint['args'], arg))
    restore = True

args.use_gpu = False
# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")
args.verbose = 0
args.use_gpu = False
args.radii_adjust = True
if args.cube_grid:
    grid_origin = args.cube_origin
    grid_extent = np.array([args.cube_extent] * 3)
    grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                      extent=grid_extent,
                      origin=np.array([grid_origin] * 3))
    sampling_fn = cubical_sampling
else:
    grid_fn = partial(spherical_grid, level=1)
    sampling_fn = partial(spherical_radial_sampling, rotate=False)
    grid_origin = 0
    grid_extent = None

args.np_dataset_test = 'datasets/ethanethiol_dft_test_small.npy'

dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density', 'dipole_moment'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           pyscf_grid=True,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=args.verbose,
                           radii_adjust=args.radii_adjust)


# %%
for basis in basis_sets:
    print('basis_set = ', basis)
    args.dens_dataset_test = 'datasets/ethanethiol_pyscf_augccpvdz_dft_test_df_' + basis + '_small.npy'
    grid_fn = partial(spherical_grid, level=1)
    grid_origin = 0
    grid_extent = None
    dataset_df = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                                  orbitals_path=args.orbitals_file,
                                  density_n_samp=10000000000,
                                  required_properties=['density', 'dipole_moment'],
                                  center_positions=False,
                                  radial_coeffs_file=args.radial_coeffs_file,
                                  dtype=args.dtype,
                                  grid_fn=grid_fn,
                                  pyscf_grid=True,
                                  grid_extent=grid_extent,
                                  grid_origin=grid_origin,
                                  verbose=args.verbose,
                                  radii_adjust=args.radii_adjust,
                                  projected_density=True)
    df_errors = []
    for i in range(len(dataset)):
        mol = dataset.get_properties([i])
        mol_df = dataset_df.get_properties([i])

        df_error = torch.mean(torch.abs(mol['density'] - mol_df['density']) * mol['coord_weights']) / torch.mean(mol['density'] * mol['coord_weights'])
        dpm_error = torch.norm(mol['dipole_moment'] - mol_df['dipole_moment'])
        print(i, 'dens_error:', df_error, 'dpm_error:', dpm_error)
        df_errors.append(df_error)

    print('average_density error', np.mean(df_errors))

# %%
# augccpvdzfit
df_errors = []
for i in range(len(dataset)):
    mol = dataset.get_properties([i])
    mol_df = dataset_df.get_properties([i])
    
    df_error = torch.mean(torch.abs(mol['density'] - mol_df['density']) * mol['coord_weights']) / torch.mean(mol['density'] * mol['coord_weights'])
    print('df_error', i, ':', df_error)
    df_errors.append(df_error)
    
print('average_density error', np.mean(df_errors))

# %%
# augccpvqzfit
df_errors = []
for i in range(len(dataset)):
    mol = dataset.get_properties([i])
    mol_df = dataset_df.get_properties([i])
    
    df_error = torch.mean(torch.abs(mol['density'] - mol_df['density']) * mol['coord_weights']) / torch.mean(mol['density'] * mol['coord_weights'])
    print('df_error', i, ':', df_error)
    df_errors.append(df_error)
    
print('average_density error', np.mean(df_errors))

# %%
# def2svpjkfit
df_errors = []
for i in range(len(dataset)):
    mol = dataset.get_properties([i])
    mol_df = dataset_df.get_properties([i])
    
    df_error = torch.mean(torch.abs(mol['density'] - mol_df['density']) * mol['coord_weights']) / torch.mean(mol['density'] * mol['coord_weights'])
    print('df_error', i, ':', df_error)
    df_errors.append(df_error)
    
print('average_density error', np.mean(df_errors))

# %%

# ccpvdzjkfit
df_errors = []
for i in range(len(dataset)):
    mol = dataset.get_properties([i])
    mol_df = dataset_df.get_properties([i])
    
    df_error = torch.mean(torch.abs(mol['density'] - mol_df['density']) * mol['coord_weights']) / torch.mean(mol['density'] * mol['coord_weights'])
    print('df_error', i, ':', df_error)
    df_errors.append(df_error)
    
print('average_density error', np.mean(df_errors))

# %%


