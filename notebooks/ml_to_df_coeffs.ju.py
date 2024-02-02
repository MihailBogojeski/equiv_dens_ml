# %%
import ase
import numpy as np
import pyscf
import time
import os
from pyscf.scf import hf
from pyscf import gto, df, lib, dft
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
from equiv_dens.utils import orbitals

from functools import partial

from pyscf.gto.mole import nao_nr
hf.MUTE_CHKFILE = True

# %load_ext autoreload
# %autoreload 2
# %cd /home/mihail/Documents/workspace/equiv_dens
# %%

mol = gto.M(atom='H 0 0 0; H 0 0 3; C 0 0 1; O 0 0 2', basis='ccpvdz')
print('nprim', mol.bas_nprim(0))
print('nctr', mol.bas_nctr(0))
print('exp', mol.bas_exp(0))
print('kappa', mol.bas_kappa(0))
print('ctr_coeff', mol.bas_ctr_coeff(0))
print('libcint_ctr_coeff', mol._libcint_ctr_coeff(0))
print('env', mol._env)
print('bas', mol._bas)
print('atm', mol._atm)
print('basis', mol._basis)
print('nao_nr', nao_nr(mol))
mf = dft.RKS(mol)
mf.chkfile=False
mf.xc = 'pbe'
mf.kernel()
print('env shape', mol._env.shape)

# %%
for a_row in mol._atm:
    print(mol._env[a_row[1]:a_row[3]])
# %%

auxmol = gto.M(atom='H 0 0 0; H 0 0 3; C 0 0 1; O 0 0 2', basis='augccpvqzjkfit')
print('36 apart', auxmol._env[60], auxmol._env[96])
auxmol._env[60] += 0.01
print('nprim', auxmol.bas_nprim(0))
print('nctr', auxmol.bas_nctr(0))
print('exp', auxmol.bas_exp(0))
print('kappa', auxmol.bas_kappa(0))
print('ctr_coeff', auxmol.bas_ctr_coeff(0))
print('libcint_ctr_coeff', auxmol._libcint_ctr_coeff(0))
print('env', auxmol._env)
print('bas', auxmol._bas)
print('atm', auxmol._atm)
print('basis', auxmol._basis)
print('nao_nr', nao_nr(auxmol))
print('env shape', auxmol._env.shape)
# %%
dm1 = hf.make_rdm1(mf.mo_coeff, mf.mo_occ)
print('density matrix', dm1.shape)

# Define the auxiliary fitting basis for 3-center integrals. Use the function
# make_auxmol to construct the auxiliary Mole object (auxmol) which will be
# used to generate integrals.

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

print('df basis regular', df_basis)
# %%
auxmol2 = gto.M(atom='H 0 0 0; H 0 0 3; C 0 0 1; O 0 0 2', basis='augccpvqzjkfit')
auxmol2._env = np.concatenate([auxmol2._env[:36], auxmol2._env[36:72], auxmol2._env[36:]], axis=0)
auxmol2._bas[auxmol2._bas[:, 0] > 0, 5:7] += 36
print('36 apart', auxmol2._env[60], auxmol2._env[96])
auxmol2._env[60] += 0.01
auxmol2._env[96] += 0.01
print('nprim', auxmol2.bas_nprim(0))
print('nctr', auxmol2.bas_nctr(0))
print('exp', auxmol2.bas_exp(0))
print('kappa', auxmol2.bas_kappa(0))
print('ctr_coeff', auxmol2.bas_ctr_coeff(0))
print('libcint_ctr_coeff', auxmol2._libcint_ctr_coeff(0))
print('env', auxmol2._env)
print('bas', auxmol2._bas)
print('atm', auxmol2._atm)
print('basis', auxmol2._basis)
print('nao_nr', nao_nr(auxmol2))
print('env shape', auxmol2._env.shape)
# %%
dm1 = hf.make_rdm1(mf.mo_coeff, mf.mo_occ)
print('density matrix', dm1.shape)

# Define the auxiliary fitting basis for 3-center integrals. Use the function
# make_auxmol to construct the auxiliary Mole object (auxmol) which will be
# used to generate integrals.

# ints_3c is the 3-center integral tensor (ij|P), where i and j are the
# indices of AO basis and P is the auxiliary basis
ints_3c2e = df.incore.aux_e2(mol, auxmol2, intor='int3c2e')
ints_2c2e = auxmol2.intor('int2c2e')

nao = mol.nao
naux = auxmol2.nao

# Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
df_coef = df_coef.reshape(naux, nao, nao)
print('df_coeff shape', df_coef.shape)
df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

print('df basis experiment', df_basis)
# %%
args, hyperparam_args = parse_command_line_arguments(arg_file='args/ethanethiol_all_006_test.txt')

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


dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density', 'dipole_moment', 'df_coeffs'],
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
basis = 'augccpvqzjkfit'
grid_fn = partial(spherical_grid, level=1)
grid_origin = 0
grid_extent = None
dataset_df = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                              orbitals_path=args.orbitals_file,
                              density_n_samp=10000000000,
                              required_properties=['density', 'dipole_moment', 'df_coeffs'],
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
for i in range(3):
    mol = dataset.get_properties([i])
    mol_df = dataset_df.get_properties([i])

    df_error = torch.mean(torch.abs(mol['density'] - mol_df['density']) * mol['coord_weights']) / torch.mean(mol['density'] * mol['coord_weights'])
    dpm_error = torch.norm(mol['dipole_moment'] - mol_df['dipole_moment'])
    print(i, 'dens_error:', df_error, 'dpm_error:', dpm_error)
    df_errors.append(df_error)

print('average_density error', np.mean(df_errors))
# %%
# Testing density fitting with standard basis
idx = 2
samp = dataset.get_properties([idx])
atom = [(int(samp['batch_atom_numbers'][0, i].detach().cpu().numpy()),
        samp['batch_positions'][0, i].detach().cpu().numpy()) for i in range(samp['batch_positions'].shape[1])]
auxmol = gto.M(atom=atom, basis=dataset.density_fitting['auxbasis'])
auxmol.build()

mol = gto.M(atom=atom, basis=dataset.mols[0].pack()['basis'])
mol.build()
mf = dft.RKS(mol)
mf.chkfile = False
mf.xc = 'pbe'
mf.kernel()

dm1 = hf.make_rdm1(mf.mo_coeff, mf.mo_occ)
print('density matrix', dm1.shape)

# Define the auxiliary fitting basis for 3-center integrals. Use the function
# make_auxmol to construct the auxiliary Mole object (auxmol) which will be
# used to generate integrals.

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
print('atoms', auxmol._atm)
df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

print('df basis experiment', df_basis)

df_coeffs_split = orbitals.split_df_coeffs(mol.pack()['atom'], df_basis, dataset.orbital_basis_size)
with np.printoptions(threshold=np.inf):
    print(auxmol._bas)
print('split coeffs', df_coeffs_split)
# %%
dens_aux = orbitals.sample_projected_density(samp, torch.tensor(df_basis).unsqueeze(0), dataset_df)
dens_coeffs = orbitals.sample_projected_density(samp, samp['df_coeffs'], dataset_df)
print('sum true density', torch.sum(samp['density'] * samp['coord_weights']))
print('sum projected density', torch.sum(dens_aux * samp['coord_weights']))
print('sum projected density', torch.sum(dens_coeffs * samp['coord_weights']))
print('projected density error', torch.sum(torch.abs(dens_aux - samp['density']) * samp['coord_weights']) / torch.sum(samp['atom_numbers']))

print('samp df basis', samp['df_coeffs'])

# %%
# Testing density fitting with modified basis
auxmol2 = gto.M(atom=atom, basis=dataset.density_fitting['auxbasis'])
auxmol2.build()

# %%
# with np.printoptions(threshold=np.inf):
#     print(auxmol2._atm)
#     print(auxmol2._bas)
#     print(auxmol2._env)
# Gathering information from basis variable
print(auxmol2._env.shape)
atom_bas = []
order = auxmol2._bas[0, 0]
start_bas = 0
start_env = auxmol2._bas[0, 5]
atom_count = {}
for i in range(auxmol2._bas.shape[0]):
    row = auxmol2._bas[i]
    if row[0] != order:
        order = row[0]
        atom_bas.append([(start_bas, end_bas), (start_env, end_env)])
        if start_env not in atom_count:
            atom_count[start_env] = 0
        start_env = row[5]
        start_bas = i
    end_bas = i
    end_env = row[6]

atom_bas.append([(start_bas, end_bas), (start_env, end_env)])
if start_env not in atom_count:
    atom_count[start_env] = 0

print(len(atom_bas))

print('atom_bas', atom_bas)
print('atom_count', atom_count)

# Extending _env variable for duplicate atoms

auxmol3 = gto.M(atom=atom, basis=dataset.density_fitting['auxbasis'])
auxmol3.build()
unseen_idx = {key: np.ones(auxmol3._bas.shape[0], dtype=bool) for key in atom_count}
offset = 0
for ab in atom_bas:
    start_bas, end_bas = ab[0]
    start_env, end_env = ab[1]
    print('start_bas', start_bas, ' end_bas', end_bas, ' start_env', start_env, ' end_env', end_env)
    print('offset start env', auxmol3._bas[start_bas, 5], 'offset end env', auxmol3._bas[end_bas, 6])
    if atom_count[start_env] != 0:
        print('new atom')
        offset_start = auxmol3._bas[start_bas, 5]
        print('offset start', offset_start)
        offset = end_env - start_env + 1
        print('offset', offset)
        offset_idx = auxmol3._bas[:, 5] >= offset_start
        offset_idx = np.logical_and(offset_idx, unseen_idx[start_env])

        auxmol3._env = np.concatenate([auxmol3._env[:auxmol3._bas[start_bas, 5] + offset],
                                       auxmol2._env[start_env:end_env + 1],
                                       auxmol3._env[auxmol3._bas[end_bas, 6] + 1:]], axis=0)
        auxmol3._bas[offset_idx, 5:7] += offset
        # auxmol3._bas[start_bas:end_bas + 1, 5:7] += offset
        print('offset start env after', auxmol3._bas[start_bas, 5], 'offset end env after', auxmol3._bas[end_bas, 6])
        print('new env shape', auxmol3._env.shape)
    atom_count[start_env] += 1
    unseen_idx[start_env][start_bas:end_bas+1] = False

print('old env shape', auxmol2._env.shape)
print('new env shape', auxmol3._env.shape)
print(atom_bas)
with np.printoptions(threshold=np.inf):
    print('combined old new bas', np.concatenate([auxmol2._bas, auxmol3._bas], axis=1))

# %%
# Define the auxiliary fitting basis for 3-center integrals. Use the function
# make_auxmol to construct the auxiliary Mole object (auxmol) which will be
# used to generate integrals.

# ints_3c is the 3-center integral tensor (ij|P), where i and j are the
# indices of AO basis and P is the auxiliary basis
# min_bas = np.min(auxmol3._bas[:, 5])
# print('auxmol env before', auxmol3._env)
# auxmol3._env[min_bas:] += np.random.normal(size=auxmol3._env[min_bas:].shape, scale=0.01)
# print('auxmol env after', auxmol3._env)
ints_3c2e = df.incore.aux_e2(mol, auxmol3, intor='int3c2e')
ints_2c2e = auxmol3.intor('int2c2e')

nao = mol.nao
naux = auxmol3.nao

# Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
df_coef = df_coef.reshape(naux, nao, nao)
print('df_coeff shape', df_coef.shape)
df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

print('df basis experiment', df_basis)

# %%
dens = orbitals.sample_projected_density(samp, torch.tensor(df_basis).unsqueeze(0), dataset_df, auxmol=auxmol3)

print('sum true density', torch.sum(samp['density'] * samp['coord_weights']))
print('sum projected density', torch.sum(dens * samp['coord_weights']))
print('projected density error', torch.sum(torch.abs(dens - samp['density']) * samp['coord_weights']) / torch.sum(samp['atom_numbers']))
samp2 = dataset.get_properties([idx])
dpm = orbitals.calc_dipole_moment(samp2, density=dens)['dipole_moment']
print('dipole moment error', 4.8 * torch.norm(dpm - samp['dipole_moment']))
# %%
model = load_model(args, dataset)
samp2 = dataset.get_properties([idx])

res = model(samp2)

print('res dens integral', torch.sum(res['density'] * samp2['coord_weights']))
print('res dens error', torch.sum(torch.abs(res['density'] - samp2['density']) *
                                  samp2['coord_weights']) / torch.sum(samp2['atom_numbers']))

print('res radial scale', [res['radial_scale'][i][list(res['radial_scale'][i].keys())[0]] for i in range(len(res['radial_scale']))])
print('res spherical L0', [res['spherical_coeffs'][i][list(res['spherical_coeffs'][i].keys())[0]] for i in range(len(res['spherical_coeffs']))])
# %%
for ab in atom_bas:
    start_bas, end_bas = ab[0]
    start_env, end_env = ab[1]
    print('start_bas', start_bas, ' end_bas', end_bas)
    print('offset start env', auxmol3._bas[start_bas, 5], 'offset end env', auxmol3._bas[end_bas, 6])
    print('L', auxmol3._bas[start_bas, 1])

old_env = auxmol3._env.copy()
# print('auxmol3 env', auxmol3._env)
atom_bas = []
order = auxmol3._bas[0, 0]
start_bas = 0
start_env = auxmol3._bas[0, 5]
atom_count = {}
for i in range(auxmol3._bas.shape[0]):
    row = auxmol3._bas[i]
    if row[0] != order:
        order = row[0]
        atom_bas.append([(start_bas, end_bas), (start_env, end_env)])
        if start_env not in atom_count:
            atom_count[start_env] = 0
        start_env = row[5]
        start_bas = i
    end_bas = i
    end_env = row[6]

atom_bas.append([(start_bas, end_bas), (start_env, end_env)])
if start_env not in atom_count:
    atom_count[start_env] = 0
print(atom_bas)
print(auxmol3._atm)

for i in range(len(res['radial_width'])):
    radial_widths = None
    radial_scales = None
    for key in res['radial_width'][i].keys():
        if radial_widths is None:
            radial_widths = res['radial_width'][i][key].squeeze()
            radial_scales = res['radial_scale'][i][key].squeeze()
        else:
            radial_widths = torch.cat([radial_widths, res['radial_width'][i][key].squeeze()])
            radial_scales = torch.cat([radial_scales, res['radial_scale'][i][key].squeeze()])
    radial_coeffs = torch.stack([radial_widths, radial_scales], dim=1)
    radial_coeffs = torch.abs(radial_coeffs.flatten())
    # print('radial_coeffs', radial_coeffs)
    # print('auxmol env old', auxmol3._env[atom_bas[i][1][0]:atom_bas[i][1][1] + 1])
    auxmol3._env[atom_bas[i][1][0]:atom_bas[i][1][1] + 1] = radial_coeffs.detach().cpu().numpy()
    # print('auxmol env new', auxmol3._env[atom_bas[i][1][0]:atom_bas[i][1][1] + 1])

# print('res radial width', res['radial_width'])
with np.printoptions(threshold=np.inf):
    print('auxmols env', np.stack([auxmol3._env, old_env], axis=1))

# %%
# Define the auxiliary fitting basis for 3-center integrals. Use the function
# make_auxmol to construct the auxiliary Mole object (auxmol) which will be
# used to generate integrals.

# ints_3c is the 3-center integral tensor (ij|P), where i and j are the
# indices of AO basis and P is the auxiliary basis
ints_3c2e = df.incore.aux_e2(mol, auxmol3, intor='int3c2e')
ints_2c2e = auxmol3.intor('int2c2e')

nao = mol.nao
naux = auxmol3.nao

# Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
df_coef = df_coef.reshape(naux, nao, nao)
print('df_coeff shape', df_coef.shape)
df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

print('df basis experiment', df_basis)

df_coeffs_split = orbitals.split_df_coeffs(mol.pack()['atom'], df_basis, dataset.orbital_basis_size)
with np.printoptions(threshold=np.inf):
    print(auxmol._bas)
print('split coeffs', df_coeffs_split)

# %%
dens = orbitals.sample_projected_density(samp, torch.tensor(df_basis).unsqueeze(0),
                                         dataset_df, auxmol=auxmol3)
print(samp['atom_numbers'])

print('sum projected density', torch.sum(samp['density'] * samp['coord_weights']))
print('sum true density', torch.sum(dens * samp['coord_weights']))
print('projected density error', torch.sum(torch.abs(dens - samp['density']) * samp['coord_weights']) / torch.sum(samp['atom_numbers']))
samp2 = dataset.get_properties([idx])
dpm = orbitals.calc_dipole_moment(samp2, density=dens)['dipole_moment']
print('dipole moment error', 4.8 * torch.norm(dpm - samp['dipole_moment']))
# %%
def atom_basis_descriptors(auxmol):
    atom_bas = []
    order = auxmol._bas[0, 0]
    start_bas = 0
    start_env = auxmol._bas[0, 5]
    atom_count = {}
    for i in range(auxmol._bas.shape[0]):
        row = auxmol._bas[i]
        if row[0] != order:
            order = row[0]
            atom_bas.append([(start_bas, end_bas), (start_env, end_env)])
            if start_env not in atom_count:
                atom_count[start_env] = 0
            start_env = row[5]
            start_bas = i
        end_bas = i
        end_env = row[6]

    atom_bas.append([(start_bas, end_bas), (start_env, end_env)])
    if start_env not in atom_count:
        atom_count[start_env] = 0

    print(len(atom_bas))

    # print('atom_bas', atom_bas)
    # print('atom_count', atom_count)
    return atom_bas, atom_count


def extend_aux_environment(auxmol, atom_bas, atom_count):
    auxmol_ext = auxmol.copy()
    auxmol_ext.build()
    unseen_idx = {key: np.ones(auxmol_ext._bas.shape[0], dtype=bool) for key in atom_count}
    offset = 0
    for ab in atom_bas:
        start_bas, end_bas = ab[0]
        start_env, end_env = ab[1]
        # print('start_bas', start_bas, ' end_bas', end_bas, ' start_env', start_env, ' end_env', end_env)
        # print('offset start env', auxmol_ext._bas[start_bas, 5], 'offset end env', auxmol_ext._bas[end_bas, 6])
        if atom_count[start_env] != 0:
            # print('new atom')
            offset_start = auxmol_ext._bas[start_bas, 5]
            # print('offset start', offset_start)
            offset = end_env - start_env + 1
            # print('offset', offset)
            offset_idx = auxmol_ext._bas[:, 5] >= offset_start
            offset_idx = np.logical_and(offset_idx, unseen_idx[start_env])

            auxmol_ext._env = np.concatenate([auxmol_ext._env[:auxmol_ext._bas[start_bas, 5] + offset],
                                           auxmol._env[start_env:end_env + 1],
                                           auxmol_ext._env[auxmol_ext._bas[end_bas, 6] + 1:]], axis=0)
            auxmol_ext._bas[offset_idx, 5:7] += offset
            # auxmol_ext._bas[start_bas:end_bas + 1, 5:7] += offset
            # print('offset start env after', auxmol_ext._bas[start_bas, 5], 'offset end env after', auxmol_ext._bas[end_bas, 6])
            # print('new env shape', auxmol_ext._env.shape)
        atom_count[start_env] += 1
        unseen_idx[start_env][start_bas:end_bas+1] = False

    # print('old env shape', auxmol._env.shape)
    # print('new env shape', auxmol_ext._env.shape)
    # print(atom_bas)
    # with np.printoptions(threshold=np.inf):
    #     print('combined old new bas', np.concatenate([auxmol._bas, auxmol_ext._bas], axis=1))

    return auxmol_ext


def ml_basis_to_pyscf_env(pred, auxmol):
    atom_bas, atom_count = atom_basis_descriptors(auxmol)
# Extending _env variable for duplicate atoms
    auxmol_ext = extend_aux_environment(auxmol, atom_bas, atom_count)

    for ab in atom_bas:
        start_bas, end_bas = ab[0]
        start_env, end_env = ab[1]
        # print('start_bas', start_bas, ' end_bas', end_bas)
        # print('offset start env', auxmol_ext._bas[start_bas, 5], 'offset end env', auxmol_ext._bas[end_bas, 6])
        # print('L', auxmol_ext._bas[start_bas, 1])

    # old_env = auxmol_ext._env.copy()
    # print('auxmol_ext env', auxmol_ext._env)
    atom_bas, atom_count = atom_basis_descriptors(auxmol_ext)

    for i in range(len(res['radial_width'])):
        radial_widths = None
        radial_scales = None
        for key in res['radial_width'][i].keys():
            if radial_widths is None:
                radial_widths = res['radial_width'][i][key].squeeze()
                radial_scales = res['radial_scale'][i][key].squeeze()
            else:
                radial_widths = torch.cat([radial_widths, res['radial_width'][i][key].squeeze()])
                radial_scales = torch.cat([radial_scales, res['radial_scale'][i][key].squeeze()])
        radial_coeffs = torch.stack([radial_widths, radial_scales], dim=1)
        radial_coeffs = torch.abs(radial_coeffs.flatten())
        # print('radial_coeffs', radial_coeffs)
        # print('auxmol env old', auxmol_ext._env[atom_bas[i][1][0]:atom_bas[i][1][1] + 1])
        auxmol_ext._env[atom_bas[i][1][0]:atom_bas[i][1][1] + 1] = radial_coeffs.detach().cpu().numpy()
        # print('auxmol env new', auxmol_ext._env[atom_bas[i][1][0]:atom_bas[i][1][1] + 1])

# print('res radial width', res['radial_width'])
    # with np.printoptions(threshold=np.inf):
    #     print('auxmols env', np.stack([auxmol_ext._env, old_env], axis=1))

    return auxmol_ext


def ml_basis_to_df_coeffs(pred, basis, auxbasis):
    atom = [(int(pred['batch_atom_numbers'][0, i].detach().cpu().numpy()),
            pred['batch_positions'][0, i].detach().cpu().numpy()) for i in range(pred['batch_positions'].shape[1])]
    auxmol = gto.M(atom=atom, basis=auxbasis)
    auxmol.build()

    mol = gto.M(atom=atom, basis=basis)
    mol.build()
    mf = dft.RKS(mol)
    mf.chkfile = False
    mf.xc = 'pbe'
    mf.kernel()
    dm1 = hf.make_rdm1(mf.mo_coeff, mf.mo_occ)

    auxmol_ext = ml_basis_to_pyscf_env(pred, auxmol)

    # Define the auxiliary fitting basis for 3-center integrals. Use the function
    # make_auxmol to construct the auxiliary Mole object (auxmol) which will be
    # used to generate integrals.

    # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
    # indices of AO basis and P is the auxiliary basis
    ints_3c2e = df.incore.aux_e2(mol, auxmol_ext, intor='int3c2e')
    ints_2c2e = auxmol_ext.intor('int2c2e')

    nao = mol.nao
    naux = auxmol_ext.nao

# Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
    df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
    df_coef = df_coef.reshape(naux, nao, nao)
    # print('df_coeff shape', df_coef.shape)
    # print('atoms', auxmol_ext._atm)
    df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

    return df_basis, auxmol_ext

# %%
idx = 0
model = load_model(args, dataset)
samp2 = dataset.get_properties([idx])
samp = dataset.get_properties([idx])

res = model(samp2)

print('res dens integral', torch.sum(res['density'] * samp2['coord_weights']))
print('res dens error', torch.sum(torch.abs(res['density'] - samp['density']) *
                                  samp2['coord_weights']) / torch.sum(samp2['atom_numbers']))

res = orbitals.calc_dipole_moment(res)
# print('res radial scale', [res['radial_scale'][i][list(res['radial_scale'][i].keys())[0]] for i in range(len(res['radial_scale']))])
# print('res spherical L0', [res['spherical_coeffs'][i][list(res['spherical_coeffs'][i].keys())[0]] for i in range(len(res['spherical_coeffs']))])
print('res dipole moment error', 4.8 * torch.norm(res['dipole_moment'] - samp2['dipole_moment']))
# %%

basis = 'augccpvdz'
auxbasis = 'augccpvqzjkfit'
ml_df_coeffs, auxmol_ext = ml_basis_to_df_coeffs(res, basis, auxbasis)
dens = orbitals.sample_projected_density(samp, torch.tensor(ml_df_coeffs).unsqueeze(0),
                                         dataset_df, auxmol=auxmol_ext)
print(samp['atom_numbers'])

print('sum projected density', torch.sum(samp['density'] * samp['coord_weights']))
print('sum true density', torch.sum(dens * samp['coord_weights']))
print('projected density error', torch.sum(torch.abs(dens - samp['density']) * samp['coord_weights']) / torch.sum(samp['atom_numbers']))
samp2 = dataset.get_properties([idx])
dpm = orbitals.calc_dipole_moment(samp2, density=dens)['dipole_moment']
print('dipole moment error', 4.8 * torch.norm(dpm - samp['dipole_moment']))
