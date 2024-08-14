# %%
import ase
import numpy as np
import pyscf
import time
import os
from pyscf import gto, dft, df, lib
from pyscf.scf import hf
import equiv_dens.utils.base as utils
import scipy
hf.MUTE_CHKFILE = True

# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..

# %%
inds = np.arange(100000)
# %%
# load data
data = np.load('md_logs/2022-12-23_pJuMyc8e/simulation_-ethanethiol_cluster_all_001_compressed_0.npy', allow_pickle=True).item()
data['positions'] = data['positions'][::1000]
print(data['positions'].shape)
# %%
atom_pos = data['positions']
atom_types = data['atom_numbers'].squeeze()
save_path = 'datasets/ethanethiol_md_traj_every1000_dft_augccpvdz_df_augccpvqzjkfit.npy'
npy_path = 'datasets/ethanethiol_md_traj_every1000_dft_augccpvdz.npy'
if os.path.exists(save_path):
    results = list(np.load(save_path, allow_pickle=True))
else:
    results = []
print('results len', len(results))
basis = 'augccpvdz'
auxbasis = 'augccpvqzjkfit'
for i in range(len(results), len(atom_pos)):
    print('calc', i)
    start = time.time()
    pos = atom_pos[i]
    atom = []
    for j in range(len(atom_types)):
        atom.append((atom_types[j], pos[j, :]))
    print(atom)
    mol = gto.M(atom=atom, basis='augccpvdz')
    # print(mol.pack())
    mf = dft.RKS(mol)
    mf.chkfile = False
    mf.xc = 'pbe'
    # mf.max_cycle = 1000
    mf.kernel()
    g = mf.nuc_grad_method()
    gradients = g.grad()
    # print(mfs[i].mo_coeff)
    res = []
    res.append(mol.pack())
    calc_dict = {}
    print('mo occ', mf.mo_occ)
    calc_dict['mo_coeff'] = mf.mo_coeff
    calc_dict['mo_occ'] = mf.mo_occ
    calc_dict['energy'] = mf.e_tot
    calc_dict['forces'] = -gradients/ase.units.Bohr

    dm1 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    auxmol = df.addons.make_auxmol(mol, auxbasis)

    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
    ints_2c2e = auxmol.intor('int2c2e')
    print('ints3c2e shape', ints_3c2e.shape)
    print('ints2c2e shape', ints_2c2e.shape)

    nao = mol.nao
    naux = auxmol.nao
    df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
    df_coef = df_coef.reshape(naux, nao, nao)
    if dm1.ndim > 2:
        df_basis = []
        for j in range(dm1.shape[0]):
            df_basis.append(lib.einsum('Pij,ij->P', df_coef, dm1[j]))
        df_basis = np.stack(df_basis, axis=0)
        print(df_basis.shape)

    else:
        df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

    calc_dict['df_coeff'] = df_basis
    calc_dict['auxbasis'] = auxbasis
    res.append(calc_dict)
    results.append(res)
    if (i+1) % 1000 == 0:
        print('i=', i, 'saving file')
        # if (i+1) == 4000:
        #     break
        np.save(save_path, results, allow_pickle=True)
np.save(save_path, results, allow_pickle=True)
npy_data = utils.calc_dict_to_npy(results, convert_forces=False, compress_atoms=False)
np.save(npy_path, npy_data, allow_pickle=True)

# %%
# inds = [5, 13, 81, 76, 65, 31]
inds = [13, 31, 65, 76]
for ind in inds:
    loc = ind * 1000
    data = np.load('md_logs/2022-12-23_pJuMyc8e/simulation_-ethanethiol_cluster_all_001_compressed_0.npy', allow_pickle=True).item()
    data['positions'] = data['positions'][loc-50:loc+50]
    print(data['positions'].shape)
    atom_pos = data['positions']
    atom_types = data['atom_numbers'].squeeze()
    save_path = 'datasets/ethanethiol_md_traj_loc_' + str(loc) + '_dft_augccpvdz_df_augccpvqzjkfit.npy'
    npy_path = 'datasets/ethanethiol_md_traj_loc_' + str(loc) + '_dft_augccpvdz.npy'
    if os.path.exists(save_path):
        results = list(np.load(save_path, allow_pickle=True))
    else:
        results = []
    print('results len', len(results))
    basis = 'augccpvdz'
    auxbasis = 'augccpvqzjkfit'
    for i in range(len(results), len(atom_pos)):
        print('calc', i)
        start = time.time()
        pos = atom_pos[i]
        atom = []
        for j in range(len(atom_types)):
            atom.append((atom_types[j], pos[j, :]))
        print(atom)
        mol = gto.M(atom=atom, basis='augccpvdz')
        # print(mol.pack())
        mf = dft.RKS(mol)
        mf.chkfile = False
        mf.xc = 'pbe'
        # mf.max_cycle = 1000
        mf.kernel()
        g = mf.nuc_grad_method()
        gradients = g.grad()
        # print(mfs[i].mo_coeff)
        res = []
        res.append(mol.pack())
        calc_dict = {}
        print('mo occ', mf.mo_occ)
        calc_dict['mo_coeff'] = mf.mo_coeff
        calc_dict['mo_occ'] = mf.mo_occ
        calc_dict['energy'] = mf.e_tot
        calc_dict['forces'] = -gradients/ase.units.Bohr

        dm1 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
        auxmol = df.addons.make_auxmol(mol, auxbasis)

        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
        ints_2c2e = auxmol.intor('int2c2e')
        print('ints3c2e shape', ints_3c2e.shape)
        print('ints2c2e shape', ints_2c2e.shape)

        nao = mol.nao
        naux = auxmol.nao
        df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        if dm1.ndim > 2:
            df_basis = []
            for j in range(dm1.shape[0]):
                df_basis.append(lib.einsum('Pij,ij->P', df_coef, dm1[j]))
            df_basis = np.stack(df_basis, axis=0)
            print(df_basis.shape)

        else:
            df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)

        calc_dict['df_coeff'] = df_basis
        calc_dict['auxbasis'] = auxbasis
        res.append(calc_dict)
        results.append(res)
        if (i+1) % 1000 == 0:
            print('i=', i, 'saving file')
            # if (i+1) == 4000:
            #     break
            np.save(save_path, results, allow_pickle=True)
    np.save(save_path, results, allow_pickle=True)
    npy_data = utils.calc_dict_to_npy(results, convert_forces=False, compress_atoms=False)
    np.save(npy_path, npy_data, allow_pickle=True)

# %%
