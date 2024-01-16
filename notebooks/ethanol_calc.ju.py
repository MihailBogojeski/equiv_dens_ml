# %%
import ase
import ase.io
import numpy as np
import pyscf
import time
import os
from pyscf import gto, dft, df, lib
from pyscf.scf import hf
import scipy
import equiv_dens.utils.base as utils
hf.MUTE_CHKFILE = True
# %load_ext autoreload
# %autoreload 2

# %%
# load data
basis = 'ccpvdz'
auxbasis = 'augccpvqzjkfit'
datasets = ['train', 'test', 'valid']

# %%

for dataset in datasets:
    load_path = 'datasets/ethanol_dft_' + dataset + '.npy'
    data = np.load(load_path, allow_pickle=True).item()
    save_path = '_'.join(load_path.split('.')[0].split('_')[:-1]) + '_pyscf_' + basis + '_df_' + auxbasis + '_' + dataset + '.npy'
    print('save path', save_path)
    npy_path = '_'.join(load_path.split('.')[0].split('_')[:-1]) + '_pyscf_' + basis + '_' + dataset + '.npy'
    print('npy path', npy_path)
    if os.path.exists(save_path):
        results = list(np.load(save_path, allow_pickle=True))
    else:
        results = []
    print('results len', len(results))
    print('atom_pos shape', data['positions'].shape)
    print('atom_nums shape', data['atom_numbers'].shape)
    for i in range(len(results), len(data['positions'])):
        print('calc', i)
        start = time.time()
        pos = data['positions'][i]
        atom_nums = data['atom_numbers'] 
        atom = []
        for j in range(len(atom_nums)):
            atom.append((atom_nums[j], pos[j, :])) 
        mol = gto.M(atom=atom, basis=basis)
        #print(mol.pack())
        mf = dft.RKS(mol)
        mf.chkfile=False
        mf.xc = 'pbe'
        #mf.max_cycle = 1000
        mf.kernel()
        g = mf.nuc_grad_method()
        gradients = g.grad()
        print('elapsed', time.time() - start)
        #print(mfs[i].mo_coeff)
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

        if i%10 == 0:
            np.save(save_path, results, allow_pickle=True)
    np.save(save_path, results, allow_pickle=True)
    npy_data = utils.calc_dict_to_npy(results, convert_forces=False, compress_atoms=False)
    np.save(npy_path, npy_data, allow_pickle=True)

# %%
for dataset in datasets:
    load_path = 'datasets/ethanol_dft_' + dataset + '.npy'
    save_path = '_'.join(load_path.split('.')[0].split('_')[:-1]) + '_pyscf_' + basis + '_df_' + auxbasis + '_' + dataset + '.npy'
    data = np.load(save_path, allow_pickle=True)
    print('save path', save_path)
    npy_path = '_'.join(load_path.split('.')[0].split('_')[:-1]) + '_pyscf_' + basis + '_' + dataset + '.npy'
    print('npy path', npy_path)
    npy_data = utils.calc_dict_to_npy(data, convert_forces=False, compress_atoms=False)
    np.save(npy_path, npy_data, allow_pickle=True)

# %%
svp_tr = np.load('datasets/ethanol_dft_train.npy', allow_pickle=True).item()
svp_te = np.load('datasets/ethanol_dft_test.npy', allow_pickle=True).item()
svp_va = np.load('datasets/ethanol_dft_valid.npy', allow_pickle=True).item()
ccpvdz_tr = np.load('datasets/ethanol_dft_pyscf_ccpvdz_train.npy', allow_pickle=True).item()
ccpvdz_te = np.load('datasets/ethanol_dft_pyscf_ccpvdz_test.npy', allow_pickle=True).item()
ccpvdz_va = np.load('datasets/ethanol_dft_pyscf_ccpvdz_valid.npy', allow_pickle=True).item()

print('train energies corr', np.corrcoef(svp_tr['energy'].T, ccpvdz_tr['energy'].T))
print('test energies corr', np.corrcoef(svp_te['energy'].T, ccpvdz_te['energy'].T))
print('valid energies corr', np.corrcoef(svp_va['energy'].T, ccpvdz_va['energy'].T))
print('train forces corr', np.corrcoef(svp_tr['forces'].flatten(), ccpvdz_tr['forces'].flatten()))
print('test forces corr', np.corrcoef(svp_te['forces'].flatten(), ccpvdz_te['forces'].flatten()))
print('valid forces corr', np.corrcoef(svp_va['forces'].flatten(), ccpvdz_va['forces'].flatten()))

# %%
print(svp_tr['forces'][0])
print(ccpvdz_tr['forces'][0])
print(ccpvdz_tr['atom_numbers'][0])

# %%


# %%


# %%
from equiv_dens.utils.grids import spherical_grid

new_data = np.copy(data).item()
new_data['atom_numbers'] = data['atom_numbers'][0]
new_data['atom_types'] = data['atom_types'][0]

print(spherical_grid(new_data, level=1))

# %%
from equiv_dens.utils.grids import spherical_grid2


print(spherical_grid2(data, level=1))

# %%


