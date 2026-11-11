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

# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..
# %%
# load data
basis = 'augccpvdz'
auxbasis = 'augccpvqzjkfit'

distances = np.arange(0.8, 2, 0.01)
print(distances.shape)

# %%
save_path = 'datasets/CO_dft_augccpvdz_df_augccpvqzjkfit.npy'
npy_path = 'datasets/CO_dft_augccpvdz_df_augccpvqzjkfit.npy'
if os.path.exists(save_path):
    results = list(np.load(save_path, allow_pickle=True))
else:
    results = []
print('results len', len(results))
atom_nums = [8, 6]
for i in range(len(results), len(distances)):
    print('calc', i)
    start = time.time()
    atom = []
    pos = np.array([[0, 0, 0], [float(distances[i]), 0, 0]])
    for j in range(len(atom_nums)):
        atom.append((atom_nums[j], pos[j, :]))
    mol = gto.M(atom=atom, basis=basis)
    mf = dft.RKS(mol)
    mf.chkfile = False
    mf.xc = 'pbe'
    mf.kernel()
    g = mf.nuc_grad_method()
    gradients = g.grad()
    print('elapsed', time.time() - start)
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

    if i % 10 == 0:
        np.save(save_path, results, allow_pickle=True)
np.save(save_path, results, allow_pickle=True)
npy_data = utils.calc_dict_to_npy(results, convert_forces=False, compress_atoms=False)
np.save(npy_path, npy_data, allow_pickle=True)
