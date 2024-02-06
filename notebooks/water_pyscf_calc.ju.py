# %%
import numpy as np
import time
import os
import ase
from pyscf import gto, dft, df, lib
from pyscf.scf import hf
import scipy
from equiv_dens.utils import base as utils
# %cd /home/mihail/Documents/workspace/equiv_dens/
# %%
hf.MUTE_CHKFILE = True

# mol = gto.M(atom='O  0  0  0.1184; H  0,  0.7532, -0.4735; H 0,  -0.7532, -0.4735 ', basis='def2svp')
# mf = dft.RKS(mol)
# mf.chkfile=False
# mf.xc = 'pbe'
# mf.kernel()
# g = mf.nuc_grad_method()
# g.kernel()
data = np.load('datasets/h2o_dynamic_centered.npy', allow_pickle=True).item()
basis = 'augccpvdz'
auxbasis = 'augccpvqzjkfit'

atom_types = data['atom_types']

print(len(data['positions']))
save_path = 'datasets/h2o_dynamic_augccpvdz_df_augccpvqzjkfit.npy'
npy_path = 'datasets/h2o_dynamic_augccpvdz.npy'
if os.path.exists(save_path):
    results = list(np.load(save_path, allow_pickle=True))
else:
    results = []
print('results len', len(results))
for i in range(len(results), len(data['positions'])):
    print('calc', i)
    start = time.time()
    pos = data['positions'][i]
    atom = []
    for j in range(len(atom_types)):
        atom.append((atom_types[j], pos[j, :])) 
    mol = gto.M(atom=atom, basis=basis)
    #print(mol.pack())
    mf = dft.RKS(mol)
    mf.chkfile=False
    mf.xc = 'pbe'
    mf.kernel()
    g = mf.nuc_grad_method()
    gradients = g.grad()
    print('elapsed', time.time() - start)
    #print(mfs[i].mo_coeff)
    res = []
    res.append(mol.pack())
    calc_dict = {}
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

    if i%100 == 0:
        np.save(save_path, results, allow_pickle=True)
np.save(save_path, results, allow_pickle=True)
npy_data = utils.calc_dict_to_npy(results, convert_forces=False, compress_atoms=False)
np.save(npy_path, npy_data, allow_pickle=True)

# %%
np.save(save_path, results, allow_pickle=True)

# %%
results = []
for i in range(len(mfs)):
    #print(mfs[i].mo_coeff)
    mol_dict = mols[i].pack()
    calc_dict = {}
    calc_dict['mo_coeff'] = mfs[i].mo_coeff
    calc_dict['mo_occ'] = mfs[i].mo_occ
    calc_dict['energy'] = mfs[i].e_tot
    calc_dict['forces'] = forces[i]
    results.append((mol_dict, calc_dict))
    results.append(res)

np.save('datasets/h2o_dynamic_pyscf_631gss_dft_f.npy', results)

# %%
data_scf = np.load('datasets/h2o_dynamic_pyscf_631gss_dft_f.npy', allow_pickle=True)
print(data_scf)
print(len(data_scf))


# %%
data_scf = np.load('datasets/h2o_dynamic_pyscf_dft_f.npy', allow_pickle=True)
print(data_scf)
print(len(data_scf))
for d in data_scf:
    print('energy', d['energy'])
    print('forces', d['forces'])

# %%
for d in data_scf:
    d['forces'] = d['forces'] * 0.529177


np.save('datasets/h2o_dynamic_pyscf_dft_f.npy', data_scf)

# %%
new_data = []
data_scf = data_scf = np.load('datasets/h2o_dynamic_pyscf_dft_f.npy', allow_pickle=True)
for d in data_scf:
    new_d = []
    mo_coeff = d.pop('mo_coeff')
    mo_occ = d.pop('mo_occ')
    en = d.pop('energy')
    f = d.pop('forces')
    new_d.append(d)
    new_d.append({'mo_coeff': mo_coeff, 'mo_occ': mo_occ, 'energy': en, 'forces': f})
    new_data.append(new_d)

np.save('datasets/h2o_dynamic_pyscf_dft_f_en.npy', new_data)

# %%
results = {'E': [], 'F': [], 'R': [], 'z': np.array([8, 1, 1])}
for i in range(len(mfs)):
    #print(mfs[i].mo_coeff)
    res = mols[i].pack()
    pos = []
    for a in res['atom']:
        pos.append(a[1])
    pos = np.array(pos)
    print('pos.shape', pos.shape)
    results['R'].append(pos)
    results['E'].append(mfs[i].e_tot)
    results['F'].append(forces[i])
    
results['R'] = np.array(results['R'])
results['E'] = np.array(results['E'])
results['F'] = np.array(results['F'])

np.savez('datasets/water_pyscf_dft_f', **results)

# %%
results = {'energy': [], 'forces': [], 'positions': [],
           'atom_numbers': [8, 1, 1], 'atom_types': ['O', 'H', 'H'],
          'mo_coeff': [], 'mo_occ': []}
for d in data_scf:
    #print(mfs[i].mo_coeff)
    pos = []
    for a in d['atom']:
        pos.append(a[1])
    pos = np.array(pos)
    print('pos.shape', pos.shape)
    results['positions'].append(pos)
    results['energies'].append(d['energy'])
    results['forces'].append(d['forces'])
    results['mo_coeff'].append(d['mo_coeff'])
    results['mo_occ'].append(d['mo_occ'])

for key in results.keys():
    results[key] = np.array(results[key])
    
np.savez('datasets/h2o_dynamic_pyscf_dft_f', **results)

# %%
print(np.load('datasets/h2o_dynamic_pyscf_dft.npy', allow_pickle=True)[0])
print(np.load('datasets/h2o_dynamic_pyscf_dft_f.npy', allow_pickle=True)[0])
print(np.load('datasets/h2o_dynamic_pyscf_dft_f_en.npy', allow_pickle=True)[0])

# %%


