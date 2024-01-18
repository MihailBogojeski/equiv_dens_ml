# %%
import numpy as np
import time
import os
from pyscf import gto, dft
from pyscf.scf import hf
hf.MUTE_CHKFILE = True
import equiv_dens.utils.base as utils

# %%
# load data

data = np.load('datasets/aspirin_gdml_random_1000_train.npy', allow_pickle=True).item()
atom_pos = data['positions']
atom_types = data['atom_numbers']

# %%
basis = 'def2svp'
save_path = 'datasets/aspirin_rot.npy'
if os.path.exists(save_path):
    results = list(np.load(save_path, allow_pickle=True))
else:
    results = []
print('results len', len(results))
for i in range(3):
    print('calc', i)
    start = time.time()
    rot_mat = utils.random_rotation_matrix()
    pos = atom_pos[0] @ rot_mat
    atom = []
    for j in range(len(atom_types)):
        atom.append((atom_types[j], pos[j, :])) 
    mol = gto.M(atom=atom, basis=basis)
    #print(mol.pack())
    mf = dft.RKS(mol)
    mf.chkfile=False
    mf.xc = 'pbe'
    #mf.max_cycle = 1000
    mf.kernel()
    g = mf.nuc_grad_method()
    forces = g.grad()
    print('elapsed', time.time() - start)
    #print(mfs[i].mo_coeff)
    res = []
    res.append(mol.pack())
    calc_dict = {}
    print('mo occ', mf.mo_occ)
    calc_dict['mo_coeff'] = mf.mo_coeff
    calc_dict['mo_occ'] = mf.mo_occ
    calc_dict['energy'] = mf.e_tot
    calc_dict['forces'] = forces
    res.append(calc_dict)
    results.append(res)
    
    if i%10 == 0:
        np.save(save_path, results, allow_pickle=True)
np.save(save_path, results, allow_pickle=True)

# %%
atom_pos = np.load('datasets/resorcinol_atom_pos.npy', allow_pickle=True)

atom_pos = utils.bohr_to_angstrom(atom_pos[:1004])
print(atom_pos.shape)
atom_types = np.load('datasets/resorcinol_atom_numbers.npy', allow_pickle=True)[0]
load_path = 'datasets/resorcinol_pyscf_augccpvdz_dft_train.npy'
pyscf_data = np.load(load_path, allow_pickle=True)
print('pyscf data len', len(pyscf_data))
data = {}
data['positions'] = atom_pos
data['atom_numbers'] = atom_types
data['atom_types'] = utils.numbers_to_symbols(atom_types)
data['energy'] = []
data['forces'] = []
for calc in pyscf_data:
    data['energy'].append(calc[1]['energy'])
    data['forces'].append(-calc[1]['forces']*utils.to_bohr)
    
data['energy'] = np.array(data['energy'])[:, None]
data['forces'] = np.array(data['forces'])
print('energy shape', data['energy'].shape)
print('forces shape', data['forces'].shape)
save_path = 'datasets/resorcinol_augccpvdz_train.npy'
np.save(save_path, data, allow_pickle=True)

# %%
# load data
data = np.load('datasets/resorcinol_combo_kmeansidx-1000_train.npy', allow_pickle=True).item()
atom_pos = data['positions'] 

atom_types = data['atom_numbers'][0]

# %%
save_path = 'datasets/resorcinol_just_checking.npy'
if os.path.exists(save_path):
    results = list(np.load(save_path, allow_pickle=True))
else:
    results = []
print('results len', len(results))
for i in range(len(results), 10):
    print('calc', i)
    start = time.time()
    pos = atom_pos[i]
    atom = []
    for j in range(len(atom_types)):
        atom.append((atom_types[j], pos[j, :])) 
    mol = gto.M(atom=atom, basis='augccpvdz')
    #print(mol.pack())
    mf = dft.RKS(mol)
    mf.chkfile=False
    mf.xc = 'pbe'
    #mf.max_cycle = 1000
    mf.kernel()
    g = mf.nuc_grad_method()
    forces = g.grad()
    print('elapsed', time.time() - start)
    #print(mfs[i].mo_coeff)
    res = []
    res.append(mol.pack())
    calc_dict = {}
    print('mo occ', mf.mo_occ)
    calc_dict['mo_coeff'] = mf.mo_coeff
    calc_dict['mo_occ'] = mf.mo_occ
    calc_dict['energy'] = mf.e_tot
    calc_dict['forces'] = forces
    res.append(calc_dict)
    results.append(res)
    
    if i%10 == 0:
        np.save(save_path, results, allow_pickle=True)
np.save(save_path, results, allow_pickle=True)

# %%
import matplotlib.pyplot as plt

save_path = 'datasets/aspirin_rot.npy'
if os.path.exists(save_path):
    results = list(np.load(save_path, allow_pickle=True))
else:
    results = []
print('results len', len(results))

mo_coeffs = [dat[1]['mo_coeff'] for dat in results]

print([mc.shape for mc in mo_coeffs])

for i in range(2, len(mo_coeffs)):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(mo_coeffs[i][::-1, ::-1], cmap='RdBu', norm='symlog')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig("figures/mo_coeff_{}.svg".format(i))

# %%
print(np.allclose(mo_coeffs[3], mo_coeffs[2], atol=1e-4))
print(np.allclose(mo_coeffs[3], mo_coeffs[2], atol=1e-4))

# %%
# load data

data = np.load('datasets/water_dyn.npy', allow_pickle=True).item()
atom_pos = data['positions']
atom_types = data['atom_numbers'][0]
print('atom pos', atom_pos)
print('atom types', atom_types)
atom_types = atom_types[[2, 0, 1]]
atom_pos = atom_pos[:, [2, 0, 1], :]
print('atom pos', atom_pos)
print('atom types', atom_types)

# %%
basis = 'sto-3g'
save_path = 'datasets/water_rot_minimal.npy'
# if os.path.exists(save_path):
#     results = list(np.load(save_path, allow_pickle=True))
# else:
results = []
print('results len', len(results))
for i in range(3):
    print('calc', i)
    start = time.time()
    rot_mat = utils.random_rotation_matrix()
    pos = atom_pos[0] @ rot_mat
    atom = []
    for j in range(len(atom_types)):
        atom.append((atom_types[j], pos[j, :])) 
    mol = gto.M(atom=atom, basis=basis)
    #print(mol.pack())
    mf = dft.RKS(mol)
    mf.chkfile=False
    mf.xc = 'pbe'
    #mf.max_cycle = 1000
    mf.kernel()
    ovr = hf.get_ovlp(mol)
    print('elapsed', time.time() - start)
    #print(mfs[i].mo_coeff)
    res = []
    res.append(mol.pack())
    calc_dict = {}
    print('mo occ', mf.mo_occ)
    calc_dict['mo_coeff'] = mf.mo_coeff
    calc_dict['mo_occ'] = mf.mo_occ
    calc_dict['mo_energy'] = mf.mo_energy
    calc_dict['ovr'] = ovr
    calc_dict['energy'] = mf.e_tot
    # calc_dict['forces'] = forces
    res.append(calc_dict)
    results.append(res)
    
    if i%10 == 0:
        np.save(save_path, results, allow_pickle=True)
np.save(save_path, results, allow_pickle=True)

# %%
import matplotlib.pyplot as plt

mo_coeffs = [dat[1]['mo_coeff'] for dat in results]

print([mc.shape for mc in mo_coeffs])

for i in range(len(mo_coeffs)):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(mo_coeffs[i][::-1, ::-1], cmap='RdBu', norm='symlog')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig("figures/mo_coeff_h2o_{}.svg".format(i))


# %%
hams = []
for i in range(len(results)):
    mo = results[i][1]['mo_coeff']
    ovr = results[i][1]['ovr']
    en = np.diag(results[i][1]['mo_energy'])
    H = ovr @ mo @ en @ np.linalg.inv(mo)
    # H = ovlp @ pyscf_orb[i][1]['mo_coeff'] @ np.diag(pyscf_orb[i][1]['mo_energy']) @ np.linalg.inv(pyscf_orb[i][1]['mo_coeff'])
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    print('mo_energies', np.diag(en))
    H = np.cbrt(H)
    ax.imshow(H, cmap='RdBu', norm='symlog')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig("figures/ham_h2o_{}.svg".format(i))


# %%
print(np.allclose(mo_coeffs[0], mo_coeffs[1], atol=1e-4))
print(np.allclose(mo_coeffs[0], mo_coeffs[2], atol=1e-4))

# %%


