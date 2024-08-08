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
hf.MUTE_CHKFILE = True
# %load_ext autoreload
# %autoreload 2
# %%

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
# calculating for a single molecule, and getting different energy components
data = np.load('datasets/h2o_dynamic_centered.npy', allow_pickle=True).item()
basis = 'augccpvdz'

atom_types = data['atom_types']

i = 0
print('calc', i)
start = time.time()
pos = data['positions'][i]
atom = []
for j in range(len(atom_types)):
    atom.append((atom_types[j], pos[j, :]))
mol = gto.M(atom=atom, basis=basis)
#print(mol.pack())
mf = hf.RHF(mol)
mf.max_cycle = 0
mf.init_guess = 'atom'
mf.chkfile=False
mf.kernel()
print(mf.energy_tot())
print(mf.e_tot)
# %%
dm = mf.make_rdm1()
m_kin = mol.intor('int1e_kin')
m_nuc = mol.intor('int1e_nuc')

e_kin = np.einsum('ij,ji', dm, m_kin)
e_nuc = np.einsum('ij,ji', dm, m_nuc)

veff = mf.get_veff()
ecoul = veff.ecoul
exc = veff.exc

print('total energy', e_kin + e_nuc + ecoul + exc + mf.energy_nuc())
print(mf.__dir__())
print('total_energy etot', mf.e_tot)
print('total_energy', mf.energy_tot())
print('nuclear energy', mf.energy_nuc())
print('eletronic energy, coulomb energy', mf.energy_elec())
print(mf.get_veff().shape)
print(mf.mo_coeff.shape)
# %%
def get_energy_components(mol, mf):
    """
    Get energy components for a single molecule.

    Args:
        mol: pyscf molecule
        mf: pyscf scf object
    Returns:
        energies: dictionary of energy components
    """
    dm = mf.make_rdm1()
    m_kin = mol.intor('int1e_kin')
    m_nuc = mol.intor('int1e_nuc')
    h1e = mf.get_hcore()
    veff = mf.get_veff()

    energies = {}
    energies['energy'] = mf.energy_tot()
    energies['energy_e_kin'] = np.einsum('ij,ji', dm, m_kin)
    energies['energy_e_nuc'] = np.einsum('ij,ji', dm, m_nuc)
    energies['energy_coul'] = veff.ecoul
    energies['energy_exc'] = veff.exc
    energies['energy_nuc'] = mf.energy_nuc()
    # print('energies', energies)
    # print('total energy', energies['energy'])
    # print('mf energy elec', mf.energy_elec())
    # print('mf energy nuc', mf.energy_nuc())
    # print('mf energy elec + nuc', mf.energy_elec() + mf.energy_nuc())
    # print('mf ecoul', energies['energy_coul'] + energies['energy_exc'])
    # print('energy h1e', energies['energy_e_kin'] + energies['energy_e_nuc'])
    # print('mf h1e', np.einsum('ij,ji', dm, h1e))
    #
    # print('total elec', energies['energy_e_kin'] + energies['energy_e_nuc'] +
    #       energies['energy_coul'] + energies['energy_exc'])
    # print('mf elec', np.einsum('ij,ji', dm, h1e) + energies['energy_coul'] + energies['energy_exc'])
    # print('summed components', energies['energy_e_kin'] + energies['energy_e_nuc'] +
    #       energies['energy_coul'] + energies['energy_exc'] + energies['energy_nuc'])

    assert np.isclose(energies['energy'], energies['energy_e_kin'] + energies['energy_e_nuc'] +
                      energies['energy_coul'] + energies['energy_exc'] + energies['energy_nuc'])
    return energies

# %%
set_types = ['train', 'valid', 'test']
for set_type in set_types:
    data = np.load('datasets/h2o_small_' + set_type + '_augccpvdz.npy', allow_pickle=True).item()
    basis = 'augccpvdz'
    auxbasis = 'augccpvqzjkfit'

    print(len(data['positions']))
    save_path = 'datasets/h2o_small_' + set_type + '_dft_augccpvdz_energy_comps_calc.npy'
    npy_path = 'datasets/h2o_small_' + set_type + '_dft_augccpvdz_energy_comps.npy'
    if os.path.exists(save_path):
        results = list(np.load(save_path, allow_pickle=True))
    else:
        results = []
    print('results len', len(results))
    for i in range(len(results), len(data['positions'])):
        print('calc', i)
        start = time.time()
        print('data positions shape', data['positions'].shape)
        pos = data['positions'][i]
        anums = data['atom_numbers'][i]
        print('pos shape', pos.shape)
        atom = []
        for j in range(len(anums)):
            atom.append((anums[j], pos[j, :])) 
        mol = gto.M(atom=atom, basis=basis)
        res = []
        res.append(mol.pack())
        #print(mol.pack())
        mf = dft.RKS(mol)
        mf.init_guess = 'atom'
        mf.max_cycle = 0
        mf.chkfile=False
        mf.xc = 'pbe'
        mf.kernel()
        g = mf.nuc_grad_method()
        gradients = g.grad()
        energies_SAD = get_energy_components(mol, mf)
        energies_SAD = {k + '_SAD': v for k, v in energies_SAD.items()}
        calc_dict = {}
        calc_dict.update(energies_SAD)
        calc_dict['forces_SAD'] = -gradients/ase.units.Bohr
        mol = gto.M(atom=atom, basis=basis)
        #print(mol.pack())
        mf = dft.RKS(mol)
        mf.chkfile=False
        mf.xc = 'pbe'
        mf.kernel()
        g = mf.nuc_grad_method()
        gradients = g.grad()
        energies = get_energy_components(mol, mf)

        calc_dict['forces'] = -gradients/ase.units.Bohr
        calc_dict.update(energies)

        print('calc_dict', calc_dict)
        res.append(calc_dict)
        results.append(res)

        if i%10 == 0:
            np.save(save_path, results, allow_pickle=True)
    np.save(save_path, results, allow_pickle=True)
    npy_data = utils.calc_dict_to_npy(results, convert_forces=False, compress_atoms=False)
    npy_data_compressed = utils.calc_dict_to_npy(results, convert_forces=False, compress_atoms=True)
    print('atom_number nc', npy_data['atom_numbers'][:3])
    print('atom_number c', npy_data_compressed['atom_numbers'][:3])
    print('pos nc', npy_data['positions'][:3])
    print('pos c', npy_data_compressed['positions'][:3])
    print('forces nc', npy_data['forces'][:3])
    print('forces c', npy_data_compressed['forces'][:3])
    print('forces sad nc', npy_data['forces_SAD'][:3])
    print('forces sad c', npy_data_compressed['forces_SAD'][:3])
    np.save(npy_path, npy_data, allow_pickle=True)
# %%
np.save(save_path, results, allow_pickle=True)

# %%
set_types = ['train', 'valid', 'test']
for set_type in set_types:
    data = np.load('datasets/h2o_small_' + set_type + '_augccpvdz_df_augccpvqzjkfit.npy', allow_pickle=True)
    basis = 'augccpvdz'
    auxbasis = 'augccpvqzjkfit'

    print(len(data))
    save_path = 'datasets/h2o_small_' + set_type + '_dft_augccpvdz_df_hm_dm_oe_calc.npy'
    npy_path = 'datasets/h2o_small_' + set_type + '_dft_augccpvdz_df_hm_dm_oe.npy'
    if os.path.exists(save_path):
        results = list(np.load(save_path, allow_pickle=True))
    else:
        results = []
    print('results len', len(results))
    for i in range(len(results), len(data)):
        print('calc', i)
        start = time.time()
        atom = data[i][0]["atom"] 
        mol = gto.M(atom=atom, basis=basis)
        res = []
        res.append(mol.pack())
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
        print('mo_coeff shape', calc_dict['mo_coeff'].shape)
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
        # print('calc_dict', calc_dict)
        oe = mf.mo_energy
        hm = hf.get_fock(mf)
        calc_dict.update({"mo_energies":oe, "density_matrix": dm1,
                          "hamiltonian_matrix": hm})
        res.append(calc_dict)
        results.append(res)
        if i%10 == 0:
            np.save(save_path, results, allow_pickle=True)
    np.save(save_path, results, allow_pickle=True)
    npy_data = utils.calc_dict_to_npy(results, convert_forces=False, compress_atoms=False)
    npy_data_compressed = utils.calc_dict_to_npy(results, convert_forces=False, compress_atoms=True)
    # print('atom_number nc', npy_data['atom_numbers'][:3])
    # print('atom_number c', npy_data_compressed['atom_numbers'][:3])
    # print('pos nc', npy_data['positions'][:3])
    # print('pos c', npy_data_compressed['positions'][:3])
    # print('forces nc', npy_data['forces'][:3])
    # print('forces c', npy_data_compressed['forces'][:3])
    # print('forces sad nc', npy_data['forces_SAD'][:3])
    # print('forces sad c', npy_data_compressed['forces_SAD'][:3])
    np.save(npy_path, npy_data, allow_pickle=True)
# %%
set_types = ['train', 'valid', 'test']
for set_type in set_types:
    data1 = np.load('datasets/h2o_small_' + set_type + '_augccpvdz_df_augccpvqzjkfit.npy', allow_pickle=True)
    data2 = np.load('datasets/h2o_small_' + set_type + '_dft_augccpvdz_hm_dm_oe_calc.npy', allow_pickle=True)
    save_path = 'datasets/h2o_small_' + set_type + '_dft_augccpvdz_df_hm_dm_oe_calc.npy'
    for i in range(len(data1)):
        data2[i][1]['auxbasis'] = data1[i][1]['auxbasis']
        data2[i][1]['df_coeff'] = data1[i][1]['df_coeff']

    np.save(save_path, data2, allow_pickle=True)

# %%
results = np.load(save_path, allow_pickle=True)
for res in results:
    calc = res[1]
    mo_coeff = calc['mo_coeff']
    mo_en = calc['mo_energies']
    mol = gto.M(atom=res[0]['atom'], basis=basis)
    s1e = mol.intor('int1e_ovlp')
    ks = calc['hamiltonian_matrix'] 
    mf = dft.RKS(mol)
    moe_calc, mo_calc = mf.eig(ks, s1e)
    print('moe calc', moe_calc)
    print('mo en', mo_en)
    print('mo en error kcal', utils.hartree_to_kcal(np.mean(np.abs(moe_calc - mo_en))))
    print('mo en error hartree', np.mean(np.abs(moe_calc - mo_en)))
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


