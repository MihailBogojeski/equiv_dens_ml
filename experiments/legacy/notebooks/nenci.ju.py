# %%
# %cd ..
# %load_ext autoreload
# %autoreload 2
# %%
import ase
import pickle
import pyscf
from pyscf import gto, dft, df
from pyscf.dft import gen_grid, radi, numint
import matplotlib.pyplot as plt
from pyscf.scf import hf
import numpy as np
from argparse import Namespace
from functools import partial
import torch
from copy import copy, deepcopy
import scipy

hf.MUTE_CHKFILE = True

from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.errors import ErrorDict
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
     CubicalGrid, spherical_grid, spherical_radial_sampling
from equiv_dens.training.model_loader import load_model
import equiv_dens.utils.base as utils
from equiv_dens.training import utils as train_utils
from equiv_dens.training import model_loader
from equiv_dens.utils import orbitals

# %%
with open('./datasets/calc_dict.pickle', 'rb') as f:
    calculations = pickle.load(f)
with open('./datasets/nenci_monomer_ase.pickle', 'rb') as f:
    monomer1, monomer2 = pickle.load( f)

# %% [markdown]
"""
calculations is a dict that stores the names of all the calculations associated with that dimer pair in a list. The keys are '001' to '066'
"""

# %%
print(calculations['001'])

# %% [markdown]
"""
Each calculation name is a key for the dicts monomer1 and monomer2. monomer1[key] is an ase.Atoms object of just the first monomer in each calc and monomer2[key] is the corresponding Atoms object for monomer2.
"""

# %%
key = calculations['002'][0]
print(key)
print(monomer1[key])
print(monomer2[key])

# %%
# #Reference calculations for two water monomers
atoms1 = monomer1[key]
atoms2 = monomer2[key]

pos1 = monomer1[key].get_positions()
z1 = monomer1[key].get_atomic_numbers()
pos2 = monomer2[key].get_positions()
z2 = monomer2[key].get_atomic_numbers()
print(pos1)
print(z1)
atom1 = [(z1[i], pos1[i]) for i in range(len(z1))]
atom2 = [(z2[i], pos2[i]) for i in range(len(z2))]

mol1 = gto.Mole(atom=atom1, basis='augccpvdz')
mf1 = dft.RKS(mol1)
mf1.chkfile = False
mf1.xc = 'pbe'
mf1.kernel()
dm1 = mf1.make_rdm1()

mol2 = gto.Mole(atom=atom2, basis='augccpvdz')
mf2 = dft.RKS(mol2)
mf2.chkfile = False
mf2.xc = 'pbe'
mf2.kernel()
dm2 = mf2.make_rdm1()
# %%
# #Generate dense integration grids
gs1 = gen_grid.gen_atomic_grids(mol1, radi_method=radi.treutler, level=9)
c1, w1 = gen_grid.get_partition(mol1, gs1)
ao1 = numint.eval_ao(mol1, c1, deriv=0)
rho1 = numint.eval_rho(mol1, ao1, dm1, xctype='LDA')
print('rho int', np.sum(rho1 * w1))
print('dm trace', np.trace(dm1))

gs2 = gen_grid.gen_atomic_grids(mol2, radi_method=radi.treutler, level=9)
c2, w2 = gen_grid.get_partition(mol2, gs2)
ao2 = numint.eval_ao(mol2, c2, deriv=0)
rho2 = numint.eval_rho(mol2, ao2, dm2, xctype='LDA')
# %%
ef_pos_1 = mol2.atom_coords(unit='Bohr')
ef_pos_2 = mol1.atom_coords(unit='Bohr')
# %%
# #Numerical integral of electric field at monomer 2 sites due to monomer 1
atc = mol1.atom_coords(unit='Bohr')
z = mol1.atom_charges()
at_diffs = ef_pos_1[None, :] - atc[:, None]
at_dists = np.linalg.norm(at_diffs, axis=-1, keepdims=True)

pos_e_field1 = np.sum(np.reshape(z, (-1, 1, 1)) * at_diffs / at_dists**3, axis=0)
print('positive e field', pos_e_field1)

rho_diffs = ef_pos_1[None, :] - c1[:, None]

rho_dists = np.linalg.norm(rho_diffs, axis=-1, keepdims=True)
neg_contrib = rho1[:, None, None] * w1[:, None, None] * rho_diffs / rho_dists**3

neg_e_field1 = np.sum(neg_contrib, axis=0)
print('negative e field', neg_e_field1)
e_field_1 = pos_e_field1 - neg_e_field1
print('e field 1', e_field_1)
# %%
# #trying out different stuff for the analytic integral
mol1_atom = mol1.atom
mol2_atom = mol2.atom
print(mol1_atom)
print(mol2_atom)

mol1_aux = [[('H0', mol2_atom[i][1])] + mol1_atom for i in range(len(mol2_atom))]
mol2_aux = [[('H0', mol1_atom[i][1])] + mol2_atom for i in range(len(mol1_atom))]

alpha = 1000.0
aux_basis_1 = {'H0': [[0, [alpha, 1.0]]]} | mol1._basis
aux_basis_2 = {'H0': [[0, [alpha, 1.0]]]} | mol2._basis

mol1_gto_aux = [[
    gto.M(atom=mol1_aux[i], basis=aux_basis_1, spin=None),
    gto.M(atom=mol1_aux[i], basis=aux_basis_1, spin=None),
    gto.M(atom=mol1_aux[i], basis=aux_basis_1, spin=None),
] for i in range(len(mol1_aux))]
mol2_gto_aux = [[
    gto.M(atom=mol2_aux[i], basis=aux_basis_2, spin=None),
    gto.M(atom=mol2_aux[i], basis=aux_basis_2, spin=None),
    gto.M(atom=mol2_aux[i], basis=aux_basis_2, spin=None),
] for i in range(len(mol2_aux))]

# #print(mol1._basis)
# #print(mol1_gto_aux[0][0]._basis)
# #print(mol2_gto_aux[0][0]._basis)

aux_mo_coeff_1 = []
aux_mo_occ_1 = []
aux_dm1 = []
# #print('mo coeffs', mf1.mo_coeff[:5, :5])
# #print('mo occs', mf1.mo_occ[:5])
# #print('dm', dm1[:5, :5])
for i in range(1):
    aux_mo_coeff_1.append(np.zeros((mol1_gto_aux[0][i].nao, mol1_gto_aux[0][i].nao)))
    aux_mo_occ_1.append(np.zeros((mol1_gto_aux[0][i].nao, )))
    aux_mo_coeff_1[i][1:, 1:] = mf1.mo_coeff
    aux_mo_occ_1[i][1:] = mf1.mo_occ
    aux_mo_coeff_1[i][i, i] = 1/alpha**(3/2)/np.pi**(3/2)
    aux_mo_occ_1[i][i] = 1
    # #print('aux occ', aux_mo_occ_1[i][:5])
    # #print('aux coeff', aux_mo_coeff_1[i][:5, :5])
    aux_dm1.append(hf.make_rdm1(aux_mo_coeff_1[i], aux_mo_occ_1[i]))

vj, _ = hf.get_jk(mol1, dm1)
coul_en_1 = np.einsum('ij,ji->', vj, dm1).real * 0.5
print('coul_en', coul_en_1)


print('aux_dm rho int', aux_dm1[0].trace())
ao_aux = numint.eval_ao(mol1_gto_aux[0][0], c1, deriv=0)
rho_aux = numint.eval_rho(mol1_gto_aux[0][0], ao_aux, aux_dm1[0])
print('rho aux int', np.sum(rho_aux * w1))
print('aux_mo_occ_1[0]', aux_mo_occ_1[0].sum())

e_field1_0_0 = []
for i in range(len(aux_dm1)):
    vj, _ = hf.get_jk(mol1_gto_aux[0][i], aux_dm1[i])
    aux_coul = np.einsum('ij,ji->', vj, aux_dm1[i]).real * 0.5
    print('aux coul', aux_coul)
    e_field1_0_0.append(aux_coul - coul_en_1)
print(' e field analytic', e_field1_0_0)

atc = mol1.atom_coords(unit='Bohr')
z = mol1.atom_charges()
at_diffs = ef_pos_1[None, :] - atc[:, None]
at_dists = np.linalg.norm(at_diffs, axis=-1, keepdims=True)

at_u = at_diffs / at_dists

out = np.einsum('i, ikx, xj, ik -> j', -z, at_u, np.eye(3), 1/at_dists.squeeze()**2)
print('einsum out', out)
print('mol1 pos', mol1.atom_coords(unit='Bohr'))
print('mol2 pos', mol2.atom_coords(unit='Bohr'))
# %%
# #calculating coulomb energy with two tiny gaussians
atom = [(1, [-0.5, 0 , 0]), (1, [0.5, 0, 0])]

alpha = 1.0
mol = gto.M(atom=atom, basis={1: [[0, [alpha, 1.0]]]}, spin=None)
print('mol intor ovlp', mol.intor('int1e_ovlp'))

mo_coeff = np.array([[1, 0], [0, 1]])
mo_occ = np.array([1, 1])
dm = hf.make_rdm1(mo_coeff, mo_occ)
vj, _ = hf.get_jk(mol, dm)
print('atom coords', mol.atom_coords(unit='Bohr'))
print(vj)
coul_en = np.einsum('ij,ji->', vj, dm).real * 0.5
print(coul_en)
# %%
mol._bas
# %%
# #Numerical integral of electric field at monomer 1 sites due to monomer 2
atc = mol2.atom_coords(unit='Bohr')
z = mol2.atom_charges()
at_diffs = ef_pos_2[None, :] - atc[:, None]
at_dists = np.linalg.norm(at_diffs, axis=-1, keepdims=True)

pos_e_field2 = np.sum(np.reshape(z, (-1, 1, 1)) * at_diffs / at_dists**3, axis=0)
print('positive e field', pos_e_field2)

rho_diffs = ef_pos_2[None, :] - c2[:, None]

rho_dists = np.linalg.norm(rho_diffs, axis=-1, keepdims=True)
neg_contrib = rho2[:, None, None] * w2[:, None, None] * rho_diffs / rho_dists**3

neg_e_field2 = np.sum(neg_contrib, axis=0)
print('negative e field', neg_e_field2)
e_field_2 = pos_e_field2 - neg_e_field2
print('e field 2', e_field_2)
# %%
# ## #using aux e2 to calculate e field from e potential calculation in pyscf
# #alpha = 1000000
# #coeff = alpha**(3/2)/(np.pi)
# ## #print('coeff', coeff)
# ## #coeff = orbitals.gto_norm(1, alpha) * orbitals.pyscf_gto_factor
# ## #print('coeff', coeff)
# #coords = mol2.atom_coords(unit='Bohr')
# #print('mol 1 coords', mol1.atom_coords(unit='Angstrom'))
# #print('mol 2 coords', utils.bohr_to_angstrom(coords))
# #print('coords shape', coords.shape)
# #fakemol = gto.fakemol_for_charges(coords)
# #fakemol._bas[:,1] = 0 #set l = 1
# #fakemol._env[-1] = coeff
# #fakemol._env[-2] = alpha
# #ints = df.incore.aux_e2(mol1, fakemol)
# #print(ints.shape)
# #print('dm shape', dm1.shape)
# #elef = -np.einsum('ijp,ij->p', ints, dm1)
# #print('e field dens', elef)
# ## #out = np.einsum('i, ikx, xj, ik -> kj', z, at_u, np.eye(3), 1/at_dists.squeeze()**2)
# ## #print('e field atom', out)
# ## #print('e field diff', out - elef)
# ## #e_field_analytic = out - elef
# #print(elef)
# %%
# #using aux e2 to calculate e field from e potential calculation in pyscf
# #analytic integral for electric field of monomer 2 sites due to monomer 1
alpha = 1000000
coeff = alpha**(3/2)/(np.pi)
# #sqrt ((2l + 1)/4pi) is missing as a normalization factor
# #from all pyscf spherical harmonics
#
# #print('coeff', coeff)
# #coeff = orbitals.gto_norm(1, alpha) * orbitals.pyscf_gto_factor
# #print('coeff', coeff)
coords = mol2.atom_coords(unit='Bohr')
fakemol = gto.fakemol_for_charges(coords)
fakemol._bas[:,1] = 0 #set l = 1
fakemol._env[-1] = coeff
fakemol._env[-2] = alpha
ints = df.incore.aux_e2(mol1, fakemol)
print(ints.shape)
print('ints shape', ints.shape)
elef1 = -np.einsum('ijp,ij->p', ints, dm1)
print('e field dens 1', elef1)
# #e_field_1_analytic = pos_e_field1 - elef1
# #print('e field 1 analytic', e_field_1_analytic)
# %%
# #using aux e2 to calculate e field from e potential calculation in pyscf
# #analytic integral for electric field of monomer 2 sites due to monomer 1
alpha = 1000000
coeff = 2 * alpha**(3/2)/(np.pi)
# #sqrt ((2l + 1)/4pi) is missing as a normalization factor
# #from all pyscf spherical harmonics
#
# #print('coeff', coeff)
# #coeff = orbitals.gto_norm(1, alpha) * orbitals.pyscf_gto_factor
# #print('coeff', coeff)
coords = mol1.atom_coords(unit='Bohr')
fakemol = gto.fakemol_for_charges(coords)
fakemol._bas[:,1] = 0 #set l = 1
fakemol._env[-1] = coeff
fakemol._env[-2] = alpha
ints = df.incore.aux_e2(mol2, fakemol)
print(ints.shape)
print('ints shape', ints.shape)
elef2 = -np.einsum('ijp,ij->p', ints, dm2)
print('e field dens 2', elef2)
# %%
# #using aux e2 to calculate e field from e potential calculation in pyscf
# #analytic integral for electric field of monomer 2 sites due to monomer 1
alpha = 1000000
coeff = 2 * np.sqrt(4/3) * alpha**(5/2)/(np.pi)
# #sqrt ((2l + 1)/4pi) is missing as a normalization factor
# #from all pyscf spherical harmonics
#
# #print('coeff', coeff)
# #coeff = orbitals.gto_norm(1, alpha) * orbitals.pyscf_gto_factor
# #print('coeff', coeff)
coords = mol2.atom_coords(unit='Bohr')
fakemol = gto.fakemol_for_charges(coords)
fakemol._bas[:,1] = 1 #set l = 1
fakemol._env[-1] = coeff
fakemol._env[-2] = alpha
ints = df.incore.aux_e2(mol1, fakemol)
print(ints.shape)
elef1 = -np.einsum('ijp,ij->p', ints, dm1)
elef1 = np.reshape(elef1, (-1, 3))
print('e field dens 1', elef1)
e_field_1_analytic = pos_e_field1 - elef1
print('e field 1 analytic', e_field_1_analytic)
# %%
# #using aux e2 to calculate e field from e potential calculation in pyscf
# #analytic integral for electric field of monomer 1 sites due to monomer 2
alpha = 1000000
coeff = 2 * np.sqrt(4/3) * alpha**(5/2)/(np.pi)
# #sqrt ((2l + 1)/4pi) is missing as a normalization factor
# #from all pyscf spherical harmonics
#
# #print('coeff', coeff)
# #coeff = orbitals.gto_norm(1, alpha) * orbitals.pyscf_gto_factor
# #print('coeff', coeff)
coords = mol1.atom_coords(unit='Bohr')
fakemol = gto.fakemol_for_charges(coords)
fakemol._bas[:,1] = 1 #set l = 1
fakemol._env[-1] = coeff
fakemol._env[-2] = alpha
ints = df.incore.aux_e2(mol2, fakemol)
print(ints.shape)
elef2 = -np.einsum('ijp,ij->p', ints, dm2)
elef2 = np.reshape(elef2, (-1, 3))
print('e field dens', elef2)
e_field_2_analytic = pos_e_field2 - elef2
print('e field 2 analytic', e_field_2_analytic)
# %%
# #density fitting projections for mol1 and mol2
auxbasis = 'augccpvqzjkfit'
auxmol1 = df.addons.make_auxmol(mol1, auxbasis)

ints_3c2e = df.incore.aux_e2(mol1, auxmol1, intor='int3c2e')
ints_2c2e = auxmol1.intor('int2c2e')
print('ints3c2e shape', ints_3c2e.shape)
print('ints2c2e shape', ints_2c2e.shape)

nao = mol1.nao
naux = auxmol1.nao
df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
df_coef = df_coef.reshape(naux, nao, nao)
df_coeffs1 = np.einsum('Pij,ij->P', df_coef, dm1)

auxmol2 = df.addons.make_auxmol(mol2, auxbasis)

ints_3c2e = df.incore.aux_e2(mol2, auxmol2, intor='int3c2e')
ints_2c2e = auxmol2.intor('int2c2e')
print('ints3c2e shape', ints_3c2e.shape)
print('ints2c2e shape', ints_2c2e.shape)

nao = mol2.nao
naux = auxmol2.nao
df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
df_coef = df_coef.reshape(naux, nao, nao)
df_coeffs2 = np.einsum('Pij,ij->P', df_coef, dm2)

# %%
# #analytic integral for electric field of monomer 2 sites due to monomer 1 using the DF density
alpha = 1000000
coeff = 2 * np.sqrt(4/3) * alpha**(5/2)/(np.pi)
# #sqrt ((2l + 1)/4pi) is missing as a normalization factor
# #from all pyscf spherical harmonics
#
# #print('coeff', coeff)
# #coeff = orbitals.gto_norm(1, alpha) * orbitals.pyscf_gto_factor
# #print('coeff', coeff)
coords = mol1.atom_coords(unit='Bohr')
fakemol = gto.fakemol_for_charges(coords)
fakemol._bas[:,1] = 1 #set l = 1
fakemol._env[-1] = coeff
fakemol._env[-2] = alpha
ints =  gto.mole.intor_cross('int2c2e', auxmol2, fakemol)
print('ints shape', ints.shape)
print('coeffs shape', df_coeffs2.shape)
elef_df2 = -np.einsum('ip, i -> p', ints, df_coeffs2)
elef_df2 = np.reshape(elef_df2, (-1, 3))
print('e field dens 2', elef_df2)
print('efield dens 2 diff', elef_df2 - elef2)
# %% [markdown]
# # Interlude: calculating overlap integral between two densities for exchange repulsion
# %%
# %cd equiv_dens
# %%
# loading data containing integral between unit charges and dipoles
with open("datasets/two_gaussians_overlap(2).pickle", "rb") as f:
    results1, results_2 = pickle.load(f)

print('results1', results1.keys())
vec_R_ab = results1['vec_R_ab']
mono_mono = results1['Mono, Mono']
dipx_dipx = results1['Dipx, Dipx']
dipx_dipy = results1['Dipx, Dipy']
dipy_dipy = results1['Dipy, Dipy']
dipz_dipz = results1['Dipz, Dipz']
print('mono_mono', mono_mono)
print('vec_R_ab', vec_R_ab)

# %%
# calculating analytic overlap integral between unit charges in pyscf
mono_mono = results1['Mono, Mono']
alpha_a = 1/4   #1/bohr^2
alpha_b = 1/9 #1/bohr^2
# #coeff_a = 2 * np.sqrt(4/3) * alpha**(5/2)/(np.pi)
coeff_a = 2 * alpha_a**(3/2)/(np.pi)
coeff_b = 2 * alpha_b**(3/2)/(np.pi)
# #num_points = 10
# #x_ab = np.linspace(0.3, 2, num_points)
# #vec_R_ab = np.zeros((num_points, 3))
# #vec_R_ab[:, 0] = x_ab
mol_a = gto.M(atom=[['H0', [0, 0, 0]]], basis={'H0': [[0, [alpha_a, 1.0]]]}, spin=None)
mol_a._env[-1] = coeff_a

intors = []
for i in range(vec_R_ab.shape[0]):
    mol_b = gto.M(atom=[['H0', vec_R_ab[i, :]]], basis={'H0': [[0, [alpha_b, 1.0]]]},
                  spin=None, unit='Bohr')
    mol_b._env[-1] = coeff_b

    intor = gto.mole.intor_cross('int1e_ovlp', mol_a, mol_b)
    intors.append(intor.item())
    # #print('R', vec_R_ab[i], 'ovlp', intor)
# print(intors)
# print(intors/mono_mono)

fig = plt.figure()
plt.plot(vec_R_ab[:, 0], np.log(mono_mono), label='qchem_ref')
plt.plot(vec_R_ab[:, 0], np.log(intors), label='pyscf_ovlp')
# #a, b, c = np.polyfit(vec_R_ab[:, 0], np.log(intors/mono_mono), 2)
# #print('a, b, c', a, b, c)
# #print(a**2)
# #print(1/(2*np.pi))
# #plt.plot(vec_R_ab[:, 0], np.log(intors/mono_mono), label='log ratio')
# #plt.plot(vec_R_ab[:, 0], quad(vec_R_ab[:, 0]), label='quadratic fit')
plt.legend()
plt.show()
# %%
# calculating analytic overlap integral between unit charges in pyscf, 2nd example
mono_mono = results_2['Mono, Mono']
alpha_a = 1   #1/bohr^2
alpha_b = 4/9 #1/bohr^2
# #coeff_a = 2 * np.sqrt(4/3) * alpha**(5/2)/(np.pi)
coeff_a = 2 * alpha_a**(3/2)/(np.pi)
coeff_b = 2 * alpha_b**(3/2)/(np.pi)
# #num_points = 10
# #x_ab = np.linspace(0.3, 2, num_points)
# #vec_R_ab = np.zeros((num_points, 3))
# #vec_R_ab[:, 0] = x_ab
mol_a = gto.M(atom=[['H0', [0, 0, 0]]], basis={'H0': [[0, [alpha_a, 1.0]]]}, spin=None)
mol_a._env[-1] = coeff_a

intors = []
for i in range(vec_R_ab.shape[0]):
    mol_b = gto.M(atom=[['H0', vec_R_ab[i, :]]], basis={'H0': [[0, [alpha_b, 1.0]]]},
                  spin=None, unit='Bohr')
    mol_b._env[-1] = coeff_b

    intor = gto.mole.intor_cross('int1e_ovlp', mol_a, mol_b)
    intors.append(intor.item())
    # #print('R', vec_R_ab[i], 'ovlp', intor)
# #print(intors)
print(intors/mono_mono)
#
fig = plt.figure()
plt.plot(vec_R_ab[:, 0], np.log(mono_mono), label='qchem_ref')
plt.plot(vec_R_ab[:, 0], np.log(intors), label='pyscf_ovlp')
# #a, b, c = np.polyfit(vec_R_ab[:, 0], np.log(intors/mono_mono), 2)
# #print('a, b, c', a, b, c)
# #print(a**2)
# #print(1/(2*np.pi))
# #plt.plot(vec_R_ab[:, 0], np.log(intors/mono_mono), label='log ratio')
# #plt.plot(vec_R_ab[:, 0], quad(vec_R_ab[:, 0]), label='quadratic fit')
plt.legend()
plt.show()
# %%
# calculating analytic overlap integral between unit dipoles in pyscf
dipx_dipx = results1['Dipx, Dipx']
alpha_a = 1/4   #1/bohr^2
alpha_b = 1/9 #1/bohr^2
coeff_a = 2 * np.sqrt(4/3) * alpha_a**(5/2)/(np.pi)
coeff_b = 2 * np.sqrt(4/3) * alpha_b**(5/2)/(np.pi)
# #coeff_a = 2 * alpha_a**(3/2)/(np.pi)
# #coeff_b = 2 * alpha_b**(3/2)/(np.pi)
# #num_points = 10
# #x_ab = np.linspace(0.3, 2, num_points)
# #vec_R_ab = np.zeros((num_points, 3))
# #vec_R_ab[:, 0] = x_ab
mol_a = gto.M(atom=[['H0', [0, 0, 0]]], basis={'H0': [[1, [alpha_a, 1.0]]]}, spin=None)
mol_a._env[-1] = coeff_a

intors = []
for i in range(vec_R_ab.shape[0]):
    mol_b = gto.M(atom=[['H0', vec_R_ab[i, :]]], basis={'H0': [[1, [alpha_b, 1.0]]]},
                  spin=None, unit='Bohr')
    mol_b._env[-1] = coeff_b

    intor = gto.mole.intor_cross('int1e_ovlp', mol_a, mol_b)
    # #print('intor.shape', intor.shape)
    intors.append(intor[0, 0])
    # #print('R', vec_R_ab[i], 'ovlp', intor)
# #print(intors)
print(intors/dipx_dipx)
#
fig = plt.figure()
plt.plot(vec_R_ab[:, 0], np.log(mono_mono), label='qchem_ref')
plt.plot(vec_R_ab[:, 0], np.log(intors), label='pyscf_ovlp')
# #a, b, c = np.polyfit(vec_R_ab[:, 0], np.log(intors/mono_mono), 2)
# #print('a, b, c', a, b, c)
# #print(a**2)
# #print(1/(2*np.pi))
# #plt.plot(vec_R_ab[:, 0], np.log(intors/mono_mono), label='log ratio')
# #plt.plot(vec_R_ab[:, 0], quad(vec_R_ab[:, 0]), label='quadratic fit')
plt.legend()
plt.show()

# %%
# overlap integral between DF densities of two water monomers
ovlp = gto.mole.intor_cross('int1e_ovlp', auxmol1, auxmol2)

ovlp_int = np.einsum('i, ij, j -> ', df_coeffs1, ovlp, df_coeffs2)
print('ovlp int', ovlp_int)

# %%
# overlap integral based between two monomer densities,
# one represented using a DF basis, and another using an MO basis
# first monomer is in MO basis, second in DF basis
ovlp = df.incore.aux_e2(mol1, auxmol2, intor='int3c1e')

ovlp_int = np.einsum('ijk, ij, k -> ', ovlp, dm1, df_coeffs2)
print('ovlp int', ovlp_int)
# %%
# overlap integral based between two monomer densities,
# one represented using a DF basis, and another using an MO basis
# first monomer is in DF basis, second in MO basis
ovlp = df.incore.aux_e2(mol2, auxmol1, intor='int3c1e')

ovlp_int = np.einsum('ijk, ij, k -> ', ovlp, dm2, df_coeffs1)
print('ovlp int', ovlp_int)
# %%
# overlap integral based between two monomer densities,
# both of the represented using an MO basis 
nbas1 = mol1.nbas
nbas2 = mol2.nbas
print('nbas1', nbas1)
print('nbas2', nbas2)
atmc, basc, envc = gto.mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                     mol2._atm, mol2._bas, mol2._env)
shls_slice = (0, nbas1, 0, nbas1, nbas1, nbas1+nbas2, nbas1, nbas1+nbas2)
#
print(shls_slice)
ovlp = gto.moleintor.getints('int4c1e_sph', atmc, basc, envc, shls_slice, None, 0)
print('ovlp shape', ovlp.shape)
ovlp_int = np.einsum('ijkp, ij, kp -> ', ovlp, dm1, dm2)
print('ovlp int', ovlp_int)

# %% [markdown]
# # End of interlude
# %%
# extracting electric field references
with open('./datasets/interaction_energies_adz.pickle', 'rb') as f:
    interaction_energies, mbd_energies = pickle.load(f)
with open('./datasets/efields_adz.pickle', 'rb') as f:
    efield_at_monomer1, efield_at_monomer2 = pickle.load(f)
with open('./datasets/sapt_adz.pickle', 'rb') as f:
    sapt_for_calc = pickle.load(f)
# %%
print(f"interatction energy between the dimer {interaction_energies[key]} kcal/mol using pbe with a aug-ccpvdz basis") 
print(f"mbd interatction energy between the dimer {mbd_energies[key]} kcal/mol using pbe with a aug-ccpvdz basis") 
print(f"electric field at monomer 1 sites due to monomer 2 in Har/(eBohr):\n {efield_at_monomer1[key]}")
print(f"electric field at monomer 2 sites due to monomer 1 in Har/(eBohr):\n {efield_at_monomer2[key]}")
# %%
print((efield_at_monomer1[key] - e_field_2_analytic) / efield_at_monomer1[key])
print((efield_at_monomer1[key] - e_field_2_analytic))
# %% [markdown]
# # Doing the same stuff with machine learned densities
# %%
# basic arguments for model loading
main_args = Namespace()

main_args.args_file = "args/qm7x250_dens_001_coreless.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
main_args.save_file = 'qm7x250_dens_001_coreless'
main_args.use_gpu = False
# %%
# #load arguments and dataset
args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)

args.fix_arguments = True

args, hyperparam_args, train_vars = train_utils.init_training_vars(args, hyperparam_args)
checkpoint = train_vars['checkpoint']

# #determine whether GPU is used for training

# #load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

args.verbose = 0
args.use_gpu = main_args.use_gpu
print('args use gpu', args.use_gpu)
args.expansion_constraint = None
args.integral_constraint = 'coeffs_in_coeff_net'
# #args.integral_constraint = None
args.ignore_missing_keywords = True

required_properties = ['density', 'dipole_moment']

args.spherical_grid_level = 1
args.cube_grid = False
grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
sampling_fn = partial(spherical_radial_sampling, rotate=False)
grid_origin = 0
grid_extent = None
args.radii_adjust = True
# #grid_vars = train_utils.init_grid_vars(args)
rotate = False

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file

# #args.np_dataset_test = "datasets/qm7x_test_dft_augccpvdz_small_base.npy"
# #args.dens_dataset_test = "datasets/qm7x_test_dft_augccpvdz_small.npy"
args.np_dataset_test = "datasets/s66x8_pyscf_augccpvdz_base.npy"
args.dens_dataset_test = "datasets/s66x8_pyscf_augccpvdz_calc.npy"

dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000000000000,
                           required_properties=required_properties,
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           pyscf_grid=True,
                           pyscf_rotate=rotate,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
                           calc_data=True,
                           radii_adjust=args.radii_adjust,
                           atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy',
                           atom_dens_type='mo_coeffs',
                           split_atom_dens=True,
                           density_grad=args.density_grad,
                           calc_basis_path='datasets/augccpvdz_orbital_basis.npy',
                           all_atom_numbers=np.array([1, 6, 7, 8, 16, 17]),
                           all_atom_coeffs=True,
                           )

print('dataset length', len(dataset))
samp = dataset.get_properties([0])
print('sample pos shape', samp['positions'].shape)
print('sample dens shape', samp['density'].shape)
print('sample dens integral', torch.sum(samp['density'][0] * samp['coord_weights'][0]))
print('args use gpu', args.use_gpu)

# %%
# #evaluate model and test density integral
model = model_loader.load_model(args, dataset)
for param in model.parameters():
    param.requires_grad = False
idx = [0, 3]
samp = dataset.get_properties(idx)
if args.use_gpu:
    for key in samp.keys():
        if isinstance(samp[key], torch.Tensor):
            samp[key] = samp[key].cuda()

res = model(samp)

print('sum atom numbers', torch.sum(samp['batch_atom_numbers'], dim=1))
print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
print('true density integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
print('density error', torch.sum(torch.abs(res['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['batch_atom_numbers'], dim=1))
# %%
# convert molecules to input format suited for ML model and generate results
input1 = {'atom_numbers': mol1.atom_charges(), 'positions': mol1.atom_coords(unit='Angstrom')}
input2 = {'atom_numbers': mol2.atom_charges(), 'positions': mol2.atom_coords(unit='Angstrom')}
samp1 = orbitals.model_input_from_atoms(input1,
                                         density_expansion=False,
                                         skip_compress=True,
                                         grid_spec=None,
                                         cutoff=args.cutoff,
                                         dtype=torch.float32,
                                         atom_dens_type="mo_coeffs",
                                         free_atom_densities=dataset.atom_dens,
                                         split_atom_densities=False,
                                         basis=None,
                                         all_atom_coeffs=True,
                                         coord_params=None,
                                         )
samp2 = orbitals.model_input_from_atoms(input2,
                                         density_expansion=False,
                                         skip_compress=True,
                                         grid_spec=None,
                                         cutoff=args.cutoff,
                                         dtype=torch.float32,
                                         atom_dens_type="mo_coeffs",
                                         free_atom_densities=dataset.atom_dens,
                                         split_atom_densities=False,
                                         basis=None,
                                         all_atom_coeffs=True,
                                         coord_params=None,
                                         )
res1 = model.density_repr_model(samp1)
res2 = model.density_repr_model(samp2)
# %%
# extract ML density coefficients and free atom coefficients for samples
i = 0
df_coeffs_ml1 = orbitals.coeffs_dict_to_vector(res1, dataset.orbital_basis_num, res1['batch_atom_numbers'],
                                                radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

auxmol_ml1 = orbitals.ml_basis_to_auxmol(res1, i, skip_zero=False)

fa_mol1 = utils.npy_to_pyscf(samp1['batch_positions'].numpy(),
                         samp1['batch_atom_numbers'].numpy(),
                         basis=samp1['atom_mo_coeffs_basis'], build=True, skip_zero=False)[0]

fa_dm1 = hf.make_rdm1(mo_coeff=samp1['atom_mo_coeffs'][0],
                         mo_occ=samp1['atom_mo_coeffs_occ'][0],)

df_coeffs_ml2 = orbitals.coeffs_dict_to_vector(res2, dataset.orbital_basis_num, res2['batch_atom_numbers'],
                                                radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].detach()

auxmol_ml2 = orbitals.ml_basis_to_auxmol(res2, i, skip_zero=False)

fa_mol2 = utils.npy_to_pyscf(samp2['batch_positions'].numpy(),
                         samp2['batch_atom_numbers'].numpy(),
                         basis=samp2['atom_mo_coeffs_basis'], build=True, skip_zero=False)[0]
fa_dm2 = hf.make_rdm1(mo_coeff=samp2['atom_mo_coeffs'][0],
                         mo_occ=samp2['atom_mo_coeffs_occ'][0],)

c1 = mol2.atom_coords(unit='Bohr')
c2 = mol1.atom_coords(unit='Bohr')

# %%
# calculate electric field values from ml densities and compare to reference
alpha = 1000000
coeff = 2 * np.sqrt(4/3) * alpha**(5/2)/(np.pi)
# #sqrt ((2l + 1)/4pi) is missing as a normalization factor
# #from all pyscf spherical harmonics
#
# #print('coeff', coeff)
# #coeff = orbitals.gto_norm(1, alpha) * orbitals.pyscf_gto_factor
# #print('coeff', coeff)
fakemol1 = gto.fakemol_for_charges(c1)
fakemol1._bas[:,1] = 1 #set l = 1
fakemol1._env[-1] = coeff
fakemol1._env[-2] = alpha

fakemol2 = gto.fakemol_for_charges(c2)
fakemol2._bas[:,1] = 1 #set l = 1
fakemol2._env[-1] = coeff
fakemol2._env[-2] = alpha

int_df = gto.mole.intor_cross('int2c2e', auxmol_ml1, fakemol1)
int_mo = df.incore.aux_e2(fa_mol1, fakemol1)
print('int mo shape', int_mo.shape)
print('fa dm shape', fa_dm1.shape)
elef_ml_df1 = -np.einsum('ip, i -> p', int_df, df_coeffs_ml1[0])
elef_ml_df1 = np.reshape(elef_ml_df1, (-1, 3))
elef_ml_mo1 = -np.einsum('ijp,ij->p', int_mo, fa_dm1)
elef_ml_mo1 = np.reshape(elef_ml_mo1, (-1, 3))
elef_ml1 = elef_ml_df1 + elef_ml_mo1
print('elef ml 1', elef_ml1)
print('elef 1', elef1)
print('elef ml 1 diff', elef_ml1 - elef1)

int_df = gto.mole.intor_cross('int2c2e', auxmol_ml2, fakemol2)
int_mo = df.incore.aux_e2(fa_mol2, fakemol2)
print('int mo shape', int_mo.shape)
print('fa dm shape', fa_dm2.shape)
elef_ml_df2 = -np.einsum('ip, i -> p', int_df, df_coeffs_ml2[0])
elef_ml_df2 = np.reshape(elef_ml_df2, (3, -1))
elef_ml_mo2 = -np.einsum('ijp,ij->p', int_mo, fa_dm2)
elef_ml_mo2 = np.reshape(elef_ml_mo2, (3, -1))
elef_ml2 = elef_ml_df2 + elef_ml_mo2
print('elef ml 2', elef_ml2)
print('elef 2', elef2)
print('elef ml 2 diff', elef_ml2 - elef2)
# %%
# calculate density overlap integrals for ML densities
# four terms: overlap between ml coeffs, overlap between free atom densities, and overlap between free atom and ML densities x2

# overlap between free atom densities
nbas1 = fa_mol1.nbas
nbas2 = fa_mol2.nbas
print('nbas1', nbas1)
print('nbas2', nbas2)
atmc, basc, envc = gto.mole.conc_env(fa_mol1._atm, fa_mol1._bas, fa_mol1._env,
                                     fa_mol2._atm, fa_mol2._bas, fa_mol2._env)
shls_slice = (0, nbas1, 0, nbas1, nbas1, nbas1+nbas2, nbas1, nbas1+nbas2)
#
print(shls_slice)
ovlp_mo = gto.moleintor.getints('int4c1e_sph', atmc, basc, envc, shls_slice, None, 0)
print('ovlp_mo shape', ovlp_mo.shape)
ovlp_int_mo = np.einsum('ijkp, ij, kp -> ', ovlp_mo, fa_dm1, fa_dm2)
print('ovlp mo int', ovlp_int_mo)
# %%
# overlap integral between ML densities
ovlp_ml = gto.mole.intor_cross('int1e_ovlp', auxmol_ml1, auxmol_ml2)
print('ovlp ml', ovlp_ml.shape)

ovlp_int_ml = np.einsum('i, ij, j -> ', df_coeffs_ml1[0], ovlp_ml, df_coeffs_ml2[0])
print('ovlp ML int', ovlp_int_ml)

# %%
# overlap integral based between two monomer densities,
# one represented using a DF basis, and another using an MO basis
# first monomer is in MO basis, second in DF basis
ovlp_mo_ml = df.incore.aux_e2(fa_mol1, auxmol_ml2, intor='int3c1e')

ovlp_int_mo_ml = np.einsum('ijk, ij, k -> ', ovlp_mo_ml, fa_dm1, df_coeffs_ml2[0])
print('ovlp MO ML int', ovlp_int_mo_ml)
# %%
# overlap integral based between two monomer densities,
# one represented using a DF basis, and another using an MO basis
# first monomer is in DF basis, second in MO basis
ovlp_int_ml_mo = df.incore.aux_e2(fa_mol2, auxmol_ml1, intor='int3c1e')

ovlp_int_ml_mo = np.einsum('ijk, ij, k -> ', ovlp_int_ml_mo, fa_dm2, df_coeffs_ml1[0])
print('ovlp ML MO int', ovlp_int_ml_mo)
# %%
ovlp_total = ovlp_int_mo_ml + ovlp_int_ml_mo + ovlp_int_ml + ovlp_int_mo
print('ovlp total', ovlp_total)
