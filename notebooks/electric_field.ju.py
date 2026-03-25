# %%
import numpy as np
import pyscf
from pyscf import gto, dft
from pyscf.dft import gen_grid, radi, numint
import matplotlib.pyplot as plt
hf.MUTE_CHKFILE = True

# %%
# %cd ..
# %load_ext autoreload
# %autoreload 2
# %%
ang2bohr = 1.88973
bohr2ang = 0.529177
Har2eV = 27.2114
eV2Har = 1/ Har2eV
V_Ang_to_Har_eBohr = 1/ (Har2eV * ang2bohr)
Har_eBohr_to_V_Ang = 1/V_Ang_to_Har_eBohr
print('ev2 har', eV2Har)
print('V_Ang_to_Har_eBohr', V_Ang_to_Har_eBohr)
print('Har_eBohr_to_V_Ang', Har_eBohr_to_V_Ang) 

# %%
mol = gto.M(atom = 'H 0 0 -0.37; H 0 0 0.37', basis = 'augccpvdz', spin=0)
num_space = 5 
z = np.linspace(2.37, 4.37, num_space)
positions = np.zeros((num_space, 3))
positions[:,2] = z
print("Check that we can calculate the electric field at the following positions:")
print(positions)
qchem_ref_field_z =np.array([0.00359850446,
                             0.00146483278,
                             0.00072567754,
                             0.00040284318,
                             0.00024133503])
qchem_ref_field = np.zeros((5,3))
qchem_ref_field[:,2] = qchem_ref_field_z
print("Here is the field at those locations in V/Ang")
print(qchem_ref_field)

# %%
fig, ax = plt.subplots()
ax.scatter(z, qchem_ref_field_z)
plt.show()

# %%
mf = dft.RKS(mol)
# don't write an output file for calculations
mf.chkfile = False
mf.xc = 'pbe'
mf.kernel()


grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=9)
coords, weights = gen_grid.get_partition(mol, grid_spec)
ao = numint.eval_ao(mol, coords, deriv=1)
rho = numint.eval_rho(mol, ao, dm, xctype='GGA')
print(rho.shape)
print('rho int', np.sum(rho * weights))

pos = positions * ang2bohr
print('positions', pos)

# %%
atom_coords = mol.atom_coords(unit='Bohr')
charges = mol.atom_charges()
at_diffs = pos[None, :] - atom_coords[:, None]
at_dists = np.linalg.norm(at_diffs, axis=-1, keepdims=True)

pos_e_field = = np.sum(np.reshape(charges, (-1, 1, 1)) * at_diffs / at_dists**3, axis=0)
print('positive e field', pos_e_field[:, 2])
# %%
rho_diffs = pos[None, :] - coords[:, None]

rho_dists = np.linalg.norm(rho_diffs, axis=-1, keepdims=True)
print('min dists', np.min(rho_dists, axis=(0,2)))

neg_contrib = rho[:, None, None] * weights[:, None, None] * rho_diffs / rho_dists**3


neg_e_field = np.sum(neg_contrib, axis=0)
print('negative e field', neg_e_field[:, 2])
# %%
e_field = pos_e_field - neg_e_field
print('e field', e_field[:, 2])
print('qchem ref', qchem_ref_field_z)
print('corr between e field and qchem ref', np.corrcoef(qchem_ref_field_z, e_field[:, 2]))

fig, axs = plt.subplots(4, 1, figsize=(5, 15))
axs[0].scatter(z, pos_e_field[:, 2])
axs[1].scatter(z, neg_e_field[:, 2])
axs[2].scatter(z, e_field[:, 2])
axs[3].scatter(z, qchem_ref_field_z)
plt.show()
# %%
# e_field = electrostatic_energy of the gradient of the density

