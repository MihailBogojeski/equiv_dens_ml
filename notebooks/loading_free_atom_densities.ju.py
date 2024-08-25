# %%
from pyscf import gto
import numpy as np
from pyscf.dft import gen_grid, radi, numint
import ase.io
# %%

# %load_ext autoreload
# %autoreload 2
# %cd ..

# %%
# # loading from npy
# mols = []
# mols_npy = np.load("datasets/ethanol_train_10.npy", allow_pickle=True).item()
# for i in range(mols_npy['positions'].shape[0]):
#     positions = mols_npy['positions'][i]
#     anums = mols_npy['atom_numbers'][i]
#     atoms = [(anums[i], positions[i]) for i in range(len(anums))]
#     print(atoms)
#     mols.append(atoms)

# %%
# loading from xyz
mols = []
mols_ase = ase.io.iread('datasets/ethanol_train_10.xyz')
for mol in mols_ase:
    positions = mol.get_positions()
    anums = mol.get_atomic_numbers()
    atoms = [(anums[i], positions[i]) for i in range(len(anums))]
    mols.append(atoms)
# %%
free_atom_dict = np.load('datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy', allow_pickle=True).item()

basis = 'augccpvdz'
pyscf_mols = []
free_atom_densities = []
for atoms in mols:
    # create full molecule in order to generate density integration grid
    mol = gto.M(atom=atoms, basis=basis)
    # generate integration grid
    grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=1)
    coords, weights = gen_grid.get_partition(mol, grid_spec)
    single_atom_dens = []
    for atom in atoms:
        z = atom[0]
        free_atom_basis = free_atom_dict[z]['mo_basis']
        # create molecule object from single atom
        at = gto.M(atom=[atom], basis=free_atom_basis, spin=None)
        ao = numint.eval_ao(at, coords, deriv=0)
        xctype = 'LDA'
        # calculate density using precomputed spherically  averaged free atom MO coefficients
        coeffs = {'mo_coeff': free_atom_dict[z]['mo_coeff'],
                  'mo_occ': free_atom_dict[z]['mo_occ']}
        rho = numint.eval_rho2(mol, ao, xctype=xctype, **coeffs)
        single_atom_dens.append(rho)
    # create dict containing free atom densities of the individual atoms evaluated on the
    # same integradtion grid, including the coordinates and integrations weights of the grid
    dens_dict = {'density': np.stack(single_atom_dens, axis=0),
                 'coords': coords, 'weights': weights}
    free_atom_densities.append(dens_dict)


for dens_dict in free_atom_densities:
    print(dens_dict['density'].shape)
    # to obtain the full free atom density for the molecule, sum the atom densities along the first dimension
    print('density integral', np.sum(np.sum(dens_dict['density'], axis=0) * dens_dict['weights']))
