from schnetpack.md.data import HDF5Loader
import numpy as np
import sys
import ase.io
import equiv_dens.utils.base as utils

file = sys.argv[1]
every = int(sys.argv[2])

data = HDF5Loader(file, load_properties=False)
print('data.properties', data.properties['_positions'].shape)
positions = data.properties['_positions'][::every] * 10
n_mols = data.n_molecules
positions = np.reshape(positions, (positions.shape[0], n_mols, -1, 3))
atom_numbers = np.reshape(data.properties['_atomic_numbers'], (n_mols, -1))
print('positions.shape', positions.shape)
print('atom_numbers.shape', atom_numbers.shape)
print('atom_numbers', atom_numbers)
for i in range(positions.shape[1]):
    pos = positions[:, i]
    mols = utils.npy_to_ase(pos, atom_numbers)
    ase.io.write(file[:-5] + '_' + str(i) + '.xyz', mols)
    print('positions', i, pos)
