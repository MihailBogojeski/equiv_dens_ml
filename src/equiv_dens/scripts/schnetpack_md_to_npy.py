from schnetpack.md.data import HDF5Loader
import numpy as np
import sys
import ase.io
import equiv_dens.utils.base as utils

file = sys.argv[1]
every = int(sys.argv[2])

data = HDF5Loader(file, load_properties=False)
n_runs = data.structure.n_molecules
positions = data.properties['_positions'][::every] * 10
atom_numbers = data.properties['_atomic_numbers']
positions = np.reshape(positions, (positions.shape[0], n_runs, -1, positions.shape[-1]))
atom_numbers = np.reshape(atom_numbers, (n_runs, -1))
print('positions.shape', positions.shape)
print('atom_numbers.shape', atom_numbers.shape)

for i in range(positions.shape[1]):
    atoms = {'positions': positions[:, i], 'atom_numbers': atom_numbers[i, :]}
    np.save(file[:-5] + '_' + str(i) + '.npy', atoms)
