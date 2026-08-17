import equiv_dens.utils.base as utils
import numpy as np
import sys
import ase.io

file = sys.argv[1]

data = dict(np.load(file, allow_pickle=True))

print('data keys', data.keys())
print('atomic numbers', data['_atomic_numbers'])
print('positions shape', data['_positions'].shape)
print('positions', data['_positions'][0])
symbols = utils.numbers_to_symbols(data['_atomic_numbers'].squeeze())
mol = utils.npy_to_ase(data['_positions'].squeeze()[:, 1] * 10, symbols)

ase.io.write(file[:-3] + 'xyz', mol)
