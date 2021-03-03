import numpy as np
# import schnetpack as spk
import argparse
import transform_hamiltonians
from ase import data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_data', type=str, help='Path to data in npy or npz format')
    parser.add_argument('convert_data', type=str, help='Path to npy database.')
    parser.add_argument('convention', type=str, help='Type of formatting convention to use.')
    args = parser.parse_args()

    matrix_types = ['hamiltonian', 'overlap', 'hamiltonian_core']

    read_filetype = args.npy_data.split('.')[-1]
    write_filetype = args.convert_data.split('.')[-1]
    if read_filetype == 'npz':
        h_data = dict(np.load(args.npy_data))
    else:
        h_data = np.load(args.npy_data).item()
    # Get all data in array form

    atom_types = h_data['atomic_numbers'][0]
    print('atom_types', atom_types)
    print('keys', h_data.keys())
    atom_types = ''.join([data.chemical_symbols[i] for i in atom_types])
    print('atom_types', atom_types)
    print('hamiltonians shape', h_data['hamiltonian'].shape)

    for mat in matrix_types:
        if mat in h_data.keys():
            h_data[mat] = transform_hamiltonians.transform(h_data[mat], atom_types, convention=args.convention)

    # Get atom types
    npy_data = {}
    for key in h_data.keys():
        if isinstance(h_data[key], np.ndarray):
            npy_data[key] = h_data[key]
        else:
            npy_data[key] = h_data[key].numpy()
    npy_data['atom_types'] = atom_types

    if write_filetype == 'npz':
        np.savez(args.convert_data, **npy_data, allow_pickle=False)
    else:
        np.save(args.convert_data, npy_data, allow_pickle=True)
