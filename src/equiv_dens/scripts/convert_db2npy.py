import numpy as np
import schnetpack as spk
import argparse
import transform_hamiltonians
from ase import data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('db_data', type=str, help='Path to data in db format')
    parser.add_argument('npy_data', type=str, help='Path to npy database.')
    parser.add_argument('convention', type=str, help='Type of formatting convention to use.')
    args = parser.parse_args()

    matrix_types = ['hamiltonian', 'overlap', 'core_hamiltonian']

    write_filetype = args.npy_data.split('.')[-1]

    # Load data
    dataset = spk.data.AtomsData(args.db_data, load_only=['hamiltonian', 'overlap'])
    loader = spk.data.AtomsLoader(dataset, batch_size=len(dataset))
    h_data = list(loader)[0]
    # loader = spk.data.AtomsLoader(dataset, batch_size=10)
    #
    # # Get all data in array form
    # for dat in loader:
    #     h_data = dat
    #     break

    # Get atom types
    atom_types = h_data['_atomic_numbers'][0].numpy()
    print('atom_types', atom_types)
    print('keys', h_data.keys())
    atom_types = ''.join([data.chemical_symbols[i] for i in atom_types])
    print('atom_types', atom_types)
    print('hamiltonians shape', h_data['hamiltonian'].shape)
    for mat in matrix_types:
        if mat in h_data.keys():
            h_data[mat] = transform_hamiltonians.transform(h_data[mat].numpy(), atom_types, convention=args.convention)

    npy_data = {}
    for key in h_data.keys():
        if isinstance(h_data[key], np.ndarray):
            npy_data[key] = h_data[key]
        else:
            npy_data[key] = h_data[key].numpy()
    npy_data['positions'] = h_data['_positions'].numpy()
    npy_data['atom_types'] = atom_types
    npy_data['basisdef'] = dataset.get_metadata('basisdef')

    if write_filetype == 'npz':
        np.savez(args.npy_data, **npy_data, allow_pickle=False)
    else:
        np.save(args.npy_data, npy_data, allow_pickle=True)
