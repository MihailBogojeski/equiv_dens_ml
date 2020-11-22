import numpy as np
import schnetpack as spk
import argparse
import transform_hamiltonians
from ase import data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dis_data', type=str, help='Path to data in db format')
    parser.add_argument('npy_data', type=str, help='Path to npy database.')
    parser.add_argument('convention', type=str, help='Type of formatting convention to use.')
    args = parser.parse_args()

    distorted = np.load(args.dis_data).item()
    # Get all data in array form

    # Get atom types
    hamiltonians, nonzero_indices = transform_hamiltonians.transform(distorted['hamiltonians'], convention=args.convention)

    npy_data = {
        'positions': distorted['positions'],
        'hamiltonians': hamiltonians,
        'overlaps': distorted['overlaps'],
        'nonzero_indices': nonzero_indices,
        'atom_types': distorted['atom_types']
    }

    np.save(args.npy_data, npy_data)
