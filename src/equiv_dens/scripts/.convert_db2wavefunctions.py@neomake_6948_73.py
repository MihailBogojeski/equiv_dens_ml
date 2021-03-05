import numpy as np
import argparse
import transform_hamiltonians
from ase import data
from equiv_dens.data.hamiltonian_dataset import HamiltonianDataset
import torch
from equiv_dens.utils.orbitals import orbitals_from_hamiltonian
from pyscf import gto

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('db_data', type=str, help='Path to data in db format')
    parser.add_argument('npy_data', type=str, help='Path to npy database.')
    parser.add_argument('orbital_data', type=str, help='Path to orbital coeffs data file.')
    parser.add_argument('convention', type=str, help='Type of formatting convention to use.')
    parser.add_argument('--basis', type=str, help='Basis for the orbitals.')
    args = parser.parse_args()

    # Load data
    dataset = HamiltonianDataset(args.db_data)
    atom_numbers = np.array(dataset.database.Z)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), collate_fn=lambda batch: dataset.collate_fn(batch))
    h_data = list(loader)[0]
    # loader = spk.data.AtomsLoader(dataset, batch_size=10)
    #
    # # Get all data in array form
    # for dat in loader:
    #     h_data = dat
    #     break
    print('data keys', h_data.keys())

    # Get atom types

    print('keys', h_data.keys())
    atom_types = ''.join([data.chemical_symbols[i] for i in atom_types])
    print('atom_types', atom_types)
    print('hamiltonians shape', h_data['hamiltonian'].shape)
    print(h_data.keys())
    if args.convention != '':
        hamiltonians = transform_hamiltonians.transform(h_data['hamiltonian'].numpy(), atom_types, convention=args.convention)
        overlap = transform_hamiltonians.transform(h_data['overlap'].numpy(), atom_types, convention=args.convention)
    else:
        hamiltonians = h_data['hamiltonian']
        overlaps = h_data['overlap']

    npy_data = {}

    h_data.pop('hamiltonian', None)
    h_data.pop('overlap', None)
    for key in h_data.keys():
        print(key)
        if isinstance(h_data[key], np.ndarray):
            npy_data[key] = h_data[key]
        else:
            npy_data[key] = h_data[key].numpy()
    npy_data['positions'] = h_data['_positions'].numpy()
    npy_data['atom_types'] = atom_types
    npy_data['basisdef'] = dataset.get_metadata('basisdef')

    np.savez(args.npy_data, **npy_data, allow_pickle=False)

    orbital_coeffs, orbital_occ = orbitals_from_hamiltonian(hamiltonians, overlap)

    mol = gto.M(atom=[(atom_types[i], h_data['_positions'][0, i, :]) for i in range(len(atom_types))], basis=args.basis)
    mol_dict = mol.pack()

    pyscf_dicts = []
    for i in range(orbital_coeffs):
        for j in range(len(mol_dict['atom'])):
            mol_dict['atom'][j][1] = h_data['_positions'][i, j, :].numpy()
        calc_dict = {}
        calc_dict['mo_coeff'] = orbital_coeffs[i]
        calc_dict['mo_occ'] = orbital_occ[i]
        pyscf_dicts.append(mol_dict, calc_dict)

    np.save(args.orbital_data, pyscf_dicts, allow_pickle=True)
