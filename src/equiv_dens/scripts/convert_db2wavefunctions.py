import numpy as np
import argparse
import transform_hamiltonians
from ase import data
from equiv_dens.data.hamiltonian_dataset import HamiltonianDataset
import torch
from equiv_dens.utils.orbitals import orbitals_from_hamiltonian
from pyscf import gto
import equiv_dens.utils.base as utils

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

    # Get atom types

    print('keys', h_data.keys())
    atom_types = ''.join([data.chemical_symbols[i] for i in atom_numbers])
    print('atom_numbers', atom_numbers)
    print('atom_types', atom_types)
    print('hamiltonians shape', h_data['full_hamiltonian'].shape)
    print(h_data.keys())
    h_data['positions'] = utils.bohr_to_angstrom(h_data['positions'])
    h_data['positions'] -= h_data['positions'][:, [0], :]
    if args.convention != '':
        hamiltonians = transform_hamiltonians.transform(h_data['full_hamiltonian'].numpy(), atom_types, convention=args.convention)
        overlap = transform_hamiltonians.transform(h_data['overlap_matrix'].numpy(), atom_types, convention=args.convention)
    else:
        hamiltonians = h_data['full_hamiltonian']
        overlaps = h_data['overlap_matrix']

    npy_data = {}

    h_data.pop('full_hamiltonian', None)
    h_data.pop('core_hamiltonian', None)
    h_data.pop('overlap_matrix', None)
    for key in h_data.keys():
        print(key)
        if isinstance(h_data[key], np.ndarray):
            npy_data[key] = h_data[key]
        else:
            npy_data[key] = h_data[key].numpy()
    npy_data['atom_types'] = atom_types
    npy_data['atom_numbers'] = atom_numbers
    npy_data['orbitals'] = dataset.orbitals

    np.save(args.npy_data, npy_data, allow_pickle=True)

    orbital_coeffs = orbitals_from_hamiltonian(hamiltonians, overlap)
    n_electrons = np.sum(atom_numbers)
    n_occ = n_electrons // 2
    occ_orbitals = np.zeros((hamiltonians.shape[1], ))
    occ_orbitals[:n_occ] = 2
    print('len orbital coeffs', len(orbital_coeffs))

    mol = gto.M(atom=[(atom_types[i], h_data['positions'][0, i, :].numpy()) for i in range(len(atom_types))], basis=args.basis)
    mol_dict = mol.pack()
    print('mol_dict', mol_dict)

    pyscf_dicts = []
    for i in range(len(orbital_coeffs)):
        mol_dict = mol.pack()
        mol_dict['atom'] = [(atom_types[j], h_data['positions'][i, j, :].numpy()) for j in range(len(atom_types))]
        calc_dict = {}
        calc_dict['mo_coeff'] = orbital_coeffs[i]
        calc_dict['mo_occ'] = occ_orbitals
        pyscf_dicts.append((mol_dict, calc_dict))

    np.save(args.orbital_data, pyscf_dicts, allow_pickle=True)
