import numpy as np
import argparse
import transform_hamiltonians
from ase import data
from equiv_dens.data.hamiltonian_dataset import HamiltonianDataset
import torch
from equiv_dens.utils.orbitals import orbitals_from_hamiltonian
from pyscf import gto, df, lib
import equiv_dens.utils.base as utils
from pyscf.scf import hf
from pyscf.gto import mole
import scipy as sp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('db_data', type=str, help='Path to data in db format')
    parser.add_argument('npy_data', type=str, help='Path to npy database.')
    parser.add_argument('orbital_data', type=str, help='Path to orbital coeffs data file.')
    parser.add_argument('convention', type=str, help='Type of formatting convention to use.')
    parser.add_argument('--basis', type=str, default='ccpvdz', help='Basis for the orbitals.')
    parser.add_argument('--auxbasis', type=str, default='augccpvqzjkfit', help='Basis for the orbitals.')
    parser.add_argument('--total_size', type=int, default=3000, help='Size of the dataset.')
    args = parser.parse_args()

    # Load data
    dataset = HamiltonianDataset(args.db_data)
    inds = np.arange(len(dataset))
    shuff_inds = np.random.choice(inds, size=(args.total_size,), replace=False)
    atom_numbers = np.array(dataset.database.Z)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=lambda batch: dataset.collate_fn(batch))
    # h_data = list(loader)[0]
    h_data = dataset.collate_fn(shuff_inds.tolist())
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
    print('hamiltonians shape', h_data['full_hamiltonian'].shape)
    print(h_data.keys())
    h_data['positions'] = utils.bohr_to_angstrom(h_data['positions'])
    # h_data['positions'] -= h_data['positions'][:, [0], :]
    if args.convention != '':
        hamiltonians = transform_hamiltonians.transform(h_data['full_hamiltonian'].numpy(), atom_types, convention=args.convention)
        overlaps = transform_hamiltonians.transform(h_data['overlap_matrix'].numpy(), atom_types, convention=args.convention)
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
    npy_data['atom_numbers'] = np.tile(atom_numbers, (h_data['positions'].shape[0], 1))
    print('atom numbers shape', npy_data['atom_numbers'].shape)
    # print('atom numbers', npy_data['atom_numbers'])
    npy_data['orbitals'] = dataset.orbitals

    np.save(args.npy_data, npy_data, allow_pickle=True)

    orbital_coeffs, _ = orbitals_from_hamiltonian(hamiltonians, overlaps)
    n_electrons = np.sum(atom_numbers)
    n_occ = n_electrons // 2
    occ_orbitals = np.zeros((hamiltonians.shape[1], ))
    occ_orbitals[:n_occ] = 2
    print('len orbital coeffs', len(orbital_coeffs))

    pyscf_dicts = []
    for i in range(len(orbital_coeffs)):
        mol = gto.M(atom=[(atom_numbers[i], h_data['positions'][0, i, :].numpy()) for i in range(len(atom_numbers))], basis=args.basis)
        mol_dict = mol.pack()
        mo_coeff = np.array(orbital_coeffs[i])
        mo_occ = occ_orbitals
        # print('mo_coeff shape', mo_coeff.shape)
        # print('mo_occ shape', mo_occ.shape)

        mol_dict = mol.pack()
        mol_dict['atom'] = [(atom_types[j], h_data['positions'][i, j, :].numpy()) for j in range(len(atom_types))]
        calc_dict = {}
        calc_dict['mo_coeff'] = orbital_coeffs[i]
        calc_dict['mo_occ'] = occ_orbitals
        calc_dict['energy'] = h_data['energy'][i]
        calc_dict['forces'] = h_data['forces'][i]

        mol = mole.unpack(mol_dict)
        mol.build()
        dm1 = hf.make_rdm1(mo_coeff, mo_occ)
        # print('density matrix', dm1.shape) 

        # Define the auxiliary fitting basis for 3-center integrals. Use the function
        # make_auxmol to construct the auxiliary Mole object (auxmol) which will be
        # used to generate integrals.
        auxbasis = 'augccpvqzjkfit'
        auxmol = df.addons.make_auxmol(mol, auxbasis)

        # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
        # indices of AO basis and P is the auxiliary basis
        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
        ints_2c2e = auxmol.intor('int2c2e')

        nao = mol.nao
        naux = auxmol.nao

        # Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
        df_coef = sp.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)
        
        mol.basis = auxbasis 
        mol.build()
        calc_dict['df_coeff'] = df_basis
        calc_dict['auxbasis'] = auxbasis
        pyscf_dicts.append((mol_dict, calc_dict))

    np.save(args.orbital_data, pyscf_dicts, allow_pickle=True)
