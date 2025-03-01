# %%
import numpy as np
import scipy as sp
import torch
from pyscf.scf import hf
from equiv_dens.data import HamiltonianDataset, seeded_random_split
from pyscf import gto
import tqdm
from equiv_dens.utils.base import calc_dict_to_npy


MOLECULE = "water"

if MOLECULE == "water":
    db_path = '../../../datasets/h2o_pbe-def2svp_4999.db'
    split_sizes = [500, 500, 3999]
elif MOLECULE == "ethanol":
    db_path = '../../../datasets/ethanol_pbe-def2svp_30000.db'
    split_sizes = [25000, 500, 4500]
elif MOLECULE == "mda-enol":
    db_path = '../../../datasets/mda-enol_pbe-def2svp_26978.db'
    split_sizes = [25000, 500, 1478]
elif MOLECULE == "uracil":
    db_path = '../../../datasets/uracil_pbe-def2svp_30000.db'
    split_sizes = [25000, 500, 4500]
elif MOLECULE == "aspirin":
    db_path = '../../../datasets/aspirin_pbe-def2svp_30000.db'
    split_sizes = [25000, 500, 4500]
else:
    raise ValueError(f"Unknown molecule: {MOLECULE}")


atom_symbols = ['n', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']

print("---------------------------------")
dataset = HamiltonianDataset(db_path)
print(len(dataset))
print(dataset.database.Z)

splits = seeded_random_split(dataset, split_sizes, seed=0)
print(f"{len(splits)} splits")
train, valid, test = splits
print(len(train), len(valid), len(test))

print("---------------------------------")

basis = 'orca_def2svp'
split_names = ['train', 'valid', 'test']
for split_name, split_data in zip(split_names, splits):
    data_loader = torch.utils.data.DataLoader(split_data, batch_size=1, collate_fn=dataset.collate_fn)

    npy_path = f'../../../datasets/{MOLECULE}_{split_name}_{basis}.npy'
    calc_path = f"../../../datasets/{MOLECULE}_{split_name}_{basis}_hm_dm_oe_calc.npy"

    # data_dict = {'energy': None, 'forces': None, 'positions': None, 'atom_numbers': None, 'atom_types': None}
    N = len(split_data)

    print(f"split: {split_name}, N: {N}")

    calc_results = []

    for batch in tqdm.tqdm(data_loader):

        # atom data
        b_energy = batch['energy']
        # if data_dict['energy'] is None:
        #     data_dict['energy'] = np.array(b_energy)
        # else:
        #     data_dict['energy'] = np.concatenate((data_dict['energy'], b_energy), axis=0)

        b_forces = batch['forces']
        # if data_dict['forces'] is None:
        #     data_dict['forces'] = np.array(b_forces)
        # else:
        #     data_dict['forces'] = np.concatenate((data_dict['forces'], b_forces), axis=0)

        b_positions = np.array(batch['positions'])
        b_positions *= 0.5291772105638411  # to angstrom
        # if data_dict['positions'] is None:
        #     data_dict['positions'] = b_positions
        # else:
        #     data_dict['positions'] = np.concatenate((data_dict['positions'], b_positions), axis=0)

        # if data_dict['atom_numbers'] is None:
        #     data_dict['atom_numbers'] = np.repeat(dataset.database.Z[None, :], N, axis=0)
        b_atom_numbers = np.repeat(dataset.database.Z[None, :], N, axis=0)
        
        # if data_dict['atom_types'] is None:
        #     data_dict['atom_types'] = [np.array(dataset.database.Z) for _ in range(N)]
        b_atom_types = [np.array(dataset.database.Z) for _ in range(N)]

        # mol_dict
        mol_dict = {'atom': [], 'basis': basis, 'unit': 'angstrom'}
        atom_types = b_atom_types[0]
        pos = np.array(b_positions[0])
        mol_dict['atom'] = [(atom_symbols[atom_types[j]], pos[j, :]) for j in range(len(atom_types))]
        # print(mol_dict)

        # calc_dict
        calc_dict = {}
        hm = batch['full_hamiltonian'][0]
        om = batch['overlap_matrix'][0]

        mo_energies, mo_coeff = sp.linalg.eig(hm, om)
        mo_energies = np.real(mo_energies)
        mo_coeff = np.real(mo_coeff)
        # print(f"mo_energies: {mo_energies.shape}, mo_coeff: {mo_coeff.shape}")

        n_electrons = b_atom_numbers[0].sum()
        mo_occ = np.zeros_like(mo_energies)
        mo_occ[:n_electrons//2] = 2
        # print(f"mo_occ: {mo_occ.shape}")

        dm = hf.make_rdm1(mo_coeff, mo_occ)

        # print(f"dm: {dm.shape}")
        calc_dict['atom_numbers'] = np.array(dataset.database.Z)
        calc_dict['atom_types'] = np.array(dataset.database.Z)
        calc_dict['positions'] = np.array(b_positions[0])
        calc_dict['forces'] = np.array(b_forces[0])
        calc_dict['energy'] = np.array(b_energy[0])

        calc_dict['mo_coeff'] = mo_coeff
        calc_dict['mo_energies'] = mo_energies
        calc_dict['mo_occ'] = mo_occ
        calc_dict['hamiltonian_matrix'] = np.array(hm)
        calc_dict['density_matrix'] = dm

        # for k in calc_dict.keys():
        #     print(f"{k}: {type(calc_dict[k])} {calc_dict[k].shape}")

        calc_results.append((mol_dict, calc_dict))


    # print(data_dict['energy'].shape)
    # print(data_dict['forces'].shape)
    # print(data_dict['positions'].shape)
    # print(data_dict['atom_numbers'].shape)
    # print(len(data_dict['atom_types']), data_dict['atom_types'][0].shape)

    # for prop in dataset_npy.keys():
    #     print(f"type of {prop}: {type(dataset_npy[prop])} vs {type(data_dict[prop])}")


    data_dict = calc_dict_to_npy(calc_results,
                                 convert_forces=False,
                                 compress_atoms=True)

    np.save(npy_path, data_dict, allow_pickle=True)
    np.save(calc_path, calc_results, allow_pickle=True)


# import numpy as np
# print("compare units")

# h2o_small = np.load('../../../datasets/h2o_small_train_augccpvdz.npy', allow_pickle=True).item()
# h2o = np.load('../../../datasets/h2o_train_augccpvdz.npy', allow_pickle=True).item()
# print(f"laoded datasets, h2o_small: {len(h2o_small['energy'])}, h2o: {len(h2o['energy'])}")

# print("energy")
# print(f"mean energy of h2o_small: {np.mean(h2o_small['energy'])}")
# print(f"vairance energy of h2o_small: {np.var(h2o_small['energy'])}")
# print(f"mean energy of h2o: {np.mean(h2o['energy'])}")
# print(f"vairance energy of h2o: {np.var(h2o['energy'])}")

# print("forces")
# print(f"mean forces of h2o_small: {np.mean(np.abs(h2o_small['forces']), axis=0)}")
# print(f"vairance forces of h2o_small: {np.var(h2o_small['forces'], axis=0)}")
# print(f"mean forces of h2o: {np.mean(np.abs(h2o['forces']), axis=0)}")
# print(f"vairance forces of h2o: {np.var(h2o['forces'], axis=0)}")

# print("positions")
# print(f"mean positions of h2o_small: {np.mean(np.abs(h2o_small['positions']), axis=0)}")
# print(f"vairance positions of h2o_small: {np.var(h2o_small['positions'], axis=0)}")
# print(f"mean positions of h2o: {np.mean(np.abs(h2o['positions']), axis=0)}")

# %%

# from equiv_dens.utils.base import calc_dict_to_npy
# import numpy as np

# MOLECULE = "waterd"
# splits = ['train', 'valid', 'test']
# calc_path = f"../../../datasets/water_train_orca_def2svp_hm_dm_oe_calc.npy"
# calc_results = np.load(calc_path, allow_pickle=True)
# print(f"calc props: {calc_results[0][1].keys()}")
# print(len(calc_results))

# np_path = f"../../../datasets/h2o_train_orca_def2svp.npy"
# np_data = calc_dict_to_npy(calc_results)
# for k in np_data.keys():
#     if isinstance(np_data[k], np.ndarray):
#         print(f"prop {k}, array shape: {np_data[k].shape}")
#     elif isinstance(np_data[k], list):
#         print(f"prop {k}, list len: {len(np_data[k])}")
#     else:
#         print(f"prop {k}, type: {type(np_data[k])}")
# %%
