import numpy as np
import torch
from argparse import Namespace

from torch.nn.functional import assert_int_or_pair


convention_dict = {
    'orca_def2svp': Namespace(
        atom_to_orbitals_map={1:  [0,0,1],
                              6:  [0,0,0,1,1,2],
                              7:  [0,0,0,1,1,2],
                              8:  [0,0,0,1,1,2],
                            },
        orbital_idx_map={0: [0], 1: [2, 0, 1], 2: [4, 2, 0, 1, 3]},
        orbital_sign_map={i: [1] * ((2 * i) + 1) for i in range(3)},
        orbital_order_map={i: [] for i in range(119)},
    ),
    'pyscf_augccpvqzjkfit': Namespace(
        atom_to_orbitals_map={1:  [0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,4,4],
                              8:  [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                              6:  [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                              7:  [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                              16: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,5,5],
                              17: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,5,5],
                             },
        orbital_idx_map={0: [0], 1: [2, 0, 1], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8], 5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        orbital_sign_map={i: [(-1)**i] * ((2 * i) + 1) for i in range(6)},
        orbital_order_map={i: [] for i in range(119)},
    ),
    'pyscf_augccpvdz': Namespace(
        atom_to_orbitals_map={1:  [0,0,0,1,1],
                              8:  [0,0,0,0,1,1,1,2,2],
                              6:  [0,0,0,0,1,1,1,2,2],
                              7:  [0,0,0,0,1,1,1,2,2],
                              16: [0,0,0,0,0,1,1,1,1,2,2],
                              17: [0,0,0,0,0,1,1,1,1,2,2],
                             },
        orbital_idx_map={0: [0], 1: [2, 0, 1], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8], 5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        orbital_sign_map={i: [(-1)**i] * ((2 * i) + 1) for i in range(6)},
        orbital_order_map={i: [] for i in range(119)},
    ),
    'ml_dft_augccpvqzjkfit': Namespace(
        atom_to_orbitals_map={1:  [0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,4,4],
                              8:  [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                              6:  [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                              7:  [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                              16: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,5,5],
                              17: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,5,5],
                             },
        orbital_idx_map={i: list(range(2 * i + 1)) for i in range(6)},
        orbital_sign_map={i: [1] * ((2 * i) + 1) for i in range(6)},
        orbital_order_map={i: [] for i in range(119)},
    ),
    'ml_dft_augccpvdz': Namespace(
        atom_to_orbitals_map={1:  [0,0,0,1,1],
                              8:  [0,0,0,0,1,1,1,2,2],
                              6:  [0,0,0,0,1,1,1,2,2],
                              7:  [0,0,0,0,1,1,1,2,2],
                              16: [0,0,0,0,0,1,1,1,1,2,2],
                              17: [0,0,0,0,0,1,1,1,1,2,2],
                             },
        orbital_idx_map={0: [0], 1: [2, 0, 1], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8], 5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        orbital_sign_map={i: [1] * ((2 * i) + 1) for i in range(6)},
        orbital_order_map={i: [] for i in range(119)},
    ),
}


def prepare_transform(atom_numbers, convention, to_internal=False):
    conv = convention_dict[convention]
    if to_internal == True:
        base = convention_dict['ml_dft_' + convention.split('_')[1]]
    else:
        base = conv
    atom_numbers, _ = torch.max(atom_numbers, dim=0)
    # print('atom_numbers', atom_numbers)
    orbitals = []
    orbitals_order = []
    for i in range(len(atom_numbers)):
        at = int(atom_numbers[i])
        offset = len(orbitals_order)
        orbitals_map = base.atom_to_orbitals_map[at]
        orbital_order_map = conv.orbital_order_map[at]
        if orbital_order_map == []:
            orbital_order_map = list(range(len(orbitals_map)))
        elif to_internal:
            orbital_order_map = inverse_permutation(orbital_order_map)
        orbitals += orbitals_map
        orbitals_order += [idx + offset for idx in orbital_order_map]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        # print(map_idx)
        if to_internal:
            map_idx = inverse_permutation(map_idx)
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(torch.LongTensor(map_idx) + offset)
        transform_signs.append(torch.LongTensor(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = torch.cat(transform_indices)
    transform_signs = torch.cat(transform_signs).unsqueeze(0)
    
    return transform_indices, transform_signs


def inverse_permutation(perm):
    inverse_perm = [0] * len(perm)
    for i, ind in enumerate(perm):
        inverse_perm[ind] = i
    return inverse_perm


def convert_ao(ao_coeffs, atom_numbers,
               convention='pyscf_augccpvqzjkfit',
               to_internal=False):
    """
    Convert atomic_orbitals from internal convention to some target convention

    Args:
        ao_coeffs: coefficients of atom centered orbital basis 
        atom_number: array containing atom numbers for each molecule in the batch 
        convention: target convention
        to_internal: if True, convert back from target convention to internal convention
    """
    transform_indices, transform_signs =\
        prepare_transform(atom_numbers, convention, to_internal)

    if transform_signs.ndim < ao_coeffs.ndim:
        transform_signs = transform_signs.unsqueeze(-1)
    print('ao coeffs shape', ao_coeffs.shape)
    print('transform indices shape', transform_indices.shape)
    print('transform signs shape', transform_signs.shape)
    ao_coeffs_new = ao_coeffs[:, transform_indices]
    ao_coeffs_new = ao_coeffs_new * transform_signs.to(ao_coeffs)

    return ao_coeffs_new


def convert_ao_matrix(ao_matrix, atom_numbers,
                      convention='pyscf_augccpvdz',
                      to_internal=False):
    """
    Convert operator matrix projected on atomic orbitals from internal convention to some target convention

    Args:
        ao_matrix: operator matrix projected on atomic orbitals  
        atom_number: array containing atom numbers for each molecule in the batch 
        convention: target convention
        to_internal: if True, convert back from target convention to internal convention
    """
    transform_indices, transform_signs =\
        prepare_transform(atom_numbers, convention, to_internal)

    # print('ao_matrix shape', ao_matrix.shape)
    ao_matrix_new = ao_matrix[:, transform_indices, :]
    ao_matrix_new = ao_matrix_new[:, :, transform_indices]
    ao_matrix_new = ao_matrix_new * transform_signs.unsqueeze(-1).to(ao_matrix_new)
    ao_matrix_new = ao_matrix_new * transform_signs.unsqueeze(1).to(ao_matrix_new)
    # print('ao_matrix old', ao_matrix[0, 0, :])
    # print('ao_matrix new', ao_matrix_new[0, 0, :])

    return ao_matrix_new
