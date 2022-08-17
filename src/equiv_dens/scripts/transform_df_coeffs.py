import numpy as np
import torch
from argparse import Namespace


convention_dict = {
    # 'orca_631Gss': Namespace(
    #     atom_to_orbitals_map={'H': 'ssp', 'O': 'sspspd', 'C': 'sspspd', 'N': 'sspspd'},
    #     orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
    #     orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
    #     orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 3, 2, 4, 5], 'C': [0, 1, 3, 2, 4, 5], 'N': [0, 1, 3, 2, 4, 5]},
    # ),
    # 'orca_def2-SVP': Namespace(
    #     atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
    #     orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
    #     orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
    #     orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    # ),
    # 'def2-SVP_to_pyscf_201': Namespace(
    #     atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
    #     orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4]},
    #     orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
    #     orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    # ),
    # 'def2-SVP_to_pyscf_210': Namespace(
    #     atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
    #     orbital_idx_map={'s': [0], 'p': [2, 1, 0], 'd': [0, 1, 2, 3, 4]},
    #     orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
    #     orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    # ),
    # 'aims': Namespace(
    #     atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
    #     orbital_idx_map={'s': [0], 'p': [0, 1, 2], 'd': [0, 1, 2, 3, 4]},
    #     orbital_sign_map={'s': [1], 'p': [1, 1, -1], 'd': [1, 1, 1, -1, 1]},
    #     orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    # ),
    # 'psi4_basic': Namespace(
    #     atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
    #     orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
    #     orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
    #     orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    # ),
    # 'psi4_lh': Namespace(
    #     atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
    #     orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
    #     orbital_sign_map={'s': [1], 'p': [-1, 1, 1], 'd': [-1, -1, 1, 1, 1]},
    #     orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    # ),
    # 'psi4_cs': Namespace(
    #     atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
    #     orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
    #     orbital_sign_map={'s': [1], 'p': [-1, 1, -1], 'd': [1, -1, 1, -1, 1]},
    #     orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    # ),
    # 'psi4_lh_cs': Namespace(
    #     atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
    #     orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
    #     orbital_sign_map={'s': [1], 'p': [1, 1, -1], 'd': [-1, 1, 1, -1, 1]},
    #     orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    # ),
    'pyscf_augccpvqzjkfit': Namespace(
            atom_to_orbitals_map={1: [0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,4,4],
                                  8: [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                                  6: [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                                  7: [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                                  16: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,5,5]},
            # orbital_idx_map={0: [0], 1: [2, 0, 1], 2: [4, 2, 0, 1, 3], 3: [6, 4, 2, 0, 1, 3, 5], 4: [8, 6, 4, 2, 0, 1, 3, 5, 7], 5: [10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9]},
            # orbital_idx_map={0: [0], 1: [1, 2, 0], 2: [2, 3, 1, 4, 0], 3: [3, 4, 2, 5, 1, 6, 0], 4: [4, 5, 3, 6, 2, 7, 1, 8, 0], 5: [5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 0]},
            orbital_idx_map={0: [0], 1: [1, 2, 0], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8], 5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            orbital_sign_map={0: [1], 1: [-1, -1, -1], 2: [1, 1, 1, 1, 1], 3: [-1, -1, -1, -1, -1, -1, -1], 4: [1, 1, 1, 1, 1, 1, 1, 1, 1], 5: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]},
            orbital_order_map={1: [], 8: [], 6: [], 7: [], 16:[]},
    ),
    'ml_dft': Namespace(
            atom_to_orbitals_map={1: [0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,4,4],
                                  8: [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                                  6: [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                                  7: [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,5,5],
                                  16: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,5,5]},
            orbital_idx_map={0: [0], 1: [0, 1, 2], 2: [0, 1, 2, 3, 4], 3: [0, 1, 2, 3, 4, 5, 6], 4: [0, 1, 2, 3, 4, 5, 6, 7, 8], 5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        orbital_sign_map={0: [1], 1: [1, 1, 1], 2: [1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1, 1], 4: [1, 1, 1, 1, 1, 1, 1, 1, 1], 5: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
        orbital_order_map={1: [], 8: [], 6: [], 7: [], 16:[]},
    ),
}


def transform(df_coeffs, atoms, convention='pyscf_augccpvqzjkfit'):
    conv = convention_dict[convention]
    orbitals = []
    orbitals_order = []
    atom_numbers, _ = torch.max(atoms, dim=0)
    # print('df coeffs shape', df_coeffs.shape)
    # print('atom numbers shape', atom_numbers.shape)
    # print('atom_numbers', atom_numbers)
    for i in range(len(atom_numbers)):
        at = int(atom_numbers[i])
        offset = len(orbitals_order)
        orbitals_map = conv.atom_to_orbitals_map[at]
        orbital_order_map = conv.orbital_order_map[at]
        if orbital_order_map == []:
            orbital_order_map = list(range(len(orbitals_map)))
        orbitals += orbitals_map
        orbitals_order += [idx + offset for idx in orbital_order_map]
    
    # print('offset', offset)
    # print('orbitals', orbitals)
    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(torch.LongTensor(map_idx) + offset)
        transform_signs.append(torch.Tensor(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = torch.cat(transform_indices)
    transform_signs = torch.cat(transform_signs).to(df_coeffs)

    # print('transform_indices', transform_indices)
    # print(transform_indices.shape)
    df_coeffs_new = df_coeffs[:, transform_indices]
    df_coeffs_new = df_coeffs_new * transform_signs

    return df_coeffs_new


def transform_back(df_coeffs, atoms, convention='pyscf_augccpvqzjkfit'):
    conv = convention_dict[convention]
    base = convention_dict['ml_dft']
    print('atoms', atoms)
    orbitals = []
    orbitals_order = []
    for a in atoms:
        at = int(a)
        offset = len(orbitals_order)
        orbitals_map = base.atom_to_orbitals_map[at]
        orbital_order_map = conv.orbital_order_map[at]
        if orbital_order_map == []:
            orbital_order_map = list(range(len(orbitals_map)))
        else:
            orbital_order_map = inverse_permutation(orbital_order_map)
        orbitals += orbitals_map
        orbitals_order += [idx + offset for idx in orbital_order_map]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        # print(map_idx)
        map_idx = inverse_permutation(map_idx)
        print(map_idx)
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(torch.LongTensor(map_idx) + offset)
        transform_signs.append(torch.LongTensor(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = torch.cat(transform_indices)
    transform_signs = torch.cat(transform_signs)
    print('transform_indices', transform_indices)
    print('transform_signs', transform_signs)

    df_coeffs_new = df_coeffs[:, transform_indices]
    df_coeffs_new = df_coeffs_new * transform_signs
    return df_coeffs_new

def inverse_permutation(perm):
    inverse_perm = [0] * len(perm)
    for i, ind in enumerate(perm):
        inverse_perm[ind] = i
    return inverse_perm
