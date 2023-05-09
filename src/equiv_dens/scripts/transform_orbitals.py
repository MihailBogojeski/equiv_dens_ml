import numpy as np


orca_orbital_idx_map = {'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]}
orca_orbital_sign_map = {'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]}
orca_atom_to_orbitals_map = {'H': 'ssp', 'O': 'sspspd', 'C': 'sspspd', 'N': 'sspspd'}
orca_orbital_order_map = {'H': [0, 1, 2], 'O': [0, 1, 3, 2, 4, 5], 'C': [0, 1, 3, 2, 4, 5], 'N': [0, 1, 3, 2, 4, 5]}

svp_orbital_idx_map = {'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]}
svp_orbital_sign_map = {'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]}
svp_atom_to_orbitals_map = {'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'}
svp_orbital_order_map = {'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]}


def transform(orbitals, atoms, convention='orca'):
    if convention == 'orca':
        transformed_orbitals = transform_from_orca(orbitals, atoms)
    elif convention == 'svp':
        transformed_orbitals = transform_from_svp(orbitals, atoms)
    else:
        raise Exception('Unsupported convention for conversion')

    return transformed_orbitals


# def transform_from_orca(orbitals):
#     orbitals = np.transpose(orbitals, axes=(1, 2, 0))  # j, i, batch
#     # orbitals[2:6, :, :] = orbitals[[5, 2, 3, 4], :, :]
#     # orbitals[:, 2:6, :] = orbitals[:, [5, 2, 3, 4], :]
#
#     orbitals_new = np.zeros((3 * 14, 3 * 14, orbitals.shape[2]))
#     mapping = [
#         (np.arange(2), np.arange(2)),
#         (np.arange(2, 6), np.array([5, 4, 2, 3])),  # move s orbital and rearange p orbitals
#         (np.arange(6, 9), np.array([8, 6, 7])),  # rearrange p orbitals
#         (np.arange(9, 14), np.array([13, 11, 9, 10, 12])),  # rearrange d orbitals
#         (np.array([14, 15, 17, 18, 19]), np.array([14, 15, 18, 16, 17])),
#         (np.array([28, 29, 31, 32, 33]), np.array([19, 20, 23, 21, 22]))
#     ]
#     for i_out, i_in in mapping:
#         for j_out, j_in in mapping:
#             print(np.meshgrid(i_out, j_out))
#             orbitals_new[tuple(np.meshgrid(i_out, j_out))] = orbitals[tuple(np.meshgrid(i_in, j_in))]
#     orbitals_new = np.transpose(orbitals_new, axes=(2, 0, 1))  # batch, i, j
#     nonzero_indices = np.concatenate([out for out, _in in mapping])
#
#     orbitals_new = orbitals_new[:, nonzero_indices][:, :, nonzero_indices]
#
#     return orbitals_new, nonzero_indices


def transform_from_orca(orbitals, atoms):
    orbitals = ''
    for a in atoms:
        orbitals += orca_atom_to_orbitals_map[a]

    print('orbitals', orbitals)

    transform_indices = np.array([])
    transform_signs = np.array([])
    for orb in orbitals:
        offset = len(transform_indices)
        map_idx = orca_orbital_idx_map[orb]
        map_sign = orca_orbital_sign_map[orb]
        transform_indices = np.concatenate((transform_indices, np.array(map_idx) + offset))
        transform_signs = np.concatenate((transform_signs, np.array(map_sign)))

    print('transform_indices', transform_indices)
    transform_indices = transform_indices.astype(np.int32)

    orbitals_new = orbitals[:, :, transform_indices]
    orbitals_new = orbitals_new * transform_signs

    return orbitals_new


def transform_from_svp(orbitals, atoms):
    print('atoms', atoms)
    orbitals = ''
    for a in atoms:
        print('svp aroms to orbs', svp_atom_to_orbitals_map[a])
        orbitals += svp_atom_to_orbitals_map[a]

    print('orbitals', orbitals)

    transform_indices = np.array([])
    transform_signs = np.array([])
    for orb in orbitals:
        offset = len(transform_indices)
        map_idx = svp_orbital_idx_map[orb]
        map_sign = svp_orbital_sign_map[orb]
        transform_indices = np.concatenate((transform_indices, np.array(map_idx) + offset))
        transform_signs = np.concatenate((transform_signs, np.array(map_sign)))

    print('transform_indices', transform_indices)
    transform_indices = transform_indices.astype(np.int32)

    orbitals_new = orbitals[:, :, transform_indices]
    orbitals_new = orbitals_new * transform_signs

    return orbitals_new


# def transform_back(orbitals, convention='orca'):
#     if convention == 'orca':
#         transformed_orbitals, nonzero_indices = transform_to_orca(orbitals)
#     else:
#         raise Exception('Unsupported convention for conversion')
#
#     return transformed_orbitals, nonzero_indices
#
#
# def transform_to_orca(orbitals):
#     orbitals = np.transpose(orbitals, axes=(1, 2, 0))  # j, i, batch
#     # orbitals[2:6, :, :] = orbitals[[5, 2, 3, 4], :, :]
#     # orbitals[:, 2:6, :] = orbitals[:, [5, 2, 3, 4], :]
#
#     orbitals_new = np.zeros((24, 24, orbitals.shape[2]))
#     mapping = [
#         (np.arange(2), np.arange(2)),
#         (np.arange(2, 6), np.array([4, 5, 3, 2])),  # move s orbital and rearange p orbitals
#         (np.arange(6, 9), np.array([7, 8, 6])),  # rearrange p orbitals
#         (np.arange(9, 14), np.array([11, 12, 10, 13, 9])),  # rearrange d orbitals
#         (np.arange(14, 19), np.array([14, 15, 17, 18, 16])),
#         (np.arange(19, 24), np.array([19, 20, 22, 23, 21]))
#     ]
#     for i_out, i_in in mapping:
#         for j_out, j_in in mapping:
#             print(np.meshgrid(i_out, j_out))
#             orbitals_new[tuple(np.meshgrid(i_out, j_out))] = orbitals[tuple(np.meshgrid(i_in, j_in))]
#     orbitals_new = np.transpose(orbitals_new, axes=(2, 0, 1))  # batch, i, j
#     nonzero_indices = np.concatenate([out for out, _in in mapping])
#
#     orbitals_new = orbitals_new[:, nonzero_indices][:, :, nonzero_indices]
#
#     return orbitals_new, nonzero_indices
