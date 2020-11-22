import numpy as np


orca_orbital_idx_map = {'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]}
orca_orbital_sign_map = {'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]}
orca_atom_to_orbitals_map = {'H': 'ssp', 'O': 'sspspd', 'C': 'sspspd', 'N': 'sspspd'}
orca_orbital_order_map = {'H': [0, 1, 2], 'O': [0, 1, 3, 2, 4, 5], 'C': [0, 1, 3, 2, 4, 5], 'N': [0, 1, 3, 2, 4, 5]}

svr_orbital_idx_map = {'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]}
svr_orbital_sign_map = {'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]}
svr_atom_to_orbitals_map = {'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'}
svr_orbital_order_map = {'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]}


def transform(hamiltonians, atoms, convention='orca'):
    if convention == 'aims':
        transformed_hamiltonians = transform_from_aims(hamiltonians)
    elif convention == 'orca':
        transformed_hamiltonians = transform_from_orca(hamiltonians, atoms)
    elif convention == 'svr':
        transformed_hamiltonians = transform_from_svr(hamiltonians, atoms)

    return transformed_hamiltonians


def transform_from_aims(hamiltonians):
    hamiltonians = np.transpose(hamiltonians, axes=(1, 2, 0))  # j, i, batch

    hamiltonians_new = np.zeros((3 * 14, 3 * 14, hamiltonians.shape[2]))
    mapping = [
        (np.arange(14), np.arange(14)),
        (np.array([14, 15, 17, 18, 19]), np.arange(14, 14 + 5)),
        (np.array([28, 29, 31, 32, 33]), np.arange(14 + 5, 14 + 5 + 5))
    ]
    for i_out, i_in in mapping:
        for j_out, j_in in mapping:
            print(np.meshgrid(i_out, j_out))
            hamiltonians_new[tuple(np.meshgrid(i_out, j_out))] = hamiltonians[tuple(np.meshgrid(i_in, j_in))]
    hamiltonians_new = np.transpose(hamiltonians_new, axes=(2, 0, 1))  # batch, i, j
    nonzero_indices = np.concatenate([out for out, _in in mapping])

    # Change of signs
    # (1, 1, -1) for l=1
    # (1, 1, 1, -1, 1) for l=2
    hamiltonians_new = hamiltonians_new.reshape(-1, 3, 14, 3, 14)
    for i in [5, 8, 12]:
        hamiltonians_new[:, :, i, :, :] = -hamiltonians_new[:, :, i, :, :]
        hamiltonians_new[:, :, :, :, i] = -hamiltonians_new[:, :, :, :, i]
    hamiltonians_new = hamiltonians_new.reshape(-1, 3 * 14, 3 * 14)
    hamiltonians_new = hamiltonians_new[:, nonzero_indices][:, :, nonzero_indices]

    return hamiltonians_new

# def transform_from_orca(hamiltonians):
#     hamiltonians = np.transpose(hamiltonians, axes=(1, 2, 0))  # j, i, batch
#     # hamiltonians[2:6, :, :] = hamiltonians[[5, 2, 3, 4], :, :]
#     # hamiltonians[:, 2:6, :] = hamiltonians[:, [5, 2, 3, 4], :]
#
#     hamiltonians_new = np.zeros((3 * 14, 3 * 14, hamiltonians.shape[2]))
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
#             hamiltonians_new[tuple(np.meshgrid(i_out, j_out))] = hamiltonians[tuple(np.meshgrid(i_in, j_in))]
#     hamiltonians_new = np.transpose(hamiltonians_new, axes=(2, 0, 1))  # batch, i, j
#     nonzero_indices = np.concatenate([out for out, _in in mapping])
#
#     hamiltonians_new = hamiltonians_new[:, nonzero_indices][:, :, nonzero_indices]
#
#     return hamiltonians_new, nonzero_indices


def transform_from_orca(hamiltonians, atoms):
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
    transform_indices = transform_indices.astype(np.int)

    hamiltonians_new = hamiltonians[:, transform_indices, :]
    hamiltonians_new = hamiltonians_new[:, :, transform_indices]
    hamiltonians_new = hamiltonians_new * transform_signs[:, None]
    hamiltonians_new = hamiltonians_new * transform_signs[None, :]

    return hamiltonians_new


def transform_from_svr(hamiltonians, atoms):
    print('atoms', atoms)
    orbitals = ''
    for a in atoms:
        print('svr aroms to orbs', svr_atom_to_orbitals_map[a])
        orbitals += svr_atom_to_orbitals_map[a]

    print('orbitals', orbitals)

    transform_indices = np.array([])
    transform_signs = np.array([])
    for orb in orbitals:
        offset = len(transform_indices)
        map_idx = svr_orbital_idx_map[orb]
        map_sign = svr_orbital_sign_map[orb]
        transform_indices = np.concatenate((transform_indices, np.array(map_idx) + offset))
        transform_signs = np.concatenate((transform_signs, np.array(map_sign)))

    print('transform_indices', transform_indices)
    transform_indices = transform_indices.astype(np.int)

    hamiltonians_new = hamiltonians[:, transform_indices, :]
    hamiltonians_new = hamiltonians_new[:, :, transform_indices]
    hamiltonians_new = hamiltonians_new * transform_signs[:, None]
    hamiltonians_new = hamiltonians_new * transform_signs[None, :]

    return hamiltonians_new


def transform_back(hamiltonians, convention='orca'):
    if convention == 'aims':
        transformed_hamiltonians, nonzero_indices = transform_to_aims(hamiltonians)
    if convention == 'orca':
        transformed_hamiltonians, nonzero_indices = transform_to_orca(hamiltonians)

    return transformed_hamiltonians, nonzero_indices


def transform_to_aims(hamiltonians):
    hamiltonians = np.transpose(hamiltonians, axes=(1, 2, 0))  # j, i, batch

    hamiltonians_new = np.zeros((3 * 14, 3 * 14, hamiltonians.shape[2]))
    mapping = [
        (np.arange(14), np.arange(14)),
        (np.array([14, 15, 17, 18, 19]), np.arange(14, 14 + 5)),
        (np.array([28, 29, 31, 32, 33]), np.arange(14 + 5, 14 + 5 + 5))
    ]
    for i_out, i_in in mapping:
        for j_out, j_in in mapping:
            print(np.meshgrid(i_out, j_out))
            hamiltonians_new[tuple(np.meshgrid(i_out, j_out))] = hamiltonians[tuple(np.meshgrid(i_in, j_in))]
    hamiltonians_new = np.transpose(hamiltonians_new, axes=(2, 0, 1))  # batch, i, j
    nonzero_indices = np.concatenate([out for out, _in in mapping])

    # Change of signs
    # (1, 1, -1) for l=1
    # (1, 1, 1, -1, 1) for l=2
    hamiltonians_new = hamiltonians_new.reshape(-1, 3, 14, 3, 14)
    for i in [5, 8, 12]:
        hamiltonians_new[:, :, i, :, :] = -hamiltonians_new[:, :, i, :, :]
        hamiltonians_new[:, :, :, :, i] = -hamiltonians_new[:, :, :, :, i]
    hamiltonians_new = hamiltonians_new.reshape(-1, 3 * 14, 3 * 14)
    hamiltonians_new = hamiltonians_new[:, nonzero_indices][:, :, nonzero_indices]

    return hamiltonians_new, nonzero_indices


def transform_to_orca(hamiltonians):
    hamiltonians = np.transpose(hamiltonians, axes=(1, 2, 0))  # j, i, batch
    # hamiltonians[2:6, :, :] = hamiltonians[[5, 2, 3, 4], :, :]
    # hamiltonians[:, 2:6, :] = hamiltonians[:, [5, 2, 3, 4], :]

    hamiltonians_new = np.zeros((24, 24, hamiltonians.shape[2]))
    mapping = [
        (np.arange(2), np.arange(2)),
        (np.arange(2, 6), np.array([4, 5, 3, 2])),  # move s orbital and rearange p orbitals
        (np.arange(6, 9), np.array([7, 8, 6])),  # rearrange p orbitals
        (np.arange(9, 14), np.array([11, 12, 10, 13, 9])),  # rearrange d orbitals
        (np.arange(14, 19), np.array([14, 15, 17, 18, 16])),
        (np.arange(19, 24), np.array([19, 20, 22, 23, 21]))
    ]
    for i_out, i_in in mapping:
        for j_out, j_in in mapping:
            print(np.meshgrid(i_out, j_out))
            hamiltonians_new[tuple(np.meshgrid(i_out, j_out))] = hamiltonians[tuple(np.meshgrid(i_in, j_in))]
    hamiltonians_new = np.transpose(hamiltonians_new, axes=(2, 0, 1))  # batch, i, j
    nonzero_indices = np.concatenate([out for out, _in in mapping])

    hamiltonians_new = hamiltonians_new[:, nonzero_indices][:, :, nonzero_indices]

    return hamiltonians_new, nonzero_indices
