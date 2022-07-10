import numpy as np
import scipy
import ase
import ase.io
import torch
import ase.data
import pyscf

to_bohr = 1/pyscf.lib.param.BOHR
to_angstrom = pyscf.lib.param.BOHR 


def angstrom_to_bohr(pos):
    return pos * to_bohr


def bohr_to_angstrom(pos):
    return pos * to_angstrom


def angstrom_to_angstrom(pos):
    return pos


def hartree_to_kcal(en):
    return en * 627.5096080305927


def millihartree_to_kcal(en):
    return en * 0.6275096080305927


def eV_to_kcal(en):
    return en * 23.060541945329334


def kelvin_to_kcal(en):
    return en * 0.001987191686485529


# def hartree_to_eV(en):
#     return en * 27.2116
#
#
# def eV_to_hartree(en):
#     return en / 27.2116


def kcal_to_kcal(en):
    return en


def kcal_to_hartree(en):
    return en / 627.5096080305927


def kcal_to_millihartree(en):
    return en / 0.6275096080305927


def kcal_to_eV(en):
    return en / 23.060541945329334


def kcal_to_kelvin(en):
    return en / 0.001987191686485529


def random_rotation_matrix():
    """
    Generates a random 3D rotation matrix from axis and angle.
    Args:
        numpy_random_state: numpy random state object
    Returns:
        Random rotation matrix.
    """

    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-8
    theta = 2 * np.pi * np.random.uniform(0.0, 1.0)
    return rotation_matrix(axis, theta)


def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))


def to_basis3d(X, n2, d=None):
    # Input:
    # X (n, d) matrix where the 3d axis are flattend into last axis
    # n2 is n/2, half the number of basis functions to use along one axis
    dx = d[0]
    dy = d[1]
    dz = d[2]
    n2x = n2[0]
    n2y = n2[1]
    n2z = n2[2]
    X2 = X.reshape(-1, dx, dy, dz)
    X2 = np.fft.rfft(X2, axis=-1)[:, :, :, :n2z]
    X2 = np.concatenate((X2.real, X2.imag), -1)
    X2 = np.fft.rfft(X2, axis=-2)[:, :, :n2y, :]
    X2 = np.concatenate((X2.real, X2.imag), -2)
    X2 = np.fft.rfft(X2, axis=-3)[:, :n2x, :, :]
    X2 = np.concatenate((X2.real, X2.imag), -3)
    return X2.reshape(X.shape[0], -1)


def rigid_transform_3D(atom_pos, base_pos):
    assert len(atom_pos) == len(base_pos)

    centroid_atom_pos = atom_pos.mean(axis=0)
    centroid_base_pos = base_pos.mean(axis=0)

    # centre the points
    atom_pos_cen = atom_pos - centroid_atom_pos
    base_pos_cen = base_pos - centroid_base_pos

    # dot is matrix multiplication for array
    H = base_pos_cen.T.dot(atom_pos_cen)

    U, S, Vt = np.linalg.svd(H)

    R = U.dot(Vt)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = U.dot(Vt)

    t = centroid_base_pos - R.dot(centroid_atom_pos)
    return R, t


def transform_molecule(A, base, heavy, return_rotation=False):
    """ Rotate molecule A so that its heavy atoms match base. """

    R, t = rigid_transform_3D(A[heavy], base[heavy])

    pos = A.dot(R.T) + t

    if len(heavy) == 3:
        # print(pos)
        pos -= pos[2, :]
        # print(pos)

        r1 = np.sqrt(np.sum((pos[0] - pos[2]) ** 2))
        r2 = np.sqrt(np.sum((pos[1] - pos[2]) ** 2))
        r3 = np.sqrt(np.sum((pos[0] - pos[1]) ** 2))
        ms = np.array([0, 0, -1])
        mr = np.sqrt(np.sum((ms - pos[2]) ** 2))
        mr1 = np.sqrt(np.sum((pos[0] - ms) ** 2))
        costheta = (r1 ** 2 + r2 ** 2 - r3 ** 2) / (2 * r1 * r2)
        cosm1 = (r1 ** 2 + mr ** 2 - mr1 ** 2) / (2 * r1 * mr)
        theta = np.arccos(costheta)
        theta1 = np.arccos(cosm1)

        rotheta = theta / 2 - theta1

        R_mat = [[1, 0, 0],
                 [0, np.cos(rotheta), -np.sin(rotheta)],
                 [0, np.sin(rotheta), np.cos(rotheta)]]

        R_mat = np.array(R_mat)

        pos = pos.dot(R_mat.T)
        pos += 10

    if return_rotation:
        return pos, R
    else:
        return pos


def reflect_through_plane(plane_points, ref_points):
    # create two vectors from plane points
    v1 = plane_points[0] - plane_points[1]
    v2 = plane_points[0] - plane_points[2]

    # calculate the normal vector by cross product
    normal_vec = np.cross(v1, v2)

    # calculate the plane constant as the dot product of the normal vector and a plane vector
    plane_const = np.sum(-plane_points[0] * normal_vec)

    # find value t, such that a line starting from a given point, moving in a perpendicular
    # direction to the plane, crosses that plane
    t = (-plane_const - np.dot(normal_vec, ref_points.T)) / np.dot(normal_vec, normal_vec)

    t = np.atleast_1d(t)
    t = t[:, np.newaxis]

    # to get reflected points, move points in the normal direction by 2t
    sym_points = ref_points + (2 * t * normal_vec)
    return sym_points


def normal_plane_on_line(plane_points, is_line_point):
    # separate points on line and other point
    line_points = plane_points[is_line_point, :]
    point = plane_points[np.logical_not(is_line_point), :]

    # calculate vector corresponding to line and another vector on the plane
    v1 = line_points[0, :] - line_points[1, :]
    v1 = v1[np.newaxis, :]
    v2 = line_points[0, :] - point

    # Calculate a vector normal to the plane from the two vectors
    normal_vec = np.cross(v1, v2)

    # Move a point on the line in the direction of the normal vector
    # to get a point on the normal plane
    normal_plane_point = line_points[0, :] + normal_vec

    # combine the point on the line with the new point to get three points
    # of the now normal plane
    normal_plane = np.concatenate((line_points, normal_plane_point), axis=0)

    return normal_plane


def symbols_to_numbers(symbols):
    numbers = []
    for s in symbols:
        numbers.append(ase.data.atomic_numbers[s])

    return numbers


def numbers_to_symbols(numbers, join=True):
    symbols = []
    for n in numbers:
        symbols.append(ase.data.chemical_symbols[n])

    if join:
        symbols = ''.join(symbols)

    return symbols


def ase_to_npy(mols):
    arr = []
    for m in mols:
        arr.append(m.get_positions())

    return np.array(arr)


def npy_to_ase(arr, atom_list):
    if atom_list.ndim == 1:
        atom_list = atom_list[None, :]
    if atom_list.shape[0] != arr.shape[0]:
        atom_list = np.tile(atom_list, (arr.shape[0], 1))
    mols = []
    for i in range(arr.shape[0]):
        mols.append(ase.Atoms(atom_list[i], positions=arr[i]))

    return mols


def energies_from_txt(filename, column, exclude_rows=0, energy_type='eV'):
    column -= 1

    with open(filename, 'r') as f:
        lines = list(f.readlines())
    lines = lines[exclude_rows:]
    words = [line.split() for line in lines]
    energies = [float(w[column]) for w in words]
    energies = np.array(energies)
    energies = np.reshape(energies, (-1, 1))
    if energy_type == 'hartree':
        energies = hartree_to_kcal(energies)
    elif energy_type == 'eV':
        energies = eV_to_kcal(energies)
    elif energy_type == 'kcal/mol':
        energies *= 1
    else:
        raise ValueError('energy type not supported for conversion')

    return energies


def positions_from_xyz(filename, convert_to_bohr=True):
    mols = ase.io.iread(filename)
    pos = ase_to_npy(mols)
    if convert_to_bohr:
        pos = angstrom_to_bohr(pos)

    return pos


def get_molecule_dists(target, neighbours, charges=None):
    neighbour_dists = np.zeros((neighbours.shape[0], 1))
    for i in range(neighbours.shape[0]):
        atom_dists = np.sum((neighbours[i] - target)**2, axis=1)
        if charges is not None:
            atom_dists *= charges

        neighbour_dists[i] = np.sum(atom_dists)
    return neighbour_dists


def distance_nearest_neighbours(targets, neighbours, num_neighbours, charges=None):
    avg_distances = np.zeros((targets.shape[0], 1))
    for i in range(targets.shape[0]):
        neighbour_dists = get_molecule_dists(targets[i], neighbours, charges=charges)
        neighbour_dists.sort()
        avg_distances[i] = np.mean(neighbour_dists[:num_neighbours])

    return avg_distances


def align_to_base(pos, base_pos, heavy):
    pos_al = np.array(pos)
    for i in range(pos.shape[0]):
        pos_al[i] = transform_molecule(pos[i], base_pos, heavy)

    return pos_al


def str_to_atom_types(string):
    str_list = list(string)
    return np.array([str_list], dtype='<U1')


"""
Given the Cartesian coordinates and index lists, calculates pairwise distances and unit displacement
vectors. Each distance / vector is specified by a pair of atom indices i and j (i != j). The total
number of interactions is num_interactions=num_atoms * (num_atoms - 1) when all pairwise distances are
calculated.

inputs:
    R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
    idx_i: indices of atoms i of shape [num_interactions] for collecting Cartesian coordinates
    idx_j: indices of atoms j of shape [num_interactions] for collecting Cartesian coordinates
outputs:
    dij: pairwise distances of shape [batch_size, num_interactions, 1]
    uij: unit displacement vectors of shape [batch_size, num_interactions, 3]
"""


def calculate_distances_and_directions(R, idx_i=None, idx_j=None, center=None):
    # print('R shape', R.shape)
    if idx_i is not None and idx_j is not None:
        Ri = torch.gather(R, -2, idx_i.view(*(1,) * len(R.shape[: -2]), -1, 1).repeat(*R.shape[: -2], 1, R.size(-1)))
        Rj = torch.gather(R, -2, idx_j.view(*(1,) * len(R.shape[: -2]), -1, 1).repeat(*R.shape[: -2], 1, R.size(-1)))
    elif center is not None:
        Ri = R
        Rj = center
    else:
        Ri = R.view(R.shape[0], 1, *R.shape[1:])
        Rj = R.view(*R.shape[:-1], 1, R.shape[-1])
    # print('Ri shape', Ri.shape)
    # print('Rj shape', Rj.shape)
    rij = Rj - Ri  # displacement vectors
    dij = torch.norm(rij, dim=-1, keepdim=True)  # distances
    uij = rij / dij  # unit displacement vectors
    return dij, uij
