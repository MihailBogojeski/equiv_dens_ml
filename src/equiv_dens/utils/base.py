import numpy as np
import scipy
import ase
import ase.io
import torch
import ase.data
import pyscf
from pyscf.data import nist
import math

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

def au_to_debye(dpm):
    return dpm * nist.AU2DEBYE

def internal_to_debye(dpm):
    return dpm * nist.AU2DEBYE * to_bohr


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


def ase_to_npy2(mols):
    positions = []
    atom_numbers = []
    for m in mols:
        positions.append(m.get_positions())
        atom_numbers.append(m.get_atomic_numbers())

    numbers, props = compress_batch_atoms(numbers, {'positions': positions}) 
    props['atom_numbers'] = numbers

    return props


def npy_to_ase(arr, atom_list):
    if atom_list.ndim == 1:
        atom_list = atom_list[None, :]
    if atom_list.shape[0] != arr.shape[0]:
        atom_list = np.tile(atom_list, (arr.shape[0], 1))
    mols = []
    for i in range(arr.shape[0]):
        nonzero = atom_list[i] != 0
        mols.append(ase.Atoms(atom_list[i][nonzero], positions=arr[i][nonzero]))

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

class TorchNeighborList:
    """
    Environment provider making use of neighbor lists as implemented in TorchAni
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py).
    Supports cutoffs, PBCs and can be performed on either CPU or GPU.

    """

    def __init__(self, cutoff):
        """
        Args:
            cutoff (float): the cutoff inside which atoms are considered pairs
            device (:class:`torch.device`): pass torch.device('cpu') or torch.device('cuda') to
                perform the calculation on a CPU or GPU, respectively.
        """
        self.cutoff = cutoff

    def get_neighbors(self, atoms, pbc=False):
        idx_is = []
        idx_js = []
        idx_Ss = []
        if pbc:
            pbc = torch.ones((3, )).to(atoms['positions']).type(torch.ByteTensor) 
        else:
            pbc = torch.zeros((3, )).to(atoms['positions']).type(torch.ByteTensor) 
        for i in range(atoms['positions'].shape[0]):
            numbers = atoms['atom_numbers'][i]
            nz = numbers > 0
            numbers = numbers[nz]
            pos = atoms['positions'][i][nz]
            
            if 'cell' in atoms.keys():
                cell = atoms['cell']
            else:
                cell = torch.max(pos, dim=(0))[0] - torch.min(pos, dim=(0))[0] + 2
                cell = torch.diag(cell)
            
            shifts = compute_shifts(cell=cell, pbc=pbc, cutoff=self.cutoff)

            # The returned indices are only one directional
            idx_i, idx_j, idx_S = neighbor_pairs(
                numbers<=0, pos, cell, shifts, self.cutoff
            )

            idx_i = idx_i
            idx_j = idx_j
            idx_S = idx_S
            # Create bidirectional id arrays, similar to what the ASE neighbor_list returns
            bi_idx_i = torch.hstack((idx_i, idx_j))
            bi_idx_j = torch.hstack((idx_j, idx_i))
            bi_idx_S = torch.vstack((-idx_S, idx_S))
            idx_is.append(bi_idx_i)
            idx_js.append(bi_idx_j)
            idx_Ss.append(bi_idx_S)


        # n_atoms = atoms.get_global_number_of_atoms()
        # if bi_idx_i.shape[0] > 0:
        #     uidx, n_nbh = np.unique(bi_idx_i, return_counts=True)
        #     n_max_nbh = np.max(n_nbh)
        #
        #     n_nbh = np.tile(n_nbh[:, np.newaxis], (1, n_max_nbh))
        #     nbh_range = np.tile(
        #         np.arange(n_max_nbh, dtype=np.int)[np.newaxis], (n_nbh.shape[0], 1)
        #     )
        #
        #     mask = np.zeros((n_atoms, np.max(n_max_nbh)), dtype=np.bool)
        #     mask[uidx, :] = nbh_range < n_nbh
        #     neighborhood_idx = -np.ones((n_atoms, np.max(n_max_nbh)), dtype=np.float32)
        #     offset = np.zeros((n_atoms, np.max(n_max_nbh), 3), dtype=np.float32)
        #
        #     # Assign neighbors and offsets according to the indices in bi_idx_i, since in contrast
        #     # to the ASE provider the bidirectional arrays are no longer sorted.
        #     # TODO: There might be a more efficient way of doing this than a loop
        #     for idx in range(n_atoms):
        #         neighborhood_idx[idx, mask[idx]] = bi_idx_j[bi_idx_i == idx]
        #         offset[idx, mask[idx]] = bi_idx_S[bi_idx_i == idx]
        #
        # else:
        #     neighborhood_idx = -np.ones((n_atoms, 1), dtype=np.float32)
        #     offset = np.zeros((n_atoms, 1, 3), dtype=np.float32)

        return idx_is, idx_js, idx_Ss


def compute_shifts(cell, pbc, cutoff):
    """Compute the shifts of unit cell along the given cell vectors to make it
    large enough to contain all pairs of neighbor atoms with PBC under
    consideration.
    Copyright 2018- Xiang Gao and other ANI developers
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

    Arguments:
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
        vectors defining unit cell:
            tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
            if pbc is enabled for that direction.

    Returns:
        :class:`torch.Tensor`: long tensor of shifts. the center cell and
            symmetric cells are not included.
    """
    # type: (Tensor, Tensor, float) -> Tensor
    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(pbc).type(torch.long)
    num_repeats = torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))

    r1 = torch.arange(1, num_repeats[0] + 1, device=cell.device)
    r2 = torch.arange(1, num_repeats[1] + 1, device=cell.device)
    r3 = torch.arange(1, num_repeats[2] + 1, device=cell.device)
    o = torch.zeros(1, dtype=torch.long, device=cell.device)

    return torch.cat(
        [
            torch.cartesian_prod(r1, r2, r3),
            torch.cartesian_prod(r1, r2, o),
            torch.cartesian_prod(r1, r2, -r3),
            torch.cartesian_prod(r1, o, r3),
            torch.cartesian_prod(r1, o, o),
            torch.cartesian_prod(r1, o, -r3),
            torch.cartesian_prod(r1, -r2, r3),
            torch.cartesian_prod(r1, -r2, o),
            torch.cartesian_prod(r1, -r2, -r3),
            torch.cartesian_prod(o, r2, r3),
            torch.cartesian_prod(o, r2, o),
            torch.cartesian_prod(o, r2, -r3),
            torch.cartesian_prod(o, o, r3),
        ]
    )


def neighbor_pairs(padding_mask, coordinates, cell, shifts, cutoff):
    """Compute pairs of atoms that are neighbors
    Copyright 2018- Xiang Gao and other ANI developers
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

    Arguments:
        padding_mask (:class:`torch.Tensor`): boolean tensor of shape
            (molecules, atoms) for padding mask. 1 == is padding.
        coordinates (:class:`torch.Tensor`): tensor of shape
            (molecules, atoms, 3) for atom coordinates.
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
            defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
    """
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    coordinates = coordinates.detach()
    cell = cell.detach()
    num_atoms = padding_mask.shape[0]
    all_atoms = torch.arange(num_atoms, device=cell.device)

    # Step 2: center cell
    p1_center, p2_center = torch.combinations(all_atoms).unbind(-1)
    shifts_center = shifts.new_zeros(p1_center.shape[0], 3)

    # Step 3: cells with shifts
    # shape convention (shift index, molecule index, atom index, 3)
    num_shifts = shifts.shape[0]
    all_shifts = torch.arange(num_shifts, device=cell.device)
    shift_index, p1, p2 = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).unbind(
        -1
    )
    shifts_outside = shifts.index_select(0, shift_index)

    # Step 4: combine results for all cells
    shifts_all = torch.cat([shifts_center, shifts_outside])
    p1_all = torch.cat([p1_center, p1])
    p2_all = torch.cat([p2_center, p2])

    shift_values = torch.mm(shifts_all.to(cell.dtype), cell)

    # step 5, compute distances, and find all pairs within cutoff
    distances = (coordinates[p1_all] - coordinates[p2_all] + shift_values).norm(2, -1)

    padding_mask = (padding_mask[p1_all]) | (padding_mask[p2_all])
    distances.masked_fill_(padding_mask, math.inf)
    in_cutoff = torch.nonzero(distances < cutoff, as_tuple=False)
    pair_index = in_cutoff.squeeze()
    atom_index1 = p1_all[pair_index]
    atom_index2 = p2_all[pair_index]
    shifts = shifts_all.index_select(0, pair_index)
    return atom_index1, atom_index2, shifts


def compress_batch_atoms(numbers, props_dict, basis_size=None):
    atom_num_count = {}
    for i in range(len(numbers)):
        nums = np.unique(numbers[i])
        for num in nums:
            if num <= 0:
                continue
            count = np.sum(numbers[i] == num)
            if num in atom_num_count.keys():
                if count > atom_num_count[num]:
                    atom_num_count[num] = count
            else:
                atom_num_count[num] = count
    common_numbers = []
    for key in atom_num_count.keys():
        common_numbers += [key] * atom_num_count[key]
    batch_nums = []
    batch_props = {}
    for i in range(len(numbers)):
        props = {key: props_dict[key][i] for key in props_dict.keys()}
        nums = np.array(numbers[i])
        new_nums = np.zeros((len(common_numbers),))
        new_props = {}
        for key in props.keys(): 
            if isinstance(props[key], np.ndarray):
                new_props[key] = np.zeros((len(common_numbers), props[key].shape[1]))
            elif basis_size is not None:
                new_props[key] = []
                for z in common_numbers:
                    new_props[key].append(np.zeros((basis_size[z], )))
            else:
                raise Exception('No basis size given for df coeffs!')

        last_idx = 0
        for z in atom_num_count.keys():
            idx = np.where(nums == z)[0]
            # print('z', z)
            # print('z idx', idx)
            new_nums[last_idx:last_idx + len(idx)] = nums[idx]
            for key in new_props.keys():
                # print('prop key', key)
                if isinstance(new_props[key], np.ndarray):
                    # print('numpy array add props')
                    new_props[key][last_idx:last_idx + len(idx)] = props[key][idx]
                else:
                    # print('df coeffs add props')
                    charges = np.array([prop[0] for prop in props[key]])
                    idx_alt = np.where(charges == z)[0]
                    new_props[key][last_idx:last_idx + len(idx_alt)] = [props[key][j][1] for j in idx_alt]
            last_idx += atom_num_count[z]
        batch_nums.append(new_nums)
        for key in new_props.keys():
            if not isinstance(new_props[key], np.ndarray):
                new_props[key] = np.concatenate(new_props[key])
            if key in batch_props.keys():
                batch_props[key].append(new_props[key])
            else:
                batch_props[key] = [new_props[key]]

    batch_nums = np.array(batch_nums)
    for key in batch_props.keys():
        batch_props[key] = np.array(batch_props[key])

    return batch_nums, batch_props


def batch_compressed_atoms(atoms, relevant_keys):
    atom_numbers = atoms['atom_numbers']
    batch_idx = atoms['atom_batch_idx']
    batch_nums = atoms['batch_atom_numbers']
    atom_count = atom_numbers.shape[1]
    batch_size = batch_nums.shape[0]
    batch_atom_count = batch_nums.shape[1]
    batch_props = {key: torch.zeros((batch_size, batch_atom_count, atoms[key].shape[-1])).to(atoms[key]) for key in relevant_keys}
    atom_idx = 0
    prev_z = int(atom_numbers[0, 0])
    prev_batch_num = int(batch_idx[:, 0])

    for i in range(atom_count):
        z = int(atom_numbers[0, i])
        batch_num = int(atoms['atom_batch_idx'][:, i])
        # print('atom count', i)
        # print('z', z)
        # print('prev z', prev_z)
        # print('batch num', batch_num)
        # print('prev batch num', prev_batch_num)
        if z != prev_z or batch_num != prev_batch_num:
            atom_idx = atoms['atom_numbers_first_positions'][z]
        prev_z = z
        # print('atom_idx', atom_idx)
        prev_batch_num = batch_num
        for key in relevant_keys:
            batch_props[key][batch_num, atom_idx] = atoms[key][0, i]
        atom_idx += 1

    for key in relevant_keys:
        atoms[key] = batch_props[key]

    return atoms


def batch_compressed_atoms_v2(atoms, relevant_keys):
    batch_idx_pos = atoms['batch_idx_pos']
    batch_nums = atoms['batch_atom_numbers']
    batch_size = batch_nums.shape[0]
    batch_atom_count = batch_nums.shape[1]
    batch_props = {key: torch.zeros((batch_size * batch_atom_count, *atoms[key].shape[2:])).to(atoms[key]) for key in relevant_keys}

    for key in relevant_keys:
        print('key', key)
        print('atom_key shape', atoms[key].shape)
        print('batch props_key shape', batch_props[key].shape)
        batch_props[key][batch_idx_pos] = atoms[key]
        print('batch props_key shape', batch_props[key].shape)
        batch_props[key] = batch_props[key].view(batch_size, batch_atom_count, *atoms[key].shape[2:])
        print('batch props_key shape', batch_props[key].shape)
        atoms[key] = batch_props[key]

    return atoms


def get_atom_num_first_positions(atom_numbers):
    if atom_numbers.ndim > 1:
        if isinstance(atom_numbers, np.ndarray):
            atom_numbers = np.max(atom_numbers, axis=0)
        else: 
            atom_numbers, _ = torch.max(atom_numbers, dim=0)
    atom_numbers_first_positions = {}
    for i in range(len(atom_numbers)):
        z = atom_numbers[i]
        if z not in atom_numbers_first_positions.keys():
            atom_numbers_first_positions[z] = i

    return atom_numbers_first_positions

            
def calc_dict_to_npy(data, convert_forces=True, compress_atoms=True):
    data_npy = {}
    data_npy['energy'] = []
    data_npy['forces'] = []
    data_npy['positions'] = []
    data_npy['atom_numbers'] = []
    data_npy['atom_types'] = []
    for calc in data:
        data_npy['energy'].append(calc[1]['energy'])
        if convert_forces:
            data_npy['forces'].append(-calc[1]['forces'] * to_bohr)
        else:
            data_npy['forces'].append(calc[1]['forces'])
        pos = []
        at = []
        z = []
        for a in calc[0]['atom']:
            if isinstance(a[0], str):
                z.append(symbols_to_numbers([a[0]])[0])
                at.append(a[0])
            else:
                z.append(a[0])
                at.append(numbers_to_symbols([a[0]])[0])
            pos.append(a[1])
        z = np.array(z)
        pos = np.array(pos)
        data_npy['positions'].append(pos)
        data_npy['atom_numbers'].append(z)
        data_npy['atom_types'].append(z)
    # print('data_npy atom numbers', data_npy['atom_numbers'][:10])
    # print('data_npy pos', data_npy['positions'][:10])
    if compress_atoms:
        atom_numbers, props = compress_batch_atoms(data_npy['atom_numbers'],
                                                   {'positions': data_npy['positions'],
                                                    'forces': data_npy['forces']})
    else:
        atom_numbers = np.array(data_npy['atom_numbers'])
        props = {'positions': np.array(data_npy['positions']), 'forces': np.array(data_npy['forces'])}
    data_npy['positions'] = props['positions'] 
    # print(data_npy['positions'].shape)
    data_npy['atom_numbers'] = atom_numbers.astype(int)
    # print(data_npy['atom_numbers'].shape)
    # print('data_npy atom numbers new', data_npy['atom_numbers'][:10])
    # print('data_npy pos new', data_npy['positions'][:10])
    data_npy['forces'] = props['forces'] 
    data_npy['energy'] = np.stack(data_npy['energy'], 0)[:, None]
    return data_npy
