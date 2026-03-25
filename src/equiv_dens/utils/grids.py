import numpy as np
from pyscf.dft import gen_grid, radi
from .base import random_rotation_matrix
from pyscf import lib
import torch
import time
# from dftpy.math_utils import bestFFTsize
# from dftpy.grid import DirectGrid
import equiv_dens.utils.base as utils
from pyscf import gto
import time


def spherical_grid(atoms, level=2, bohr=False):
    numbers = np.unique(atoms['atom_numbers'].flatten()).astype(int)
    numbers = numbers[numbers > 0]
    positions = []
    for n in numbers:
        positions.append([0, 0, n])
    positions = np.array(positions)
    mol_dict = list(zip(numbers, positions))
    if (np.sum(numbers) % 2 == 1):
        mol = gto.M(atom=mol_dict, spin=1)
    else:
        mol = gto.M(atom=mol_dict)

    grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=level)
    for key in grid_spec.keys():
        if bohr:
            grid_spec[key] = (torch.tensor(grid_spec[key][0]),
                              torch.tensor(grid_spec[key][1]))  # convert Bohr grid to Angstrom
        else:
            grid_spec[key] = (torch.tensor(grid_spec[key][0] * utils.to_angstrom),
                              torch.tensor(grid_spec[key][1]))  # convert Bohr grid to Angstrom

    return grid_spec


def cubical_grid(atoms, nx=125, ny=125, nz=125, resolution=None,
                 margin=2, origin=None, extent=[10, 10, 10]):
    numbers = atoms['atom_numbers'][0]
    numbers = numbers[numbers > 0]
    positions = atoms['positions'][0]
    positions = positions[numbers > 0, :]
    mol_dict = list(zip(numbers, positions))
    if (np.sum(atoms['atom_numbers'][0]) % 2 == 1):
        mol = gto.M(atom=mol_dict, spin=1)
    else:
        mol = gto.M(atom=mol_dict)
    coord = mol.atom_coords(unit='Angstrom')  # positions in angstrom
    if extent is None:
        box = np.max(coord, axis=0) - np.min(coord, axis=0) + margin * 2
        box = np.diag(box)
    else:
        box = np.diag(extent)
    if origin is None:
        boxorig = np.min(coord, axis=0) - margin
    else:
        boxorig = np.array(origin)
    if resolution is not None:
        nx, ny, nz = np.ceil(np.diag(box) / resolution).astype(int)

    # .../(nx-1) to get symmetric mesh
    # see also the discussion https://github.com/sunqm/pyscf/issues/154
    xs = np.arange(nx) * (np.diag(box)[0] / (nx - 1))
    ys = np.arange(ny) * (np.diag(box)[1] / (ny - 1))
    zs = np.arange(nz) * (np.diag(box)[2] / (nz - 1))

    sample_volume = extent / np.array([nx, ny, nz])

    sample_volume *= utils.to_bohr
    sample_volume = np.prod(sample_volume)

    coords = lib.cartesian_prod([xs, ys, zs])
    coords = np.asarray(coords, order='C') - (-boxorig)
    return (coords, sample_volume)


def becke_scheme(g):
    '''Becke, JCP 88, 2547 (1988); DOI:10.1063/1.454033'''
#    This funciton has been optimized in the C code VXCgen_grid
    g = (3 - g**2) * g * .5
    g = (3 - g**2) * g * .5
    g = (3 - g**2) * g * .5

    return g


def treutler_atomic_radii_adjust(charges, atomic_radii):
    charges = np.amax(charges, axis=0).astype(int)
    rad = np.sqrt(atomic_radii[charges]) + 1e-200
    rr = rad.reshape(-1, 1) * (1.0 / rad)
    a = .25 * (rr.T - rr)
    a[a < -0.5] = -0.5
    a[a > 0.5] = 0.5

    def fadjust(i, j, g):
        g1 = g**2
        g1 -= 1.
        g1 *= -a[i, j]
        g1 += g
        return g1

    return fadjust


def gen_grid_partition(positions, coords, becke_scheme, f_radii_adjust=None):
    ngrids = coords.shape[1]
    natm = positions.shape[1]
    nbatch = positions.shape[0]
    atm_dist, _ = utils.calculate_distances_and_directions(positions)
    # print('atom distances', atm_dist)
    # print('atm dist shape', atm_dist.shape)
    dc = coords[:, None] - positions[:, :, None]
    # print('dc shape', dc.shape)
    grid_dist = torch.sqrt(torch.sum(dc**2, dim=-1))
    pbecke = torch.ones((nbatch, natm, ngrids)).to(positions)
    for i in range(natm):
        for j in range(i):
            g = 1 / atm_dist[:, i, j] * (grid_dist[:, i] - grid_dist[:, j])
            if f_radii_adjust is not None:
                g = f_radii_adjust(i, j, g)
            g = becke_scheme(g)
            pbecke[:, i] *= .5 * (1 - g)
            pbecke[:, j] *= .5 * (1 + g)
    return pbecke


def spherical_radial_sampling(grid_spec, n_samp, atom_numbers, positions,
                              radii_adjust=None,
                              rotate=False):
    grid_coords = []
    grid_weights = []
    # print('pos type', pos.type())
    start = time.time()
    atom_numbers_max = np.amax(atom_numbers, axis=0).astype(int)
    for i in range(len(atom_numbers)):
        grid_coords.append([])
        grid_weights.append([])
        start_i = time.time()
        # print('i', i)
        pos = positions[[i]]
        mask = atom_numbers[i] > 0
        pos_nz = pos[:, mask, :]
        atm_dist, _ = utils.calculate_distances_and_directions(positions)
        # print('atom_numbers', atom_numbers.squeeze())
        # print('atom distances', utils.angstrom_to_bohr(atm_dist.squeeze()))
        pos_idx = -1
        for j, z in enumerate(atom_numbers_max):
            # print('i', j)
            start_jz = time.time()
            if z <= 0:
                continue
            if rotate:
                rot_mat = torch.tensor(random_rotation_matrix()).to(pos)
            else:
                rot_mat = torch.eye(3).to(pos)
            if atom_numbers[i, j] > 0:
                pos_idx += 1
            t = utils.numbers_to_symbols([z])[0]
            # print('rot_mat type', rot_mat.type())
            # print('grid spec type', grid_spec[t][0].type())
            # print('atom coords', pos[:, [j], :])
            # print('atom num', z)
            # print('grid coords', grid_spec[t][0][:3])
            coords = pos[:, [j], :] + (grid_spec[t][0].unsqueeze(0) @ rot_mat)
            # print('grid + atom_coords', coords[:3])
            weights = grid_spec[t][1] * (atom_numbers[i][j] > 0)
            pbecke = gen_grid_partition(pos_nz, coords, becke_scheme, radii_adjust)
            weights = weights * pbecke[:, pos_idx] * (1.0 / pbecke.sum(1))
            grid_coords[i].append(coords)
            grid_weights[i].append(weights)
            # print('sampling time for atom', j, z, time.time() - start_jz)
            # print('i', i)
        # print('sampling time for mol', i, time.time() - start_i)
    # print('len grid coords', len(grid_coords))
    # print('len grid coords[0]', len(grid_coords[0]))
    # print('shape grid coords[0][0]', grid_coords[0][0].shape)
    # print('sampling time after loop', time.time() - start)

    grid_coords = [list(coord) for coord in zip(*grid_coords)]
    grid_weights = [list(coord) for coord in zip(*grid_weights)]
    # print('len grid coords', len(grid_coords))
    # print('len grid coords[0]', len(grid_coords[0]))
    # print('shape grid coords[0][0]', grid_coords[0][0].shape)
    if isinstance(grid_coords[0][0], torch.Tensor):
        grid_coords = [torch.cat(atoms, dim=0) for atoms in grid_coords]
        grid_weights = [torch.cat(atoms, dim=0) for atoms in grid_weights]
    else:
        grid_coords = [np.concatenate(atoms, axis=0) for atoms in grid_coords]
        grid_weights = [np.concatenate(atoms, axis=0) for atoms in grid_weights]
    # print('sampling time before collect', time.time() - start)

    return collect_and_sample_grid(grid_coords, grid_weights, n_samp)


def spherical_radial_sampling_fast(grid_spec, n_samp, atom_numbers, positions,
                                   radii_adjust=None,
                                   rotate=False):
    grid_coords = []
    grid_weights = []
    # print('pos type', pos.type())
    start = time.time()
    atom_numbers_max = np.amax(atom_numbers, axis=0).astype(int)
    atom_symbols_max = utils.number_to_symbols(atom_numbers_max)

    pos = positions.unqsqueeze(2)
    coords = [grid_spec[atom_symbols_max[i]][0].unsqueeze(0) @
              torch.tensor(random_rotation_matrix().to(positions)) + pos[:, i]
              for i in range(len(atom_symbols_max))]
    # print('sampling time after loop', time.time() - start)

    grid_coords = [list(coord) for coord in zip(*grid_coords)]
    grid_weights = [list(coord) for coord in zip(*grid_weights)]
    # print('len grid coords', len(grid_coords))
    # print('len grid coords[0]', len(grid_coords[0]))
    # print('shape grid coords[0][0]', grid_coords[0][0].shape)
    if isinstance(grid_coords[0][0], torch.Tensor):
        grid_coords = [torch.cat(atoms, dim=0) for atoms in grid_coords]
        grid_weights = [torch.cat(atoms, dim=0) for atoms in grid_weights]
    else:
        grid_coords = [np.concatenate(atoms, axis=0) for atoms in grid_coords]
        grid_weights = [np.concatenate(atoms, axis=0) for atoms in grid_weights]
    # print('sampling time before collect', time.time() - start)

    return collect_and_sample_grid(grid_coords, grid_weights, n_samp)


def spherical_sampling(grid_spec, n_samp, atom_numbers, pos):
    grid_coords = []
    grid_weights = []
    atom_numbers = np.amax(atom_numbers, axis=0).astype(int)
    for i, n in enumerate(atom_numbers):
        t = utils.numbers_to_symbols([n])[0]
        grid_coords.append(pos[:, [i], :] + (grid_spec[t][0][None, :]))
        grid_weights.append(grid_spec[t][1] / len(atom_numbers))

    return collect_and_sample_grid(grid_coords, grid_weights, n_samp)


def rot_spherical_sampling(grid_spec, n_samp, atom_numbers, pos):
    grid_coords = []
    grid_weights = []
    atom_numbers = np.amax(atom_numbers, axis=0).astype(int)
    for i, n in enumerate(atom_numbers):
        t = utils.numbers_to_symbols([n])[0]
        rot_mat = random_rotation_matrix()
        grid_coords.append(pos[:, [i], :] + (grid_spec[t][0][None, :]) @ rot_mat)
        grid_weights.append(grid_spec[t][1] / len(atom_numbers))

    return collect_and_sample_grid(grid_coords, grid_weights, n_samp)


def collect_and_sample_grid(grid_coords, grid_weights, n_samp):
    if isinstance(grid_coords[0], torch.Tensor):
        grid_coords = torch.cat(grid_coords, dim=1)
        grid_weights = torch.cat(grid_weights, dim=1)
    else:
        grid_coords = np.concatenate(grid_coords, axis=1)
        grid_weights = np.concatenate(grid_weights, axis=1)

    if n_samp > grid_coords.shape[1]:
        return grid_coords, grid_weights
    else:
        rand_idx = np.random.choice(np.arange(grid_coords.shape[1]), size=n_samp, replace=False)
        # print('grid_coords shape', grid_coords.shape)
        # print('grid_weights shape', grid_weights.shape)
        # print('rand idx', rand_idx)
        return grid_coords[:, rand_idx, :], grid_weights[:, rand_idx]


def cubical_sampling(grid_spec, n_samp, _, pos):
    flat_coords = np.reshape(grid_spec[0], (-1, 3))
    flat_coords = flat_coords[None, :]
    flat_coords = np.repeat(flat_coords, pos.shape[0], axis=0)
    if isinstance(pos, torch.Tensor):
        flat_coords = torch.tensor(flat_coords).to(pos)
    if n_samp > flat_coords.shape[1]:
        return flat_coords, torch.ones((flat_coords.shape[0], flat_coords.shape[1], )) * grid_spec[1]
    else:
        rand_idx = np.random.choice(np.arange(flat_coords.shape[1]), size=n_samp, replace=False)
        return flat_coords[:, rand_idx, :], torch.ones((flat_coords.shape[0], n_samp, )) * grid_spec[1]

def spherical_grid_atom_cutoff(coords, pos, atom_numbers, grid_spec):
    """
    Cut off the extent of the spherical grid for each atom based on the maximum extent of that atom's atomic grid.
    Args:
        coords (n_mols, n_grid, 3): Array containing grid coordinates
        pos: (n_mols, n_atoms, 3): Array containing atomic coordinates
        atom_numbers: (n_mols, n_atoms): Array containing atomic numbers
        grid_spec: (dict): Dictionary containing atomic grid specifications
    Returns:
        cutoff_coords (n_mols, n_atoms, n_grid, 3): Boolean array used to mask out the coordinates outside of the cutoff
    """
    pos_dists = torch.norm(pos.unsqueeze(2) - coords.unsqueeze(1), dim=-1)
    atom_numbers = torch.amax(atom_numbers, axis=0).type(torch.long)
    grid_dist = torch.zeros(size=(1, len(atom_numbers), 1)).to(pos_dists)
    for i in range(atom_numbers.shape[0]):
        t = utils.numbers_to_symbols([atom_numbers[i]])[0]
        grid_dist[0, i] = torch.max(torch.norm(grid_spec[t][0], dim=-1))
    cutoff_coords = pos_dists <= grid_dist
    return cutoff_coords

# def dftpy_grid(lattice, gap):
#     nr = np.zeros(3, dtype='int32')
#     metric = np.dot(lattice.T, lattice)
#     # print('lattice', lattice)
#     # print('metric', np.sqrt(metric[0, 0]))
#     # print('gap', gap)
#     for i in range(3):
#         nr[i] = int(np.sqrt(metric[i, i]) / gap)
#     # print('The initial grid size is ', nr)
#     for i in range(3):
#         nr[i] = bestFFTsize(nr[i])
#     # print('The final grid size is ', nr)
#     grid = DirectGrid(lattice=lattice, nr=nr, units=None, full=False)
#     return grid


class CubicalGrid():

    def __init__(self, atoms, nx=125, ny=125, nz=125, resolution=None,
                 margin=2, origin=None, extent=[10, 10, 10], use_gpu=False, dtype=torch.double):
        self.use_gpu = use_gpu
        self.dtype = dtype
        numbers, _ = torch.max(atoms['atom_numbers'], dim=1)
        positions = atoms['positions'][0]
        mol_dict = list(zip(numbers, positions))
        # print('mol_dict', mol_dict)
        mol = gto.M(atom=mol_dict)
        if extent is None:
            coord = mol.atom_coords(unit='Angstrom')  # positions in angstrom
            box = np.max(coord, axis=0) - np.min(coord, axis=0) + margin * 2
            box = np.diag(box)
        else:
            box = np.diag(extent)
        if origin is None:
            coord = mol.atom_coords(unit='Angstrom')  # positions in angstrom
            self.boxorig = np.min(coord, axis=0) - margin
        else:
            self.boxorig = np.array(origin)
        if resolution is not None:
            nx, ny, nz = np.ceil(np.diag(box) / resolution).astype(int)
        self.shape = torch.LongTensor([nx, ny, nz])
        self.box = torch.tensor(box).type(dtype)
        if self.use_gpu:
            self.box = self.box.cuda()
        self.lattice = self.box
        # .../(nx-1) to get symmetric mesh
        # see also the discussion https://github.com/sunqm/pyscf/issues/154
        xs = np.linspace(0, np.diag(box)[0], nx, endpoint=False)
        ys = np.linspace(0, np.diag(box)[0], nx, endpoint=False)
        zs = np.linspace(0, np.diag(box)[0], nx, endpoint=False)

        self.point_volume = torch.diag(self.box) / torch.tensor([nx, ny, nz]).type(dtype)
        if self.use_gpu:
            self.point_volume = self.point_volume.cuda()
        self.point_volume = torch.prod(self.point_volume)

        self.volume = torch.prod(torch.diag(self.lattice))

        self.coords = lib.cartesian_prod([xs, ys, zs])
        self.coords = np.asarray(self.coords, order='C') - (-self.boxorig)
        self.coords = torch.tensor(self.coords).type(dtype)
        if self.use_gpu:
            self.coords = self.coords.cuda()

        self._rr = None
        self.rec_grid = None

    @property
    def rr(self):
        if self._rr is None:
            rr = torch.einsum("lijk,lijk->ijk", self.coords, self.coords)
            # self._rr = np.reshape(rr, [self.nr[0], self.nr[1], self.nr[2], 1])
            self._rr = rr
        return self._rr

    def get_reciprocal_grid(self):
        if self.rec_grid is None:
            fac = 2 * np.pi
            bg = fac * torch.inverse(self.lattice)
            reciprocal_lat = bg.T
            self.rec_grid = ReciprocalGrid(np.array(self.shape).astype(np.int),
                                           reciprocal_lat, use_gpu=self.use_gpu,
                                           dtype=self.dtype)
        return self.rec_grid


class ReciprocalGrid():

    def __init__(self, shape, lattice, use_gpu=False, dtype=torch.double):
        ax = []
        self.use_gpu = use_gpu
        self.dtype = dtype
        for i in range(3):
            dd = 1 / shape[i]
            if i == 2:
                ax.append(np.fft.rfftfreq(shape[i], d=dd))
            else:
                freq = np.fft.fftfreq(shape[i], d=dd)

                ax.append(freq)
        S0, S1, S2 = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")

        # S_cart = s2r(S, self)
        self.lattice = lattice
        S_cart = np.asarray([S0, S1, S2])
        S_cart = torch.tensor(S_cart).type(dtype)
        if self.use_gpu:
            S_cart = S_cart.cuda()
        self.coords = torch.einsum("j...,kj->k...", S_cart, self.lattice)
        self.shape = self.coords.shape[1:]
        self.full_shape = shape
        self._mask = None
        self._gg = None
        self._q = None

    @property
    def gg(self):
        if self._gg is None:
            gg = torch.einsum("lijk,lijk->ijk", self.coords, self.coords)
            self._gg = gg
        return self._gg

    @property
    def q(self):
        if self._q is None:
            self._q = torch.sqrt(self.gg)
        return self._q

    @property
    def mask(self):
        if self._mask is None:
            grid_shape = np.array(self.full_shape)
            # Dnr = nr[:3]//2
            # Dmod = nr[:3]%2
            # mask = np.ones((nr[0], nr[1], Dnr[2]+1), dtype = bool)
            Dnr = grid_shape // 2
            Dmod = grid_shape % 2
            mask = torch.ones(self.shape, dtype=torch.bool)
            mask[:, :, Dnr[2] + 1:] = False

            mask[0, 0, 0] = False
            mask[0, Dnr[1] + 1:, 0] = False
            mask[Dnr[0] + 1:, :, 0] = False
            if Dmod[2] == 0:
                mask[0, 0, Dnr[2]] = False
                mask[0, Dnr[1] + 1:, Dnr[2]] = False
                mask[Dnr[0] + 1:, :, Dnr[2]] = False
                if Dmod[1] == 0:
                    mask[0, Dnr[1], Dnr[2]] = False
                if Dmod[0] == 0:
                    mask[Dnr[0], 0, Dnr[2]] = False
                    mask[Dnr[0], Dnr[1] + 1:, Dnr[2]] = False
            if Dmod[0] == 0:
                mask[Dnr[0], Dnr[1] + 1:, 0] = False
                if Dmod[1] == 0:
                    mask[Dnr[0], Dnr[1], 0] = False
            if Dmod[1] == 0:
                mask[0, Dnr[1], 0] = False
            if all(Dmod == 0):
                mask[Dnr[0], Dnr[1], Dnr[2]] = False
            if self.use_gpu:
                self._mask = mask.cuda()
            else:
                self._mask = mask
        return self._mask
