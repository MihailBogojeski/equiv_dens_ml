import numpy as np
from pyscf.dft import gen_grid, radi
from .base import random_rotation_matrix
from pyscf import lib
import torch
from dftpy.math_utils import bestFFTsize
from dftpy.grid import DirectGrid
import equiv_dens.utils.base as utils
from pyscf import gto


def spherical_grid(atoms, level=2):
    symbols = atoms['atom_types']
    positions = atoms['positions'][0]
    mol_dict = list(zip(symbols, positions))
    if atoms['atom_types'] == ['H']:
        mol = gto.M(atom=mol_dict, spin=1)
    else:
        mol = gto.M(atom=mol_dict)

    print('level', level)
    grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=level)
    for key in grid_spec.keys():
        grid_spec[key] = (torch.tensor(grid_spec[key][0] * utils.to_angstrom),
                          torch.tensor(grid_spec[key][1]))  # convert Bohr grid to Angstrom

    return grid_spec


def cubical_grid(atoms, nx=125, ny=125, nz=125, resolution=None,
                 margin=2, origin=None, extent=[10, 10, 10]):
    symbols = atoms['atom_types']
    positions = atoms['positions'][0]
    mol_dict = list(zip(symbols, positions))
    print('mol_dict', mol_dict)
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


def gen_grid_partition(positions, atom_types, coords, becke_scheme, f_radii_adjust=None):
    ngrids = coords.shape[1]
    natm = positions.shape[1]
    nbatch = positions.shape[0]
    atm_dist, _ = utils.calculate_distances_and_directions(positions)
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


def spherical_radial_sampling(grid_spec, n_samp, atom_types, pos,
                              radii_adjust=None,
                              rotate=False):
    grid_coords = []
    grid_weights = []
    # print('pos type', pos.type())
    for i, t in enumerate(atom_types):
        if rotate:
            rot_mat = torch.tensor(random_rotation_matrix()).to(pos)
        else:
            rot_mat = torch.eye(3).to(pos)
        # print('rot_mat type', rot_mat.type())
        # print('grid spec type', grid_spec[t][0].type())
        coords = pos[:, [i], :] + (grid_spec[t][0].unsqueeze(0) @ rot_mat)
        weights = grid_spec[t][1]
        pbecke = gen_grid_partition(pos, atom_types, coords, becke_scheme, radii_adjust)
        weights = weights * pbecke[:, i] * (1.0 / pbecke.sum(1))
        grid_coords.append(coords)
        grid_weights.append(weights)

    return collect_and_sample_grid(grid_coords, grid_weights, n_samp)


def spherical_sampling(grid_spec, n_samp, atom_types, pos):
    grid_coords = []
    grid_weights = []
    for i, t in enumerate(atom_types):
        grid_coords.append(pos[:, [i], :] + (grid_spec[t][0][None, :]))
        grid_weights.append(grid_spec[t][1] / len(atom_types))

    return collect_and_sample_grid(grid_coords, grid_weights, n_samp)


def rot_spherical_sampling(grid_spec, n_samp, atom_types, pos):
    grid_coords = []
    grid_weights = []
    for i, t in enumerate(atom_types):
        rot_mat = random_rotation_matrix()
        grid_coords.append(pos[:, [i], :] + (grid_spec[t][0][None, :]) @ rot_mat)
        grid_weights.append(grid_spec[t][1] / len(atom_types))

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
        return grid_coords[:, rand_idx, :], grid_weights[rand_idx]


def cubical_sampling(grid_spec, n_samp, atom_types, pos):
    flat_coords = np.reshape(grid_spec[0], (-1, 3))
    flat_coords = flat_coords[None, :]
    flat_coords = np.repeat(flat_coords, pos.shape[0], axis=0)
    if n_samp > flat_coords.shape[1]:
        return flat_coords, np.ones((flat_coords.shape[1], )) * grid_spec[1]
    else:
        rand_idx = np.random.choice(np.arange(flat_coords.shape[1]), size=n_samp, replace=False)
        return flat_coords[:, rand_idx, :], np.ones((n_samp, ) * grid_spec[1])


def dftpy_grid(lattice, gap):
    nr = np.zeros(3, dtype='int32')
    metric = np.dot(lattice.T, lattice)
    print('lattice', lattice)
    print('metric', np.sqrt(metric[0, 0]))
    print('gap', gap)
    for i in range(3):
        nr[i] = int(np.sqrt(metric[i, i]) / gap)
    print('The initial grid size is ', nr)
    for i in range(3):
        nr[i] = bestFFTsize(nr[i])
    print('The final grid size is ', nr)
    grid = DirectGrid(lattice=lattice, nr=nr, units=None, full=False)
    return grid


class CubicalGrid():

    def __init__(self, atoms, nx=125, ny=125, nz=125, resolution=None,
                 margin=2, origin=None, extent=[10, 10, 10], use_gpu=False, dtype=torch.double):
        self.use_gpu = use_gpu
        self.dtype = dtype
        symbols = atoms['atom_types']
        positions = atoms['positions'][0]
        mol_dict = list(zip(symbols, positions))
        print('mol_dict', mol_dict)
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
