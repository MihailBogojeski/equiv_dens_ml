import numpy as np
from pyscf.dft import gen_grid, radi
from .rotation import random_rotation_matrix
from pyscf import lib
from pyscf.lib import param


def spherical_grid(mols, level=2):
    print('level', level)
    grid_spec = gen_grid.gen_atomic_grids(mols[0], radi_method=radi.treutler_ahlrichs, level=level)
    for key in grid_spec.keys():
        grid_spec[key] = (grid_spec[key][0] * param.BOHR, grid_spec[key][1])  # convert Bohr grid to Angstrom

    return grid_spec


def cubical_grid(mols, nx=125, ny=125, nz=125, resolution=None,
                 margin=2, origin=None, extent=[10, 10, 10]):
    coord = mols[0].atom_coords(unit='Angstrom')  # positions in angstrom
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

    sample_volume /= param.BOHR
    sample_volume = np.prod(sample_volume)

    coords = lib.cartesian_prod([xs, ys, zs])
    coords = np.asarray(coords, order='C') - (-boxorig)
    return (coords, sample_volume)


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
    grid_coords = np.concatenate(grid_coords, axis=1)
    grid_weights = np.concatenate(grid_weights)

    if n_samp > grid_coords.shape[1]:
        return grid_coords, grid_weights
    else:
        rand_idx = np.random.choice(np.arange(grid_coords.shape[1]), size=n_samp, replace=False)
        return grid_coords[:, rand_idx, :], grid_weights[rand_idx]


def cubical_sampling(grid_spec, n_samp, atom_types, pos):
    flat_coords = np.reshape(grid_spec[0], (-1 , 3))
    flat_coords = flat_coords[None, :]
    flat_coords = np.repeat(flat_coords, pos.shape[0], axis=0)
    if n_samp > flat_coords.shape[1]:
        return flat_coords, np.ones((flat_coords.shape[1], )) * grid_spec[1]
    else:
        rand_idx = np.random.choice(np.arange(flat_coords.shape[1]), size=n_samp, replace=False)
        return flat_coords[:, rand_idx, :], np.ones((n_samp, ) * grid_spec[1])
