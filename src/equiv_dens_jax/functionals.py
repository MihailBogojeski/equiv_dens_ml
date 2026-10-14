"""Native current LDA/Ewald equations with explicit host grid preparation.

``prepare_grid`` freezes the source grid's numeric reciprocal data and Ewald
image enumeration. ``prepare_pseudopotential`` calls the existing DFTpy object
outside AD. Supply their outputs as ``functional_grid`` and ``v_real``; the
latter remains detached exactly as in the current Torch functional. Positive
density and an even final FFT axis are the existing equations' valid domain.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc

from .common import attr, child, parameter, buffer

BOHR = 0.52917721092
AXES = (1, 2, 3)


def make_grid_metadata(shape, lattice, *, volume=None, point_volume=None):
    """Build the current ReciprocalGrid metadata on the host using only NumPy.

    Inputs are bohr and fixed integers. Default volumes preserve CubicalGrid's
    product of lattice diagonal entries; callers supplying DFTpy metadata may
    pass its explicit volumes. Reciprocal Nyquist masking follows the original
    grid exactly, including its boundary-plane exclusions.
    """
    shape = np.asarray(shape, dtype=int)
    lattice = np.asarray(lattice)
    if shape.shape != (3,) or np.any(shape <= 0) or lattice.shape != (3, 3):
        raise ValueError("Functional grid requires three positive dimensions and a 3x3 lattice")
    axes = [np.fft.fftfreq(shape[i], d=1 / shape[i]) if i < 2
            else np.fft.rfftfreq(shape[i], d=1 / shape[i]) for i in range(3)]
    frequencies = np.stack(np.meshgrid(*axes, indexing="ij"))
    reciprocal_lattice = (2 * np.pi * np.linalg.inv(lattice)).T
    coords = np.einsum("j...,kj->k...", frequencies, reciprocal_lattice)
    gg = np.sum(coords**2, axis=0)
    mask = np.ones(gg.shape, dtype=bool)
    half, odd = shape // 2, shape % 2
    mask[:, :, half[2] + 1:] = False
    mask[0, 0, 0] = False
    mask[0, half[1] + 1:, 0] = False
    mask[half[0] + 1:, :, 0] = False
    if odd[2] == 0:
        mask[0, 0, half[2]] = False
        mask[0, half[1] + 1:, half[2]] = False
        mask[half[0] + 1:, :, half[2]] = False
        if odd[1] == 0:
            mask[0, half[1], half[2]] = False
        if odd[0] == 0:
            mask[half[0], 0, half[2]] = False
            mask[half[0], half[1] + 1:, half[2]] = False
    if odd[0] == 0:
        mask[half[0], half[1] + 1:, 0] = False
        if odd[1] == 0:
            mask[half[0], half[1], 0] = False
    if odd[1] == 0:
        mask[0, half[1], 0] = False
    if np.all(odd == 0):
        mask[half[0], half[1], half[2]] = False
    volume = np.prod(np.diag(lattice)) if volume is None else volume
    point_volume = volume / np.prod(shape) if point_volume is None else point_volume
    return {"shape": shape, "lattice": lattice.copy(), "volume": np.asarray(volume),
            "point_volume": np.asarray(point_volume), "gg": gg,
            "reciprocal_coords": coords, "reciprocal_mask": mask}


def prepare_grid(grid_host, *, ewald_node, initial_eta_dtype=np.float32):
    """Extract fixed metadata from the actual source grid, without importing Torch.

    ``grid_host`` may instead be a dictionary with shape, lattice, volume,
    point_volume, gg, reciprocal_coords and reciprocal_mask. A warmed module's
    exported eta is honored; a cold module uses the source's default FP32 eta
    initialization unless the source default dtype was explicitly overridden.
    The original lattice @ image-index convention and grid volume are retained.
    Prepare again when the cell changes, retaining eta if the source retained it.
    """
    if isinstance(grid_host, dict):
        values = grid_host
    else:
        reciprocal = grid_host.get_reciprocal_grid()
        values = {"shape": grid_host.shape, "lattice": grid_host.lattice,
                  "volume": grid_host.volume, "point_volume": grid_host.point_volume,
                  "gg": reciprocal.gg, "reciprocal_coords": reciprocal.coords,
                  "reciprocal_mask": reciprocal.mask}
    shape = tuple(int(value) for value in np.asarray(values["shape"]))
    if len(shape) != 3 or min(shape) < 1 or shape[-1] % 2:
        raise ValueError("Current LDA inverse real FFT requires a positive 3D grid with even final axis")
    lattice = np.asarray(values["lattice"])
    gg = np.asarray(values["gg"])
    coords = np.asarray(values["reciprocal_coords"])
    mask = np.asarray(values["reciprocal_mask"], dtype=bool)
    reciprocal_shape = (*shape[:2], shape[2] // 2 + 1)
    if (lattice.shape != (3, 3) or gg.shape != reciprocal_shape
            or coords.shape != (3, *reciprocal_shape) or mask.shape != reciprocal_shape):
        raise ValueError("Functional grid and reciprocal metadata shapes differ")
    if not all(np.isfinite(value).all() for value in (lattice, gg, coords)):
        raise ValueError("Functional grid contains nonfinite metadata")
    if np.any(gg < 0) or gg.flat[0] != 0 or np.any(gg.reshape(-1)[np.flatnonzero(mask)] <= 0):
        raise ValueError("Ewald reciprocal mask must select strictly positive squared frequencies")
    volume, point_volume = float(np.asarray(values["volume"])), float(np.asarray(values["point_volume"]))
    if not (np.isfinite(volume) and volume > 0 and np.isfinite(point_volume) and point_volume > 0):
        raise ValueError("Functional grid volumes must be positive and finite")
    charges = np.asarray(attr(ewald_node, "a_num"), dtype=gg.dtype)
    if charges.ndim != 1 or len(charges) == 0 or not np.isfinite(charges).all():
        raise ValueError("Ewald requires a nonempty finite charge vector")
    precision = float(attr(ewald_node, "prec", 1e-8))
    if not 0 < precision < 3:
        raise ValueError("Ewald precision is outside the source cutoff domain")
    from scipy.special import erfc as host_erfc, erfcinv
    eta = attr(ewald_node, "eta")
    if eta is None:
        gmax = max(np.sqrt(gg[:, 0, 0].max()), np.sqrt(gg[0, :, 0].max()),
                   np.sqrt(gg[0, 0, :].max()))
        eta = np.asarray(1.6, dtype=initial_eta_dtype).astype(gg.dtype)
        while eta > 0:
            bound = (4 * np.pi * len(charges) * np.sum(charges**2)
                     * np.sqrt(eta / np.pi) * host_erfc(gmax / 2 * np.sqrt(1 / eta)))
            if bound < precision:
                break
            eta = np.asarray(eta - 0.01, dtype=gg.dtype)
        if eta <= 0:
            raise ValueError("Current reciprocal grid cannot select a positive Ewald eta")
    eta = float(np.asarray(eta))
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("Ewald eta must be positive and finite")
    cutoff = np.asarray(erfcinv(precision / 3), dtype=gg.dtype) / np.sqrt(np.asarray(eta, dtype=gg.dtype))
    lengths = np.sqrt(np.sum(lattice**2, axis=1))
    if np.any(lengths <= 0):
        raise ValueError("Ewald lattice row lengths must be positive")
    count = np.ceil(cutoff / lengths).astype(int)
    indices = np.stack(np.meshgrid(*(np.arange(-n, n + 1) for n in count), indexing="ij"), axis=-1).reshape(-1, 3)
    translations = np.einsum("ij,tj->ti", lattice, indices)
    return {"gg": gg.copy(), "reciprocal_coords": coords.copy(),
            "reciprocal_mask_indices": np.flatnonzero(mask).astype(np.int32),
            "volume": np.asarray(volume, dtype=gg.dtype),
            "point_volume": np.asarray(point_volume, dtype=gg.dtype),
            "eta": np.asarray(eta, dtype=gg.dtype),
            "real_cutoff": np.asarray(cutoff, dtype=gg.dtype),
            "translations": translations.astype(gg.dtype)}


def prepare_pseudopotential(atoms_host, *, shape):
    """Run the actual mutable DFTpy pseudopotential adapter outside all AD.

    This mirrors source ``idx`` selection, restart and density evaluation. Keep
    the returned field fixed during differentiation; refresh explicitly after
    geometry/cell/input changes. Its values are never inferred from density.
    """
    density = np.asarray(atoms_host["density"]).reshape((-1, *shape))
    pseudo = atoms_host["pseudo_pot"]
    fields = []
    for index in atoms_host["idx"]:
        index = int(index)
        pseudo.restart(grid=atoms_host["dftpy_grid"], ions=atoms_host["ions"][index])
        pseudo(density[index])
        fields.append(np.asarray(pseudo.vreal).copy())
    result = np.stack(fields).astype(density.dtype)
    if result.shape != density.shape or not np.isfinite(result).all():
        raise ValueError("Current pseudopotential batch selection or output shape is invalid")
    return result


def prepare_functional_inputs(node, atoms_host, *, initial_eta_dtype=np.float32):
    """Explicit host preparation after the current density head has evaluated.

    The returned dictionary still contains caller host objects; pass only its
    numeric fields into JIT. No learned density/radial values enter grid metadata.
    """
    if node["type"] != "LDAFunctional":
        raise NotImplementedError(f"Unsupported density functional: {node['type']}")
    grid = atoms_host["grid"]
    shape = tuple(int(value) for value in np.asarray(grid["shape"] if isinstance(grid, dict) else grid.shape))
    return dict(atoms_host,
                functional_grid=prepare_grid(grid, ewald_node=child(node, "ewald"), initial_eta_dtype=initial_eta_dtype),
                v_real=prepare_pseudopotential(atoms_host, shape=shape))


def thomas_fermi_en(rho, grid):
    return jnp.sum(rho**(5 / 3), axis=AXES) * (3 / 10) * (3 * np.pi**2)**(2 / 3) * grid["point_volume"]


def von_weizsacker_pot(rho, grid):
    root = jnp.sqrt(rho)
    transformed = jnp.fft.rfftn(root, axes=AXES) * grid["point_volume"]
    laplacian = jnp.fft.irfftn(transformed * grid["gg"], axes=AXES) / grid["point_volume"]
    return 0.5 * laplacian / root


def von_weizsacker_en(rho, grid):
    return jnp.sum(rho * von_weizsacker_pot(rho, grid), axis=AXES) * grid["point_volume"]


def lda_en(rho, grid):
    root = rho**(1 / 3)
    rs = (3 / (4 * np.pi))**(1 / 3) / root
    exchange = -3 / 4 * (3 / np.pi)**(1 / 3) * root
    corr1 = (0.0311 * jnp.log(rs) - 0.048 + 0.002 * rs * jnp.log(rs) - 0.0116 * rs) * (rs < 1)
    corr2 = -0.1423 / (1 + 1.0529 * jnp.sqrt(rs) + 0.3334 * rs) * (rs >= 1)
    return jnp.sum((exchange + corr1 + corr2) * rho, axis=AXES) * grid["point_volume"]


def hartree_en(rho, grid):
    gg = grid["gg"].at[0, 0, 0].set(1)
    potential = jnp.fft.rfftn(rho, axes=AXES) / gg * 4 * np.pi
    potential = potential.at[:, 0, 0, 0].set(0)
    real_potential = jnp.fft.irfftn(potential, axes=AXES)
    return jnp.sum(real_potential * rho, axis=AXES) * grid["point_volume"] / 2


def pseudo_en(rho, grid, v_real):
    if v_real is None:
        return 0
    return jnp.sum(jax.lax.stop_gradient(v_real) * rho, axis=AXES) * grid["point_volume"]


def ewald_terms(node, grid, positions):
    """Source Ewald decomposition; positions and translations are in bohr."""
    charges = jnp.asarray(attr(node, "a_num"), dtype=positions.dtype)
    if positions.ndim != 3 or positions.shape[1:] != (len(charges), 3):
        raise ValueError("Ewald positions must have shape (batch, fixed_atoms, 3)")
    eta = grid["eta"]
    images = (positions[:, None, :, :] - grid["translations"][None, :, None, :]).reshape(positions.shape[0], -1, 3)
    image_charges = jnp.tile(charges, grid["translations"].shape[0])
    real = jnp.zeros((positions.shape[0],), dtype=positions.dtype)
    for i in range(len(charges)):
        squared = jnp.sum((images - positions[:, i, None, :])**2, axis=-1)
        # cdist has zero first derivative at a self pair. Avoid sqrt(0)'s NaN
        # cotangent while preserving the source exclusion distances <= 0.01.
        nonzero = squared > 0
        distance = jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, squared, 1)), 0)
        included = (distance < grid["real_cutoff"]) & (distance > 0.01)
        denominator = distance + (distance == 0)
        real += charges[i] * jnp.sum(included * image_charges * erfc(jnp.sqrt(eta) * distance) / denominator, axis=1)
    real /= 2
    correction = (-jnp.sqrt(eta / np.pi) * jnp.sum(charges**2)
                  - 4 * np.pi / (4 * eta * grid["volume"]) / 2 * jnp.sum(charges)**2)
    coords = grid["reciprocal_coords"]
    structure = charges[0] * jnp.exp(-1j * jnp.einsum("lijk,bl->bijk", coords, positions[:, 0]))
    for i in range(1, len(charges)):
        structure += charges[i] * jnp.exp(-1j * jnp.einsum("lijk,bl->bijk", coords, positions[:, i]))
    squared_structure = jnp.conj(structure) * structure
    gg = grid["gg"].at[0, 0, 0].set(1)
    inverse = (1 / gg).at[0, 0, 0].set(0)
    gg = gg.at[0, 0, 0].set(0)
    indices = grid["reciprocal_mask_indices"]
    selected = lambda value: jnp.take(value.reshape(-1), indices)
    reciprocal = jnp.sum(jnp.take(squared_structure.reshape(positions.shape[0], -1), indices, axis=1)
                         * jnp.exp(-selected(gg) / (4 * eta)) * selected(inverse), axis=1)
    reciprocal = 4 * np.pi * reciprocal.real / grid["volume"]
    return {"real": real, "correction": correction, "reciprocal": reciprocal}


def apply_functional(node, state, atoms):
    """Return current energy_min and optional energy; do not derive forces here."""
    if node["type"] != "LDAFunctional":
        raise NotImplementedError(f"Unsupported density functional: {node['type']}")
    if "functional_grid" not in atoms or "v_real" not in atoms:
        raise ValueError("LDA requires explicitly prepared functional_grid and detached v_real")
    grid, potential = atoms["functional_grid"], atoms["v_real"]
    if potential.ndim != 4 or potential.shape[-1] % 2:
        raise ValueError("Current LDA requires a 4D pseudopotential field with even final FFT axis")
    rho = atoms["density"].reshape(potential.shape)
    positions = atoms["shifted_positions"] / BOHR
    terms = ewald_terms(child(node, "ewald"), grid, positions)
    total = (terms["real"] + terms["correction"] + terms["reciprocal"]
             + thomas_fermi_en(rho, grid) + von_weizsacker_en(rho, grid)
             + lda_en(rho, grid) + hartree_en(rho, grid) + pseudo_en(rho, grid, potential))
    result = dict(atoms, energy_min=total)
    if attr(node, "store_energy", False):
        offset = parameter(node, state, "en_offset") if "en_offset" in node["params"] else buffer(node, state, "en_offset")
        result["energy"] = total + offset
    return result
