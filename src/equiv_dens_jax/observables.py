"""Native array dipoles and an explicit, nondifferentiable PySCF host boundary.

Grid dipoles retain the current clipping, neutral electron count and centering.
Analytic integrals are prepared from current host geometry/radial parameters;
only the later coefficient contraction and nuclear term are differentiated, as
in Torch. Host caching is explicit and must never be called from a jitted core.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import threading
import warnings

import numpy as np
import jax
import jax.numpy as jnp

from .common import attr

_PYSCF_LOCK = threading.RLock()


def prepare_layout(node, atoms_numpy):
    """Static PySCF shell ordering; no learned coefficient or radial values."""
    if node["type"] == "DipoleMomentCalc":
        return {}
    if node["type"] != "DipoleMomentIntorCalc":
        raise NotImplementedError(f"Unsupported observable: {node['type']}")
    species = tuple(map(int, np.max(np.asarray(atoms_numpy["batch_atom_numbers"]), axis=0)))
    if any(z == 0 for z in species):
        raise ValueError("The current analytic dipole layout requires an occupied species per slot")
    shells = []
    for i, z in enumerate(species):
        count = {}
        for orbital in attr(node, "orbital_basis")[z]:
            key = (z, orbital[2])
            index = count.get(key, 0)
            shells.append((i, key, index))
            count[key] = index + 1
    return {"species": species, "shells": tuple(shells)}


def coefficients_to_pyscf_vector(atoms, layout):
    """Keep the current p-component permutation and odd-L sign convention."""
    values = []
    for slot, key, index in layout["shells"]:
        value = atoms["spherical_coeffs"][slot][key][..., index]
        if key[1] == 1:
            value = value[..., jnp.asarray([2, 0, 1])]
        if key[1] % 2:
            value = -value
        values.append(value)
    return jnp.concatenate(values, axis=2).squeeze(axis=1)


def apply_observable(node, state, atoms, *, layout=None, center_coordinates=True,
                     integrals=None):
    """Pure-array observable; analytic mode requires explicitly prepared integrals."""
    result = dict(atoms)
    charges = atoms["batch_atom_numbers"]
    positions = atoms["batch_positions"]
    if node["type"] == "DipoleMomentCalc":
        density = jnp.where(atoms["density"] >= 0, atoms["density"], 0)
        electrons = jnp.sum(charges, axis=-1, keepdims=True)
        total = jnp.sum(density * atoms["coord_weights"], axis=1, keepdims=True)
        # Torch clamp retains derivative 1 at the lower boundary.
        total = jnp.where(total < 1e-30, 1e-30, total)
        density = density * (electrons / total)
        center = (jnp.sum(positions * charges[..., None], axis=1, keepdims=True)
                  / jnp.sum(charges[..., None], axis=1, keepdims=True)
                  if center_coordinates else jnp.zeros_like(positions[:, :1]))
        nuclear = jnp.sum((positions - center) * charges[..., None], axis=1)
        electronic = jnp.sum((density * atoms["coord_weights"])[..., None]
                             * (atoms["coords"] - center), axis=1)
    elif node["type"] == "DipoleMomentIntorCalc":
        if layout is None:
            raise ValueError("Analytic dipoles require a host-prepared shell layout")
        if integrals is None:
            integrals = atoms.get("dipole_integrals")
        if integrals is None:
            raise ValueError("Analytic dipoles require explicit current/frozen PySCF integrals")
        vector = coefficients_to_pyscf_vector(atoms, layout)
        integrals = jax.lax.stop_gradient(jnp.asarray(integrals, dtype=positions.dtype))
        if integrals.shape != (positions.shape[0], 3, vector.shape[-1]):
            raise ValueError("Analytic dipole integral and coefficient shapes differ")
        nuclear = jnp.sum(charges[..., None] * positions, axis=1)
        electronic = jnp.einsum("bji,bi->bj", integrals, vector)
        if attr(node, "remove_atom_density", False):
            if "atom_dipole_moment" not in atoms:
                raise ValueError("Free-atom analytic dipoles require host atom_dipole_moment")
            electronic = electronic + atoms["atom_dipole_moment"]
    else:
        raise NotImplementedError(f"Unsupported observable: {node['type']}")
    result["dipole_moment"] = nuclear - electronic
    return result


@contextmanager
def _unnormalized_gto():
    from pyscf import gto
    with _PYSCF_LOCK:
        previous = gto.mole.NORMALIZE_GTO
        gto.mole.NORMALIZE_GTO = False
        try:
            yield
        finally:
            gto.mole.NORMALIZE_GTO = previous


def prepare_integrals(node, atoms_numpy, *, layout=None):
    """Actual PySCF int1e_r preparation, outside JIT/AD, in electron-Angstrom units.

    Preserve the current supported basis helper: one-dimensional squeezed radial
    arrays per angular key. Historical scalar/multi-primitive helper failures
    are explicit instead of silently changing its basis construction.
    """
    from pyscf import gto, lib
    from pyscf.data.elements import ELEMENTS
    BOHR = lib.param.BOHR
    if layout is None:
        layout = prepare_layout(node, atoms_numpy)
    positions = np.asarray(atoms_numpy["batch_positions"])
    atom_types, counts = [], {}
    for z in layout["species"]:
        atom_types.append(ELEMENTS[z] + str(counts.get(z, 0)))
        counts[z] = counts.get(z, 0) + 1
    batches = []
    factor = 2 * math.sqrt(math.pi)
    for batch in range(positions.shape[0]):
        basis = {}
        for i, slot in enumerate(atoms_numpy["radial_scale"]):
            for key in slot:
                L = key[1]
                widths = np.maximum(np.asarray(atoms_numpy["radial_width"][i][key][batch]).squeeze(), 1e-10)
                scales = np.asarray(slot[key][batch]).squeeze()
                if widths.ndim != 1 or scales.shape != widths.shape:
                    raise NotImplementedError(
                        "Current PySCF ML basis helper requires 1D squeezed radial arrays per key"
                    )
                exponent = (2 * L + 3) / 2
                normalization = 1 / np.sqrt(math.gamma(exponent) / (2 * (2 * widths)**exponent))
                scales = scales / normalization
                shells = [[L, [float(width), float(scale)]] for width, scale in zip(widths, scales)]
                if atom_types[i] in basis:
                    basis[atom_types[i]].extend([[shell] for shell in shells])
                else:
                    basis[atom_types[i]] = [shells]
        atoms = list(zip(atom_types, positions[batch]))
        spin = sum(layout["species"]) % 2
        with _unnormalized_gto():
            molecule = gto.M(atom=atoms, basis=basis, spin=spin, verbose=0)
            molecule.build()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*divide by zero.*")
                helper = gto.M(atom=molecule.atom, basis={"default": [[0, (0.0, 1.0)]]},
                               spin=int(np.sum(molecule.atom_charges()) % 2), verbose=0)
        for shell in range(helper.nbas):
            pointer = helper._bas[shell, gto.mole.PTR_COEFF]
            length = helper.bas_nprim(shell) * helper.bas_nctr(shell)
            helper._env[pointer:pointer + length] = factor
        owners = [molecule.bas_atom(shell) for shell in range(molecule.nbas)
                  for _ in range(2 * molecule.bas_angular(shell) + 1)]
        integrals = gto.mole.intor_cross("int1e_r", helper, molecule)
        batches.append(integrals[:, owners, np.arange(molecule.nao)] * BOHR)
    return np.stack(batches)


def prepare_free_atom_dipole(atoms):
    """Explicit current MO/DF host free-atom moment, with the original AD boundary.

    MO density matrices and integrals are NumPy constants. DF moments retain
    their later JAX coefficient contraction; their integral/basis/position
    dependence remains detached. Positions and basis metadata must be host data.
    """
    from pyscf import gto, scf, lib
    BOHR = lib.param.BOHR
    positions = np.asarray(atoms["batch_positions"])
    numbers = np.asarray(atoms["batch_atom_numbers"])
    is_mo = "atom_mo_coeffs" in atoms
    if not is_mo and "atom_df_coeffs" not in atoms:
        raise ValueError("No free-atom MO or DF coefficients found")
    prefix = "atom_mo_coeffs" if is_mo else "atom_df_coeffs"
    basis = atoms[prefix + "_basis"]
    if is_mo:
        numbers = np.broadcast_to(np.max(numbers, axis=0), numbers.shape)
    moments = []
    for batch in range(positions.shape[0]):
        current_basis = basis[batch] if isinstance(basis, list) else basis
        specification = [(int(z), position) for z, position in zip(numbers[batch], positions[batch]) if z]
        with _PYSCF_LOCK:
            molecule = gto.M(atom=specification, basis=current_basis,
                             spin=int(np.sum(numbers[batch]) % 2), verbose=0)
        if is_mo:
            density_matrix = scf.hf.make_rdm1(np.asarray(atoms[prefix][batch]),
                                             np.asarray(atoms["atom_mo_coeffs_occ"][batch]))
            integrals = molecule.intor_symmetric("int1e_r", comp=3)
            moment = np.einsum("xij,ji->x", integrals, density_matrix).real * BOHR
            moments.append(jax.lax.stop_gradient(jnp.asarray(moment, dtype=positions.dtype)))
        else:
            with _unnormalized_gto():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*divide by zero.*")
                    helper = gto.M(atom=molecule.atom, basis={"default": [[0, (0.0, 1.0)]]},
                                   spin=int(np.sum(molecule.atom_charges()) % 2), verbose=0)
            for shell in range(helper.nbas):
                pointer = helper._bas[shell, gto.mole.PTR_COEFF]
                size = helper.bas_nprim(shell) * helper.bas_nctr(shell)
                helper._env[pointer:pointer + size] = 2 * math.sqrt(math.pi)
            owners = [molecule.bas_atom(shell) for shell in range(molecule.nbas)
                      for _ in range(2 * molecule.bas_angular(shell) + 1)]
            integral = gto.mole.intor_cross("int1e_r", helper, molecule)[:, owners, np.arange(molecule.nao)] * BOHR
            coefficients = jnp.asarray(atoms[prefix][batch])
            if coefficients.shape != (molecule.nao,):
                raise ValueError("Free-atom DF coefficient and actual molecule basis sizes differ")
            integral = jax.lax.stop_gradient(jnp.asarray(integral, dtype=positions.dtype))
            moments.append(jnp.einsum("ji,i->j", integral, coefficients))
    return jnp.stack(moments)


@dataclass
class HostDipoleCache:
    """Explicit legacy cache approximation; radial changes do not invalidate it."""
    positions: np.ndarray | None = None
    integrals: np.ndarray | None = None
    electronic: np.ndarray | None = None
    call_count: int = 0
    last_action: str = field(default="uninitialized", init=False)


def apply_host_observable(node, state, atoms, *, layout=None, cache=None):
    """Host orchestration preserving the existing displacement/cadence cache policy.

    Retain ``cache`` explicitly across calls to request that approximation.
    Reused electronic dipoles are constants, exactly as the detached Torch cache.
    For training, prepare current integrals outside AD and call apply_observable.
    """
    if node["type"] != "DipoleMomentIntorCalc":
        return apply_observable(node, state, atoms, layout=layout)
    if cache is None:
        cache = HostDipoleCache()
    if layout is None:
        layout = prepare_layout(node, atoms)
    if attr(node, "remove_atom_density", False):
        # The Torch dispatcher recomputes free-atom moments on every call, even
        # when the predicted electronic contribution is reused by cadence.
        atoms = dict(atoms)
        atoms["atom_dipole_moment"] = prepare_free_atom_dipole(atoms)
    cache.call_count += 1
    cadence = max(1, int(attr(node, "dipole_every_n_steps", 1)))
    nuclear = jnp.sum(atoms["batch_atom_numbers"][..., None] * atoms["batch_positions"], axis=1)
    compute = cadence <= 1 or cache.call_count % cadence == 1 or cache.electronic is None
    if not compute:
        result = dict(atoms)
        electronic = jax.lax.stop_gradient(jnp.asarray(cache.electronic, dtype=nuclear.dtype))
        if attr(node, "remove_atom_density", False):
            electronic = electronic + atoms["atom_dipole_moment"]
        result["dipole_moment"] = nuclear - electronic
        cache.last_action = "reused_electronic"
        return result
    positions = np.asarray(atoms["batch_positions"])
    reuse = (attr(node, "cache_integrals", True) and cache.positions is not None
             and cache.positions.shape == positions.shape and cache.integrals is not None
             and len(cache.integrals) == positions.shape[0]
             and np.max(np.abs(positions - cache.positions))
             < attr(node, "integral_cache_threshold", 0.5))
    if reuse:
        integrals = cache.integrals
        cache.last_action = "reused_integrals"
    else:
        integrals = prepare_integrals(node, atoms, layout=layout)
        cache.last_action = "prepared_integrals"
        if attr(node, "cache_integrals", True):
            cache.positions, cache.integrals = positions.copy(), integrals.copy()
    result = apply_observable(node, state, atoms, layout=layout, integrals=integrals)
    if cadence > 1:
        electronic = nuclear - result["dipole_moment"]
        if attr(node, "remove_atom_density", False):
            electronic = electronic - atoms["atom_dipole_moment"]
        cache.electronic = np.asarray(electronic).copy()
    return result
