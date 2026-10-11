"""Host preparation callback for current-geometry native JAX MD inputs.

PySCF grid construction and free-atom density sampling are intentionally kept
outside the differentiated/JIT pipeline.  No SCF is run as the geometry moves:
the optional atomic MO/DF coefficients and bases are loaded once from the
configured source free-atom table.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .preparation import (
    BOHR_ANGSTROM,
    _spline_from_entry,
    load_array_mapping,
    make_pyscf_grid,
    sample_sad_coeff_density,
    sample_sad_spline,
)


def _number_to_symbol(number: int) -> str:
    from pyscf.data.elements import ELEMENTS

    return ELEMENTS[int(number)]


def _basis_for_entry(value, atomic_number, key):
    if key not in value:
        raise KeyError(f"free-atom Z={atomic_number} entry is missing {key}")
    basis = value[key]
    if isinstance(basis, Mapping):
        if atomic_number in basis:
            return basis[atomic_number]
        if str(atomic_number) in basis:
            return basis[str(atomic_number)]
        symbol = _number_to_symbol(atomic_number)
        if symbol in basis:
            return basis[symbol]
        raise KeyError(f"free-atom Z={atomic_number} {key} has no basis for that element")
    return basis


class MDPropertyPreparer:
    """Refresh grid and optional SAD fields from each current geometry."""

    def __init__(self, config):
        if not isinstance(config, Mapping):
            raise TypeError("MD preparation config must be a mapping")
        config = dict(config)
        basis_path = config.get("basis_path", config.get("orbital_basis_path"))
        if basis_path is None:
            raise ValueError("MD preparation requires basis_path (the model orbital-basis metadata)")
        trusted_legacy = bool(config.get("trusted_legacy", False))
        self.orbital_basis = load_array_mapping(basis_path, trusted_legacy=trusted_legacy)
        if not self.orbital_basis:
            raise ValueError("basis metadata is empty")

        self.position_unit = str(config.get("position_unit", "angstrom")).lower()
        if self.position_unit not in {"angstrom", "bohr"}:
            raise ValueError("position_unit must be 'angstrom' or 'bohr'")
        self.grid_level = int(config.get("grid_level", 2))
        self.grid_count = config.get("grid_count", config.get("grid_samples"))
        self.grid_count = None if self.grid_count is None else int(self.grid_count)
        self.seed = int(config.get("seed", 0))
        self.make_grid = bool(config.get("density_expansion", True))
        self.density_grad = bool(config.get("density_grad", False))
        self.atom_density_type = config.get("atom_density_type", "none")
        if self.atom_density_type not in {"none", "spline", "mo_coeffs", "df_coeffs"}:
            raise ValueError("atom_density_type must be none, spline, mo_coeffs, or df_coeffs")
        self.include_atom_density = bool(config.get("remove_atom_density", False))
        self.include_atom_dipole = bool(config.get("include_atom_dipole", False))
        if self.include_atom_density and self.atom_density_type == "none":
            raise ValueError("remove_atom_density requires configured free-atom density data")
        if self.include_atom_dipole and self.atom_density_type not in {"mo_coeffs", "df_coeffs"}:
            raise ValueError("free-atom dipoles require MO or DF free-atom coefficients")
        self.atom_entries = None
        self.logr_splines = None
        if self.atom_density_type != "none":
            atom_path = config.get("atom_density_path")
            if atom_path is None:
                raise ValueError("atom_density_type requires atom_density_path")
            self.atom_entries = load_array_mapping(atom_path, trusted_legacy=trusted_legacy)
            normalized = {}
            for key, entry in self.atom_entries.items():
                try:
                    number = int(key)
                except (TypeError, ValueError):
                    from pyscf.data.elements import charge
                    number = int(charge(str(key)))
                if not isinstance(entry, Mapping):
                    raise TypeError(f"free-atom Z={number} entry must be a mapping")
                normalized[number] = dict(entry)
            self.atom_entries = normalized
            for number, entry in self.atom_entries.items():
                if self.atom_density_type == "spline":
                    if not ("spline_interp" in entry or {"x_samples", "y_samples"} <= entry.keys()
                            or {"t", "c", "k"} <= entry.keys()):
                        raise ValueError(f"free-atom Z={number} has no supported spline data")
                else:
                    basis_key = "mo_basis" if self.atom_density_type == "mo_coeffs" else "df_basis"
                    coeff_key = "mo_coeff" if self.atom_density_type == "mo_coeffs" else "df_coeffs"
                    _basis_for_entry(entry, number, basis_key)
                    if coeff_key not in entry:
                        raise KeyError(f"free-atom Z={number} entry is missing {coeff_key}")
                    if self.atom_density_type == "mo_coeffs" and "mo_occ" not in entry:
                        raise KeyError(f"free-atom Z={number} entry is missing mo_occ")
            if self.atom_density_type == "spline":
                # The spline-SAD energy path uses these host fields even when
                # grid subtraction is disabled.
                self.logr_splines = {
                    number: _spline_from_entry(entry)
                    for number, entry in self.atom_entries.items()
                }

        self._check_basis_metadata = bool(config.get("validate_elements", True))

    def _validate_species(self, numbers):
        if not self._check_basis_metadata:
            return
        available = set(self.orbital_basis)
        for z in np.unique(numbers[numbers > 0]):
            symbol = _number_to_symbol(int(z))
            if int(z) not in available and str(int(z)) not in available and symbol not in available:
                raise ValueError(f"model basis metadata has no entry for atomic number {int(z)} ({symbol})")

    def _padded_grid(self, numbers, positions):
        from pyscf import gto

        coords_by_batch, weights_by_batch = [], []
        to_bohr = 1.0 / BOHR_ANGSTROM if self.position_unit == "angstrom" else 1.0
        for batch, (charges, coords) in enumerate(zip(numbers, positions)):
            atoms = [(int(z), (xyz * to_bohr).tolist())
                     for z, xyz in zip(charges, coords) if z > 0]
            spin = int(sum(int(z) for z, _ in atoms) % 2)
            # Match the native grid/SAD conversion, including translated
            # geometries; PySCF's Angstrom constant differs in the last digits.
            molecule = gto.M(atom=atoms, unit="Bohr", spin=spin, verbose=0)
            molecule.build()
            grid_coords, grid_weights = make_pyscf_grid(
                molecule, level=self.grid_level, count=self.grid_count,
                seed=self.seed + batch, coords_unit="angstrom")
            coords_by_batch.append(grid_coords)
            weights_by_batch.append(grid_weights)
        if not coords_by_batch:
            raise ValueError("MD callback received an empty batch")
        max_points = max(map(len, coords_by_batch))
        coords_out = np.zeros((len(numbers), max_points, 3), dtype=np.float64)
        weights_out = np.zeros((len(numbers), max_points), dtype=np.float64)
        for batch, (coords, weights) in enumerate(zip(coords_by_batch, weights_by_batch)):
            coords_out[batch, :len(coords)] = coords
            weights_out[batch, :len(weights)] = weights
        return coords_out, weights_out

    def _add_joined_coefficients(self, atoms, numbers):
        if self.atom_density_type not in {"mo_coeffs", "df_coeffs"}:
            return
        prefix = "atom_mo_coeffs" if self.atom_density_type == "mo_coeffs" else "atom_df_coeffs"
        basis_key = "mo_basis" if self.atom_density_type == "mo_coeffs" else "df_basis"
        coeff_key = "mo_coeff" if self.atom_density_type == "mo_coeffs" else "df_coeffs"
        species = tuple(map(int, np.max(numbers, axis=0)))
        if any(z == 0 for z in species):
            raise ValueError("free-atom coefficient observables do not support padded species columns")
        basis_by_species = {}
        arrays = []
        occ_arrays = []
        for z in species:
            entry = self.atom_entries.get(z)
            if entry is None:
                raise KeyError(f"No free-atom coefficient entry for atomic number {z}")
            basis = _basis_for_entry(entry, z, basis_key)
            if isinstance(basis, Mapping) and z in basis:
                basis = basis[z]
            elif isinstance(basis, Mapping) and str(z) in basis:
                basis = basis[str(z)]
            elif isinstance(basis, Mapping) and _number_to_symbol(z) in basis:
                basis = basis[_number_to_symbol(z)]
            basis_by_species[z] = basis
            coeff = np.asarray(entry[coeff_key], dtype=np.float64)
            if self.atom_density_type == "df_coeffs":
                if isinstance(entry[coeff_key], (tuple, list)):
                    coeff = np.concatenate([np.asarray(x[1] if isinstance(x, tuple) else x).ravel()
                                            for x in entry[coeff_key]])
                arrays.append(coeff.reshape(-1))
            else:
                if coeff.ndim != 2:
                    raise ValueError(f"free-atom Z={z} mo_coeff must have shape [AO, MO]")
                arrays.append(coeff)
                occ_arrays.append(np.asarray(entry["mo_occ"], dtype=np.float64).reshape(-1))

        batch_size = numbers.shape[0]
        if self.atom_density_type == "df_coeffs":
            joined = np.concatenate(arrays)
            widths = [len(a) for a in arrays]
            mask = np.concatenate([
                np.repeat((numbers[:, i] > 0)[:, None], width, axis=1)
                for i, width in enumerate(widths)
            ], axis=1)
            atoms[prefix] = np.broadcast_to(joined, (batch_size, len(joined))).copy() * mask
            atoms[prefix + "_basis"] = basis_by_species
        else:
            nmo = sum(a.shape[1] for a in arrays)
            nao = sum(a.shape[0] for a in arrays)
            joined = np.zeros((batch_size, nao, nmo), dtype=np.float64)
            occupations = np.zeros((batch_size, nmo), dtype=np.float64)
            row = col = 0
            for i, (z, coeff, occ) in enumerate(zip(species, arrays, occ_arrays)):
                nr, nc = coeff.shape
                if occ.size != nc:
                    raise ValueError(f"free-atom Z={z} mo_occ size does not match mo_coeff")
                active = (numbers[:, i] > 0).astype(np.float64)
                joined[:, row:row + nr, col:col + nc] = coeff[None] * active[:, None, None]
                occupations[:, col:col + nc] = occ[None] * active[:, None]
                row += nr
                col += nc
            atoms[prefix] = joined
            atoms["atom_mo_coeffs_occ"] = occupations
            atoms[prefix + "_basis"] = basis_by_species

    def __call__(self, atoms):
        result = dict(atoms)
        numbers = np.asarray(result["batch_atom_numbers"], dtype=np.int64)
        positions = np.asarray(result["batch_positions"], dtype=np.float64)
        if numbers.ndim != 2 or positions.shape != (*numbers.shape, 3):
            raise ValueError("prepared batch geometry must have numbers [B,N] and positions [B,N,3]")
        if np.any(numbers < 0) or np.any(np.sum(numbers > 0, axis=1) == 0):
            raise ValueError("MD preparation requires nonnegative numbers and nonempty molecules")
        self._validate_species(numbers)
        positions_ang = positions if self.position_unit == "angstrom" else positions * BOHR_ANGSTROM
        if self.logr_splines is not None:
            result["logr_splines"] = self.logr_splines
        if self.make_grid:
            coords, weights = self._padded_grid(numbers, positions)
            result["coords"] = coords
            result["coord_weights"] = weights
        elif self.include_atom_density:
            if "coords" not in result or "coord_weights" not in result:
                raise ValueError("SAD subtraction with density_expansion=False requires existing coords and coord_weights")
            coords = np.asarray(result["coords"], dtype=np.float64)
            weights = np.asarray(result["coord_weights"], dtype=np.float64)
            if coords.ndim != 3 or coords.shape[0] != len(numbers) or coords.shape[-1] != 3:
                raise ValueError("existing coords must have shape [B,G,3]")
            if weights.shape != coords.shape[:2]:
                raise ValueError("existing coord_weights must have shape [B,G]")
        if self.include_atom_density:
            if self.atom_density_type == "spline":
                sampled = sample_sad_spline(positions_ang, numbers, coords,
                                            self.atom_entries, density_grad=self.density_grad)
            else:
                sampled, _ = sample_sad_coeff_density(
                    positions_ang, numbers, coords, self.atom_entries,
                    coefficient_type=self.atom_density_type,
                    density_grad=self.density_grad)
            if self.density_grad:
                result["atom_density"] = sampled[..., 0]
                result["atom_density_grad"] = sampled[..., 1:]
            else:
                result["atom_density"] = sampled
        if self.include_atom_dipole:
            self._add_joined_coefficients(result, numbers)
            from .observables import prepare_free_atom_dipole
            dipole_atoms = dict(result, batch_positions=positions_ang)
            result["atom_dipole_moment"] = np.asarray(prepare_free_atom_dipole(dipole_atoms))
        return result


def make_md_preparation(config):
    """Return a callable suitable for ``Calculator(..., prepare_properties=...)``.

    Required config key is ``basis_path`` (or ``orbital_basis_path``), a model
    orbital-basis NPZ or trusted legacy NPY. Optional free-atom settings use
    ``atom_density_type`` (``spline``, ``mo_coeffs`` or ``df_coeffs``),
    ``atom_density_path``, ``remove_atom_density``, ``density_grad``, and
    ``include_atom_dipole``. ``position_unit`` describes input batch positions;
    generated grid coordinates are always Angstrom. Spline mode also supplies
    host-only ``logr_splines`` as a mapping from atomic number to SciPy spline,
    including when ``remove_atom_density`` is false. Grids use ``grid_level``,
    optional deterministic ``grid_count``/``grid_samples``, and ``seed``.
    Object NPY inputs require explicit ``trusted_legacy=True``. Calculations
    only use the frozen source atomic data; no molecular SCF is launched here.
    """
    return MDPropertyPreparer(config)
