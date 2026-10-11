"""Host-side PySCF preparation for the native JAX DenSnet pipeline.

This module deliberately imports no Torch code. PySCF molecules, AO evaluation,
grid construction, and spline interpolation stay on the host; returned values
are ordinary NumPy arrays that callers may pass to ``data.prepare_atoms`` and
the JAX pipeline.

Legacy ``.npy`` files containing Python objects require pickle. Loading them is
therefore opt-in with ``trusted_legacy=True``. New datasets should use numeric
``.npz`` archives and are loaded without pickle support.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

BOHR_ANGSTROM = 0.529177210903


def load_array_mapping(path: str | Path, *, trusted_legacy: bool = False) -> dict[str, Any]:
    """Load a numeric NPZ mapping, or an explicitly trusted object NPY mapping.

    ``atoms.npy`` and historical ``QC-results.npy`` are commonly object arrays.
    Because NumPy pickle loading can execute code, callers must opt in for those
    files. No pickle is enabled for NPZ inputs.
    """
    path = Path(path)
    if path.is_dir():
        from .serialization import load_tree
        value = load_tree(path)
        if not isinstance(value, dict):
            raise TypeError(f"expected a mapping in portable bundle {path}")
        return value
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    if not trusted_legacy:
        raise ValueError("object NPY loading requires trusted_legacy=True")
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping in {path}, got {type(value).__name__}")
    return value


def load_legacy_qc_results(path: str | Path, *, trusted_legacy: bool = False):
    """Load historical ``QC-results.npy`` records as ``(mol_dict, calc_dict)``.

    The tuple/list ordering matches ``AtomsDensityData``. This helper does not
    import or deserialize model objects beyond the explicitly trusted NumPy
    object archive.
    """
    if not trusted_legacy:
        raise ValueError("QC-results NPY loading requires trusted_legacy=True")
    raw = np.load(Path(path), allow_pickle=True)
    records = raw.tolist() if isinstance(raw, np.ndarray) else raw
    if not isinstance(records, (list, tuple)):
        raise TypeError("QC results must contain a sequence of (mol_dict, calc_dict) records")
    result = []
    for index, record in enumerate(records):
        if not isinstance(record, (tuple, list)) or len(record) != 2:
            raise ValueError(f"QC result {index} is not a (mol_dict, calc_dict) pair")
        mol_dict, calc_dict = record
        if not isinstance(mol_dict, dict) or not isinstance(calc_dict, dict):
            raise TypeError(f"QC result {index} must contain mapping values")
        result.append((mol_dict, calc_dict))
    return result


def load_current_dataset(atom_path: str | Path, orbital_basis_path: str | Path, *,
                         qc_results_path: str | Path | None = None,
                         trusted_legacy: bool = False):
    """Load the three current pipeline inputs without hiding pickle use.

    The returned mapping contains ``atoms``, ``orbital_basis`` and, when
    supplied, ``qc_results``. Legacy object NPY files require the explicit
    ``trusted_legacy`` opt-in; numeric NPZ mappings remain safe by default.
    """
    atoms = load_array_mapping(atom_path, trusted_legacy=trusted_legacy)
    orbital_basis = load_array_mapping(orbital_basis_path, trusted_legacy=trusted_legacy)
    result = {"atoms": atoms, "orbital_basis": orbital_basis}
    if qc_results_path is not None:
        result["qc_results"] = load_legacy_qc_results(
            qc_results_path, trusted_legacy=trusted_legacy)
    return result


def build_pyscf_molecules(records, *, basis=None):
    """Reconstruct and build PySCF molecules from legacy QC records.

    ``basis`` optionally overrides each saved molecule's basis, as in the
    projected-density path. Coordinates and basis metadata remain in PySCF's
    native units/order.
    """
    from pyscf import gto

    molecules = []
    for index, (mol_dict, _calc_dict) in enumerate(records):
        kwargs = dict(mol_dict)
        if basis is not None:
            kwargs["basis"] = basis
        mol = gto.Mole(**kwargs)
        try:
            mol.build()
        except Exception as exc:
            raise ValueError(f"could not build PySCF molecule {index}") from exc
        molecules.append(mol)
    return molecules


def make_pyscf_grid(mol, *, level: int = 2, count: int | None = None,
                    seed: int = 0, coords_unit: str = "angstrom"):
    """Construct a PySCF atom-centered integration grid for one molecule.

    Returns ``(coords, weights)`` as float64 arrays. PySCF produces Bohr
    coordinates; the default result converts only coordinates to Angstrom, as
    the current dataset's ``get_pyscf_coords`` does. If ``count`` is smaller
    than the full grid, a deterministic sample is drawn without replacement.
    """
    from pyscf.dft import gen_grid, radi

    if level < 0:
        raise ValueError("grid level must be nonnegative")
    if coords_unit not in {"angstrom", "bohr"}:
        raise ValueError("coords_unit must be 'angstrom' or 'bohr'")
    grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=level)
    coords, weights = gen_grid.get_partition(mol, grid_spec)
    if count is not None:
        if count <= 0:
            raise ValueError("grid count must be positive")
        if count < len(coords):
            idx = np.random.default_rng(seed).choice(len(coords), count, replace=False)
            coords, weights = coords[idx], weights[idx]
    coords = np.asarray(coords, dtype=np.float64)
    if coords_unit == "angstrom":
        coords = coords * BOHR_ANGSTROM
    return coords, np.asarray(weights, dtype=np.float64)


def batch_pyscf_grids(molecules, *, level: int = 2, count: int | None = None,
                      seed: int = 0, coords_unit: str = "angstrom"):
    """Create equal-length batches of molecular grids, zero-padding shorter grids.

    Padding weights are zero so they contribute nothing to grid integrals.
    Returns ``coords[B,G,3]`` and ``weights[B,G]``.
    """
    samples = [make_pyscf_grid(mol, level=level, count=count, seed=seed + i,
                               coords_unit=coords_unit) for i, mol in enumerate(molecules)]
    if not samples:
        raise ValueError("at least one molecule is required")
    max_points = max(len(coords) for coords, _ in samples)
    coords_out = np.zeros((len(samples), max_points, 3), dtype=np.float64)
    weights_out = np.zeros((len(samples), max_points), dtype=np.float64)
    for i, (coords, weights) in enumerate(samples):
        coords_out[i, :len(coords)] = coords
        weights_out[i, :len(weights)] = weights
    return coords_out, weights_out


def _molecular_density_one(mol, coords_bohr, coefficients, *, projected, density_grad):
    from pyscf.dft import numint

    ao = numint.eval_ao(mol, coords_bohr, deriv=int(density_grad))
    if projected:
        coeffs = np.asarray(coefficients)
        if density_grad:
            if coeffs.ndim > 1:
                rho = sum(np.einsum("ijk,k->ij", ao, coeffs[j])
                          for j in range(coeffs.shape[0]))
            else:
                rho = np.einsum("ijk,k->ij", ao, coeffs)
        elif coeffs.ndim > 1:
            rho = sum(np.einsum("ij,j->i", ao, coeffs[j])
                      for j in range(coeffs.shape[0]))
        else:
            rho = np.einsum("ij,j->i", ao, coeffs)
    else:
        xctype = "GGA" if density_grad else "LDA"
        mo_coeff = np.asarray(coefficients["mo_coeff"])
        mo_occ = np.asarray(coefficients["mo_occ"])
        if mo_occ.ndim > 1:
            rho = sum(numint.eval_rho2(mol, ao, mo_occ=mo_occ[j],
                                       mo_coeff=mo_coeff[j], xctype=xctype)
                      for j in range(mo_occ.shape[0]))
        else:
            rho = numint.eval_rho2(mol, ao, xctype=xctype,
                                   mo_occ=mo_occ, mo_coeff=mo_coeff)
    # Match sample_density_base: PySCF's derivative channels move from [4,G] to [G,4].
    return np.asarray(rho).T if density_grad else np.asarray(rho)


def sample_molecular_density(molecules, coords, coefficients, *, projected=False,
                             density_grad=False, coords_unit="angstrom"):
    """Sample MO or projected-AO density and optional explicit spatial gradient.

    ``coords`` is ``[B,G,3]`` and ``coefficients`` has one item per molecule.
    The projected path contracts coefficients directly against PySCF AO values
    in the basis ordering that created those coefficients. For a legacy
    ``df_coeff`` the matching auxiliary basis must therefore be installed on
    each molecule before calling this function (or use ``build_pyscf_molecules``
    with ``basis=auxbasis``). It intentionally does not reinterpret projected
    AO coefficients as the model's spherical-shell coefficient vector.

    Density-gradient output is ``[B,G,4]``: density followed by x/y/z
    derivatives in PySCF's per-Bohr convention, matching the historical
    molecular sampler.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape [batch, grid, 3]")
    if len(molecules) != coords.shape[0] or len(coefficients) != len(molecules):
        raise ValueError("molecule, coordinate, and coefficient batch sizes must match")
    if coords_unit not in {"angstrom", "bohr"}:
        raise ValueError("coords_unit must be 'angstrom' or 'bohr'")
    scale = 1.0 / BOHR_ANGSTROM if coords_unit == "angstrom" else 1.0
    result = []
    width = 4 if density_grad else 1
    for mol, points, coeff in zip(molecules, coords, coefficients):
        if len(mol.atom_charges()) == 0:
            vals = np.zeros((len(points), width), dtype=np.float64)
        else:
            vals = _molecular_density_one(mol, points * scale, coeff,
                                          projected=projected, density_grad=density_grad)
            if density_grad:
                # PySCF AO derivatives are per Bohr; source scaled Angstrom coords
                # to Bohr before evaluation and kept PySCF's derivative units.
                # Preserve that contract for dataset callers.
                pass
            else:
                vals = vals.reshape((len(points),))
                vals = vals[:, None]
        result.append(vals)
    max_points = coords.shape[1]
    out = np.zeros((len(molecules), max_points, width), dtype=np.float64)
    for i, vals in enumerate(result):
        out[i, :len(vals)] = vals
    return out if density_grad else out[..., 0]


def sample_qc_results(records, coords, *, projected=False, density_grad=False,
                      basis=None, coords_unit="angstrom"):
    """Sample density directly from legacy QC result records.

    MO density reads ``mo_coeff``/``mo_occ`` from each calculation mapping.
    Projected density reads the current ``df_coeff`` (or ``df_coeffs``) and
    uses the corresponding ``auxbasis`` unless ``basis`` is explicitly given.
    The coefficient vector is passed through in PySCF AO order; no spherical
    orbital reordering is inferred here.
    """
    coeffs = []
    per_record_basis = basis
    for i, (mol_dict, calc_dict) in enumerate(records):
        if projected:
            if "df_coeff" in calc_dict:
                coeff = calc_dict["df_coeff"]
            elif "df_coeffs" in calc_dict:
                coeff = calc_dict["df_coeffs"]
            else:
                raise KeyError(f"QC record {i} has no df_coeff/df_coeffs")
            if isinstance(coeff, (list, tuple)):
                coeff = np.concatenate([np.asarray(x[1] if isinstance(x, tuple) else x).ravel()
                                        for x in coeff])
            coeffs.append(np.asarray(coeff))
            if per_record_basis is None:
                per_record_basis = calc_dict.get("auxbasis")
                if per_record_basis is None:
                    raise KeyError(f"QC record {i} has no auxbasis for projected density")
        else:
            if "mo_coeff" not in calc_dict or "mo_occ" not in calc_dict:
                raise KeyError(f"QC record {i} has no mo_coeff/mo_occ")
            coeffs.append({"mo_coeff": calc_dict["mo_coeff"], "mo_occ": calc_dict["mo_occ"]})
    molecules = build_pyscf_molecules(records, basis=per_record_basis if projected else basis)
    return sample_molecular_density(molecules, coords, coeffs, projected=projected,
                                    density_grad=density_grad, coords_unit=coords_unit)


def prepare_dataset_batches(atom_path: str | Path, basis_path: str | Path,
                            qc_path: str | Path, count: int | None = None,
                            seed: int = 0, trusted_legacy: bool = False, *,
                            projected: bool = False, density_grad: bool = False,
                            grid_level: int = 2, center_positions: bool = True):
    """Yield one-molecule native JAX input/target mappings from current raw files.

    This adapts the current ``atoms.npy`` + orbital-basis + ``QC-results.npy``
    inputs. Each item is a flat mapping accepted by ``data.prepare_atoms`` and
    ``data.load_batches``; labels are fields alongside inputs (``energy``,
    ``forces``, ``density`` and optionally ``density_grad``). Density is
    evaluated with the saved QC MO or projected-DF coefficients, and input
    positions/grid points are centered with the same charge-weighted shift.

    If ``count`` is below a molecule's full PySCF grid size, a reproducible
    uniform subset of integration-grid nodes is selected and the original
    per-node quadrature weights are retained unchanged. This mirrors the
    legacy subset policy; it is not a stratified or unbiased reweighting
    estimator. PySCF AO coefficients remain in their native basis ordering.

    ``basis_path`` is the project's orbital-basis metadata file. It is loaded
    and validated here for provenance/format consistency; density evaluation
    itself uses the AO basis saved in each QC record (or ``auxbasis`` for
    projected density), not the model's spherical-shell representation.
    """
    from .data import prepare_atoms

    current = load_current_dataset(atom_path, basis_path, qc_results_path=qc_path,
                                   trusted_legacy=trusted_legacy)
    atoms = current["atoms"]
    orbital_basis = current["orbital_basis"]
    if not isinstance(orbital_basis, dict):
        raise TypeError("orbital-basis metadata must be a mapping")
    records = current["qc_results"]
    positions = np.asarray(atoms["positions"], dtype=np.float64)
    numbers = np.asarray(atoms["atom_numbers"], dtype=np.int64)
    if positions.ndim == 2:
        positions = positions[None]
    if numbers.ndim == 1:
        numbers = np.broadcast_to(numbers[None], positions.shape[:2]).copy()
    if positions.ndim != 3 or positions.shape[-1] != 3 or numbers.shape != positions.shape[:2]:
        raise ValueError("atoms.npy positions/atom_numbers must have matching [M,N,3]/[M,N] shapes")
    if len(records) != len(positions):
        raise ValueError("QC-results record count does not match atoms.npy molecule count")
    if count is not None and count <= 0:
        raise ValueError("grid count must be positive")

    from pyscf import gto
    for index, ((mol_dict, _calc_dict), pos_row, num_row) in enumerate(zip(records, positions, numbers)):
        active = num_row > 0
        if not np.any(active):
            raise ValueError(f"molecule {index} contains no atoms")
        mol = gto.Mole(**dict(mol_dict))
        mol.build()
        mol_charges = np.asarray(mol.atom_charges(), dtype=np.int64)
        if not np.array_equal(mol_charges, num_row[active]):
            raise ValueError(f"molecule {index}: QC atom order/numbers do not match atoms.npy")
        if len(mol_charges) != int(active.sum()):
            raise ValueError(f"molecule {index}: QC and atom array sizes do not match")
        raw_positions = pos_row[active]
        qc_positions = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64)
        if not np.allclose(qc_positions, raw_positions, atol=1e-6, rtol=0):
            raise ValueError(f"molecule {index}: QC coordinates do not match atoms.npy")

        coords, weights = make_pyscf_grid(mol, level=grid_level, count=count,
                                          seed=seed + index, coords_unit="angstrom")
        density_pack = sample_qc_results(
            [records[index]], coords[None], projected=projected,
            density_grad=density_grad, coords_unit="angstrom")
        shift = (np.sum(raw_positions * num_row[active, None], axis=0)
                 / np.sum(num_row[active])) if center_positions else np.zeros(3)
        coords_centered = coords - shift
        centered_positions = pos_row - shift
        batch = {
            "positions": centered_positions[None],
            "atom_numbers": num_row[None],
            "coords": coords_centered[None],
            "coord_weights": weights[None],
            "idx": np.asarray([[index]], dtype=np.int64),
            "pos_shift": (-shift)[None, None, :],
        }
        if density_grad:
            batch["density"] = density_pack[..., 0]
            batch["density_grad"] = density_pack[..., 1:]
        else:
            batch["density"] = density_pack
        for name in ("energy", "forces"):
            if name not in atoms:
                continue
            value = np.asarray(atoms[name][index])
            if name == "energy":
                batch[name] = value.reshape(1, *value.shape)
            else:
                force = value[None] if value.ndim == 2 else value
                if force.shape != positions[index:index + 1].shape:
                    raise ValueError(f"molecule {index}: force labels do not match positions")
                batch[name] = force
        yield prepare_atoms(batch)


def _spline_from_entry(entry):
    if "spline_interp" in entry:
        # Legacy scipy splines can only enter through an explicitly trusted object file.
        return entry["spline_interp"]
    from scipy.interpolate import BSpline, make_interp_spline
    if {"t", "c", "k"} <= set(entry):
        return BSpline(np.asarray(entry["t"]), np.asarray(entry["c"]), int(entry["k"]))
    if {"x_samples", "y_samples"} <= set(entry):
        radii = np.asarray(entry["x_samples"], dtype=np.float64)
        values = np.asarray(entry["y_samples"], dtype=np.float64)
        if np.any(radii <= 0):
            raise ValueError("SAD spline radii must be positive")
        return make_interp_spline(np.log(radii), values,
                                  k=int(entry.get("spline_order", 7)))
    raise ValueError("SAD spline entry needs spline_interp, t/c/k, or x_samples/y_samples")


def sample_sad_spline(positions, atom_numbers, coords, spline_entries, *, density_grad=False):
    """Evaluate legacy radial SAD splines for padded batches of atoms.

    Coordinates and atom positions are Angstrom. Radial tables are interpolated
    against log(radius in Bohr), and negative spline density is clamped to zero
    exactly as in ``gpu_splines.eval_spline_density_gpu``. Returned gradients
    are per Angstrom and have shape ``[B,G,4]`` when requested.
    """
    positions = np.asarray(positions, dtype=np.float64)
    numbers = np.asarray(atom_numbers, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B,N,3]")
    if numbers.shape != positions.shape[:2] or coords.ndim != 3 or coords.shape[0] != len(positions):
        raise ValueError("atom_numbers/coords batch shape mismatch")
    width = 4 if density_grad else 1
    result = np.zeros((len(positions), coords.shape[1], width), dtype=np.float64)
    splines = {int(z): _spline_from_entry(entry) for z, entry in spline_entries.items()}
    for b in range(len(positions)):
        for a, z in enumerate(numbers[b]):
            if z == 0:
                continue
            if int(z) not in splines:
                raise KeyError(f"No SAD spline for atomic number {int(z)}")
            delta = coords[b] - positions[b, a]
            radius_ang = np.linalg.norm(delta, axis=-1)
            radius_bohr = radius_ang / BOHR_ANGSTROM
            with np.errstate(divide="ignore", invalid="ignore"):
                log_r = np.log(radius_bohr)
            spline = splines[int(z)]
            density = np.nan_to_num(np.maximum(np.asarray(spline(log_r)), 0.0), nan=0.0)
            result[b, :, 0] += density
            if density_grad:
                deriv = np.asarray(spline.derivative()(log_r))
                with np.errstate(divide="ignore", invalid="ignore"):
                    grad = deriv[:, None] * delta / (radius_bohr[:, None] ** 2 * BOHR_ANGSTROM)
                result[b, :, 1:] += grad
    return result if density_grad else result[..., 0]


def sample_sad_coeff_density(positions, atom_numbers, coords, atom_entries, *,
                             coefficient_type="df_coeffs", density_grad=False):
    """Sample per-atom MO or projected-DF SAD densities without Torch.

    Entries follow the legacy ``atom_dens`` mappings: MO entries contain
    ``mo_basis/mo_coeff/mo_occ``; DF entries contain ``df_basis/df_coeffs``.
    This retains each entry's native PySCF AO coefficient ordering.
    """
    positions = np.asarray(positions, dtype=np.float64)
    numbers = np.asarray(atom_numbers, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64)
    if coefficient_type not in {"mo_coeffs", "df_coeffs"}:
        raise ValueError("coefficient_type must be 'mo_coeffs' or 'df_coeffs'")
    if positions.ndim != 3 or positions.shape[-1] != 3 or numbers.shape != positions.shape[:2]:
        raise ValueError("positions and atom_numbers must be [B,N,3] and [B,N]")
    if coords.ndim != 3 or coords.shape[0] != positions.shape[0]:
        raise ValueError("coords must have shape [B,G,3]")
    width = 4 if density_grad else 1
    atomwise = np.zeros((len(positions), positions.shape[1], coords.shape[1], width), dtype=np.float64)
    from pyscf import gto
    for b in range(len(positions)):
        for a, z in enumerate(numbers[b]):
            if z == 0:
                continue
            entry = atom_entries[int(z)]
            if coefficient_type == "mo_coeffs":
                basis = entry["mo_basis"]
                coeff = {"mo_coeff": entry["mo_coeff"], "mo_occ": entry["mo_occ"]}
                projected = False
            else:
                basis = entry["df_basis"]
                coeff = entry["df_coeffs"]
                if isinstance(coeff, (list, tuple)):
                    coeff = np.concatenate([np.asarray(item[1] if isinstance(item, tuple) else item).ravel()
                                            for item in coeff])
                projected = True
            # Use the same conversion for atom centers and evaluation points.
            # PySCF's Angstrom constant differs slightly from this pipeline's;
            # mixing them breaks translations near steep atomic core densities.
            mol = gto.M(atom=[(int(z), (positions[b, a]/BOHR_ANGSTROM).tolist())], basis=basis,
                        unit="Bohr", spin=int(z % 2))
            mol.build()
            atomwise[b, a] = _molecular_density_one(
                mol, coords[b] / BOHR_ANGSTROM,
                coeff, projected=projected, density_grad=density_grad,
            ).reshape((len(coords[b]), width))
    total = atomwise.sum(axis=1)
    return (total, atomwise) if density_grad else (total[..., 0], atomwise[..., 0])

