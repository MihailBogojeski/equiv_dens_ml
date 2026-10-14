"""Paper-system geometry and auxiliary-basis fixtures with initialized weights.

These fixtures measure computation, not prediction accuracy. The architecture
comes from the checked-in ethanol configuration; all numerical arrays are
exported once and reused by every backend.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CASES = ("ethanol-small", "ethanol-paper")


def ethanol_frames():
    """Read the real local trajectory, retaining its original species order."""
    from ase.io import read
    frames = read(ROOT / "datasets/ethanol_train_10.xyz", index=":")
    numbers = frames[0].numbers.astype(np.int64)
    if len(frames) != 10 or len(numbers) != 9:
        raise ValueError("Unexpected ethanol trajectory dimensions")
    if any(not np.array_equal(frame.numbers, numbers) for frame in frames):
        raise ValueError("Trajectory atom order changes between frames")
    positions = np.stack([frame.positions for frame in frames]).astype(np.float64)
    center = (positions * numbers[None, :, None]).sum(1) / numbers.sum()
    return numbers, positions - center[:, None, :]


def application_spec(case):
    if case not in CASES:
        raise ValueError(case)
    orbital_file = ROOT / "datasets/augccpvqzjkfit_orbital_basis_df.npy"
    radial_file = ROOT / "datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy"
    # These are trusted, tracked legacy metadata assets. Generated benchmark
    # bundles themselves use the non-pickle JSON/NPZ serialization.
    orbitals = np.load(orbital_file, allow_pickle=True).item()
    radial = np.load(radial_file, allow_pickle=True).item()
    symbols = {1: "H", 6: "C", 8: "O"}
    orbital_basis = {z: [tuple(map(int, shell)) for shell in orbitals[symbol]]
                     for z, symbol in symbols.items()}
    radial_coeffs = {z: [(np.asarray(width).tolist(), np.asarray(scale).tolist())
                        for width, scale in radial[symbol]]
                     for z, symbol in symbols.items()}
    paper = case == "ethanol-paper"
    return {
        "case": case, "dtype": "float64", "seed": 1729,
        "density_orders": [1, 3, 5] if paper else [1],
        "energy_orders": [5, 2, 2] if paper else [1],
        "num_features": 128 if paper else 8,
        "num_basis_functions": 128 if paper else 8,
        "num_radial_components": 4, "num_neighbours": 8,
        "cutoff_angstrom": 7.937658158457616,
        "orbital_basis": orbital_basis, "radial_coeffs": radial_coeffs,
        "learned_radial": True, "L0_start": True,
        "policies": {"normalization": 0, "SAD": False, "core": False,
                     "integral_constraint": None, "grid_screening": False},
        "units": {"positions": "angstrom", "coords": "angstrom",
                  "density": "electrons/bohr^3", "density_grad": "electrons/bohr^4",
                  "energy": "identity internal units", "forces": "internal energy/angstrom"},
        "source_config": "config/training/ethanol_all_001_md.txt",
        "asset_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (orbital_file, radial_file,
                                      ROOT / "datasets/ethanol_train_10.xyz",
                                      ROOT / "config/training/ethanol_all_001_md.txt")},
    }


def application_atoms(grid_points=1000, batch_size=1):
    """Real centered geometries with a deterministic, supplied evaluation grid.

    Grid construction and QC labels are outside the benchmark. Low-discrepancy
    points span the molecule rather than repeating the old tiny grid cluster.
    """
    from equiv_dens_jax.data import prepare_atoms
    numbers, positions = ethanol_frames()
    if not 1 <= batch_size <= len(positions) or grid_points < 1:
        raise ValueError("Invalid batch/grid dimensions")
    positions = positions[:batch_size]
    index = np.arange(1, grid_points + 1, dtype=np.float64)
    fractions = (index[:, None] * np.sqrt([2., 3., 5.])) % 1.
    lower = positions.min(axis=1) - 1.5
    upper = positions.max(axis=1) + 1.5
    coords = lower[:, None, :] + fractions[None, :, :] * (upper-lower)[:, None, :]
    atoms = prepare_atoms({"atom_numbers": np.tile(numbers, (batch_size, 1)),
                           "positions": positions, "coords": coords,
                           "coord_weights": np.full((batch_size, grid_points), .11)})
    return atoms, {"numbers": numbers, "positions": ethanol_frames()[1]}


def build_application_model(spec, mode="ef"):
    from equiv_dens.nn.representation.spherical_harmonic import EquivariantSphericalHarmonics
    from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DensityExpansion
    from equiv_dens.nn.property_output.energy import SphericalHarmonicsEnergyNetwork
    from equiv_dens.nn.dft_network import DFTNetwork
    from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
    from equiv_dens.utils.scaling import UnitConversion
    torch.manual_seed(spec["seed"])
    max_order = max(max(spec["density_orders"]), max(spec["energy_orders"]),
                    max(shell[2] for shells in spec["orbital_basis"].values() for shell in shells))
    cg = ClebschGordanMatrix(order_max=max_order)
    common = dict(orbital_basis=spec["orbital_basis"], num_features=spec["num_features"],
                  num_basis_functions=spec["num_basis_functions"],
                  num_radial_components=spec["num_radial_components"],
                  cutoff=spec["cutoff_angstrom"], normalize=0,
                  num_neighbours=spec["num_neighbours"], clebsch_gordan=cg)
    representation = EquivariantSphericalHarmonics(
        **common, order=spec["density_orders"], num_modules=len(spec["density_orders"]))
    head = DensityCoeffsNetwork(
        orbital_basis=spec["orbital_basis"], order=spec["density_orders"][-1],
        num_features=spec["num_features"], init_radial_coeffs=spec["radial_coeffs"],
        pred_radial_coeffs=spec["learned_radial"], normalize=0, clebsch_gordan=cg)
    forces = mode in ("ef", "energy_force_update")
    energy = SphericalHarmonicsEnergyNetwork(
        **common, order=spec["energy_orders"], num_modules=len(spec["energy_orders"]),
        calculate_forces=forces, pred_radial_coeffs=spec["learned_radial"],
        L0_start=spec["L0_start"])
    identity = UnitConversion(torch.nn.Identity(), torch.nn.Identity())
    model = DFTNetwork(torch.nn.Sequential(representation, head),
                       {"energy": energy, "density": DensityExpansion(spec["orbital_basis"], density_grad=True)},
                       {"energy": forces, "density": False},
                       conversions_in=identity, conversions_out=identity).to(torch.float64)
    model.force_props = ["energy"] if forces else []
    model.no_force_props = ["density"] if mode in ("density", "density_update") else []
    return model.train(mode.endswith("_update"))
