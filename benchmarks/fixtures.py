"""Generated full DenSNet fixtures, independent of datasets/checkpoints.

Call ``select_source`` before constructing a model in a fresh process. Fixtures
store actual initialized state tensors, not only an initialization seed.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

import torch

MODES = ("coefficients", "density", "ef", "force_training")
CASES = ("tiny", "batch", "wide")

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "b15a0ff7c1e6070265c36d80a75706749778e9da"
REFERENCE_ROOT = Path(os.environ.get("DENSNET_REFERENCE_ROOT", REPO_ROOT / "scratch" / "equiv_dens_ml-paper-reference"))


def ensure_reference_source(root=REFERENCE_ROOT):
    """Extract src/ of the paper-reference commit into root if it is missing."""
    root = Path(root)
    if not (root / "src" / "equiv_dens").is_dir():
        root.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "-C", REPO_ROOT, "archive", REFERENCE_COMMIT, "src"],
                                 capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", root], input=archive, check=True)
    return root


def select_source(root):
    root = Path(root).resolve()
    existing = [name for name in sys.modules if name == "equiv_dens" or name.startswith("equiv_dens.")]
    if existing:
        raise RuntimeError("Source selection requires a fresh Python process")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    package = importlib.import_module("equiv_dens")
    if not Path(package.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f"Wrong equiv_dens import: {package.__file__}")
    return root


def tensor_hash(value):
    """Stable hash of tensor names, shapes, types and exact contiguous bytes."""
    digest = hashlib.sha256()
    def visit(item, path=""):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"{path}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}/{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
        else:
            digest.update(f"{path}:{item!r}".encode())
    visit(value)
    return digest.hexdigest()


def clone_tree(value, device=None):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device=device or value.device)
    if isinstance(value, dict):
        return {key: clone_tree(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [clone_tree(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(child, device) for child in value)
    return value


def make_atoms(case="tiny", dtype=torch.float64, grid_points=None):
    """Prepared source-independent compressed layout; positions in angstrom."""
    if case not in CASES:
        raise ValueError(case)
    numbers = [[1, 1, 8]]
    geometry = [[[0.91, 0.13, -0.12], [-0.23, 0.86, 0.17], [0.03, -0.07, 0.04]]]
    if case == "batch":
        numbers.append([1, 0, 8])
        geometry.append([[0.82, -0.31, 0.21], [0, 0, 0], [-0.08, 0.09, -0.12]])
    if case == "wide":
        # Twelve atoms, four separated water-like motifs in one molecule.
        numbers = [[1] * 8 + [8] * 4]
        geometry = [[
            [0.91 + 2.1*k, 0.13 + 0.18*k, -0.12] for k in range(4)
        ] + [[-0.23 + 2.1*k, 0.86 + 0.18*k, 0.17] for k in range(4)]
          + [[0.03 + 2.1*k, -0.07 + 0.18*k, 0.04] for k in range(4)]]
    z = torch.tensor(numbers, dtype=torch.long)
    r = torch.tensor(geometry, dtype=dtype)
    mask = z > 0
    center = (r * z[..., None]).sum(1) / z.sum(1)[:, None]
    r = torch.where(mask[..., None], r - center[:, None], torch.zeros_like(r))
    b, n = z.shape
    active = mask.flatten().nonzero().flatten()
    atom_batches = torch.arange(b).repeat_interleave(n)[active]
    edges_i, edges_j = [], []
    offset = 0
    for row in mask:
        size = int(row.sum())
        for i in range(size):
            for j in range(size):
                if i != j:
                    edges_i.append(offset + i)
                    edges_j.append(offset + j)
        offset += size
    count = grid_points or (256 if case == "wide" else 9)
    grid_index = torch.arange(count, dtype=dtype)
    coords = torch.stack((0.38 + 0.17*grid_index.sin(),
                          -0.41 + 0.21*(grid_index*0.7).cos(),
                          0.31 + 0.13*(grid_index*1.3).sin()), -1)
    return {
        "positions": r.reshape(1, -1, 3)[:, active],
        "atom_numbers": z.flatten()[active][None],
        "atom_mask": mask.flatten(),
        "batch_atom_numbers": z, "batch_atom_mask": mask,
        "batch_positions": r, "batch_idx_pos": active,
        "atom_batch_idx": atom_batches[None],
        "atom_numbers_first_positions": {1: 0, 8: int((z.max(0).values == 8).nonzero()[0])},
        "idx_i": torch.tensor(edges_i, dtype=torch.long),
        "idx_j": torch.tensor(edges_j, dtype=torch.long),
        "coords": coords[None].repeat(b, 1, 1),
        "coord_weights": torch.full((b, count), 0.11, dtype=dtype),
    }


def fixture_spec(case="tiny", dtype=torch.float64, learned_radial=True, *,
                 num_features=None, order=None, num_modules=None, energy_normalize=None):
    spec = {
        "case": case, "dtype": str(dtype).split(".")[-1], "seed": 1729,
        "num_features": 4 if case != "wide" else 16,
        "num_modules": 1 if case != "wide" else 2,
        "order": 1 if case != "wide" else 2,
        "num_basis_functions": 4 if case != "wide" else 8,
        "orbital_basis": {1: [(1, 2, 0), (1, 1, 0), (1, 2, 1)],
                          8: [(8, 2, 0), (8, 1, 0), (8, 2, 1)]},
        "radial_coeffs": {1: [([0.7, 1.4], [0.8, 0.3]), ([0.24], [0.6]), ([0.5, 1.2], [0.7, 0.2])],
                          8: [([1.1, 2.0], [0.9, 0.4]), ([0.32], [0.5]), ([0.7, 1.5], [0.6, 0.3])]},
        "learned_radial": learned_radial, "cutoff_angstrom": 5.0,
        "units": {"positions": "angstrom", "coords": "angstrom",
                  "density": "electrons/bohr^3", "density_grad": "electrons/bohr^4",
                  "energy": "fixture internal identity units", "forces": "energy/angstrom"},
        "policies": {"normalization": 0, "SAD": False, "core": False,
                     "integral_constraint": None, "grid_screening": False},
    }
    for key, value in (("num_features", num_features), ("order", order), ("num_modules", num_modules)):
        if value is not None:
            if value < 1:
                raise ValueError(f"{key} must be positive")
            spec[key] = value
    if energy_normalize is not None:
        if energy_normalize not in (0, 1, 2):
            raise ValueError("energy_normalize must be 0, 1, or 2")
        spec["energy_normalize"] = energy_normalize
    return spec


def build_model(spec, mode="ef", kernel_backend="broadcast", *,
                density_backend="legacy", energy_backend="legacy"):
    """Every E/F mode uses the actual density-mediated energy encoder."""
    from equiv_dens.nn.representation.spherical_harmonic import EquivariantSphericalHarmonics
    from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DensityExpansion
    from equiv_dens.nn.property_output.energy import SphericalHarmonicsEnergyNetwork
    from equiv_dens.nn.dft_network import DFTNetwork
    from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
    from equiv_dens.utils.scaling import UnitConversion
    if mode not in MODES:
        raise ValueError(mode)
    torch.manual_seed(spec["seed"])
    # Build only the degrees actually used by these networks. The default L10
    # table is unnecessary here and makes tiny fixture serialization enormous.
    clebsch_gordan = ClebschGordanMatrix(order_max=spec["order"])
    common = dict(orbital_basis=spec["orbital_basis"], order=spec["order"],
                  num_features=spec["num_features"], num_modules=spec["num_modules"],
                  num_basis_functions=spec["num_basis_functions"], cutoff=spec["cutoff_angstrom"],
                  normalize=0, num_neighbours=1, clebsch_gordan=clebsch_gordan)
    representation = EquivariantSphericalHarmonics(**common)
    head = DensityCoeffsNetwork(orbital_basis=spec["orbital_basis"], order=spec["order"],
                               num_features=spec["num_features"],
                               init_radial_coeffs=spec["radial_coeffs"],
                               pred_radial_coeffs=spec["learned_radial"], normalize=0,
                               clebsch_gordan=clebsch_gordan)
    # Keep the identical state schema in every requested-output mode.
    energy_common = dict(common, normalize=spec.get("energy_normalize", 0))
    energy = SphericalHarmonicsEnergyNetwork(**energy_common, calculate_forces=mode in ("ef", "force_training"),
                                             pred_radial_coeffs=spec["learned_radial"], L0_start=False)
    expansion = DensityExpansion(spec["orbital_basis"], density_grad=True)
    identity = UnitConversion(torch.nn.Identity(), torch.nn.Identity())
    model = DFTNetwork(torch.nn.Sequential(representation, head),
                       {"energy": energy, "density": expansion},
                       {"energy": mode in ("ef", "force_training"), "density": False},
                       conversions_in=identity, conversions_out=identity)
    if density_backend == "contracted":
        from benchmarks.experimental_density import install_experimental_density_evaluator
        install_experimental_density_evaluator(model, density_backend)
    elif density_backend != "legacy":
        raise ValueError("Unknown benchmark density backend")
    if energy_backend == "device":
        from benchmarks.experimental_energy import install_energy_accumulation_experiment
        install_energy_accumulation_experiment(model, energy_backend)
    elif energy_backend != "legacy":
        raise ValueError("Unknown benchmark energy backend")
    # Requested branch selection is explicit, while retaining all saved tensors.
    model.force_props = ["energy"] if mode in ("ef", "force_training") else []
    model.no_force_props = ["density"] if mode == "density" else []
    model = model.to(dtype=getattr(torch, spec["dtype"]))
    if kernel_backend == "matmul":
        from benchmarks.experimental_kernels import install_matrix_experiment
        install_matrix_experiment(model)
    elif kernel_backend != "broadcast":
        raise ValueError("Unknown benchmark kernel backend")
    if mode == "force_training":
        model.train()
        for parameter in model.density_repr_model.parameters():
            parameter.requires_grad_(False)
    else:
        model.eval()
    return model


def make_fixture(case="tiny", dtype=torch.float64, learned_radial=True, *, grid_points=None, **overrides):
    spec = fixture_spec(case, dtype, learned_radial, **overrides)
    model = build_model(spec)
    atoms = make_atoms(case, dtype, grid_points=grid_points)
    state = clone_tree(model.state_dict(), "cpu")
    return {"schema": 1, "spec": spec, "atoms": atoms, "state_dict": state,
            "fixture_hash": tensor_hash({"spec": spec, "atoms": atoms}),
            "state_hash": tensor_hash(state)}


def load_fixture(path):
    fixture = torch.load(path, map_location="cpu", weights_only=False)
    if fixture["fixture_hash"] != tensor_hash({"spec": fixture["spec"], "atoms": fixture["atoms"]}):
        raise ValueError("Fixture input hash mismatch")
    if fixture["state_hash"] != tensor_hash(fixture["state_dict"]):
        raise ValueError("Fixture parameter hash mismatch")
    return fixture
