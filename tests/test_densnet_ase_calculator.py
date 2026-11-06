"""Unit tests for the ASE DenSNetCalculator wrapper."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator

from equiv_dens.md.dft_network_calculator import (
    DenSNetCalculator,
    ase_atoms_to_model_inputs,
    infer_append_atom_density,
    remap_stale_dataset_path,
    resolve_inference_np_dataset,
    should_enable_inference_density_head,
    should_enable_inference_intor_head,
)


class _DummyEnergyForceModel(torch.nn.Module):
    def forward(self, inputs):
        n = int(inputs["atom_numbers"].numel())
        energy = torch.tensor([n * 10.0], dtype=torch.float32)
        forces = torch.zeros((1, n, 3), dtype=torch.float32)
        forces[..., 0] = 1.0
        return {"energy": energy, "forces": forces}


def test_ase_atoms_to_model_inputs_shapes():
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    inputs = ase_atoms_to_model_inputs(atoms, cutoff=6.0, use_gpu=False)
    assert inputs["positions"].shape[-2] == 3
    assert inputs["atom_numbers"].shape[-1] == 3
    assert inputs["idx_i"].ndim == 1
    assert inputs["idx_j"].ndim == 1


def test_densnet_calculator_kcal_to_ev():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    calc = DenSNetCalculator(
        _DummyEnergyForceModel(),
        cutoff=6.0,
        use_gpu=False,
        energy_unit="kcal/mol",
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    # dummy energy is 2 * 10 kcal/mol
    assert energy == pytest.approx(20.0 * 0.0433641153087705, rel=1e-6)
    assert forces.shape == (2, 3)
    assert forces[0, 0] == pytest.approx(0.0433641153087705, rel=1e-6)
    assert isinstance(calc, Calculator)


class _DummyDensityEnergyModel(torch.nn.Module):
    """Mirrors ``DFTNetwork.forward`` after the energy-only skip:

    density/core_density run only when ``coords`` is present. A trained
    density head in ``property_models`` is not enough to require a grid.
    """

    def __init__(self, require_coords=False):
        super().__init__()
        self.require_coords = require_coords
        self.property_models = {"energy": None, "density": None}

    def forward(self, inputs):
        n = int(inputs["atom_numbers"].numel())
        if self.require_coords:
            _ = inputs["coords"]
        elif "coords" in inputs:
            _ = inputs["coords"]
        energy = torch.tensor([n * 10.0], dtype=torch.float32)
        forces = torch.zeros((1, n, 3), dtype=torch.float32)
        forces[..., 0] = 1.0
        return {"energy": energy, "forces": forces}


def _dummy_grid_sampling_fn(grid_spec, n_samp, atom_numbers, positions):
    coords = positions.new_zeros((1, 4, 3))
    weights = positions.new_ones((1, 4))
    return coords, weights


def test_densnet_calculator_energy_forces_do_not_require_density_grid():
    """H-bond / NVE path: energy+forces with a density-trained checkpoint
    must not attach or require an integration grid."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    calc = DenSNetCalculator(
        _DummyDensityEnergyModel(require_coords=False),
        cutoff=6.0,
        use_gpu=False,
        need_density_grid=False,
    )
    atoms.calc = calc
    forces = atoms.get_forces()
    assert np.isfinite(forces).all()
    assert forces.shape == (2, 3)


def test_densnet_calculator_attaches_grid_when_requested():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    calc = DenSNetCalculator(
        _DummyDensityEnergyModel(require_coords=True),
        cutoff=6.0,
        use_gpu=False,
        need_density_grid=True,
        grid_sampling_fn=_dummy_grid_sampling_fn,
        grid_spec={},
    )
    atoms.calc = calc
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)


class _DummyDipoleDensityModel(torch.nn.Module):
    """Returns a grid density only when coords are attached."""

    def __init__(self):
        super().__init__()
        self.saw_coords = False
        self.property_models = {"energy": None, "density": None}

    def forward(self, inputs):
        n = int(inputs["atom_numbers"].numel())
        energy = torch.tensor([n * 10.0], dtype=torch.float32)
        forces = torch.zeros((1, n, 3), dtype=torch.float32)
        out = {"energy": energy, "forces": forces}
        if "coords" in inputs:
            self.saw_coords = True
            density = torch.zeros(inputs["coords"].shape[:2], dtype=torch.float32)
            density[..., 0] = 2.0
            out["density"] = density
        return out


def _dipole_grid_sampling_fn(grid_spec, n_samp, atom_numbers, positions):
    coords = positions.new_zeros((1, 4, 3))
    coords[0, 0] = torch.tensor([1.0, 0.0, 0.0], dtype=positions.dtype)
    weights = positions.new_ones((1, 4))
    return coords, weights


def test_densnet_energy_does_not_attach_grid_when_dipole_is_not_requested():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    model = _DummyDipoleDensityModel()
    calc = DenSNetCalculator(
        model,
        cutoff=6.0,
        use_gpu=False,
        need_density_grid=False,
        grid_sampling_fn=_dipole_grid_sampling_fn,
        grid_spec={},
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert model.saw_coords is False
    assert "dipole" not in calc.results


def test_densnet_get_dipole_moment_attaches_grid_and_returns_length_three():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    model = _DummyDipoleDensityModel()
    calc = DenSNetCalculator(
        model,
        cutoff=6.0,
        use_gpu=False,
        need_density_grid=False,
        grid_sampling_fn=_dipole_grid_sampling_fn,
        grid_spec={},
        energy_unit="eV",
    )
    atoms.calc = calc
    dipole = atoms.get_dipole_moment()
    assert model.saw_coords is True
    assert np.asarray(dipole).shape == (3,)
    assert np.all(np.isfinite(dipole))
    assert "dipole" in calc.results


def test_densnet_calculator_reset_between_calls():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    calc = DenSNetCalculator(_DummyEnergyForceModel(), use_gpu=False)
    atoms.calc = calc
    e1 = atoms.get_potential_energy()
    atoms.positions[1, 0] += 0.01
    e2 = atoms.get_potential_energy()
    assert e1 == pytest.approx(e2)
    assert np.isfinite(e2)


def test_remap_stale_ml_dft_dataset_path(tmp_path):
    prior = tmp_path / "datasets" / "free_atom_densities_augccpvdz_augccpvqzjkfit.npy"
    prior.parent.mkdir()
    prior.write_bytes(b"x")
    remapped = remap_stale_dataset_path(
        "/home/ml-dft/equiv_dens/datasets/free_atom_densities_augccpvdz_augccpvqzjkfit.npy",
        tmp_path,
    )
    assert remapped == str(prior)
    assert remap_stale_dataset_path(str(prior), tmp_path) == str(prior)
    assert remap_stale_dataset_path("/missing/nowhere.npy", tmp_path) is None


def test_infer_append_atom_density_from_energy_width():
    assert infer_append_atom_density(
        {"property_models.energy.input_layer.0.weight": torch.zeros(128, 60)}
    ) is True
    assert infer_append_atom_density(
        {"property_models.energy.input_layer.0.weight": torch.zeros(128, 30)}
    ) is False
    assert infer_append_atom_density({}) is None


def test_resolve_inference_np_dataset_keeps_args_file_when_caller_omits_it(tmp_path):
    """12-mer timing used to overwrite the thiophene npy with ethanol_train_10.

    Ethanol is C/H/O (L=0 width 27). The polythiophene checkpoint was built
    with C/H/S (width 30), so the energy head then failed to load.
    """
    repo = tmp_path
    (repo / "datasets").mkdir()
    thiophene = repo / "datasets" / "thiophene_all_train_d4.npy"
    ethanol = repo / "datasets" / "ethanol_train_10.npy"
    thiophene.write_bytes(b"thio")
    ethanol.write_bytes(b"etoh")

    chosen = resolve_inference_np_dataset(
        explicit=None,
        args_np_dataset=str(thiophene),
        repo=repo,
    )
    assert chosen == str(thiophene)

    override = repo / "datasets" / "thiophene12mer_init.npy"
    override.write_bytes(b"12mer")
    chosen = resolve_inference_np_dataset(
        explicit=str(override),
        args_np_dataset=str(thiophene),
        repo=repo,
    )
    assert chosen == str(override)


def test_should_enable_inference_density_head_for_energy_only_ext_checkpoint():
    """ext4 latest_checkpoint zeros density_weight but still stores dens_model."""
    energy_only_args = type("Args", (), {"density_weight": 0.0, "density_coeffs": True})()
    assert should_enable_inference_density_head(
        {"density_repr_model.1.spherical_output.linear.0.weight": torch.zeros(30, 128)},
        energy_only_args,
    ) is True
    assert should_enable_inference_density_head(
        {"property_models.energy.input_layer.0.weight": torch.zeros(128, 60)},
        energy_only_args,
    ) is False


def test_should_enable_inference_intor_head_when_dens_model_weights_exist():
    energy_only_args = type("Args", (), {"dipole_moment_weight": 0.0, "dpm_intor": False})()
    assert should_enable_inference_intor_head(
        {"density_repr_model.1.spherical_output.linear.0.weight": torch.zeros(30, 128)},
        energy_only_args,
    ) is True
    assert should_enable_inference_intor_head(
        {"property_models.energy.input_layer.0.weight": torch.zeros(128, 60)},
        energy_only_args,
    ) is False


class _DummyIntorDipoleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.saw_coords = False

    def forward(self, inputs):
        n = int(inputs["atom_numbers"].numel())
        self.saw_coords = "coords" in inputs
        energy = torch.tensor([n * 10.0], dtype=torch.float32)
        forces = torch.zeros((1, n, 3), dtype=torch.float32)
        dipole = torch.tensor([[0.4, -0.1, 0.2]], dtype=torch.float32)
        return {"energy": energy, "forces": forces, "dipole_moment": dipole}


def test_densnet_get_dipole_moment_reads_model_dipole_moment_without_grid():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    model = _DummyIntorDipoleModel()
    calc = DenSNetCalculator(
        model,
        cutoff=6.0,
        use_gpu=False,
        need_density_grid=False,
        energy_unit="eV",
    )
    atoms.calc = calc
    dipole = atoms.get_dipole_moment()
    np.testing.assert_allclose(dipole, [0.4, -0.1, 0.2])
    assert model.saw_coords is False


def test_densnet_get_dipole_moment_builds_grid_when_sampler_and_spec_are_missing():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    model = _DummyDipoleDensityModel()
    calc = DenSNetCalculator(
        model,
        cutoff=6.0,
        use_gpu=False,
        need_density_grid=False,
        grid_sampling_fn=None,
        grid_spec={},
        pyscf_grid=False,
        energy_unit="eV",
    )
    atoms.calc = calc
    dipole = atoms.get_dipole_moment()
    assert model.saw_coords is True
    assert np.asarray(dipole).shape == (3,)
    assert np.all(np.isfinite(dipole))


def test_ase_analytic_dipole_restores_free_atom_density():
    """A normalized H 1s prior exactly cancels a displaced proton's moment."""
    from equiv_dens.utils import orbitals
    from pyscf import gto
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    prior = {1: {"mo_coeff": np.eye(1), "mo_occ": np.ones(1),
                 "mo_basis": {1: mol._basis["H"]}}}

    class AtomicModel(_DummyEnergyForceModel):
        def forward(self, inputs):
            output = super().forward(inputs)
            result = orbitals.intor_dipole_moment_free_atom(inputs)
            nuclear = (inputs["batch_atom_numbers"].unsqueeze(-1) *
                       inputs["batch_positions"]).sum(dim=1)
            output["dipole_moment"] = nuclear - result["atom_dipole_moment"]
            return output

    atoms = Atoms("H", positions=[[1.2, -0.7, 0.3]])
    atoms.calc = DenSNetCalculator(AtomicModel(), atom_dens=prior,
                                  atom_dens_type="mo_coeffs", remove_atom_density=True)
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.get_dipole_moment() == pytest.approx([0, 0, 0], abs=2e-6)
