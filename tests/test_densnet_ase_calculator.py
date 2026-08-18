"""Unit tests for the ASE DenSNetCalculator wrapper."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator

from equiv_dens.md.dft_network_calculator import (
    DenSNetCalculator,
    ase_atoms_to_model_inputs,
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


def test_densnet_calculator_reset_between_calls():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    calc = DenSNetCalculator(_DummyEnergyForceModel(), use_gpu=False)
    atoms.calc = calc
    e1 = atoms.get_potential_energy()
    atoms.positions[1, 0] += 0.01
    e2 = atoms.get_potential_energy()
    assert e1 == pytest.approx(e2)
    assert np.isfinite(e2)
