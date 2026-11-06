"""Dipole extraction and SO3LR kwargs for ethanethiol IR MD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import Trajectory

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

import run_mlip_ir_md as ir  # noqa: E402


class _ResultsCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, results_blob, **kwargs):
        super().__init__(**kwargs)
        self._blob = dict(results_blob)

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, **self._blob}


def test_calculator_dipole_reads_dipole_vec_on_results():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.calc = _ResultsCalculator({"dipole_vec": np.array([0.1, -0.2, 0.3])})
    np.testing.assert_allclose(ir._calculator_dipole(atoms), [0.1, -0.2, 0.3])


def test_calculator_dipole_reads_aux_dipole_vec():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.calc = _ResultsCalculator({"aux": {"dipole_vec": np.array([[0.4, 0.5, 0.6]])}})
    np.testing.assert_allclose(ir._calculator_dipole(atoms), [0.4, 0.5, 0.6])


def test_so3lr_calculator_kwargs_request_native_dipole():
    kwargs = ir.so3lr_calculator_kwargs()
    assert kwargs["has_aux"] is True
    assert kwargs["observables"] == ["dipole_vec"]
    assert kwargs["calculate_stress"] is False


class _TrackingDipoleCalculator(Calculator):
    implemented_properties = ["energy", "forces", "dipole"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.property_calls: list[tuple[str, ...]] = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        super().calculate(atoms, properties, system_changes)
        self.property_calls.append(tuple(properties))
        n = len(atoms) if atoms is not None else 1
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((n, 3)),
            "dipole": np.array([0.2, 0.0, -0.1]),
        }


def test_calculator_dipole_requests_dipole_in_one_calculate():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    calc = _TrackingDipoleCalculator()
    atoms.calc = calc
    np.testing.assert_allclose(ir._calculator_dipole(atoms), [0.2, 0.0, -0.1])
    assert calc.property_calls == [("dipole",)]


def test_load_resume_dipoles_and_remaining_steps(tmp_path: Path):
    out_dir = tmp_path / "ir"
    out_dir.mkdir()
    existing = np.arange(15, dtype=float).reshape(5, 3)
    np.save(out_dir / "dipoles.npy", existing)
    loaded = ir.load_resume_dipoles(out_dir)
    assert len(loaded) == 5
    np.testing.assert_allclose(loaded[0], [0.0, 1.0, 2.0])
    assert ir.remaining_production_steps(
        n_samples=len(loaded), production_steps=20, sample_every=2
    ) == 10
    assert ir.load_resume_dipoles(tmp_path / "empty") == []


def test_run_md_appends_to_existing_dipoles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ase.io import write

    structure = tmp_path / "start.xyz"
    start = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    from ase.constraints import FixCom
    start.set_constraint(FixCom())
    write(structure, start)
    out_dir = tmp_path / "ir"
    out_dir.mkdir()
    existing = np.ones((3, 3))
    np.save(out_dir / "dipoles.npy", existing)
    traj = Trajectory(str(out_dir / "trajectory.traj"), "w", start)
    for _ in range(3):
        traj.write()
    traj.close()

    monkeypatch.setattr(ir, "_dynamics_calculator", lambda *a, **k: _TrackingDipoleCalculator())
    args = argparse.Namespace(
        method="densnet",
        structure=structure,
        out_dir=out_dir,
        device="cpu",
        seed=0,
        temperature_k=300.0,
        timestep_fs=1.0,
        sample_every=1,
        production_ps=0.005,
        equilibration_ps=0.0,
        friction=0.01,
        windows_ps=(0.025, 0.05),
    )
    args.production_steps = 5
    args.equilibration_steps = 0
    metadata = ir._run_md(args)
    saved = np.load(out_dir / "dipoles.npy")
    assert saved.shape[0] == 5
    np.testing.assert_allclose(saved[:3], existing)
    assert metadata["n_samples"] == 5
    assert metadata["resumed_samples"] == 3


def test_mesa_ir_returns_finite_frequencies():
    memspectrum = pytest.importorskip("memspectrum")
    assert memspectrum is not None
    from overlay_mace_vs_densnet_ir import _mesa_ir

    t = np.arange(64, dtype=float)
    signal = np.sin(2.0 * np.pi * t / 16.0)
    dipoles = np.column_stack((signal, np.zeros_like(signal), np.zeros_like(signal)))
    freq, intensity = _mesa_ir(dipoles, timestep_fs=1.0)
    assert np.all(np.isfinite(freq))
    assert np.all(np.isfinite(intensity))
    assert freq.size == intensity.size
    assert freq.size > 1


def test_fresh_md_counts_only_production_intervals(tmp_path, monkeypatch):
    from ase.io import write
    structure = tmp_path / "start.xyz"
    write(structure, Atoms("H2", positions=[[0, 0, 0], [.74, 0, 0]]))
    monkeypatch.setattr(ir, "_dynamics_calculator", lambda *a, **k: _TrackingDipoleCalculator())
    args = argparse.Namespace(method="densnet", structure=structure, out_dir=tmp_path / "run",
        device="cpu", seed=1, temperature_k=300., timestep_fs=.5, sample_every=2,
        production_steps=8, equilibration_steps=3, friction=.01, windows_ps=())
    result = ir._run_md(args)
    assert result["n_samples"] == 4
    assert len(Trajectory(str(args.out_dir / "trajectory.traj"))) == 4
    assert result["thermostat"] == "Langevin(fixcm=False) + FixCom"
