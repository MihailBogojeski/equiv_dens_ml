"""Goedecker minima hopping: SO(3) dedup, accept/reject, one-cycle smoke."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

_REPO = Path(__file__).resolve().parents[3]
_THIO = _REPO / "scripts" / "revision" / "thiophene"
sys.path.insert(0, str(_THIO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minima_hopping import (  # noqa: E402
    adapt_ediff,
    adapt_temperature,
    decide_accept,
    minima_identical,
    run_minima_hopping,
    so3_rmsd,
)
from synthetic_3mer import build_synthetic_3mer, twist_ring  # noqa: E402


class HarmonicWell(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, reference: np.ndarray, k: float = 8.0) -> None:
        super().__init__()
        self.reference = np.asarray(reference, dtype=float)
        self.k = float(k)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions()
        delta = pos - self.reference
        energy = 0.5 * self.k * float(np.sum(delta * delta))
        forces = -self.k * delta
        self.results = {"energy": energy, "forces": forces}


def test_so3_rmsd_vanishes_for_rotated_copy() -> None:
    _, pos = build_synthetic_3mer()
    angle = np.deg2rad(37.0)
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = pos @ rot.T + np.array([4.0, -2.0, 1.5])
    assert so3_rmsd(pos, moved) < 1e-8
    assert minima_identical(pos, moved, threshold=0.1)


def test_so3_rmsd_nonfinite_is_infinite() -> None:
    _, pos = build_synthetic_3mer()
    bad = np.full_like(pos, np.nan)
    assert so3_rmsd(bad, pos) == float("inf")
    assert not minima_identical(bad, pos, threshold=0.1)


def test_twisted_3mer_is_a_distinct_minimum() -> None:
    symbols, pos = build_synthetic_3mer()
    twisted = twist_ring(pos, symbols, ring_index=2, angle_deg=90.0)
    assert so3_rmsd(pos, twisted) > 0.2
    assert not minima_identical(pos, twisted, threshold=0.1)


def test_accept_new_minimum_below_ediff() -> None:
    decision = decide_accept(
        energy_new=1.0,
        energy_current=0.5,
        is_current=False,
        e_diff=2.0,
    )
    assert decision is True


def test_reject_new_minimum_above_ediff() -> None:
    decision = decide_accept(
        energy_new=4.0,
        energy_current=0.5,
        is_current=False,
        e_diff=2.0,
    )
    assert decision is False


def test_adapt_temperature_rises_on_repeat_minimum() -> None:
    t_new = adapt_temperature(1000.0, found_new=False, beta=1.05)
    assert t_new == pytest.approx(1050.0)


def test_adapt_ediff_falls_after_accept() -> None:
    e_new = adapt_ediff(2.0, accepted=True, alpha=1.05)
    assert e_new == pytest.approx(2.0 / 1.05)


def test_one_escape_cycle_returns_single_unique_minimum() -> None:
    symbols, pos = build_synthetic_3mer()
    atoms = Atoms(symbols=symbols, positions=pos)
    atoms.calc = HarmonicWell(pos, k=12.0)
    result = run_minima_hopping(
        atoms,
        n_escapes=1,
        escape_steps=4,
        timestep_fs=0.5,
        temperature_k=50.0,
        e_diff_ev=2.0,
        fmax=0.2,
        relax_steps=20,
        seed=0,
    )
    assert result["n_unique"] == 1
    assert len(result["minima"]) == 1
    assert so3_rmsd(result["minima"][0]["positions"], pos) < 0.15
