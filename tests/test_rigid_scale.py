"""Rigid-monomer intermolecular scaling (OMol25 H.6)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "lib"))

import rigid_scale as rs  # noqa: E402


def _ideal_water(com: np.ndarray) -> np.ndarray:
    """TIP3P-like monomer: OH = 0.9572 A, HOH = 104.52 deg."""
    roh = 0.9572
    half = math.radians(104.52) / 2.0
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [roh * math.sin(half), 0.0, roh * math.cos(half)],
            [-roh * math.sin(half), 0.0, roh * math.cos(half)],
        ]
    )
    return local + com


def _dimer(r_oo: float = 2.8) -> tuple[list[str], np.ndarray]:
    mols = [_ideal_water(np.array([r_oo / 2.0, 0.0, 0.0])), _ideal_water(np.array([-r_oo / 2.0, 0.0, 0.0]))]
    return rs.molecules_to_water_frame(mols)


def test_intramolecular_oh_and_hoh_unchanged_after_scale():
    symbols, coords = _dimer()
    mols = rs.split_water_molecules(symbols, coords)
    before = [rs.monomer_internal_geometry(m) for m in mols]
    scaled = rs.scale_rigid_coms(mols, 2.0)
    after = [rs.monomer_internal_geometry(m) for m in scaled]
    for (oh_b, hoh_b), (oh_a, hoh_a) in zip(before, after):
        np.testing.assert_allclose(oh_a, oh_b, atol=1e-12)
        assert abs(hoh_a - hoh_b) < 1e-12


def test_com_com_distance_scales_by_lambda():
    symbols, coords = _dimer(r_oo=2.8)
    mols = rs.split_water_molecules(symbols, coords)
    d0 = rs.min_intermolecular_oo(mols)
    scaled = rs.scale_rigid_coms(mols, 2.0)
    d1 = rs.min_intermolecular_oo(scaled)
    assert abs(d1 / d0 - 2.0) < 1e-12


def test_lambda_two_doubles_dimer_r_oo():
    symbols, coords = _dimer(r_oo=2.8)
    mols = rs.split_water_molecules(symbols, coords)
    scaled = rs.scale_rigid_coms(mols, 2.0)
    assert abs(rs.min_intermolecular_oo(scaled) - 5.6) < 1e-12


def test_clash_filter_rejects_crushed_dimer():
    symbols, coords = _dimer(r_oo=2.8)
    mols = rs.split_water_molecules(symbols, coords)
    crushed = rs.scale_rigid_coms(mols, 0.15)
    sym, xyz = rs.molecules_to_water_frame(crushed)
    flagged, pairs = rs.covalent_overlap_pairs(sym, xyz)
    assert flagged
    assert pairs


def test_clash_filter_accepts_equilibrium_dimer():
    symbols, coords = _dimer(r_oo=2.8)
    flagged, pairs = rs.covalent_overlap_pairs(symbols, coords)
    assert not flagged
    assert pairs == []


def test_sr_lr_split_at_six_angstrom():
    assert rs.sr_lr_label(5.99) == "SR"
    assert rs.sr_lr_label(6.00) == "SR"
    assert rs.sr_lr_label(6.01) == "LR"


def test_omol25_scale_grid_has_twenty_two_factors():
    factors = rs.omol25_scale_factors()
    assert len(factors) == 22
    assert min(factors) == pytest.approx(0.70)
    assert max(factors) == pytest.approx(3.00)
    assert sum(f < 1.0 for f in factors) == 2
    assert sum(1.0 < f <= 1.80 + 1e-12 for f in factors) == 10
    assert sum(f > 1.80 + 1e-12 for f in factors) == 10
