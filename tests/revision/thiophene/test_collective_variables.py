"""Inter-ring S–Cα–Cα–S dihedrals for α-terthiophene (SO3krates CV analogue)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_THIO = _REPO / "scripts" / "revision" / "thiophene"
sys.path.insert(0, str(_THIO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from thiophene_collective_variables import (  # noqa: E402
    backbone_cv_pair,
    cv_histogram_2d,
    dihedral_angle,
    inter_ring_dihedral_quads,
    inter_ring_dihedrals,
    low_density_bins,
    fold_abs_deg,
    nearest_train_cv_distance,
    wrap_deg,
)
from synthetic_3mer import build_synthetic_3mer, twist_ring  # noqa: E402


def test_fold_abs_deg_identifies_plus_minus_anti() -> None:
    folded = fold_abs_deg(np.array([-170.0, 170.0, -10.0, 10.0, 180.0, -180.0]))
    assert folded[0] == pytest.approx(170.0)
    assert folded[1] == pytest.approx(170.0)
    assert folded[2] == pytest.approx(10.0)
    assert folded[3] == pytest.approx(10.0)
    assert folded[4] == pytest.approx(180.0)
    assert folded[5] == pytest.approx(180.0)


def test_dihedral_angle_right_angle() -> None:
    p0 = np.array([1.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 1.0, 1.0])
    assert dihedral_angle(p0, p1, p2, p3) == pytest.approx(90.0, abs=1e-6)


def test_backbone_cv_pair_matches_3mer_dihedrals() -> None:
    symbols, pos = build_synthetic_3mer()
    pair = backbone_cv_pair(pos, symbols)
    angles = inter_ring_dihedrals(pos, symbols)
    assert pair[0] == pytest.approx(angles[0])
    assert pair[1] == pytest.approx(angles[1])


def test_planar_3mer_has_two_near_zero_inter_ring_dihedrals() -> None:
    symbols, pos = build_synthetic_3mer()
    quads = inter_ring_dihedral_quads(pos, symbols)
    assert len(quads) == 2
    angles = inter_ring_dihedrals(pos, symbols)
    assert angles.shape == (2,)
    assert abs(wrap_deg(angles[0])) < 5.0
    assert abs(wrap_deg(angles[1])) < 5.0


def test_twisting_terminal_ring_moves_only_theta23() -> None:
    symbols, pos = build_synthetic_3mer()
    twisted = twist_ring(pos, symbols, ring_index=2, angle_deg=90.0)
    base = inter_ring_dihedrals(pos, symbols)
    new = inter_ring_dihedrals(twisted, symbols)
    assert abs(wrap_deg(new[0] - base[0])) < 8.0
    assert abs(wrap_deg(new[1])) == pytest.approx(90.0, abs=12.0)


def test_low_density_bins_flag_empty_quadrant() -> None:
    rng = np.random.default_rng(0)
    theta12 = rng.normal(0.0, 15.0, size=400)
    theta23 = rng.normal(0.0, 15.0, size=400)
    hist, edges = cv_histogram_2d(theta12, theta23, bins=12, range_deg=(-180.0, 180.0))
    empty = low_density_bins(hist, edges, max_count=0)
    assert any(x0 >= 90.0 and y0 >= 90.0 for x0, x1, y0, y1 in empty)


def test_nearest_train_cv_distance_is_small_for_train_point() -> None:
    train = np.array([[0.0, 0.0], [10.0, -5.0], [-8.0, 4.0]])
    near = nearest_train_cv_distance(np.array([0.2, -0.1]), train)
    far = nearest_train_cv_distance(np.array([160.0, 160.0]), train)
    assert near < 1.0
    assert far > near
