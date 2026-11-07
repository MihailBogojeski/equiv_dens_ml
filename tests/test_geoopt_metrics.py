"""Geometry comparison helpers for R2.11 DenSNet vs DFT minima."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "revision" / "lib"))

from geoopt_metrics import (  # noqa: E402
    align_onto,
    angle_deg,
    covalent_bonds,
    heavy_atom_rmsd,
    kabsch_rmsd,
    named_angle_groups,
    named_bond_groups,
    pair_distance,
    summarize_internal_errors,
)


def _ethanol_coords() -> tuple[list[str], np.ndarray]:
    symbols = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.00, 0.00, 0.00],
            [1.52, 0.00, 0.00],
            [-0.70, 1.20, 0.00],
            [-0.40, -0.70, 0.70],
            [-0.40, -0.70, -0.70],
            [1.88, 0.70, 0.70],
            [1.88, 0.70, -0.70],
            [1.88, -0.90, 0.00],
            [-1.40, 1.70, 0.00],
        ],
        dtype=float,
    )
    return symbols, coords


def test_kabsch_rmsd_vanishes_for_rigid_motion() -> None:
    _, coords = _ethanol_coords()
    shifted = coords + np.array([2.0, -1.0, 0.5])
    angle = np.pi / 3.0
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = shifted @ rotation.T
    assert kabsch_rmsd(moved, coords) < 1e-10
    assert np.allclose(align_onto(moved, coords), coords, atol=1e-10)


def test_heavy_atom_rmsd_ignores_hydrogen_shift() -> None:
    symbols, coords = _ethanol_coords()
    perturbed = coords.copy()
    perturbed[3] += np.array([0.4, 0.0, 0.0])
    assert heavy_atom_rmsd(perturbed, coords, symbols) < 1e-10
    assert kabsch_rmsd(perturbed, coords) > 0.05


def test_ethanol_covalent_groups() -> None:
    symbols, coords = _ethanol_coords()
    bonds = covalent_bonds(symbols, coords)
    groups = named_bond_groups(symbols, bonds)
    assert len(groups["C-C"]) == 1
    assert len(groups["C-O"]) == 1
    assert len(groups["C-H"]) == 5
    assert len(groups["O-H"]) == 1
    angles = named_angle_groups(symbols, bonds)
    assert len(angles["C-C-O"]) == 1
    assert abs(angle_deg(coords, angles["C-C-O"][0]) - 120.0) < 8.0


def test_internal_cc_mae_is_exact() -> None:
    symbols, coords = _ethanol_coords()
    other = coords.copy()
    other[1, 0] += 0.02
    other[2, 1] += 0.05
    summary = summarize_internal_errors(symbols, other, coords)
    assert summary["bond_mae_ang"]["C-C"] == pytest.approx(0.02)
    assert summary["angle_mae_deg"]["C-C-O"] > 0.0
    assert pair_distance(coords, (0, 1)) == pytest.approx(1.52)
