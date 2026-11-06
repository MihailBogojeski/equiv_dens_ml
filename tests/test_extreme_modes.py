"""Extreme-amplitude normal-mode displacements and q-space projection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "lib"))
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

import generate_extreme_modes as gen  # noqa: E402
import normal_modes as nm  # noqa: E402
import project_mode_cv as pmc  # noqa: E402


def _orthonormal_modes(n_atoms: int, n_keep: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(3 * n_atoms, 3 * n_atoms + 6))
    q, _ = np.linalg.qr(raw)
    evecs = q[:, 6 : 6 + n_keep]
    evals = np.arange(1, n_keep + 1, dtype=float)
    return evals, evecs


def test_drop_rigid_modes_keeps_highest_frequency_block():
    evals = np.array([1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 100.0, 200.0, 300.0])
    evecs = np.eye(9)
    kept_w, kept_v = nm.drop_rigid_modes(evals, evecs, n_drop=6)
    np.testing.assert_allclose(kept_w, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(kept_v, evecs[:, 6:])


def test_displacement_along_mode_recovers_that_coordinate():
    n_atoms = 4
    masses = np.ones(n_atoms)
    _, evecs = _orthonormal_modes(n_atoms, 3)
    positions = np.zeros((n_atoms, 3))
    mode = evecs[:, 1].reshape(n_atoms, 3)
    displaced = nm.displace_along_mode(positions, mode, rmsd=0.5)
    assert nm.cartesian_rmsd(positions, displaced) == pytest.approx(0.5)
    q = nm.project_modes(displaced - positions, evecs, masses)
    np.testing.assert_allclose(q[0], 0.0, atol=1e-12)
    assert abs(q[1]) > 0.1
    np.testing.assert_allclose(q[2], 0.0, atol=1e-12)


def test_equilibrium_projection_is_zero():
    n_atoms = 3
    masses = np.array([12.0, 1.0, 1.0])
    _, evecs = _orthonormal_modes(n_atoms, 2)
    q = nm.project_modes(np.zeros((n_atoms, 3)), evecs, masses)
    np.testing.assert_allclose(q, 0.0, atol=1e-12)


def test_align_kabsch_superposes_rotated_copy():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(5, 3))
    angle = 0.7
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mobile = ref @ rot.T + np.array([3.0, -1.0, 2.0])
    aligned = nm.align_kabsch(mobile, ref)
    assert nm.cartesian_rmsd(aligned, ref) < 1e-12


def test_mode_nearest_neighbour_distance_zero_on_train_point():
    cloud = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    d = nm.nearest_mode_distance(np.array([1.0, 0.0]), cloud)
    assert d == pytest.approx(0.0)
    assert nm.nearest_mode_distance(np.array([0.0, 3.0]), cloud) == pytest.approx(1.0)


def test_mode_scan_hits_requested_rmsd_and_keeps_stretched_frames():
    n_atoms = 3
    symbols = ["C", "H", "H"]
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 0.9, 0.0]])
    masses = np.ones(n_atoms)
    _, evecs = _orthonormal_modes(n_atoms, 2, seed=3)
    frames, records = gen.build_mode_scan(
        symbols,
        positions,
        masses,
        evecs,
        frequencies=np.array([100.0, 200.0]),
        rmsd_ladder=(0.5, 10.0),
        n_modes=2,
    )
    assert len(frames) == 8  # 2 modes * 2 signs * 2 amplitudes
    assert {round(r["rmsd"], 6) for r in records} == {0.5, 10.0}
    assert all("q_k" in r and "mode_index" in r for r in records)
    assert any(r["bond_stretched"] or r["rmsd"] == 10.0 for r in records)


def test_mode_cv_projects_displaced_frame_and_measures_train_distance(tmp_path: Path):
    n_atoms = 4
    masses = np.ones(n_atoms)
    _, evecs = _orthonormal_modes(n_atoms, 2, seed=4)
    eq = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    displaced = nm.displace_along_mode(eq, evecs[:, 0].reshape(n_atoms, 3), 0.4)
    np.savez(
        tmp_path / "modes.npz",
        equilibrium=eq,
        modes=evecs,
        masses=masses,
        frequencies=np.array([80.0, 120.0]),
        symbols=np.array(["C", "H", "H", "H"]),
    )
    train_q = pmc.project_cloud([eq], eq, evecs, masses)
    test_q = pmc.project_cloud([displaced], eq, evecs, masses)
    assert train_q.shape == (1, 2)
    np.testing.assert_allclose(train_q[0], 0.0, atol=1e-12)
    assert abs(test_q[0, 0]) > abs(test_q[0, 1])
    assert nm.nearest_mode_distance(test_q[0], train_q) == pytest.approx(float(np.linalg.norm(test_q[0])))
