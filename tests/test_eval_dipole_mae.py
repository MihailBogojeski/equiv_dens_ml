"""Dipole MAE: stored DFT vector vs prediction, same origin."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision/lib"))
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision/analysis"))

from eval_dipole_mae import (  # noqa: E402
    AU_TO_DEBYE,
    dipole_mae_debye,
    net_charge,
    nuclear_com,
    shift_dipole_to_com,
)


def test_neutral_water_dipole_is_origin_independent() -> None:
    z = np.array([8, 1, 1], dtype=int)
    xyz = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=float)
    mu0 = np.array([0.4, 0.1, 0.0], dtype=float)
    assert net_charge(z, n_electrons=10) == 0
    shifted = shift_dipole_to_com(mu0, z, xyz, n_electrons=10)
    np.testing.assert_allclose(shifted, mu0)


def test_charged_dipole_shifts_by_q_times_com() -> None:
    z = np.array([8, 1], dtype=int)
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    mu0 = np.array([1.0, 0.0, 0.0], dtype=float)
    com = nuclear_com(z, xyz)
    q = net_charge(z, n_electrons=8)  # OH+ : 9 protons, 8 electrons
    shifted = shift_dipole_to_com(mu0, z, xyz, n_electrons=8)
    np.testing.assert_allclose(shifted, mu0 - q * com)


def test_mae_rmse_in_debye() -> None:
    ref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    pred = np.array([[1.1, 0.0, 0.0], [0.0, 0.8, 0.0]], dtype=float)
    stats = dipole_mae_debye(ref, pred, already_debye=True)
    assert stats["n_frames"] == 2
    assert stats["mae_debye"] == pytest.approx(0.15)
    assert stats["rmse_debye"] == pytest.approx(np.sqrt((0.1**2 + 0.2**2) / 2))


def test_eval_cli_on_stub_npy(tmp_path: Path) -> None:
    from eval_dipole_mae import main

    base = {
        "positions": np.zeros((2, 3, 3), dtype=float),
        "atom_numbers": np.array([[8, 1, 1], [8, 1, 1]], dtype=int),
        "dipole_moment": np.array([[0.5, 0.0, 0.0], [0.0, 0.4, 0.0]], dtype=float),
    }
    pred = {"dipole_moment": base["dipole_moment"] + np.array([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]])}
    base_path = tmp_path / "split_base.npy"
    pred_path = tmp_path / "pred.npy"
    out = tmp_path / "dipole_mae.json"
    np.save(base_path, base, allow_pickle=True)
    np.save(pred_path, pred, allow_pickle=True)
    rc = main(
        [
            "--base-npy",
            str(base_path),
            "--pred-npy",
            str(pred_path),
            "--already-debye",
            "--json-out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["n_frames"] == 2
    assert payload["mae_debye"] > 0.0
    assert AU_TO_DEBYE > 2.0
