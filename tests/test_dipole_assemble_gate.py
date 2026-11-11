"""Assembled base npy must carry a finite dipole_moment on every frame."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision/lib"))
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision/qm7x"))

from dipole_gate import (  # noqa: E402
    DIPOLE_MISSING_FRACTION_MAX,
    assert_dipole_moment,
    dipole_is_finite,
)
from qm7x_orca_common import pad_frames  # noqa: E402


def _frame(*, dipole=None, include_dipole=True) -> dict:
    fr = {
        "index": 0,
        "atom_numbers": [8, 1, 1],
        "positions": [[0.0, 0.0, 0.0], [0.76, 0.58, 0.0], [-0.76, 0.58, 0.0]],
        "energy": -76.0,
        "forces": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    }
    if include_dipole:
        fr["dipole"] = dipole
    return fr


def test_pad_frames_writes_dipole_moment_from_worker_dipole() -> None:
    packed = pad_frames([_frame(dipole=[0.1, -0.2, 0.3])])
    assert "dipole_moment" in packed
    np.testing.assert_allclose(packed["dipole_moment"][0], [0.1, -0.2, 0.3])
    assert_dipole_moment(packed)


def test_gate_fails_when_dipole_missing() -> None:
    packed = pad_frames([_frame(include_dipole=False)])
    assert "dipole_moment" not in packed
    with pytest.raises(ValueError, match="dipole"):
        assert_dipole_moment(packed)


def test_gate_fails_when_dipole_is_nan() -> None:
    packed = pad_frames([_frame(dipole=[np.nan, 0.0, 0.0])])
    with pytest.raises(ValueError, match="dipole"):
        assert_dipole_moment(packed)


def test_gate_allows_small_missing_fraction() -> None:
    good = _frame(dipole=[1.0, 0.0, 0.0])
    bad = _frame(include_dipole=False)
    # One missing in 100 is under the few-percent ceiling.
    frames = [dict(good, index=i, dipole=[1.0, 0.0, float(i)]) for i in range(99)]
    frames.append(dict(bad, index=99))
    packed = pad_frames(frames)
    # pad_frames writes a zero row when the key is absent on some frames
    # once any frame has a dipole. The gate counts non-finite / all-zero? 
    # Missing key on a subset still produces a (n, 3) array; zeros are not
    # a valid DFT dipole for water. Treat all-zero as missing.
    with pytest.raises(ValueError):
        assert_dipole_moment(packed, max_missing_frac=0.005)
    assert_dipole_moment(packed, max_missing_frac=0.02)
    assert DIPOLE_MISSING_FRACTION_MAX == pytest.approx(0.02)


def test_dipole_is_finite_rejects_none_and_wrong_shape() -> None:
    assert dipole_is_finite([0.1, 0.2, 0.3])
    assert not dipole_is_finite(None)
    assert not dipole_is_finite([0.0, 0.0, 0.0])
    assert not dipole_is_finite([np.nan, 0.0, 0.0])
