"""Require a finite dipole_moment on assembled DFT base npy files.

The ORCA worker already stores ``calc["dipole"]`` from int1e_r. Assembly
copies that to ``dipole_moment``. This gate fails the write when too many
frames are missing a usable 3-vector, so later dipole MAE eval does not
discover an empty column after 100k SCF jobs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DIPOLE_MISSING_FRACTION_MAX = 0.02


def dipole_is_finite(value: Any) -> bool:
    """True for a finite, non-zero 3-vector (a.u. or Debye)."""
    if value is None:
        return False
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        return False
    if not np.isfinite(arr).all():
        return False
    return bool(np.linalg.norm(arr) > 0.0)


def _missing_mask(dipole: np.ndarray) -> np.ndarray:
    dipole = np.asarray(dipole, dtype=float)
    if dipole.ndim != 2 or dipole.shape[1] != 3:
        raise ValueError(f"dipole_moment must have shape (n, 3), got {dipole.shape}")
    finite = np.isfinite(dipole).all(axis=1)
    nonzero = np.linalg.norm(dipole, axis=1) > 0.0
    return ~(finite & nonzero)


def assert_dipole_moment(
    base: dict,
    max_missing_frac: float = DIPOLE_MISSING_FRACTION_MAX,
) -> None:
    """Raise if ``dipole_moment`` is absent or too incomplete."""
    if "dipole_moment" not in base:
        raise ValueError("assembled base npy is missing dipole_moment")
    missing = _missing_mask(base["dipole_moment"])
    n = int(missing.size)
    n_missing = int(missing.sum())
    frac = n_missing / n if n else 1.0
    if frac > max_missing_frac:
        raise ValueError(
            f"dipole_moment missing or non-finite on {n_missing}/{n} frames "
            f"({frac:.1%} > {max_missing_frac:.1%})"
        )
