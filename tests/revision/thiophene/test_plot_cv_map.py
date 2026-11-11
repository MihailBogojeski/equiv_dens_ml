"""CV-map writer produces a PNG from synthetic clouds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_THIO = _REPO / "scripts" / "revision" / "thiophene"
sys.path.insert(0, str(_THIO))

from plot_thiophene_mh_figures import plot_cv_map  # noqa: E402


def test_plot_cv_map_writes_png(tmp_path: Path) -> None:
    dest = plot_cv_map(
        np.array([[0.0, 0.0], [8.0, -4.0]]),
        np.array([[0.0, 0.0], [140.0, 150.0]]),
        [{"label": "unsampled", "theta12": 140.0, "theta23": 150.0}],
        tmp_path / "thiophene_mh_cv_map.png",
    )
    assert dest.is_file()
    assert dest.stat().st_size > 100
