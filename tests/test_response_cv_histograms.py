"""Binning and series extraction for Response Figure 2 (water CV histograms)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

import plot_response_letter_figures as figs  # noqa: E402


def test_cv_hist_panels_are_continuous():
    keys = [key for key, _title in figs.CV_HIST_PANELS]
    assert "n_water" not in keys
    assert "q3_per_molecule" in keys
    assert keys == [
        "q3_per_molecule",
        "n_hbond",
        "local_density",
        "mean_nn_oo",
    ]


def test_cv_series_pools_per_molecule_q3():
    frames = [
        {"q3_tetrahedral": 0.5, "q3_per_molecule": [0.4, 0.6], "n_hbond": 1.0},
        {"q3_tetrahedral": float("nan"), "q3_per_molecule": [], "n_hbond": 0.5},
    ]
    q3 = figs.cv_series(frames, "q3_per_molecule")
    assert list(q3) == [0.4, 0.6]
    hb = figs.cv_series(frames, "n_hbond")
    assert list(hb) == [1.0, 0.5]


def test_shared_hist_bins_span_all_series_and_leave_no_empty_wings():
    a = np.array([1.0, 1.2, 1.4])
    b = np.array([2.8, 3.0])
    edges = figs.shared_hist_bins(a, b, n_bins=20, pad_frac=0.05)
    assert len(edges) == 21
    assert edges[0] < 1.0
    assert edges[-1] > 3.0
    span = edges[-1] - edges[0]
    data_span = 3.0 - 1.0
    assert span < 1.3 * data_span


def test_water_cv_includes_rackers_split():
    assert "Rackers/Rakshit" in figs.WATER_CV
    assert figs.WATER_CV["Rackers/Rakshit"].endswith("rackers.json")


def test_iter_water_cv_splits_skips_missing(tmp_path, monkeypatch):
    missing = tmp_path / "no_such_cv.json"
    present = tmp_path / "present.json"
    present.write_text('{"frames": [{"n_hbond": 1.0}]}')
    monkeypatch.setattr(
        figs,
        "WATER_CV",
        {"Keep": present.as_posix(), "Drop": missing.as_posix()},
    )
    loaded = dict(figs.iter_water_cv_splits())
    assert list(loaded) == ["Keep"]
    assert loaded["Keep"][0]["n_hbond"] == 1.0


def test_dataset_cv_histograms_write_png(tmp_path, monkeypatch):
    dest = tmp_path / "dataset_water_cv_histograms.pdf"
    monkeypatch.setattr(figs, "OUT", tmp_path)
    written = figs.plot_dataset_cv_histograms()
    assert written == dest
    assert dest.is_file()
    assert dest.stat().st_size > 0
