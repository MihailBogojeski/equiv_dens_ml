"""Select SO3krates A–D regions and an unsampled CV box for DFT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_THIO = _REPO / "scripts" / "revision" / "thiophene"
sys.path.insert(0, str(_THIO))

from analyze_mh_minima import select_regions_for_dft  # noqa: E402
from validate_mh_minima_dft import (  # noqa: E402
    density_l1_per_electron,
    energy_error_mev,
    summarize_validation,
)


def test_select_regions_picks_four_dense_and_one_unsampled() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(0.0, 12.0, size=(200, 2))
    # Four MH clumps plus one in a train-empty corner.
    clumps = [
        rng.normal([-20.0, -20.0], 4.0, size=(40, 2)),
        rng.normal([20.0, -20.0], 4.0, size=(35, 2)),
        rng.normal([-20.0, 20.0], 4.0, size=(30, 2)),
        rng.normal([20.0, 20.0], 4.0, size=(25, 2)),
        np.full((8, 2), 150.0),
    ]
    mh = np.vstack(clumps)
    energies = np.linspace(-1.0, 0.0, len(mh))
    selected = select_regions_for_dft(train, mh, energies, n_dense=4)
    labels = {row["label"] for row in selected}
    assert labels >= {"A", "B", "C", "D", "unsampled"}
    unsampled = next(row for row in selected if row["label"] == "unsampled")
    assert unsampled["theta12"] > 90.0
    assert unsampled["theta23"] > 90.0


def test_density_l1_per_electron_is_normalized() -> None:
    pred = np.array([1.0, 2.0, 0.0])
    ref = np.array([1.0, 1.0, 0.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert density_l1_per_electron(pred, ref, weights, n_electrons=2.0) == pytest.approx(0.5)


def test_energy_error_mev_matches_ev_difference() -> None:
    assert energy_error_mev(ml_ev=-10.0, dft_ev=-10.041) == pytest.approx(41.0, abs=0.05)


def test_summarize_validation_reports_mean_abs_energy() -> None:
    rows = [
        {"label": "A", "energy_ml_ev": 0.0, "energy_dft_sp_ev": 0.003, "rmsd_relax_ang": 0.02},
        {"label": "unsampled", "energy_ml_ev": 0.1, "energy_dft_sp_ev": 0.141, "rmsd_relax_ang": 0.08},
    ]
    summary = summarize_validation(rows)
    assert summary["n"] == 2
    assert summary["mae_energy_mev"] == pytest.approx(22.0, abs=0.5)
    assert "unsampled" in summary["labels"]


def test_select_regions_writes_json_roundtrip(tmp_path: Path) -> None:
    train = np.zeros((10, 2))
    mh = np.array([[0.0, 0.0], [140.0, 140.0]])
    selected = select_regions_for_dft(train, mh, np.array([0.0, 0.4]), n_dense=1)
    dest = tmp_path / "selected_for_dft.json"
    dest.write_text(json.dumps({"selected": selected}))
    loaded = json.loads(dest.read_text())
    assert loaded["selected"]
