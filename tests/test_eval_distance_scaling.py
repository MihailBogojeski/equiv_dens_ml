"""OMol25 ΔE / ΔF relative to the lowest-energy short-range scan point."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

pytest.importorskip("tblite")  # GFN2-xTB backend of the script; optional in CI
import eval_distance_scaling as eds  # noqa: E402


def test_delta_energy_uses_lowest_sr_point_per_parent():
    records = [
        {"parent": "n2", "lambda": 0.95, "regime": "SR"},
        {"parent": "n2", "lambda": 1.2, "regime": "SR"},
        {"parent": "n2", "lambda": 2.5, "regime": "LR"},
        {"parent": "n4", "lambda": 1.1, "regime": "SR"},
        {"parent": "n4", "lambda": 2.8, "regime": "LR"},
    ]
    energies = np.array([-10.0, -12.0, -8.0, -20.0, -15.0])
    rows = eds.scan_deltas(records, energies)
    n2 = [r for r in rows if r["parent"] == "n2"]
    assert {r["lambda"] for r in n2 if r["is_sr_reference"]} == {1.2}
    assert next(r["delta_e"] for r in n2 if r["lambda"] == 1.2) == 0.0
    assert next(r["delta_e"] for r in n2 if r["lambda"] == 2.5) == 4.0
    summary = eds.summarize_sr_lr(rows)
    assert summary["SR"]["n"] == 3
    assert summary["LR"]["n"] == 2
    assert summary["LR"]["delta_e_mae"] == 4.5  # (4.0 + 5.0) / 2


def test_delta_force_mae_against_sr_reference():
    records = [
        {"parent": "a", "lambda": 1.0, "regime": "SR"},
        {"parent": "a", "lambda": 3.0, "regime": "LR"},
    ]
    energies = np.array([0.0, 1.0])
    forces = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )
    rows = eds.scan_deltas(records, energies, forces=forces)
    lr = next(r for r in rows if r["regime"] == "LR")
    assert lr["delta_f_mae"] == pytest.approx(4.0 / 6.0)
