"""Response-letter figures for the two new OOD CVs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

import plot_response_letter_figures as figs  # noqa: E402


def test_distance_scale_plot_writes_png(tmp_path: Path):
    payload = {
        "records": [
            {"regime": "SR", "r_oo_min": 2.8, "delta_e": 0.0},
            {"regime": "SR", "r_oo_min": 4.0, "delta_e": 0.2},
            {"regime": "LR", "r_oo_min": 8.4, "delta_e": 1.5},
        ]
    }
    src = tmp_path / "eval.json"
    src.write_text(json.dumps(payload))
    dest = tmp_path / "scale.pdf"
    written = figs.plot_distance_scale(src, dest)
    assert written == dest
    assert dest.is_file()
    assert dest.stat().st_size > 0


def test_extreme_mode_plot_writes_png(tmp_path: Path):
    payload = {
        "molecule": "ethanol",
        "records": [
            {"rmsd": 0.5, "q_k": 1.2},
            {"rmsd": 10.0, "q_k": -20.0},
        ],
    }
    src = tmp_path / "manifest.json"
    src.write_text(json.dumps(payload))
    dest = tmp_path / "modes.pdf"
    written = figs.plot_extreme_modes(src, dest)
    assert written == dest
    assert dest.is_file()


def test_extreme_mode_error_plot_writes_png(tmp_path: Path):
    payload = {
        "systems": {
            "ethanol_pbe": {
                "records": [
                    {"abs_q": 1.2, "rmsd": 0.5, "afe": 0.04},
                    {"abs_q": 20.0, "rmsd": 10.0, "afe": 0.11},
                ]
            }
        }
    }
    src = tmp_path / "score.json"
    src.write_text(json.dumps(payload))
    dest = tmp_path / "mode_error.pdf"
    written = figs.plot_extreme_mode_error(src, dest)
    assert written == dest
    assert dest.is_file()
    assert dest.stat().st_size > 0
