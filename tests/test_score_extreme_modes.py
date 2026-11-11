"""Join DenSNet density error onto extreme-mode frames by source_index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

import score_extreme_modes as sem  # noqa: E402


def _manifest_records() -> list[dict]:
    # XYZ order: frame i is records[i]. AFE will be made to track |q_k|.
    return [
        {"mode_index": 0, "q_k": -1.0, "rmsd": 0.5, "rmsd_target": 0.5, "bond_stretched": False},
        {"mode_index": 0, "q_k": -4.0, "rmsd": 2.0, "rmsd_target": 2.0, "bond_stretched": True},
        {"mode_index": 1, "q_k": 2.0, "rmsd": 1.0, "rmsd_target": 1.0, "bond_stretched": False},
        {"mode_index": 1, "q_k": 8.0, "rmsd": 4.0, "rmsd_target": 4.0, "bond_stretched": True},
    ]


def test_join_uses_source_index_not_assembled_row():
    """ORCA drops close the assembled rows; the XYZ index is the join key."""
    manifest = _manifest_records()
    # Frames 0 and 2 failed SCF. Assembled rows 0,1 are XYZ 1,3.
    eval_records = [
        {"index": 0, "source_index": 1, "afe": 0.04},
        {"index": 1, "source_index": 3, "afe": 0.08},
    ]
    joined = sem.join_mode_eval(manifest, eval_records)
    assert [row["source_index"] for row in joined] == [1, 3]
    assert joined[0]["abs_q"] == pytest.approx(4.0)
    assert joined[0]["afe"] == pytest.approx(0.04)
    assert joined[1]["abs_q"] == pytest.approx(8.0)
    assert joined[0]["rmsd"] == pytest.approx(2.0)
    assert joined[1]["bond_stretched"] is True


def test_join_skips_eval_rows_without_source_index():
    manifest = _manifest_records()
    eval_records = [
        {"index": 0, "afe": 0.01},
        {"index": 1, "source_index": 2, "afe": 0.02},
    ]
    joined = sem.join_mode_eval(manifest, eval_records)
    assert [row["source_index"] for row in joined] == [2]
    assert joined[0]["afe"] == pytest.approx(0.02)


def test_summarize_recovers_afe_vs_abs_q_trend():
    manifest = _manifest_records()
    eval_records = [
        {"source_index": i, "afe": 0.01 * abs(rec["q_k"])}
        for i, rec in enumerate(manifest)
    ]
    joined = sem.join_mode_eval(manifest, eval_records)
    summary = sem.summarize_mode_errors(joined)
    assert summary["n"] == 4
    assert summary["afe_mean"] == pytest.approx(0.0375)
    assert summary["spearman_afe_vs_abs_q"] == pytest.approx(1.0)
    assert summary["spearman_afe_vs_rmsd"] == pytest.approx(1.0)


def test_cli_writes_joined_report(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"molecule": "ethanol", "records": _manifest_records()}))
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "records": [
                    {"index": 0, "source_index": 0, "afe": 0.01},
                    {"index": 1, "source_index": 3, "afe": 0.08},
                ]
            }
        )
    )
    out = tmp_path / "score.json"
    fig = tmp_path / "score.png"
    sys.argv = [
        "score_extreme_modes.py",
        "--pair",
        f"ethanol:{manifest_path}:{eval_path}",
        "--out",
        str(out),
        "--figure",
        str(fig),
    ]
    assert sem.main() == 0
    report = json.loads(out.read_text())
    ethanol = report["systems"]["ethanol"]
    assert ethanol["n"] == 2
    assert ethanol["afe_mean"] == pytest.approx(0.045)
    abs_q = {row["source_index"]: row["abs_q"] for row in ethanol["records"]}
    assert abs_q[0] == pytest.approx(1.0)
    assert abs_q[3] == pytest.approx(8.0)
    from mpl_classic_tex import response_figure_path

    dest = response_figure_path(fig)
    assert dest.is_file()
    assert dest.stat().st_size > 0


def test_series_labels_are_tex_safe_and_human_readable():
    for key, spec in sem.SERIES_STYLE.items():
        label = spec["label"]
        assert "_" not in label, key
        assert "water_n" not in label
        assert spec["color"]
        assert spec["marker"]


def test_fit_mode_saturation_uses_rung_medians():
    rng = np.random.default_rng(0)
    rmsd = np.repeat([0.5, 1.0, 2.0, 5.0, 10.0], 8)
    true = {"floor": 0.04, "sigma2": 0.06, "length_scale": 2.5}
    afe = sem.saturation_curve(rmsd, true["sigma2"], true["length_scale"], true["floor"])
    afe = afe + rng.normal(0.0, 0.003, size=afe.size)
    fit = sem.fit_mode_saturation(rmsd, afe)
    assert fit["floor"] == pytest.approx(true["floor"], abs=0.01)
    assert fit["sigma2"] == pytest.approx(true["sigma2"], abs=0.015)
    assert fit["length_scale"] == pytest.approx(true["length_scale"], rel=0.35)
    wrecked = np.append(afe, 3.45)
    wrecked_x = np.append(rmsd, 1.0)
    clean = sem.fit_mode_saturation(wrecked_x, wrecked)
    assert clean["sigma2"] == pytest.approx(fit["sigma2"], rel=0.05)


def test_mode_error_panel_draws_a_thick_saturation_line():
    systems = {
        "ethanol_pbe": {
            "records": [
                {"abs_q": 1.0, "rmsd": 0.5, "afe": 0.035},
                {"abs_q": 2.0, "rmsd": 1.0, "afe": 0.048},
                {"abs_q": 4.0, "rmsd": 2.0, "afe": 0.070},
                {"abs_q": 8.0, "rmsd": 5.0, "afe": 0.095},
                {"abs_q": 12.0, "rmsd": 10.0, "afe": 0.106},
            ]
        },
    }
    fig = sem.draw_mode_error_figure(systems)
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    thick = [line for ax in visible for line in ax.lines if line.get_linewidth() >= 2.5]
    assert len(thick) == 1
    x, y = thick[0].get_data()
    assert x[0] == pytest.approx(0.0)
    assert y[-1] > y[0]
    plt.close(fig)


def test_make_figure_colors_molecules_and_marks_theory(tmp_path: Path):
    systems = {
        "water_n2_pbe": {
            "records": [
                {"abs_q": 1.0, "rmsd": 0.5, "afe": 0.05},
                {"abs_q": 8.0, "rmsd": 10.0, "afe": 0.06},
            ]
        },
        "water_n2_hybrid": {
            "records": [
                {"abs_q": 1.2, "rmsd": 0.5, "afe": 0.05},
                {"abs_q": 9.0, "rmsd": 10.0, "afe": 0.17},
            ]
        },
        "malonaldehyde_pbe": {
            "records": [
                {"abs_q": 2.0, "rmsd": 0.5, "afe": 0.04},
                {"abs_q": 7.0, "rmsd": 10.0, "afe": 0.09},
            ]
        },
        "resorcinol_pbe": {
            "records": [
                {"abs_q": 3.0, "rmsd": 5.0, "afe": 12.0},
            ]
        },
    }
    dest = tmp_path / "modes.pdf"
    fig = sem.draw_mode_error_figure(systems)
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert any("H_2O" in title for title in titles)
    assert "Malonaldehyde" in titles
    assert "Resorcinol" not in titles
    assert len(visible) == 2
    for ax in visible:
        assert ax.get_yscale() == "linear"
    from mpl_classic_tex import save_response_figure

    save_response_figure(fig, dest)
    assert dest.is_file()
