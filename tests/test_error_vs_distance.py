"""The headline error-versus-distance curve.

This figure is the answer to the reviewer's train/test-overlap objection, so the
thing to guard against is it producing a convincing-looking curve from a bug:
frames silently dropped, errors joined to the wrong frame, or a trend that comes
out of the binning rather than the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision/lib"))
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision/analysis"))

import error_vs_distance as evd  # noqa: E402

CVS = ("q_tetrahedral", "n_hbond", "ring_6_frac", "local_density", "surface_frac", "mean_nn_oo")


def write_cv(path: Path, rows: list[dict], label: str = "x") -> Path:
    for i, row in enumerate(rows):
        row.setdefault("index", i)
    path.write_text(json.dumps({"label": label, "frames": rows}))
    return path


def write_eval(path: Path, afes: dict[int, float]) -> Path:
    path.write_text(
        json.dumps({"records": [{"index": i, "afe": v, "natoms": 6} for i, v in afes.items()]})
    )
    return path


def cv_row(offset: float, rng) -> dict:
    base = {name: float(rng.normal(1.0, 0.05)) for name in CVS}
    base["local_density"] = 1.0 + offset
    base["n_hbond"] = 2.0 + offset
    return base


def test_spearman_recovers_a_monotone_trend():
    x = np.arange(20.0)
    assert evd.spearman(x, x**3) == pytest.approx(1.0)
    assert evd.spearman(x, -(x**3)) == pytest.approx(-1.0)


def test_spearman_is_flat_for_unrelated_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = rng.normal(size=400)
    assert abs(evd.spearman(x, y)) < 0.2


def test_exp_kernel_is_one_at_the_origin_and_equals_e_minus_x():
    assert evd.exp_kernel(0.0) == pytest.approx(1.0)
    assert evd.exp_kernel(1.0) == pytest.approx(np.exp(-1.0))
    assert evd.exp_kernel(2.0) == pytest.approx(np.exp(-2.0))
    assert evd.exp_kernel(0.0) > evd.exp_kernel(1.0) > evd.exp_kernel(2.0)


def test_saturation_curve_starts_at_the_floor_and_approaches_the_plateau():
    x = np.array([0.0, 20.0])
    y = evd.saturation_curve(x, sigma2=0.002, length_scale=0.5, floor=0.003)
    assert y[0] == pytest.approx(0.003)
    assert y[1] == pytest.approx(0.005)


def test_fit_saturation_curve_recovers_known_params():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 4.0, 80)
    y = evd.saturation_curve(x, sigma2=0.0012, length_scale=0.4, floor=0.003)
    y = y + rng.normal(0.0, 1.0e-5, size=x.size)
    fit = evd.fit_saturation_curve(x, y)
    assert fit["floor"] == pytest.approx(0.003, abs=2.0e-4)
    assert fit["sigma2"] == pytest.approx(0.0012, abs=2.0e-4)
    assert fit["length_scale"] == pytest.approx(0.4, abs=0.08)


def test_fit_saturation_curve_keeps_a_physical_floor_on_a_near_flat_rise():
    x = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    y = np.array([0.051, 0.053, 0.055, 0.058, 0.061])
    fit = evd.fit_saturation_curve(x, y)
    assert fit["floor"] >= 0.0
    assert fit["floor"] < 0.08
    assert 0.0 <= fit["sigma2"] < 0.08
    assert evd.saturation_curve(0.0, fit["sigma2"], fit["length_scale"], fit["floor"]) >= 0.0


def test_series_saturation_fit_tracks_tier_medians_not_the_id_pileup():
    series = [
        ("id_test", np.full(80, 0.006), np.full(80, 0.00274)),
        ("ood_size", np.full(20, 0.36), np.full(20, 0.00390)),
        ("ood_order", np.full(20, 1.90), np.full(20, 0.00437)),
        ("ood_density", np.full(20, 2.79), np.full(20, 0.00400)),
    ]
    x, y = evd.saturation_fit_xy(series)
    assert len(x) == 4
    assert y[0] == pytest.approx(0.00274)
    fit = evd.fit_series_saturation(series)
    assert evd.saturation_curve(0.0, **fit) == pytest.approx(0.00274, abs=3.0e-4)
    assert evd.saturation_curve(0.36, **fit) == pytest.approx(0.00390, abs=3.0e-4)
    assert evd.saturation_curve(8.0, **fit) > evd.saturation_curve(0.0, **fit)


def test_bin_curve_counts_every_frame_once():
    x = np.array([0.0, 0.4, 0.9, 1.4, 1.9])
    y = np.ones_like(x)
    edges = np.linspace(0.0, 2.0, 5)
    rows = evd.bin_curve(x, y, edges)
    assert sum(r["n"] for r in rows) == len(x)


def test_bin_curve_reports_spread_not_just_the_mean():
    x = np.zeros(4)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    rows = evd.bin_curve(x, y, np.array([-1.0, 1.0]))
    assert rows[0]["afe_mean"] == pytest.approx(2.5)
    assert rows[0]["afe_std"] > 0
    assert rows[0]["afe_sem"] == pytest.approx(rows[0]["afe_std"] / 2.0)


def test_end_to_end_recovers_error_growing_with_distance(tmp_path, capsys):
    rng = np.random.default_rng(7)
    train = write_cv(tmp_path / "train.json", [cv_row(0.0, rng) for _ in range(120)])

    # Error is built to grow with the offset that also drives the descriptor
    # distance, so a correct pipeline must report a strong positive rank
    # correlation; a join on the wrong index would wash it out.
    rows, afes = [], {}
    for i in range(40):
        offset = 0.1 * i
        rows.append(cv_row(offset, rng))
        afes[i] = 0.001 * (1.0 + offset)
    test_cv = write_cv(tmp_path / "test.json", rows)
    test_eval = write_eval(tmp_path / "eval.json", afes)

    out = tmp_path / "report.json"
    sys.argv = [
        "error_vs_distance.py",
        "--train-cv", str(train),
        "--tier", f"tier_a:{test_cv}:{test_eval}",
        "--out", str(out),
    ]
    assert evd.main() == 0

    report = json.loads(out.read_text())
    entry = report["tiers"]["tier_a"]
    assert entry["n"] == 40
    assert entry["spearman_afe_vs_distance"] > 0.9
    assert sum(c["n"] for c in entry["curve"]) == 40


def test_frames_without_an_error_are_dropped_not_defaulted(tmp_path):
    rng = np.random.default_rng(3)
    train = write_cv(tmp_path / "train.json", [cv_row(0.0, rng) for _ in range(60)])
    rows = [cv_row(0.1 * i, rng) for i in range(10)]
    test_cv = write_cv(tmp_path / "test.json", rows)
    # Only half the frames were labelled, as happens mid-campaign.
    test_eval = write_eval(tmp_path / "eval.json", {i: 0.002 for i in range(0, 10, 2)})

    out = tmp_path / "report.json"
    sys.argv = [
        "error_vs_distance.py",
        "--train-cv", str(train),
        "--tier", f"partial:{test_cv}:{test_eval}",
        "--out", str(out),
    ]
    assert evd.main() == 0
    assert json.loads(out.read_text())["tiers"]["partial"]["n"] == 5


def test_physical_axis_uses_the_named_coordinate(tmp_path):
    rng = np.random.default_rng(11)
    train = write_cv(tmp_path / "train.json", [cv_row(0.0, rng) for _ in range(60)])

    rows, afes = [], {}
    for i in range(30):
        delta = -0.8 + 0.05 * i
        row = cv_row(0.0, rng)
        row["delta_pt"] = delta
        rows.append(row)
        afes[i] = 0.001 * (1.0 + delta)
    pt_cv = write_cv(tmp_path / "pt.json", rows)
    pt_eval = write_eval(tmp_path / "pt_eval.json", afes)

    out = tmp_path / "report.json"
    sys.argv = [
        "error_vs_distance.py",
        "--train-cv", str(train),
        "--tier", f"pt:{pt_cv}:{pt_eval}",
        "--physical-axis", f"pt:{pt_cv}:{pt_eval}:delta_pt",
        "--out", str(out),
    ]
    assert evd.main() == 0
    entry = json.loads(out.read_text())["physical_axis"]
    assert entry["coordinate"] == "delta_pt"
    assert entry["n"] == 30
    assert entry["spearman_afe_vs_coordinate"] > 0.99
    assert min(entry["x"]) < 0.0
    assert max(entry["x"]) > 0.0


def test_water_error_panel_is_scatter_only():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    evd.draw_water_error_panel(
        ax,
        [
            ("id_test", np.array([0.1, 0.2]), np.array([0.005, 0.006])),
            ("ood_order", np.array([1.8, 1.9]), np.array([0.006, 0.007])),
        ],
    )
    assert len(ax.collections) == 2
    assert ax.containers == []
    c0 = ax.collections[0].get_facecolor()[0]
    c1 = ax.collections[1].get_facecolor()[0]
    assert not np.allclose(c0, c1)
    plt.close(fig)


def test_water_error_panel_draws_a_thick_saturation_line():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    evd.draw_water_error_panel(
        ax,
        [
            ("id_test", np.array([0.05, 0.08, 0.12]), np.array([0.0030, 0.0031, 0.0032])),
            ("ood_order", np.array([1.7, 1.9, 2.1]), np.array([0.0043, 0.0044, 0.0045])),
        ],
    )
    thick = [line for line in ax.lines if line.get_linewidth() >= 2.5]
    assert len(thick) == 1
    x, y = thick[0].get_data()
    assert x[0] == pytest.approx(0.0)
    assert y[-1] > y[0]
    plt.close(fig)


def test_make_water_figure_writes_single_panel(tmp_path):
    dest = tmp_path / "water_only.pdf"
    evd.make_water_figure(
        [
            ("id_test", np.array([0.1, 0.2]), np.array([0.005, 0.006])),
            ("ood_order", np.array([1.8, 1.9]), np.array([0.006, 0.007])),
        ],
        dest,
    )
    assert dest.is_file()
    assert dest.stat().st_size > 0


def test_figure_water_flag_writes_without_combined_figure(tmp_path):
    rng = np.random.default_rng(19)
    train = write_cv(tmp_path / "train.json", [cv_row(0.0, rng) for _ in range(40)])
    rows = [cv_row(0.1 * i, rng) for i in range(12)]
    test_cv = write_cv(tmp_path / "test.json", rows)
    test_eval = write_eval(tmp_path / "eval.json", {i: 0.002 for i in range(12)})
    water_fig = tmp_path / "water.pdf"
    sys.argv = [
        "error_vs_distance.py",
        "--train-cv", str(train),
        "--tier", f"id_test:{test_cv}:{test_eval}",
        "--out", str(tmp_path / "report.json"),
        "--figure-water", str(water_fig),
    ]
    assert evd.main() == 0
    assert water_fig.is_file()
    assert water_fig.stat().st_size > 0


def test_malon_error_panel_uses_signed_delta_and_shades_training_well():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    evd.draw_malon_error_panel(
        ax,
        np.array([-0.8, -0.2, 0.3]),
        np.array([0.004, 0.005, 0.006]),
        train_delta_max=-0.40,
    )
    assert ax.xaxis.get_label().get_text() == r"$\delta$ / $\mathrm{\AA}$"
    assert any(getattr(artist, "get_linestyle", lambda: None)() == "--" for artist in ax.lines)
    assert ax.containers == []
    plt.close(fig)


def test_errors_join_on_the_source_frame_not_the_dataset_row(tmp_path):
    """A dropped frame must not shift every later error onto the wrong geometry.

    Frames ORCA fails on never reach the assembled dataset, so the evaluator's
    row numbers close up over the gaps while the collective variables stay
    numbered over the geometry file. Joining on the row number would slide the
    whole tail of the tier along the x-axis, which is invisible in the output --
    the curve still looks like a curve. Here the labelled frames are the even
    ones, carrying their true indices as `source_index`, and error is made to
    track the descriptor exactly, so a positional join breaks the correlation
    that a correct one recovers perfectly.
    """
    rng = np.random.default_rng(13)
    train = write_cv(tmp_path / "train.json", [cv_row(0.0, rng) for _ in range(80)])

    rows = [cv_row(0.1 * i, rng) for i in range(20)]
    test_cv = write_cv(tmp_path / "test.json", rows)

    kept = list(range(0, 20, 2))
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "records": [
                    # `index` is the row in the assembled dataset, which has closed
                    # up over the dropped odd frames; `source_index` is the truth.
                    {"index": row, "source_index": src, "afe": 0.001 * src, "natoms": 6}
                    for row, src in enumerate(kept)
                ]
            }
        )
    )

    # The join itself, stated exactly: errors land on frames 0,2,4,... and not on
    # rows 0,1,2,..., which is what a positional read would have given.
    assert evd.load_errors(eval_path) == {src: 0.001 * src for src in kept}

    out = tmp_path / "report.json"
    sys.argv = [
        "error_vs_distance.py",
        "--train-cv", str(train),
        "--tier", f"gappy:{test_cv}:{eval_path}",
        "--out", str(out),
    ]
    assert evd.main() == 0

    entry = json.loads(out.read_text())["tiers"]["gappy"]
    assert entry["n"] == len(kept)
    # Error was made to rise with the same offset that drives the descriptor
    # distance, so a correct join leaves a strong monotone trend. It is not
    # exactly 1 because the other descriptors carry noise, as real ones do.
    assert entry["spearman_afe_vs_distance"] > 0.85


def test_reports_without_a_source_index_still_join_positionally(tmp_path):
    """Older reports predate the key and meant the row number when they said index."""
    errors = evd.load_errors(
        write_eval(tmp_path / "old.json", {0: 0.1, 1: 0.2, 2: 0.3})
    )
    assert errors == {0: 0.1, 1: 0.2, 2: 0.3}


def test_missing_inputs_are_reported_not_fatal(tmp_path):
    rng = np.random.default_rng(5)
    train = write_cv(tmp_path / "train.json", [cv_row(0.0, rng) for _ in range(40)])
    out = tmp_path / "report.json"
    sys.argv = [
        "error_vs_distance.py",
        "--train-cv", str(train),
        "--tier", f"absent:{tmp_path / 'nope.json'}:{tmp_path / 'nope2.json'}",
        "--out", str(out),
    ]
    assert evd.main() == 0
    assert json.loads(out.read_text())["tiers"] == {}


def test_dw_error_axis_is_linear_nonnegative_and_uses_varepsilon_rho():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots()
    ax.scatter([0.0, 1.0], [3.0e-3, 6.0e-3])
    evd.style_dw_error_axis(ax)
    assert ax.get_yscale() == "linear"
    assert ax.get_xlim()[0] == pytest.approx(0.0)
    assert ax.yaxis.get_label().get_text() == evd.EPSILON_RHO
    formatter = ax.yaxis.get_major_formatter()
    assert isinstance(formatter, ScalarFormatter)
    assert formatter._powerlimits == (-3, -3)
    plt.close(fig)
