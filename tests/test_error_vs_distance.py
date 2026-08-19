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
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision"))

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
        row["abs_delta_pt"] = abs(delta)
        rows.append(row)
        afes[i] = 0.001 * (1.0 + abs(delta))
    pt_cv = write_cv(tmp_path / "pt.json", rows)
    pt_eval = write_eval(tmp_path / "pt_eval.json", afes)

    out = tmp_path / "report.json"
    sys.argv = [
        "error_vs_distance.py",
        "--train-cv", str(train),
        "--tier", f"pt:{pt_cv}:{pt_eval}",
        "--physical-axis", f"pt:{pt_cv}:{pt_eval}:abs_delta_pt",
        "--out", str(out),
    ]
    assert evd.main() == 0
    entry = json.loads(out.read_text())["physical_axis"]
    assert entry["coordinate"] == "abs_delta_pt"
    assert entry["n"] == 30
    assert entry["spearman_afe_vs_coordinate"] > 0.99


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
