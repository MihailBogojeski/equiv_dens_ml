"""Learning-curve JSON schema and checkpoint scoring helpers.

The trainer logs in-distribution validation MAE. The reviewer water tests
(ID, size-OOD, ice, droplets) have to be scored from kept checkpoints. Records
are {n_train or step, split, afe_mean, arm} so the two figures share a collector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ANALYSIS = _REPO_ROOT / "scripts" / "revision" / "analysis"
sys.path.insert(0, str(_ANALYSIS))

from learning_curve_records import (  # noqa: E402
    SCORE_STEPS,
    VALID_ARMS,
    VALID_SPLITS,
    collect_curve_records,
    ntrain_record,
    series_for_figure,
    step_record,
    validate_curve_record,
)
from score_learning_curve_ckpts import (  # noqa: E402
    checkpoint_step,
    find_checkpoint,
    prepare_eval_run_dir,
)


def test_ntrain_record_schema() -> None:
    rec = ntrain_record(n_train=50, split="water_id_test", afe_mean=0.012, arm="sad")
    validate_curve_record(rec, kind="ntrain")
    assert rec["n_train"] == 50
    assert rec["split"] == "water_id_test"
    assert rec["afe_mean"] == pytest.approx(0.012)
    assert rec["arm"] == "sad"


def test_step_record_schema() -> None:
    rec = step_record(step=5000, split="water_ood_size", afe_mean=0.04, arm="direct")
    validate_curve_record(rec, kind="step")
    assert rec["step"] == 5000
    assert "n_train" not in rec


def test_invalid_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="split"):
        ntrain_record(50, "train", 0.1, "sad")


def test_score_steps_are_log_spaced() -> None:
    assert SCORE_STEPS == (5000, 10000, 25000, 50000, 75000, 100000)
    assert VALID_SPLITS == (
        "water_id_test",
        "water_ood_size",
        "water_ood_order",
        "water_ood_density",
    )
    assert VALID_ARMS == ("sad", "direct")


def test_collect_curve_records_from_eval_jsons(tmp_path: Path) -> None:
    rows = []
    for n_train, afe in ((50, 0.08), (125, 0.05)):
        rec = ntrain_record(n_train, "water_id_test", afe, "sad")
        path = tmp_path / f"n{n_train}_id.json"
        path.write_text(json.dumps({"summary": {"afe_mean": afe}, "curve": rec}))
        rows.append(path)
    collected = collect_curve_records(rows, kind="ntrain")
    assert [r["n_train"] for r in collected["records"]] == [50, 125]
    assert collected["records"][0]["afe_mean"] == pytest.approx(0.08)


def test_series_for_figure_groups_arm_and_split() -> None:
    records = [
        ntrain_record(50, "water_id_test", 0.08, "sad"),
        ntrain_record(125, "water_id_test", 0.05, "sad"),
        ntrain_record(50, "water_id_test", 0.11, "direct"),
        ntrain_record(50, "water_ood_size", 0.20, "sad"),
    ]
    series = series_for_figure(records, x_key="n_train")
    assert series[("sad", "water_id_test")] == [(50, 0.08), (125, 0.05)]
    assert series[("direct", "water_id_test")] == [(50, 0.11)]
    assert series[("sad", "water_ood_size")] == [(50, 0.20)]


def test_checkpoint_step_from_filename_and_payload(tmp_path: Path) -> None:
    import torch

    numbered = tmp_path / "checkpoint_0000075000_density.pth"
    numbered.write_bytes(b"x")
    assert checkpoint_step(numbered) == 75000
    latest = tmp_path / "latest_checkpoint.pth"
    torch.save({"ID": "abc", "step": 114000}, latest)
    assert checkpoint_step(latest) == 114000


def test_find_checkpoint_numbered_and_final(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    numbered = ckpt / "checkpoint_0000050000.pth"
    numbered.write_bytes(b"x")
    latest = ckpt / "latest_checkpoint.pth"
    latest.write_bytes(b"y")
    assert find_checkpoint(tmp_path, 50000) == numbered
    assert find_checkpoint(tmp_path, None) == latest


def test_find_checkpoint_accepts_density_phase_suffix(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    numbered = ckpt / "checkpoint_0000025000_density.pth"
    numbered.write_bytes(b"x")
    assert find_checkpoint(tmp_path, 25000) == numbered


def test_plot_ntrain_and_step_write_png(tmp_path: Path) -> None:
    from plot_response_letter_figures import (
        plot_ntrain_learning_curves,
        plot_ood_step_learning_curves,
    )

    ntrain = {
        "kind": "ntrain",
        "records": [
            ntrain_record(50, "water_id_test", 0.08, "sad"),
            ntrain_record(125, "water_id_test", 0.05, "sad"),
            ntrain_record(50, "water_ood_size", 0.20, "sad"),
            ntrain_record(50, "water_id_test", 0.11, "direct"),
        ],
    }
    step = {
        "kind": "step",
        "records": [
            step_record(5000, "water_id_test", 0.09, "sad"),
            step_record(10000, "water_id_test", 0.04, "sad"),
            step_record(5000, "water_ood_density", 0.15, "sad"),
        ],
    }
    n_json = tmp_path / "ntrain.json"
    s_json = tmp_path / "step.json"
    n_json.write_text(json.dumps(ntrain))
    s_json.write_text(json.dumps(step))
    n_pdf = tmp_path / "ntrain.pdf"
    s_pdf = tmp_path / "step.pdf"
    assert plot_ntrain_learning_curves(n_json, n_pdf) == n_pdf
    assert plot_ood_step_learning_curves(s_json, s_pdf) == s_pdf
    assert n_pdf.is_file() and n_pdf.stat().st_size > 0
    assert s_pdf.is_file() and s_pdf.stat().st_size > 0


def test_plot_learning_curves_skip_missing(tmp_path: Path) -> None:
    from plot_response_letter_figures import plot_ntrain_learning_curves

    assert plot_ntrain_learning_curves(tmp_path / "missing.json", tmp_path / "out.pdf") is None


def test_prepare_eval_run_dir_symlinks_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "args.txt").write_text("--cutoff=6.0\n")
    src = run / "checkpoints" / "checkpoint_0000005000.pth"
    src.write_bytes(b"weights")
    dest = tmp_path / "eval_wrap"
    wrap = prepare_eval_run_dir(run, src, dest)
    latest = wrap / "checkpoints" / "latest_checkpoint.pth"
    assert latest.is_symlink() or latest.is_file()
    assert latest.resolve() == src.resolve()
    assert (wrap / "args.txt").read_text() == "--cutoff=6.0\n"
    # load_model looks for best_{ID}.pth at the run root; the wrapper must
    # expose the same weights there so evaluation does not silently load
    # a later best_* from the original run.
    bests = list(wrap.glob("best_*.pth"))
    assert len(bests) == 1
    assert bests[0].resolve() == src.resolve()
