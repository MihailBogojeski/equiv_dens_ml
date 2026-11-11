"""Parse density learning-curve points from a campaign train log."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

from train_log_curve import parse_density_curve, parse_density_curve_result


def test_parse_density_curve_drops_sad_fine_tune_tail(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "restore before training density False",
                "000000/300000 epoch:      0",
                "  density:",
                "    train loss:   1.0    train density_perc_mae:   1.000    valid 0 mae:   1.100    valid 0 loss:   0.100",
                "000100/300000 epoch:      1",
                "  density:",
                "    train loss:   0.5    train density_perc_mae:   0.500    valid 0 mae:   0.550    valid 0 loss:   0.050",
                "Learning rate is smaller than 1e-05! Training stopped.",
                "finished trainer run of phase density",
                "restore before training density_fine_tuning False",
                "000000/300000 epoch:      0",
                "  density:",
                "    train loss:   1.8    train density_perc_mae:   1.800    valid 0 mae:  17.900    valid 0 loss:   1.800",
                "Nans found in predicted density",
            ]
        )
        + "\n"
    )
    result = parse_density_curve_result(log)
    assert [p["step"] for p in result.points] == [0, 100]
    assert result.stop_reason == "lr"
    assert result.points[1]["valid_perc_mae"] == 0.050


def test_parse_density_curve_stitches_resume_fragments(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "restore before training density False",
                "000000/300000 epoch:      0",
                "  density:",
                "    train density_perc_mae:   0.900    valid 0 mae:   0.950    valid 0 loss:   0.090",
                "010000/300000 epoch:     50",
                "  density:",
                "    train density_perc_mae:   0.800    valid 0 mae:   0.850    valid 0 loss:   0.080",
                "restore before training density False",
                "009500/300000 epoch:     48",
                "  density:",
                "    train density_perc_mae:   0.810    valid 0 mae:   0.860    valid 0 loss:   0.081",
                "020000/300000 epoch:    100",
                "  density:",
                "    train density_perc_mae:   0.700    valid 0 mae:   0.750    valid 0 loss:   0.070",
            ]
        )
        + "\n"
    )
    result = parse_density_curve_result(log)
    steps = [p["step"] for p in result.points]
    assert steps == [0, 9500, 10000, 20000]
    by_step = {p["step"]: p for p in result.points}
    assert by_step[9500]["valid_perc_mae"] == 0.081
    assert result.stop_reason == "unfinished"


def test_parse_density_curve_ignores_watchdog_loop_after_lr_stop(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
                [
                    "restore before training density False",
                    "000000/300000 epoch:      0",
                    "  density:",
                    "    train density_perc_mae:   0.600    valid 0 mae:   0.650    valid 0 loss:   0.060",
                    "000100/300000 epoch:      1",
                    "  density:",
                    "    train density_perc_mae:   0.500    valid 0 mae:   0.550    valid 0 loss:   0.050",
                    "Learning rate is smaller than 1e-05! Training stopped.",
                "finished trainer run of phase density",
                "restore before training density False",
                "000200/300000 epoch:      2",
                "  density:",
                "    train density_perc_mae:   0.400    valid 0 mae:   0.450    valid 0 loss:   0.040",
                "Learning rate is smaller than 1e-05! Training stopped.",
                "finished trainer run of phase density",
            ]
        )
        + "\n"
    )
    result = parse_density_curve_result(log)
    assert [p["step"] for p in result.points] == [0, 100]
    assert result.stop_reason == "lr"


def test_parse_density_curve_backward_compat_list(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "restore before training density False\n"
        "000000/300000 epoch:      0\n"
        "  density:\n"
        "    train density_perc_mae:   1.000    valid 0 mae:   1.100    valid 0 loss:   0.100\n"
        "000100/300000 epoch:      1\n"
        "  density:\n"
        "    train density_perc_mae:   0.500    valid 0 mae:   0.550    valid 0 loss:   0.050\n"
    )
    curve = parse_density_curve(log)
    assert len(curve) == 2
    assert curve[1]["valid_perc_mae"] == 0.050
