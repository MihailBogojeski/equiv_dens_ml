"""Campaign density-completion detection from train logs."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "water"))

from density_log_finished import density_training_complete


def test_sad_lr_stop_counts_as_finished(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "restore before training density False\n"
        "000000/300000 epoch:      0\n"
        "  density:\n"
        "    train density_perc_mae:   1.000    valid 0 mae:   1.100    valid 0 loss:   0.100\n"
        "000100/300000 epoch:      1\n"
        "  density:\n"
        "    train density_perc_mae:   0.500    valid 0 mae:   0.550    valid 0 loss:   0.050\n"
        "Learning rate is smaller than 1e-05! Training stopped.\n"
        "finished trainer run of phase density\n"
    )
    assert density_training_complete(log, "water_pbe_d4_avdz")


def test_wave1_unfinished(tmp_path: Path):
    log = tmp_path / "train.log"
    log.write_text(
        "restore before training df False\n"
        "finished trainer run of phase df\n"
    )
    assert not density_training_complete(log, "water_n50_pbe_d4_avdz")
