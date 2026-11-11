"""Tests for validate_training_recipe.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts/revision/analysis/validate_training_recipe.py"


def test_rejects_joint_style_args(tmp_path: Path):
    run_dir = tmp_path / "2026-08-25_test"
    run_dir.mkdir()
    (run_dir / "args.txt").write_text(
        "\n".join(
            [
                "--density_weight=1.0",
                "--energy_weight=0.01",
                "--forces_weight=0.99",
                "--density_fine_tuning=False",
            ]
        )
    )
    (run_dir.parent / ".density_training_done").touch()

    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "co-optimise" in proc.stderr


def test_accepts_energy_stage_args(tmp_path: Path):
    run_dir = tmp_path / "2026-08-25_energy"
    run_dir.mkdir()
    (run_dir / "args.txt").write_text(
        "\n".join(
            [
                "--density_weight=0.0",
                "--energy_weight=1.0",
                "--forces_weight=0.99",
            ]
        )
    )

    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
