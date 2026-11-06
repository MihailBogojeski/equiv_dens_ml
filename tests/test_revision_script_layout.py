"""Revision campaign scripts live in themed folders; old entry paths still work.

The peer-review tree grew into a flat dump of ~95 files. The layout split is
only useful if (1) shared modules import from ``scripts/revision/lib`` and
(2) the campaign commands already in the watchdog and the JACS todo still
resolve at their original ``scripts/revision/<name>`` paths.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REVISION = _REPO_ROOT / "scripts" / "revision"
_LIB = _REVISION / "lib"


def test_lib_paths_finds_repo_root() -> None:
    sys.path.insert(0, str(_LIB))
    import paths

    assert paths.repo_root() == _REPO_ROOT
    assert paths.revision_lib() == _LIB


def test_shared_modules_import_from_lib() -> None:
    sys.path.insert(0, str(_LIB))
    import shard_claim
    import theory_levels

    assert theory_levels.DEFAULT_LEVEL
    assert callable(shard_claim.acquire_claim)


@pytest.mark.parametrize(
    "relpath",
    [
        "lib/theory_levels.py",
        "lib/shard_claim.py",
        "lib/qm7x_orca_worker.py",
        "lib/qm7x_orca_common.py",
        "lib/paths.py",
        "water/campaign_status.py",
        "water/launch_water_campaign.sh",
        "water/thicken_with_md.py",
        "water/generate_new_motif_parents.py",
        "water/select_stratified_subset.py",
        "water/make_learning_curve_splits.py",
        "analysis/learning_curve_records.py",
        "analysis/score_learning_curve_ckpts.py",
        "water/launch_md_expansion_waves.sh",
        "water/run_geometry_expansion.sh",
        "water/watch_water_campaign.sh",
        "hbonds/generate_all.py",
        "hbonds/overlap_gates.py",
        "hbonds/launch_hbond_campaign.sh",
        "hbonds/launch_wave1_orca.sh",
        "hbonds/assemble_hbond_labels.sh",
        "qm7x/qm7x_assemble_npy.py",
        "csh/csh_evaluate.py",
        "analysis/ood_overlap_report.py",
        "analysis/error_vs_distance.py",
        "analysis/run_ood_analysis.sh",
        "analysis/launch_ood_geometry_dft.sh",
        "thiophene/thiophene_collective_variables.py",
        "thiophene/minima_hopping.py",
        "thiophene/minima_hopping_densnet.py",
        "thiophene/analyze_train_cv_density.py",
        "thiophene/analyze_mh_minima.py",
        "thiophene/validate_mh_minima_dft.py",
        "thiophene/plot_thiophene_mh_figures.py",
    ],
)
def test_themed_files_exist(relpath: str) -> None:
    path = _REVISION / relpath
    assert path.is_file(), f"missing {path}"


@pytest.mark.parametrize(
    "name",
    [
        "launch_water_campaign.sh",
        "watch_water_campaign.sh",
        "submit_water_watchdog.sbatch",
        "run_ood_analysis.sh",
        "campaign_status.py",
        "submit_water_train.sbatch",
        "submit_water_train_energy.sbatch",
        "submit_retrain_campaign.sh",
        "submit_learning_curves.sh",
        "submit_score_curve_ckpts.sbatch",
        "submit_score_learning_curves.sh",
        "collect_learning_curves.sh",
        "submit_water_orca.sbatch",
        "submit_ood_analysis.sbatch",
        "submit_ir_md.sbatch",
        "submit_qm7x_orca.sh",
        "submit_csh_cpu.sbatch",
        "launch_ood_geometry_dft.sh",
    ],
)
def test_stable_entry_paths_exist(name: str) -> None:
    path = _REVISION / name
    assert path.is_file(), f"missing stable entry {path}"
    if path.suffix in {".sh", ".py"}:
        assert path.stat().st_mode & 0o111, f"{path} is not executable"


def test_submit_learning_curves_dry_run_skips_full_n() -> None:
    import os

    result = subprocess.run(
        ["bash", str(_REVISION / "submit_learning_curves.sh")],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "DRY_RUN": "1"},
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout + result.stderr
    assert "water_n50" in out
    assert "water_direct_n50" in out
    assert "water_curve" in out
    assert "water_direct_curve" in out
    assert "water_n1250" not in out
    assert "water_direct_n1250" not in out


@pytest.mark.parametrize(
    ("script", "argv"),
    [
        ("water/campaign_status.py", ["--help"]),
        ("analysis/ood_overlap_report.py", ["--help"]),
        ("campaign_status.py", ["--help"]),
    ],
)
def test_cli_help(script: str, argv: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, str(_REVISION / script), *argv],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in (result.stdout + result.stderr).lower()
