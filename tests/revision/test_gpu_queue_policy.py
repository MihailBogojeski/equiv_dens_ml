"""Regression tests for the project-level GPU partition policy."""

from __future__ import annotations

import re
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts" / "revision"
_POLICY = _SCRIPTS / "lib" / "gpu_queue_policy.sh"
_LIVE_PARTITION = re.compile(r"^\s*(?:#SBATCH\s+)?--partition[=\s]+(?P<value>\S+)")


def test_equiv_gpu_submit_paths_never_request_l40s_public(implementation_text) -> None:
    """No active revision submit directive may compete with TPS on L40S."""
    offenders: list[str] = []
    for pattern in ("*.sh", "*.sbatch"):
        for path in _SCRIPTS.rglob(pattern):
            for line_number, line in enumerate(implementation_text(path).splitlines(), start=1):
                match = _LIVE_PARTITION.match(line)
                if match and "l40s_public" in match.group("value"):
                    offenders.append(f"{path.relative_to(_REPO)}:{line_number}")

    assert offenders == [], "l40s_public is reserved for tps-cofolding: " + ", ".join(offenders)


def test_12mer_timing_sbatch_cds_via_slurm_submit_dir(implementation_text) -> None:
    """Slurm copies the script; dirname $0 is the spool, not the repo."""
    text = implementation_text(_SCRIPTS / "submit_12mer_timing.sbatch")
    assert 'cd "${SLURM_SUBMIT_DIR:-$PWD}"' in text
    assert 'dirname "$0"' not in text
    assert "--densnet-np-dataset" in text
    assert "thiophene" in text


def test_shared_gpu_policy_targets_a100_then_h200(implementation_text) -> None:
    """The shared policy keeps equiv_dens_ml off the L40S partition."""
    text = implementation_text(_POLICY)
    assert "EQUIV_GPU_PARTITIONS=a100_chemistry,h200_public" in text
    assert "EQUIV_GPU_QOS=gpu48" in text


def test_energy_sbatch_is_one_gpu_with_mps_and_sixteen_cpus(implementation_text) -> None:
    """Two concurrent energy trainers need MPS and 2 x (1 + 4 workers) cores."""
    text = implementation_text(_SCRIPTS / "submit_water_train_energy.sbatch")
    assert "--gres=gpu:1" in text
    assert "--cpus-per-task=16" in text
    assert 'gpu_mps=yes' in text
    assert "ENERGY_PACK" in text
    assert "run_energy_pack.py" in text
    assert "l40s_public" not in text


def test_watchdogs_submit_one_energy_pack_not_n_tre_jobs(implementation_text) -> None:
    """Ready energy specs share one tre-pack job; density stays per-run."""
    for rel in (
        "watch_water_campaign.sh",
        "water/watch_water_campaign.sh",
    ):
        text = implementation_text(_SCRIPTS / rel)
        assert "--job-name=tre-pack" in text or '--job-name="$pack_job"' in text
        assert "ENERGY_PACK=" in text
        assert "submitting energy stage" not in text
        assert "submit_water_train_energy.sbatch" in text
        assert "l40s_public" not in text
