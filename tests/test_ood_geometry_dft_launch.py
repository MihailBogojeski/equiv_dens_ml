"""OOD-geometry DFT launch stays on sibling trees and does not retrain 3k."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_LAUNCH = _REPO / "scripts" / "revision" / "analysis" / "launch_ood_geometry_dft.sh"
_STABLE = _REPO / "scripts" / "revision" / "launch_ood_geometry_dft.sh"
_SMALL = _REPO / "scripts" / "revision" / "submit_water_orca.sbatch"
_LARGE = _REPO / "scripts" / "revision" / "submit_water_orca_large.sbatch"

_MODE_SYSTEMS = (
    "water_n2",
    "water_n6",
    "ethanol",
    "ethanethiol",
    "resorcinol",
    "thiophene_2mer",
    "thiophene_12mer",
    "malonaldehyde",
)


def test_stable_entry_points_at_themed_launcher() -> None:
    assert _LAUNCH.is_file()
    assert _STABLE.is_file()
    text = _STABLE.read_text()
    assert "analysis/launch_ood_geometry_dft.sh" in text


def test_launcher_writes_sibling_trees_not_3k(implementation_text) -> None:
    text = implementation_text(_LAUNCH)
    assert "shards_ood_geom" in text
    assert "ood_geom_orca" in text
    assert "pbe_d4_avdz" in text
    assert "wb97mv_def2tzvpd" in text
    assert "datasets/revision/ood_scale" in text
    for name in _MODE_SYSTEMS:
        assert name in text
    # Default shard/out roots must not be the finished 3k campaign.
    assert "SHARD_ROOT=${SHARD_ROOT:-datasets/revision/shards_ood_geom}" in text
    assert "OUT_ROOT=${OUT_ROOT:-results/revision/ood_geom_orca}" in text
    assert "SHARD_ROOT=${SHARD_ROOT:-datasets/revision/shards}" not in text
    assert "OUT_ROOT=${OUT_ROOT:-results/revision/water_orca}" not in text
    assert "tr-water" not in text
    assert "submit_water_train" not in text


def test_twelve_mer_is_energy_only_large_tier(implementation_text) -> None:
    text = implementation_text(_LAUNCH)
    assert "thiophene_12mer" in text
    assert "NO_DF" in text
    assert "submit_water_orca_large.sbatch" in text


def test_orca_sbatch_forwards_no_df(implementation_text) -> None:
    for path in (_SMALL, _LARGE):
        text = implementation_text(path)
        assert "NO_DF" in text
        assert "--no-df" in text


def test_large_tier_keeps_gbw_and_asks_for_two_days(implementation_text) -> None:
    """A 12-mer SCF does not finish in 24 h; the next task must see the GBW."""
    text = implementation_text(_LARGE)
    assert "#SBATCH --time=48:00:00" in text
    assert "KEEP_ORCA" in text
    assert "--keep-orca" in text


def test_ood_orca_stays_on_cpu_partitions(implementation_text) -> None:
    """ORCA is CPU-only. cpu_short includes gl (L40S) and gr (RTX 6000)."""
    small = implementation_text(_SMALL)
    large = implementation_text(_LARGE)
    watch = implementation_text(_REPO / "scripts/revision/watch_ood_geometry_dft.sh")
    launch = implementation_text(_LAUNCH)
    assert "#SBATCH --partition=cs" in small
    assert "#SBATCH --partition=cs,cpu_short" not in small
    assert "#SBATCH --partition=cs" in large
    assert "partition=all" not in small
    assert "partition=all" not in large
    assert 'PARTITION="${PARTITION:-cs}"' in watch
    assert 'PARTITION_LARGE:-cs' in watch
    assert 'PARTITION="${PARTITION:-cs}"' in launch
    assert launch.count(" 1 cs") >= 1


def test_watchdog_uses_clean_exports_and_forwards_resume_flags(implementation_text) -> None:
    """Chained watchdogs must not grow their environment until ARG_MAX fails."""
    submit = implementation_text(_REPO / "scripts/revision/submit_ood_geometry_watch.sbatch")
    resubmit = implementation_text(_REPO / "scripts/revision/water/watch_and_resubmit.sh")
    assert "--export=NONE" in submit
    for name in ("KEEP_ORCA", "RETRY_ERRORS", "NO_DF"):
        assert f'{name}="${{{name}:-' in resubmit


def test_watchdog_drains_all_unfinished_ood_mode_splits(implementation_text) -> None:
    """Medium OOD systems must not be stranded outside the retry loop."""
    watch = implementation_text(_REPO / "scripts/revision/watch_ood_geometry_dft.sh")
    for split in (
        "ood_modes_water_n6",
        "ood_modes_resorcinol",
        "ood_modes_malonaldehyde",
        "ood_modes_thiophene_2mer",
        "ood_modes_thiophene_12mer",
    ):
        assert split in watch
