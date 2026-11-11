"""md40x launch/assemble/geometry scripts point at new trees, not the 3k campaign."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_WATER = _REPO / "scripts" / "revision" / "water"


def test_launch_md40x_waves_uses_md40x_trees(implementation_text) -> None:
    text = implementation_text(_WATER / "launch_md40x_waves.sh")
    assert "water_clusters_md40x" in text
    assert "shards_md40x" in text
    assert "water_orca_md40x" in text
    assert "wb97mv_def2tzvpd" in text
    assert "pbe_d4_avdz" in text
    assert "hybrid10" not in text
    assert "select_stratified_subset" not in text
    assert "shards_md40x" in text


def test_geometry_expansion_has_md40x_stages(implementation_text) -> None:
    text = implementation_text(_WATER / "run_geometry_expansion.sh")
    assert "phase_md40x" in text
    assert "phase_b_md40x" in text
    assert "--n-keep 40" in text
    assert "--stride 80" in text
    assert "water_clusters_md40x" in text


def test_thicken_default_output_is_still_md10x(implementation_text) -> None:
    text = implementation_text(_WATER / "thicken_with_md.py")
    assert "water_clusters_md10x" in text
    assert '"--n-keep", type=int, default=10' in text


def test_assemble_supports_md40x_env_and_dipole_gate(implementation_text) -> None:
    text = implementation_text(_WATER / "assemble_water_labels.sh")
    assert "REQUIRE_DIPOLE" in text
    assert "--require-dipole" in text
    assert "OUT_ROOT=" in text
    assert "water_ood_rackers" in text
