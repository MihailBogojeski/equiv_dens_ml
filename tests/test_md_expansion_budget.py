"""Budget/yield gate for the md40x water/malon expansion.

The 100k campaign is a count and a core-hour estimate, not a hope. If
n_keep=40 times the current parents, after a realistic dedup keep-fraction,
lands outside 90k–120k, the operator should see that before GFN2 starts.
Hybrid n=6 cost must stay within a factor of two of the measured 608 s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WATER = _REPO_ROOT / "scripts" / "revision" / "water"
_LIB = _REPO_ROOT / "scripts" / "revision" / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_WATER))

from build_shards_from_xyz import COST_MODEL, predicted_cost_s  # noqa: E402
from plan_md40x_budget import (  # noqa: E402
    DEFAULT_KEEP_FRACTION,
    TARGET_BAND,
    estimate_budget,
    parent_counts_from_manifests,
)


def test_current_parents_times_40_land_in_90k_120k_band() -> None:
    counts = parent_counts_from_manifests(_REPO_ROOT)
    total_parents = sum(counts.values())
    assert total_parents == 2967
    plan = estimate_budget(counts, n_keep=40, keep_fraction=0.85)
    assert TARGET_BAND[0] <= plan["n_expected_frames"] <= TARGET_BAND[1]
    assert plan["n_parents"] == 2967
    assert plan["n_keep"] == 40
    assert plan["keep_fraction"] == pytest.approx(0.85)
    assert DEFAULT_KEEP_FRACTION == pytest.approx(0.85)


def test_hybrid_n6_cost_within_factor_two_of_calibration() -> None:
    # Water hexamer: 18 atoms. Measured ωB97M-V wall time is 608 s on 8 cores.
    n_atoms = 18
    cost_s = predicted_cost_s(n_atoms, "wb97mv_def2tzvpd")
    assert cost_s == max(60.0, COST_MODEL["wb97mv_def2tzvpd"][0] * float(n_atoms) ** COST_MODEL["wb97mv_def2tzvpd"][1])
    assert cost_s / 608.0 < 2.0
    assert 608.0 / cost_s < 2.0
    plan = estimate_budget({"water_train": 1}, n_keep=1, keep_fraction=1.0, n_atoms_by_split={"water_train": 18})
    hybrid = plan["theories"]["wb97mv_def2tzvpd"]
    assert hybrid["frame_cost_s"]["water_train"] / 608.0 < 2.0


def test_estimate_budget_reports_core_hours_per_theory() -> None:
    counts = {"water_train": 100, "malon_train": 20}
    n_atoms = {"water_train": 9, "malon_train": 9}  # trimer / malon
    plan = estimate_budget(counts, n_keep=40, keep_fraction=0.85, n_atoms_by_split=n_atoms)
    assert plan["n_expected_frames"] == int(round(120 * 40 * 0.85))
    for theory in ("pbe_d4_avdz", "wb97mv_def2tzvpd"):
        assert plan["theories"][theory]["core_hours"] > 0.0
    assert plan["theories"]["wb97mv_def2tzvpd"]["core_hours"] > plan["theories"]["pbe_d4_avdz"]["core_hours"]


def test_budget_cli_writes_json(tmp_path: Path, capsys) -> None:
    from plan_md40x_budget import main

    out = tmp_path / "budget.json"
    rc = main(["--n-keep", "40", "--keep-fraction", "0.85", "--json-out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert TARGET_BAND[0] <= payload["n_expected_frames"] <= TARGET_BAND[1]
    printed = capsys.readouterr().out
    assert "expected frames" in printed.lower() or "n_expected_frames" in printed
