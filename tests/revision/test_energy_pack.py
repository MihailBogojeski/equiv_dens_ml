"""Concurrent energy-stage packer: two trainings share one GPU allocation.

A single energy job on ga* was SIGTERM-cancelled after ~2.5 h because
collate ran on the trainer (num_workers=0) and SM util stayed below 50%.
Packing two small energy heads on one device fills those host-bound gaps.
Sequential A-then-B would not raise util and is not what this tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_REVISION = _REPO / "scripts" / "revision"
_LIB = _REVISION / "lib"
sys.path.insert(0, str(_REVISION))
sys.path.insert(0, str(_LIB))

from energy_pack import run_pack  # noqa: E402
from energy_train_spec import (  # noqa: E402
    build_dry_run_argv,
    campaign_name,
    config_stack,
    parse_spec,
    should_submit_pack,
    spec_is_done,
)


class _AfterPolls:
    def __init__(self, code: int, n: int) -> None:
        self.code = code
        self.n = n
        self.i = 0

    def poll(self) -> int | None:
        self.i += 1
        return self.code if self.i >= self.n else None


def test_parse_spec_run_theory_and_optional_seed() -> None:
    assert parse_spec("cutoff_5:pbe_d4_avdz") == ("cutoff_5", "pbe_d4_avdz", None)
    assert parse_spec("water:wb97mv_def2tzvpd:7") == ("water", "wb97mv_def2tzvpd", "7")


def test_campaign_name_matches_sbatch() -> None:
    assert campaign_name(*parse_spec("water:pbe_d4_avdz")) == "water_pbe_d4_avdz"
    assert campaign_name(*parse_spec("cutoff_5:pbe_d4_avdz")) == "water_pbe_orca_cutoff_5"
    assert campaign_name(*parse_spec("water:wb97mv_def2tzvpd:7")) == (
        "water_wb97mv_def2tzvpd_seed7"
    )
    assert campaign_name(*parse_spec("malonaldehyde:wb97mv_def2tzvpd")) == (
        "malonaldehyde_wb97mv_def2tzvpd"
    )


def test_config_stack_is_energy_not_density() -> None:
    stack = config_stack(*parse_spec("cutoff_5:pbe_d4_avdz"))
    assert "@config/training/water_energy_base.txt" in stack
    assert "@config/training/water_energy_pbe.txt" in stack
    assert "@config/training/water_clusters_cutoff_5.txt" in stack
    assert not any("water_multiphase_overlay" in part for part in stack)


def test_skip_specs_with_training_done(tmp_path: Path) -> None:
    done = tmp_path / "results" / "revision" / "water_pbe_d4_avdz"
    done.mkdir(parents=True)
    (done / ".training_done").touch()
    assert spec_is_done(tmp_path, "water:pbe_d4_avdz")
    assert not spec_is_done(tmp_path, "cutoff_5:pbe_d4_avdz")


def test_packer_never_exceeds_slots() -> None:
    live: list[str] = []
    max_live = 0
    started: list[str] = []
    polls: dict[str, int] = {}

    def launch(spec: str) -> object:
        nonlocal max_live
        started.append(spec)
        live.append(spec)
        polls[spec] = 0
        max_live = max(max_live, len(live))

        class _Handle:
            def poll(self) -> int | None:
                nonlocal max_live
                max_live = max(max_live, len(live))
                polls[spec] += 1
                if polls[spec] >= 2:
                    if spec in live:
                        live.remove(spec)
                    return 0
                return None

        return _Handle()

    results = run_pack(
        ["a:t", "b:t", "c:t"],
        slots=2,
        is_done=lambda _s: False,
        launch=launch,
        poll_interval=0.0,
    )
    assert max_live == 2
    assert {spec for spec, _code in results} == {"a:t", "b:t", "c:t"}
    assert started == ["a:t", "b:t", "c:t"]


def test_packer_starts_next_when_a_slot_exits_zero() -> None:
    order: list[str] = []

    def launch(spec: str) -> _AfterPolls:
        order.append(f"start:{spec}")
        return _AfterPolls(0, n=1)

    results = run_pack(
        ["a:t", "b:t", "c:t"],
        slots=2,
        is_done=lambda _s: False,
        launch=launch,
        poll_interval=0.0,
    )
    assert results == [("a:t", 0), ("b:t", 0), ("c:t", 0)]
    assert order[0] == "start:a:t"
    assert order[1] == "start:b:t"
    assert "start:c:t" in order


def test_nonzero_child_does_not_abort_siblings() -> None:
    codes = {"ok:t": 0, "bad:t": 3, "also:t": 0}

    def launch(spec: str) -> _AfterPolls:
        return _AfterPolls(codes[spec], n=1)

    results = run_pack(
        ["ok:t", "bad:t", "also:t"],
        slots=2,
        is_done=lambda _s: False,
        launch=launch,
        poll_interval=0.0,
    )
    by_spec = dict(results)
    assert by_spec["ok:t"] == 0
    assert by_spec["bad:t"] == 3
    assert by_spec["also:t"] == 0


def test_packer_skips_done_specs() -> None:
    started: list[str] = []

    def launch(spec: str) -> _AfterPolls:
        started.append(spec)
        return _AfterPolls(0, n=1)

    results = run_pack(
        ["done:t", "live:t"],
        slots=2,
        is_done=lambda spec: spec == "done:t",
        launch=launch,
        poll_interval=0.0,
    )
    assert started == ["live:t"]
    assert results == [("live:t", 0)]


def test_cli_dry_run_does_not_write_training_done(tmp_path: Path) -> None:
    """A dry-run argv print must not plant .training_done on a real campaign."""
    import run_energy_pack as cli

    dest = tmp_path / "results" / "revision" / "water_pbe_d4_avdz"
    dest.mkdir(parents=True)
    assert cli.main(["--dry-run", "--repo", str(tmp_path), "--specs", "water:pbe_d4_avdz"]) == 0
    assert not (dest / ".training_done").exists()


def test_dry_run_does_not_call_sbatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import energy_pack as pack_mod

    def _fail(*_a, **_k):
        raise AssertionError("dry-run must not launch sbatch or train")

    monkeypatch.setattr(pack_mod.subprocess, "Popen", _fail)
    monkeypatch.setattr(pack_mod.subprocess, "call", _fail)
    monkeypatch.setattr(pack_mod.subprocess, "run", _fail)

    recorded: list[list[str]] = []

    def launch(spec: str) -> _AfterPolls:
        recorded.append(build_dry_run_argv(spec, repo=_REPO))
        return _AfterPolls(0, n=1)

    run_pack(
        ["water:pbe_d4_avdz"],
        slots=1,
        is_done=lambda _s: False,
        launch=launch,
        poll_interval=0.0,
    )
    assert recorded
    flat = " ".join(recorded[0])
    assert "sbatch" not in flat
    assert "run.py" in flat
    assert "--mode" in flat and "energy" in flat


def test_should_submit_pack_when_ready_and_none_queued() -> None:
    assert should_submit_pack(
        ["water:pbe_d4_avdz", "cutoff_5:pbe_d4_avdz"],
        queued_job_names=["tr-water-pbe_d4_avdz"],
    )
    assert should_submit_pack(["water:pbe_d4_avdz"], queued_job_names=[])


def test_should_not_submit_pack_if_tre_already_queued() -> None:
    assert not should_submit_pack(
        ["water:pbe_d4_avdz", "cutoff_5:pbe_d4_avdz"],
        queued_job_names=["tre-pack"],
    )
    assert not should_submit_pack(
        ["water:pbe_d4_avdz", "cutoff_5:pbe_d4_avdz"],
        queued_job_names=["tre-cutoff_5-pbe_d4_avdz"],
    )


def test_should_not_submit_pack_when_none_ready() -> None:
    assert not should_submit_pack([], queued_job_names=[])


def test_next_energy_pack_keeps_water_ahead_of_malonaldehyde() -> None:
    """One tre-pack at a time. Water resumes first; malonaldehyde energy waits."""
    from energy_train_spec import next_energy_pack

    water = ["water:pbe_d4_avdz", "water:wb97mv_def2tzvpd"]
    malon = ["malonaldehyde:pbe_d4_avdz", "malonaldehyde:wb97mv_def2tzvpd"]
    assert next_energy_pack([], [], malon_density_ready=True) == water
    assert next_energy_pack([], ["tre-pack"], malon_density_ready=True) == []
    assert (
        next_energy_pack(water, [], malon_density_ready=False) == []
    )
    assert next_energy_pack(water, [], malon_density_ready=True) == malon
    assert next_energy_pack(water + malon, [], malon_density_ready=True) == []
    # A live water pack blocks malonaldehyde even after the water parents are done.
    assert next_energy_pack(water, ["tre-pack"], malon_density_ready=True) == []


@pytest.mark.parametrize("theory", ["pbe_d4_avdz", "wb97mv_def2tzvpd"])
def test_malon_energy_keeps_malon_split_sizes(theory):
    values = {}
    for token in config_stack("malonaldehyde", theory):
        lines = (_REPO / token[1:]).read_text().splitlines() if token.startswith("@") else [token]
        for line in lines:
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    assert [values["--num_" + key] for key in ("train", "valid", "test")] == ["400", "80", "125"]
    assert values["--energy_weight"] == "1.0"
    assert "malonaldehyde_train" in values["--np_dataset"]
