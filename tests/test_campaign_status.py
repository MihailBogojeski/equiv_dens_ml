"""Which shards the watchdog considers outstanding.

The consequences of getting this wrong are asymmetric and both bad: call a live
worker's shard outstanding and the campaign duplicates work, call a dead
worker's shard claimed and the shard silently never runs. The second is the one
that bit us -- a cancelled array left claims that a purely age-based check would
have honoured for a full day.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision"))

import campaign_status as cs  # noqa: E402


def make_shards(tmp_path: Path, n: int) -> tuple[Path, Path]:
    shard_dir = tmp_path / "shards"
    outdir = tmp_path / "out"
    shard_dir.mkdir()
    outdir.mkdir()
    for i in range(n):
        (shard_dir / f"shard_{i:04d}.json").write_text(json.dumps({"shard_id": i, "frames": []}))
    return shard_dir, outdir


def mark(outdir: Path, index: int, *, done=False, claim=None, partial=False) -> Path:
    d = outdir / f"shard_{index:04d}"
    d.mkdir(parents=True, exist_ok=True)
    if done:
        (d / "shard.done").touch()
    if partial:
        (d / "status.jsonl").write_text("{}\n")
    if claim is not None:
        (d / "shard.claim").write_text(f"node 123 theory/orca {claim}\n")
    return d


def test_array_spec_collapses_runs():
    assert cs.array_spec([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert cs.array_spec([4]) == "4"
    assert cs.array_spec([]) == ""


def test_done_shards_are_not_outstanding(tmp_path):
    shard_dir, outdir = make_shards(tmp_path, 4)
    mark(outdir, 0, done=True)
    mark(outdir, 2, done=True)
    info = cs.survey(shard_dir, outdir, cs.DEFAULT_STALE_S, live=set())
    assert info["n_done"] == 2
    assert info["outstanding"] == [1, 3]


def test_claim_owned_by_a_live_task_is_left_alone(tmp_path):
    shard_dir, outdir = make_shards(tmp_path, 2)
    mark(outdir, 0, claim="900_3")
    info = cs.survey(shard_dir, outdir, cs.DEFAULT_STALE_S, live={"900_3"})
    assert info["counts"]["claimed"] == 1
    assert info["outstanding"] == [1]


def test_claim_owned_by_a_dead_task_is_reclaimed_immediately(tmp_path):
    shard_dir, outdir = make_shards(tmp_path, 2)
    # Fresh enough that the age rule alone would call it live.
    mark(outdir, 0, claim="900_3")
    info = cs.survey(shard_dir, outdir, cs.DEFAULT_STALE_S, live={"901_0"})
    assert "claimed" not in info["counts"]
    assert info["outstanding"] == [0, 1]


def test_squeue_failure_does_not_resubmit_the_world(tmp_path):
    shard_dir, outdir = make_shards(tmp_path, 2)
    mark(outdir, 0, claim="900_3")
    info = cs.survey(shard_dir, outdir, cs.DEFAULT_STALE_S, live=None)
    assert info["counts"]["claimed"] == 1


def test_old_claim_without_a_task_id_still_ages_out(tmp_path):
    shard_dir, outdir = make_shards(tmp_path, 1)
    d = outdir / "shard_0000"
    d.mkdir(parents=True)
    (d / "shard.claim").write_text("node 123 theory/orca\n")
    import os

    old = time.time() - 10 * cs.DEFAULT_STALE_S
    os.utime(d / "shard.claim", (old, old))
    info = cs.survey(shard_dir, outdir, cs.DEFAULT_STALE_S, live=set())
    assert info["outstanding"] == [0]


def test_partial_shard_is_outstanding_but_distinguished(tmp_path):
    shard_dir, outdir = make_shards(tmp_path, 2)
    mark(outdir, 0, partial=True)
    info = cs.survey(shard_dir, outdir, cs.DEFAULT_STALE_S, live=set())
    assert info["counts"]["partial"] == 1
    assert info["outstanding"] == [0, 1]


def test_discover_skips_directories_without_shards(tmp_path):
    root = tmp_path / "shards"
    (root / "split_a" / "theory_x").mkdir(parents=True)
    (root / "split_a" / "theory_x" / "shard_0000.json").write_text("{}")
    (root / "split_b" / "theory_y").mkdir(parents=True)
    pairs = cs.discover(root, tmp_path / "out")
    assert [(p[0], p[1]) for p in pairs] == [("split_a", "theory_x")]


@pytest.mark.parametrize("write_id", [True, False])
def test_claim_task_id_roundtrip(tmp_path, monkeypatch, write_id):
    sys.path.insert(0, str(_REPO_ROOT / "scripts/revision"))
    import shard_claim

    if write_id:
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "555")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    else:
        monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    claim = tmp_path / "shard.claim"
    assert shard_claim.acquire_claim(claim, 3600.0, "theory/orca")
    assert shard_claim.claim_task_id(claim) == ("555_7" if write_id else None)
