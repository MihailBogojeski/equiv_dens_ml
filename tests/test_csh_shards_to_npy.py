"""Unit tests for the packaging completeness gate in csh_shards_to_npy.py.

No PySCF/`build_pair` calls here - only the gate/classification logic that
decides whether packaging should proceed, run, refuse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "revision"
sys.path.insert(0, str(SCRIPT_DIR))

from csh_shards_to_npy import (  # noqa: E402
    check_completeness,
    classify_failure,
    failure_size_stats,
    load_shards,
)


# --------------------------------------------------------------------------
# classify_failure
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error,expected",
    [
        ("KeyError: 17", "keyerror"),
        ("KeyError: 35", "keyerror"),
        ("n_elec drift -1.97e-03", "drift"),
        ("n_elec drift 5.0e-02", "drift"),
        ("ValueError: Z=17: table width 4 != 5", "other"),
        ("RuntimeError: disk full", "other"),
        ("", "other"),
    ],
)
def test_classify_failure(error, expected):
    assert classify_failure(error) == expected


# --------------------------------------------------------------------------
# failure_size_stats
# --------------------------------------------------------------------------


def test_failure_size_stats_joins_on_path():
    failures = [{"path": "/a", "error": "n_elec drift 1e-2"}, {"path": "/b", "error": "n_elec drift 1e-2"}]
    manifest_entries = [{"path": "/a", "natoms": 100}, {"path": "/b", "natoms": 140}, {"path": "/c", "natoms": 5}]
    stats = failure_size_stats(failures, manifest_entries)
    assert stats["n"] == 2
    assert stats["natoms_max"] == 140
    assert stats["natoms_mean"] == 120.0


def test_failure_size_stats_none_when_no_manifest():
    assert failure_size_stats([{"path": "/a"}], None) is None


def test_failure_size_stats_none_when_no_paths_match():
    assert failure_size_stats([{"path": "/z"}], [{"path": "/a", "natoms": 10}]) is None


# --------------------------------------------------------------------------
# check_completeness
# --------------------------------------------------------------------------


def _write_done(outdir: Path, shard: int) -> None:
    (outdir / f"shard_{shard:05d}.done").write_text(json.dumps({"shard": shard}))


def test_check_completeness_passes_when_fully_done_and_clean(tmp_path):
    for i in range(3):
        _write_done(tmp_path, i)
    manifest_entries = [{"path": f"/e{i}", "natoms": 10} for i in range(3 * 64 - 10)]  # ceil -> 3 shards
    warnings = check_completeness(tmp_path, manifest_entries, 64, [], allow_incomplete=False)
    assert warnings == []


def test_check_completeness_refuses_when_done_count_short(tmp_path):
    _write_done(tmp_path, 0)  # only 1 of 2 expected shards done
    manifest_entries = [{"path": f"/e{i}", "natoms": 10} for i in range(100)]  # ceil(100/64) = 2
    with pytest.raises(SystemExit, match="only 1/2 shards"):
        check_completeness(tmp_path, manifest_entries, 64, [], allow_incomplete=False)


def test_check_completeness_allow_incomplete_bypasses_refusal(tmp_path):
    _write_done(tmp_path, 0)
    manifest_entries = [{"path": f"/e{i}", "natoms": 10} for i in range(100)]
    warnings = check_completeness(tmp_path, manifest_entries, 64, [], allow_incomplete=True)
    assert any("only 1/2 shards" in w for w in warnings)


def test_check_completeness_refuses_on_in_flight_claim(tmp_path):
    _write_done(tmp_path, 0)
    (tmp_path / "shard_00001.claim").write_text("host 1 cpu\n")
    with pytest.raises(SystemExit, match="still claimed"):
        check_completeness(tmp_path, None, 64, [], allow_incomplete=False)


def test_check_completeness_refuses_on_residual_retryable_keyerror(tmp_path):
    _write_done(tmp_path, 0)
    failures = [{"path": "/x", "error": "KeyError: 17"}]
    with pytest.raises(SystemExit, match="KeyErrors that look retryable"):
        check_completeness(tmp_path, None, 64, failures, allow_incomplete=False)


def test_check_completeness_ignores_non_retryable_failures(tmp_path):
    _write_done(tmp_path, 0)
    failures = [{"path": "/x", "error": "n_elec drift 5.0e-02"}]
    warnings = check_completeness(tmp_path, None, 64, failures, allow_incomplete=False)
    assert warnings == []


def test_check_completeness_no_manifest_no_gate_on_done_count(tmp_path):
    """Without --manifest, the .done-count check is skipped (nothing to compare against)."""
    warnings = check_completeness(tmp_path, None, 64, [], allow_incomplete=False)
    assert warnings == []


# --------------------------------------------------------------------------
# load_shards closes its np.load handles (context manager)
# --------------------------------------------------------------------------


def test_load_shards_reads_records_and_failures(tmp_path):
    np.savez_compressed(
        tmp_path / "shard_00000.npz",
        records=np.array([{"path": "/ok"}], dtype=object),
        failures=np.array([{"path": "/bad", "error": "KeyError: 17"}], dtype=object),
    )
    records, failures = load_shards(tmp_path)
    assert len(records) == 1
    assert len(failures) == 1
