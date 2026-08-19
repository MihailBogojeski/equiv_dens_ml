"""Unit tests for the CSH shard retry / stale-claim logic (no live HDF5, no Slurm).

`process()` is heavy (builds a PySCF mol, solves the Fock equation, runs the
density fit), so every test here mocks it out and only exercises the
bookkeeping around it: which recorded failures are eligible for a retry, and
how the npz/.done files are rewritten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "revision"
sys.path.insert(0, str(SCRIPT_DIR))

import csh_shard_to_df as shard_mod  # noqa: E402
from csh_shard_to_df import (  # noqa: E402
    acquire_claim,
    atomic_savez_compressed,
    is_claim_stale,
    is_missing_element_failure,
    missing_element_z,
    retry_failed_shard,
    touch_claim,
)

TABLE = {1: "H", 6: "C", 17: "Cl", 35: "Br"}


def make_args(**overrides):
    args = MagicMock()
    args.shard = 0
    args.n_elec_tol = 1e-3
    args.s_thresh = 1e-5
    args.auxbasis = "augccpvqzjkfit"
    args.block = 40
    args.gpu_budget = 8_000_000_000
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def make_h5(tmp_path) -> Path:
    """A tiny, valid, but otherwise-empty HDF5 file (never actually read)."""
    path = tmp_path / "fake.h5"
    with h5py.File(path, "w"):
        pass
    return path


def write_npz(out: Path, records, failures) -> None:
    np.savez_compressed(
        out,
        records=np.array(records, dtype=object),
        failures=np.array(failures, dtype=object),
        allow_pickle=True,
    )


# --------------------------------------------------------------------------
# missing_element_z / is_missing_element_failure
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error,expected",
    [
        ("KeyError: 17", 17),
        ("KeyError: 35", 35),
        ("KeyError: 6", 6),
        ("n_elec drift -1.97e-03", None),
        ("ValueError: Z=17: table width 4 != 5", None),
        ("KeyError: not-an-int", None),
        ("", None),
    ],
)
def test_missing_element_z_parses_keyerror_integer(error, expected):
    assert missing_element_z(error) == expected


def test_is_missing_element_failure_requires_z_in_table():
    assert is_missing_element_failure("KeyError: 17", TABLE) is True
    assert is_missing_element_failure("KeyError: 99", TABLE) is False
    assert is_missing_element_failure("n_elec drift -1.97e-03", TABLE) is False


# --------------------------------------------------------------------------
# retry_failed_shard
# --------------------------------------------------------------------------


def test_retry_failed_shard_reprocesses_only_missing_element_failures(tmp_path, monkeypatch):
    out = tmp_path / "shard_00000.npz"
    done = tmp_path / "shard_00000.done"
    done.write_text(json.dumps({"shard": 0, "n_ok": 1, "n_failed": 3, "device": "cpu", "seconds": 5.0}))

    records = [{"path": "/kept/ok", "n_elec_err": 0.0}]
    failures = [
        {"path": "/ani2x/mol_cl", "error": "KeyError: 17"},
        {"path": "/omol/drift", "error": "n_elec drift -1.97e-03"},
        {"path": "/other/unknown_elem", "error": "KeyError: 99"},
    ]
    write_npz(out, records, failures)

    chunk = [
        {"path": "/ani2x/mol_cl", "charge": 0},
        {"path": "/omol/drift", "charge": 0},
        {"path": "/other/unknown_elem", "charge": 0},
        {"path": "/not/in/failures", "charge": 0},
    ]

    fake_rec = {"path": "/ani2x/mol_cl", "n_elec_err": 1e-6, "extra": "ok"}
    mock_process = MagicMock(return_value=fake_rec)
    monkeypatch.setattr(shard_mod, "process", mock_process)

    args = make_args(h5=str(make_h5(tmp_path)))
    retry_failed_shard(out, done, chunk, TABLE, args, "cpu")

    assert mock_process.call_count == 1
    called_entry = mock_process.call_args[0][0]
    assert called_entry["path"] == "/ani2x/mol_cl"

    saved = np.load(out, allow_pickle=True)
    saved_records = list(saved["records"])
    saved_failures = list(saved["failures"])

    assert records[0] in saved_records
    assert fake_rec in saved_records
    assert len(saved_records) == 2

    saved_failure_paths = {f["path"] for f in saved_failures}
    assert saved_failure_paths == {"/omol/drift", "/other/unknown_elem"}

    done_info = json.loads(done.read_text())
    assert done_info["n_ok"] == 2
    assert done_info["n_failed"] == 2
    assert done_info["retried"] == 1


def test_retry_failed_shard_drops_retry_that_still_fails(tmp_path, monkeypatch):
    out = tmp_path / "shard_00001.npz"
    done = tmp_path / "shard_00001.done"
    done.write_text(json.dumps({"shard": 1, "n_ok": 0, "n_failed": 1, "device": "cpu", "seconds": 1.0}))

    write_npz(out, [], [{"path": "/x/y", "error": "KeyError: 35"}])
    chunk = [{"path": "/x/y", "charge": -1}]

    mock_process = MagicMock(side_effect=RuntimeError("still broken"))
    monkeypatch.setattr(shard_mod, "process", mock_process)

    args = make_args(h5=str(make_h5(tmp_path)))
    retry_failed_shard(out, done, chunk, TABLE, args, "cpu")

    assert mock_process.call_count == 1
    saved = np.load(out, allow_pickle=True)
    assert len(list(saved["records"])) == 0
    saved_failures = list(saved["failures"])
    assert len(saved_failures) == 1
    assert saved_failures[0]["path"] == "/x/y"
    assert "RuntimeError" in saved_failures[0]["error"]


def test_retry_failed_shard_noop_when_nothing_retryable(tmp_path, monkeypatch):
    out = tmp_path / "shard_00002.npz"
    done = tmp_path / "shard_00002.done"
    done.write_text(json.dumps({"shard": 2, "n_ok": 0, "n_failed": 1, "device": "cpu", "seconds": 1.0}))

    write_npz(out, [], [{"path": "/z", "error": "n_elec drift 5.0e-02"}])
    chunk = [{"path": "/z", "charge": 0}]

    mock_process = MagicMock()
    monkeypatch.setattr(shard_mod, "process", mock_process)

    before_npz_bytes = out.read_bytes()
    before_done_text = done.read_text()

    args = make_args(h5=str(make_h5(tmp_path)))
    retry_failed_shard(out, done, chunk, TABLE, args, "cpu")

    mock_process.assert_not_called()
    assert out.read_bytes() == before_npz_bytes
    assert done.read_text() == before_done_text


def test_retry_failed_shard_ignores_failure_not_in_current_chunk(tmp_path, monkeypatch):
    """A recorded failure whose path is absent from the manifest chunk can't be retried."""
    out = tmp_path / "shard_00003.npz"
    done = tmp_path / "shard_00003.done"
    done.write_text(json.dumps({"shard": 3, "n_ok": 0, "n_failed": 1, "device": "cpu", "seconds": 1.0}))

    write_npz(out, [], [{"path": "/missing/from/chunk", "error": "KeyError: 17"}])
    chunk = [{"path": "/some/other/entry", "charge": 0}]

    mock_process = MagicMock()
    monkeypatch.setattr(shard_mod, "process", mock_process)

    args = make_args(h5=str(make_h5(tmp_path)))
    retry_failed_shard(out, done, chunk, TABLE, args, "cpu")

    mock_process.assert_not_called()
    saved = np.load(out, allow_pickle=True)
    assert len(list(saved["failures"])) == 1


# --------------------------------------------------------------------------
# is_claim_stale
# --------------------------------------------------------------------------


def test_is_claim_stale_false_when_no_claim(tmp_path):
    assert is_claim_stale(tmp_path / "nope.claim", 14400) is False


def test_is_claim_stale_false_when_fresh(tmp_path):
    claim = tmp_path / "shard_00060.claim"
    claim.write_text("host 123 cpu\n")
    assert is_claim_stale(claim, 14400) is False


def test_is_claim_stale_true_when_old(tmp_path):
    import os
    import time

    claim = tmp_path / "shard_00060.claim"
    claim.write_text("host 123 cpu\n")
    old = time.time() - 20000
    os.utime(claim, (old, old))
    assert is_claim_stale(claim, 14400) is True


# --------------------------------------------------------------------------
# acquire_claim / touch_claim
# --------------------------------------------------------------------------


def test_acquire_claim_creates_file_when_absent(tmp_path):
    claim = tmp_path / "shard_00000.claim"
    assert acquire_claim(claim, 14400, "cpu") is True
    assert claim.exists()
    assert "cpu" in claim.read_text()


def test_acquire_claim_refuses_live_claim_held_by_another_worker(tmp_path):
    """A fresh claim (like 15992619_60's live claim) must never be stolen."""
    claim = tmp_path / "shard_00060.claim"
    claim.write_text("cs711 123 cpu\n")
    assert acquire_claim(claim, 14400, "cpu") is False
    assert claim.read_text() == "cs711 123 cpu\n"


def test_acquire_claim_reclaims_stale_claim(tmp_path):
    import os
    import time

    claim = tmp_path / "shard_00061.claim"
    claim.write_text("dead-host 999 cpu\n")
    old = time.time() - 20000
    os.utime(claim, (old, old))
    assert acquire_claim(claim, 14400, "cpu") is True
    assert "dead-host" not in claim.read_text()


def test_acquire_claim_retakes_a_claim_left_by_this_same_task(tmp_path, monkeypatch):
    """The preemption case: a requeued task must be able to retake its own shard.

    A preempted task is killed outright, so the `finally` that removes its claim
    never runs, and Slurm requeues it under the same id. If the restarted task
    treated that claim as somebody else's it would exit immediately, and since
    the scheduler still lists the id as live nothing else would touch the shard
    either -- it would idle until the stale timeout, which on the large tier is
    a day.
    """
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "777")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    claim = tmp_path / "shard_00003.claim"
    claim.write_text("cs601 4242 wb97mv/orca 777_3\n")

    assert acquire_claim(claim, 14400, "wb97mv/orca") is True
    assert "4242" not in claim.read_text()


def test_acquire_claim_still_refuses_a_fresh_claim_from_a_different_task(tmp_path, monkeypatch):
    """The reclaim above keys on the Slurm id, so a neighbour's claim is untouched."""
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "777")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    claim = tmp_path / "shard_00004.claim"
    claim.write_text("cs602 4243 wb97mv/orca 777_4\n")

    assert acquire_claim(claim, 14400, "wb97mv/orca", lambda: {"777_3", "777_4"}) is False
    assert claim.read_text() == "cs602 4243 wb97mv/orca 777_4\n"


def test_acquire_claim_reclaims_from_an_owner_the_scheduler_has_forgotten(tmp_path):
    """The livelock: a cancelled array's claims must not hold shards for `stale_s`.

    campaign_status.py already resolved this the same way, so while acquire_claim
    aged claims out instead, the watchdog re-submitted these shards every pass and
    every worker refused them -- for four hours on the small tier and a day on the
    large one, with both logs reading as normal throughout.
    """
    claim = tmp_path / "shard_00006.claim"
    claim.write_text("cs602 4243 pbe_d4_avdz/orca 900_6\n")

    assert acquire_claim(claim, 14400, "pbe_d4_avdz/orca", lambda: {"901_0"}) is True
    assert "4243" not in claim.read_text()


def test_acquire_claim_keeps_a_fresh_claim_when_the_scheduler_cannot_be_reached(tmp_path):
    """A squeue that times out must not read as "every worker died"."""
    claim = tmp_path / "shard_00007.claim"
    claim.write_text("cs602 4243 pbe_d4_avdz/orca 900_7\n")

    assert acquire_claim(claim, 14400, "pbe_d4_avdz/orca", lambda: None) is False
    assert claim.read_text() == "cs602 4243 pbe_d4_avdz/orca 900_7\n"


def test_acquire_claim_does_not_ask_the_scheduler_about_a_claim_with_no_task_id(tmp_path):
    """Pre-Slurm-id claims still fall back to ageing out, with no squeue call."""
    calls = []

    def probe():
        calls.append(1)
        return set()

    claim = tmp_path / "shard_00008.claim"
    claim.write_text("laptop 111 cpu\n")

    assert acquire_claim(claim, 14400, "cpu", probe) is False
    assert calls == []


def test_acquire_claim_ages_out_before_asking_the_scheduler(tmp_path):
    """An old claim is reclaimed on age alone, so squeue is not on the common path."""
    import os
    import time

    calls = []

    def probe():
        calls.append(1)
        return set()

    claim = tmp_path / "shard_00009.claim"
    claim.write_text("cs602 4243 pbe_d4_avdz/orca 900_9\n")
    old = time.time() - 20000
    os.utime(claim, (old, old))

    assert acquire_claim(claim, 14400, "pbe_d4_avdz/orca", probe) is True
    assert calls == []


def test_acquire_claim_outside_slurm_does_not_treat_dashes_as_a_match(tmp_path, monkeypatch):
    """Two interactive runs both record `-`, which must not read as the same task."""
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    claim = tmp_path / "shard_00005.claim"
    claim.write_text("laptop 111 cpu -\n")

    assert acquire_claim(claim, 14400, "cpu") is False


def test_touch_claim_advances_mtime(tmp_path):
    import os
    import time

    claim = tmp_path / "shard_00000.claim"
    claim.write_text("host 1 cpu\n")
    old = time.time() - 5000
    os.utime(claim, (old, old))
    touch_claim(claim)
    assert (time.time() - claim.stat().st_mtime) < 5


def test_touch_claim_noop_when_claim_missing(tmp_path):
    touch_claim(tmp_path / "nope.claim")  # must not raise


# --------------------------------------------------------------------------
# atomic_savez_compressed
# --------------------------------------------------------------------------


def test_atomic_savez_compressed_writes_readable_npz(tmp_path):
    out = tmp_path / "shard_00000.npz"
    atomic_savez_compressed(out, records=np.array([1, 2, 3]), failures=np.array([]))
    assert out.exists()
    with np.load(out) as data:
        assert list(data["records"]) == [1, 2, 3]
    assert list(tmp_path.glob("*.tmp*")) == []


def test_atomic_savez_compressed_leaves_original_untouched_on_failure(tmp_path, monkeypatch):
    """A crash mid-write must not corrupt the only copy of successful records."""
    out = tmp_path / "shard_00000.npz"
    write_npz(out, [{"path": "/kept"}], [])
    before = out.read_bytes()

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(shard_mod.np, "savez_compressed", boom)
    with pytest.raises(RuntimeError):
        atomic_savez_compressed(out, records=np.array([]), failures=np.array([]))

    assert out.read_bytes() == before
    assert list(tmp_path.glob("*.tmp*")) == []


def test_atomic_savez_compressed_does_not_pollute_npz_with_allow_pickle_key(tmp_path):
    """np.savez_compressed(..., allow_pickle=True) would save a spurious
    'allow_pickle' array (kwargs become array names), so it must never be
    passed through."""
    out = tmp_path / "shard_00000.npz"
    atomic_savez_compressed(out, records=np.array([1]), failures=np.array([]))
    with np.load(out) as data:
        assert set(data.files) == {"records", "failures"}


# --------------------------------------------------------------------------
# retry_failed_shard + claim integration
# --------------------------------------------------------------------------


def test_retry_failed_shard_touches_claim_after_each_retry(tmp_path, monkeypatch):
    import os
    import time

    out = tmp_path / "shard_00000.npz"
    done = tmp_path / "shard_00000.done"
    done.write_text(json.dumps({"shard": 0, "n_ok": 0, "n_failed": 1, "device": "cpu", "seconds": 1.0}))
    write_npz(out, [], [{"path": "/x", "error": "KeyError: 17"}])
    chunk = [{"path": "/x", "charge": 0}]

    claim = tmp_path / "shard_00000.claim"
    claim.write_text("host 1 cpu\n")
    old = time.time() - 5000
    os.utime(claim, (old, old))

    monkeypatch.setattr(shard_mod, "process", MagicMock(return_value={"path": "/x", "n_elec_err": 0.0}))
    args = make_args(h5=str(make_h5(tmp_path)))
    retry_failed_shard(out, done, chunk, TABLE, args, "cpu", claim=claim)

    assert (time.time() - claim.stat().st_mtime) < 5


def test_retry_failed_shard_survives_missing_prior_done_key(tmp_path, monkeypatch):
    """`done` always exists by the time retry_failed_shard runs (checked by
    the caller); no `else {}` fallback should be needed to read it."""
    out = tmp_path / "shard_00000.npz"
    done = tmp_path / "shard_00000.done"
    done.write_text(json.dumps({"shard": 0, "n_ok": 0, "n_failed": 1, "device": "cpu", "seconds": 2.5}))
    write_npz(out, [], [{"path": "/x", "error": "KeyError: 17"}])
    chunk = [{"path": "/x", "charge": 0}]

    monkeypatch.setattr(shard_mod, "process", MagicMock(return_value={"path": "/x", "n_elec_err": 0.0}))
    args = make_args(h5=str(make_h5(tmp_path)))
    retry_failed_shard(out, done, chunk, TABLE, args, "cpu")

    updated = json.loads(done.read_text())
    assert updated["seconds"] >= 2.5
