#!/usr/bin/env python3
"""Filesystem claims that let several array jobs drain one shard list.

Extracted from csh_shard_to_df.py, where this protocol was worked out the hard
way after a crashed worker's claim stranded a shard for a whole campaign. The
water campaign runs a CPU pool and a GPU pool against the same queue, so it
needs the same guarantees:

- exclusive-create so two workers never take the same shard,
- a liveness heartbeat so a slow worker is not mistaken for a dead one,
- stale recovery so a killed worker's shard is picked up again,
- atomic writes so a crash mid-save cannot leave a truncated result.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

#: Longer than any single frame is expected to take, shorter than the walltime,
#: so a task killed at the wall is reclaimed on the next pass. Only the fallback:
#: when the claim names a Slurm task, `acquire_claim` asks the scheduler instead.
DEFAULT_STALE_S = 14400.0


def claim_age_s(claim: Path) -> float | None:
    """Seconds since `claim` was last touched, or None if it does not exist."""
    try:
        return time.time() - claim.stat().st_mtime
    except FileNotFoundError:
        return None


def is_claim_stale(claim: Path, stale_s: float) -> bool:
    """True if `claim` exists and has not been refreshed for `stale_s` seconds."""
    age = claim_age_s(claim)
    return age is not None and age > stale_s


def slurm_task_id() -> str:
    """This task's Slurm id in the form `squeue -o %i` prints, or ``-``.

    Written into the claim so a reader can ask the scheduler whether the owner is
    still alive, which is a far sharper test than the claim's age: a task killed
    at the walltime leaves a claim that looks fresh for as long as the stale
    timeout, and on the 24 h large tier that would idle a shard for a whole day.
    """
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and task:
        return f"{array_job}_{task}"
    return os.environ.get("SLURM_JOB_ID", "-")


def claim_task_id(claim: Path) -> str | None:
    """The Slurm id recorded in `claim`, or None if it has none."""
    try:
        fields = claim.read_text().split()
    except OSError:
        return None
    if len(fields) >= 4 and fields[3] != "-":
        return fields[3]
    return None


def live_task_ids(timeout_s: float = 60.0) -> set[str] | None:
    """Slurm ids of this user's queued and running tasks, or None if unavailable.

    None rather than an empty set on failure, so a transient squeue error is not
    read as "every worker died" -- which would hand every shard in the campaign
    to whichever worker asked next, on top of the ones already running them.
    """
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except Exception:
        return None
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def owner_has_exited(owner: str, live_tasks: Callable[[], set[str] | None] | None = None) -> bool:
    """True only when the scheduler positively says task `owner` is gone.

    Sound without a grace period, because of the order the two facts are read: a
    task has to be running to have written the claim we just read, so it was in
    the queue before this snapshot was taken. Absent from the snapshot therefore
    means exited since, never "has not started yet". For the same reason the
    snapshot is taken fresh on every call rather than cached -- a cached one
    could predate the claim, and then absence would prove nothing.
    """
    live = (live_tasks or live_task_ids)()
    return live is not None and owner not in live


def touch_claim(claim: Path) -> None:
    """Refresh a claim's mtime so staleness measures liveness, not age."""
    try:
        os.utime(claim, None)
    except FileNotFoundError:
        pass


def acquire_claim(
    claim: Path,
    stale_s: float,
    tag: str = "",
    live_tasks: Callable[[], set[str] | None] | None = None,
) -> bool:
    """Take the exclusive-create claim on `claim`, recovering a stale one.

    Returns False (without raising) when another live worker holds it, so the
    caller can simply move on instead of treating contention as an error.

    A claim recorded by *this* Slurm task is always reclaimed, however fresh it
    looks. That is the preemption case: on the scavenger partition Slurm requeues
    a preempted task under the same id, and the SIGKILL means the previous
    incarnation never reached the `finally` that removes its claim. Without this
    the restarted task would find its own claim, decide the shard was busy, and
    exit -- and because the scheduler still lists that id as live, nothing else
    would touch the shard until the stale timeout. The test is unambiguous: the
    only process that can be running under this id right now is this one.

    A claim whose owner the scheduler no longer lists is reclaimed on the same
    reasoning, one task removed. This has to agree with campaign_status.py, which
    decides what to re-submit and has always asked the scheduler; while only this
    side aged claims out, the two halves of the protocol disagreed for up to
    `stale_s` after any cancellation, and the disagreement was a livelock rather
    than a delay. The watchdog saw twelve orphaned ood_size shards and submitted
    twelve tasks; each one read a claim left by an array cancelled ninety minutes
    earlier, called it live because it was younger than four hours, and exited
    after six seconds; half an hour later the watchdog counted the same twelve
    and did it again. Nothing in either log says anything is wrong -- the
    watchdog reports work dispatched, the workers report a shard already taken.
    """
    if claim.exists():
        age = claim_age_s(claim)
        owner = claim_task_id(claim)
        if age is not None and age > stale_s:
            print(f"claim stale ({age:.0f}s > {stale_s:.0f}s), reclaiming: {claim}")
            claim.unlink(missing_ok=True)
        elif owner == slurm_task_id() != "-":
            print(f"claim left by an earlier run of this task ({slurm_task_id()}), reclaiming: {claim}")
            claim.unlink(missing_ok=True)
        elif owner is not None and owner_has_exited(owner, live_tasks):
            print(f"claim owner {owner} is no longer queued or running, reclaiming: {claim}")
            claim.unlink(missing_ok=True)
        else:
            return False
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.uname().nodename} {os.getpid()} {tag} {slurm_task_id()}\n".encode())
        os.close(fd)
    except FileExistsError:
        return False
    return True


def atomic_savez_compressed(out: Path, **arrays) -> None:
    """Write `arrays` to `out` via a temp file plus `os.replace`."""
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def atomic_save_npy(out: Path, array) -> None:
    """Write one `.npy` via a temp file plus `os.replace`."""
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}.npy")
    try:
        np.save(tmp, array, allow_pickle=True)
        os.replace(tmp, out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(out: Path, payload: dict) -> None:
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        os.replace(tmp, out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
