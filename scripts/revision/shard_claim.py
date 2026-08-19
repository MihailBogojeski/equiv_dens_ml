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
import time
from pathlib import Path

import numpy as np

#: Longer than any single frame is expected to take, shorter than the walltime,
#: so a task killed at the wall is reclaimed on the next pass.
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


def touch_claim(claim: Path) -> None:
    """Refresh a claim's mtime so staleness measures liveness, not age."""
    try:
        os.utime(claim, None)
    except FileNotFoundError:
        pass


def acquire_claim(claim: Path, stale_s: float, tag: str = "") -> bool:
    """Take the exclusive-create claim on `claim`, recovering a stale one.

    Returns False (without raising) when another live worker holds it, so the
    caller can simply move on instead of treating contention as an error.
    """
    if claim.exists():
        age = claim_age_s(claim)
        if age is not None and age > stale_s:
            print(f"claim stale ({age:.0f}s > {stale_s:.0f}s), reclaiming: {claim}")
            claim.unlink(missing_ok=True)
        else:
            return False
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.uname().nodename} {os.getpid()} {tag}\n".encode())
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
