"""Run several energy trainings with a fixed number of concurrent slots."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from typing import Any


def run_pack(
    specs: list[str],
    slots: int,
    *,
    is_done: Callable[[str], bool],
    launch: Callable[[str], Any],
    poll_interval: float = 0.05,
) -> list[tuple[str, int]]:
    """Launch unfinished specs, at most ``slots`` at a time.

    A non-zero child exit does not cancel siblings or the remaining queue.
    ``launch`` must return an object with ``poll() -> int | None``.
    """
    if slots < 1:
        raise ValueError(f"slots must be >= 1, got {slots}")
    pending = [spec for spec in specs if not is_done(spec)]
    live: dict[str, Any] = {}
    results: list[tuple[str, int]] = []
    while pending or live:
        while pending and len(live) < slots:
            spec = pending.pop(0)
            live[spec] = launch(spec)
        if not live:
            break
        finished: list[tuple[str, int]] = []
        for spec, handle in list(live.items()):
            code = handle.poll()
            if code is not None:
                finished.append((spec, int(code)))
        if not finished:
            if poll_interval:
                time.sleep(poll_interval)
            continue
        for spec, code in finished:
            del live[spec]
            results.append((spec, code))
    return results


# Imported so tests can monkeypatch subprocess on this module.
__all__ = ["run_pack", "subprocess"]
