"""Policies shared by the response-letter timing table and figure."""

from __future__ import annotations

import math


def gpu_speedup_is_clear(
    cpu_mean_s: float,
    gpu_mean_s: float,
    *,
    maximum_ratio: float = 0.8,
) -> bool:
    """Return whether a GPU result is clearly faster than its CPU counterpart.

    Parameters
    ----------
    cpu_mean_s
        Mean CPU wall time per step, in seconds.
    gpu_mean_s
        Mean GPU wall time per step, in seconds.
    maximum_ratio
        Strict upper bound on ``gpu_mean_s / cpu_mean_s``.

    Returns
    -------
    bool
        ``True`` only for finite, positive timings with a speedup of at least
        ``1 / maximum_ratio``.
    """
    values = (cpu_mean_s, gpu_mean_s, maximum_ratio)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        return False
    return gpu_mean_s / cpu_mean_s < maximum_ratio
