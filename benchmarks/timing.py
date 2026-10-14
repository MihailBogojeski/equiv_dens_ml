"""Controlled paired timings; callers hold the shared measurement lock."""

from __future__ import annotations

import os
import statistics
import time
from typing import Callable

import torch
from benchmarks.precision import precision_metadata


def synchronize(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def timed_block(fn: Callable, *, device="cpu", iterations: int = 30) -> dict:
    """Synchronized wall time includes Python/host work inside ``fn``."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    synchronize(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
    synchronize(device)
    elapsed = time.perf_counter() - start
    # Keep the last result alive through synchronization; do not accumulate all
    # output/derivative graphs across repetitions.
    del result
    memory = {}
    if torch.device(device).type == "cuda":
        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return {"seconds_per_call": elapsed / iterations,
            "iterations": iterations, **memory}


def paired_benchmark(reference_fn: Callable, candidate_fn: Callable, *,
                     device="cpu", warmup: int = 10, iterations: int = 30,
                     rounds: int = 5) -> dict:
    """Alternate order, preserving raw round timings rather than only a ratio.

    Stateful callables must replay matched lifecycle/optimizer sequences. Setup,
    state reset and cache-policy decisions belong to the benchmark case, not to
    this timing helper. This helper does not acquire an allocation or lock.
    """
    if rounds < 5 or warmup < 0:
        raise ValueError("at least five paired rounds and nonnegative warmup required")
    for fn in (reference_fn, candidate_fn):
        for _ in range(warmup):
            fn()
    synchronize(device)
    results = []
    for index in range(rounds):
        order = ("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")
        row = {"round": index, "order": list(order)}
        for arm in order:
            fn = reference_fn if arm == "reference" else candidate_fn
            row[arm] = timed_block(fn, device=device, iterations=iterations)
        row["speedup"] = row["reference"]["seconds_per_call"] / row["candidate"]["seconds_per_call"]
        results.append(row)
    reference = [r["reference"]["seconds_per_call"] for r in results]
    candidate = [r["candidate"]["seconds_per_call"] for r in results]
    speedups = [r["speedup"] for r in results]
    device_info = {"requested": str(device)}
    if torch.device(device).type == "cuda":
        props = torch.cuda.get_device_properties(device)
        device_info.update(name=props.name, uuid=str(getattr(props, "uuid", "unknown")))
    return {
        "protocol": "P1", "torch": torch.__version__,
        "precision": precision_metadata(),
        "device": device_info, "threads": torch.get_num_threads(),
        "allocation": os.environ.get("SLURM_JOB_ID"), "warmup_calls": warmup,
        "rounds": results,
        "reference_median_seconds": statistics.median(reference),
        "candidate_median_seconds": statistics.median(candidate),
        "speedup": statistics.median(reference) / statistics.median(candidate),
        "paired_speedup_min": min(speedups), "paired_speedup_max": max(speedups),
    }
