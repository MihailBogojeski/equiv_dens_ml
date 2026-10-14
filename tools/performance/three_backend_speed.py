#!/usr/bin/env python3
"""Host-side runner for matched Torch eager/compile/JAX application timings.

This file intentionally uses only the Python standard library. Numerical work
and array comparison stay inside the project's qualified runtime images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

BACKENDS = ("torch-eager", "torch-compile", "jax")
MODES = ("ef", "calculator", "density", "density_update", "energy_force_update")
DEFAULT_CASES = ("ethanol-paper",)
STOP_SIGNAL: int | None = None


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stop_handler(signum: int, _frame: object) -> None:
    global STOP_SIGNAL
    STOP_SIGNAL = signum


def run_child(command: list[str], log_path: Path, timeout: int,
              env: dict[str, str] | None = None) -> dict[str, object]:
    """Run a child in its own process group, preserving its complete log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("wb") as log:
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
        except OSError as exc:
            log.write((f"Could not start child: {exc}\n").encode())
            return {"status": "launch-error", "returncode": None,
                    "wall_seconds": time.monotonic() - started,
                    "error": str(exc)}
        deadline = started + timeout
        try:
            while True:
                if STOP_SIGNAL is not None:
                    _kill_group(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        _kill_group(child.pid, signal.SIGKILL)
                        child.wait()
                    return {"status": "interrupted", "returncode": child.returncode,
                            "wall_seconds": time.monotonic() - started,
                            "signal": STOP_SIGNAL}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    returncode = child.wait(timeout=min(1.0, remaining))
                    status = "ok" if returncode == 0 else "exit"
                    return {"status": status, "returncode": returncode,
                            "wall_seconds": time.monotonic() - started}
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired:
            _kill_group(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_group(child.pid, signal.SIGKILL)
                child.wait()
            log.write(f"\nCENSORED: exceeded {timeout}s timeout; process group terminated\n".encode())
            return {"status": "timeout", "returncode": child.returncode,
                    "wall_seconds": time.monotonic() - started,
                    "timeout_seconds": timeout, "censored": True}
        except BaseException:
            _kill_group(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_group(child.pid, signal.SIGKILL)
                child.wait()
            raise


def _kill_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def validate_compile_threads(value: int, environ: dict[str, str] | None = None) -> int:
    if value < 1:
        raise ValueError("compile-threads must be a positive integer")
    environ = os.environ if environ is None else environ
    allocated = environ.get("SLURM_CPUS_PER_TASK")
    if allocated:
        try:
            cpu_count = int(allocated)
        except ValueError as exc:
            raise ValueError(f"SLURM_CPUS_PER_TASK is not an integer: {allocated!r}") from exc
        if cpu_count < 1 or value > cpu_count:
            raise ValueError(f"compile-threads={value} exceeds SLURM_CPUS_PER_TASK={allocated}")
    return value


def seed_compiler_caches(seed_root: Path, cache_root: Path, cases: list[str],
                         modes: list[str],
                         selected_backends: tuple[str, ...] = BACKENDS) -> dict[str, object]:
    """Copy only persistent compiler-cache files into a clean destination."""
    source = seed_root.resolve(strict=True)
    destination = cache_root.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("seed-cache-root and cache-root must be separate, non-nested directories")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"cache-root must be fresh or empty before seeding: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    entries: dict[str, object] = {}
    for case in cases:
        for backend in selected_backends:
            for mode in modes:
                key = f"{case}/{backend}/{mode}"
                source_parent = source / case / backend / mode
                target_parent = destination / case / backend / mode
                copied: list[str] = []
                for cache_name in ("inductor", "jax", "triton", "cuda"):
                    source_cache = source_parent / cache_name
                    if source_cache.is_dir() and any(path.is_file() for path in source_cache.rglob("*")):
                        target_cache = target_parent / cache_name
                        target_cache.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(source_cache, target_cache)
                        copied.append(cache_name)
                entries[key] = {"seeded": bool(copied), "source": str(source_parent),
                                "destination": str(target_parent),
                                "compiler_cache_directories_copied": copied,
                                "provenance": "copied from --seed-cache-root; first call is continuation with seeded compiler cache"}
    return {"status": "complete", "source": str(source), "destination": str(destination),
            "entries": entries}


def cache_environment(root: Path, case: str, backend: str, round_index: int,
                      mode: str, policy: str, cache_root: Path | None = None,
                      compile_threads: int = 1, seeded: bool = False) -> tuple[list[str], dict[str, object]]:
    compile_threads = validate_compile_threads(compile_threads)
    cache_root = (cache_root or (root / "caches")).resolve()
    cache_parent = cache_root / case / backend / mode
    cache = (cache_parent / f"round-{round_index + 1:02d}"
             if policy == "isolated" else cache_parent)
    cache_preexisted = cache.exists()
    files_before = sum(1 for path in cache.rglob("*") if path.is_file()) if cache_preexisted else 0
    cache.mkdir(parents=True, exist_ok=(policy == "shared-after-first"))
    initially_empty = not cache_preexisted or files_before == 0
    for name in ("densnet", "inductor", "jax", "triton", "cuda"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    files_before = sum(1 for path in cache.rglob("*") if path.is_file())
    values = {
        "DENSNET_CACHE_ROOT": str(cache / "densnet"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1" if policy == "shared-after-first" else "0",
        "TORCHINDUCTOR_COMPILE_THREADS": str(compile_threads),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "true",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
        "JAX_COMPILATION_CACHE_DIR": str(cache / "jax"),
        "JAX_ENABLE_COMPILATION_CACHE": "true" if policy == "shared-after-first" else "false",
        # run.sh uses --cleanenv, so job identity must cross explicitly.
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
    }
    if policy == "shared-after-first":
        values["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        values["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
    metadata = {"cache_policy": policy, "cache_root": str(cache_root),
                "cache_directory": str(cache), "compile_threads": compile_threads,
                "cache_initially_empty": initially_empty and not seeded,
                "cache_reused": not initially_empty,
                "cache_seeded": seeded,
                "cache_files_before": files_before,
                "cache_reuse_definition": "a nonempty persistent cache directory was available at process start; individual cache hits are backend-reported separately",
                "first_call_interpretation": ("fresh-process first call with seeded compiler caches available; continuation, not cold"
                                               if seeded else
                                               "fresh-process first call with a nonempty persistent cache available"
                                               if not initially_empty else
                                               "fresh-process first call with an initially empty persistent cache")}
    return ["env", *(f"{key}={value}" for key, value in values.items())], metadata


def count_cache_files(cache_directory: str) -> int:
    path = Path(cache_directory)
    return sum(1 for child in path.rglob("*") if child.is_file()) if path.exists() else 0


def arm_order(round_index: int, selected_backends: tuple[str, ...] = BACKENDS) -> tuple[str, ...]:
    rotations = (BACKENDS, (BACKENDS[1], BACKENDS[2], BACKENDS[0]),
                 (BACKENDS[2], BACKENDS[0], BACKENDS[1]))
    rotation = rotations[round_index % len(rotations)]
    ordered = rotation if (round_index // len(rotations)) % 2 == 0 else tuple(reversed(rotation))
    return tuple(backend for backend in ordered if backend in selected_backends)


def write_register(path: Path, state: dict[str, object]) -> None:
    atomic_json(path, state)


def arm_record_path(root: Path, case: str, mode: str, round_index: int,
                    backend: str) -> Path:
    return root / "timings" / f"{case}-{mode}-round-{round_index + 1:02d}-{backend}.json"


def invoke_arm(root: Path, case: str, mode: str, round_index: int,
               backend: str, warmup: int, iterations: int, timeout: int,
               cache_policy: str, cache_root: Path | None = None,
               compile_threads: int = 1, seeded: bool = False,
               compile_strategy: str = "fullgraph") -> dict[str, object]:
    fixture = root / "fixtures" / case
    output = arm_record_path(root, case, mode, round_index, backend)
    output.parent.mkdir(parents=True, exist_ok=True)
    if backend == "torch-eager":
        runtime, worker_backend = "torch", "torch-eager"
    elif backend == "torch-compile":
        runtime, worker_backend = "torch", "torch-compile"
    else:
        runtime, worker_backend = "jax", "jax"
    cache_env, cache_metadata = cache_environment(root, case, backend, round_index, mode,
                                                  cache_policy, cache_root, compile_threads, seeded)
    command = ["tools/runtime/run.sh", runtime, *cache_env,
               "python", "-u", "-m", "benchmarks.torch_compile_speed",
               "--backend", worker_backend, "--fixture", str(fixture),
               "--mode", mode, "--output", str(output), "--warmup", str(warmup),
               "--iterations", str(iterations)]
    if backend == "torch-compile":
        command.extend(["--compile-strategy", compile_strategy])
    result = run_child(command, output.with_suffix(".log"), timeout)
    result.update({"backend": backend, "file": str(output), "log": str(output.with_suffix(".log")),
                   "order_index": None, "compile_strategy": compile_strategy,
                   **cache_metadata})
    if result["status"] == "ok" and not output.is_file():
        result.update(status="missing-report", error="worker exited successfully without its JSON report")
    result["cache_files_after"] = count_cache_files(str(cache_metadata["cache_directory"]))
    if result["status"] != "ok":
        # Keep a machine-readable censor/failure record even when the worker
        # never reached the point where it could write its own report.
        failure = {"status": result["status"], "backend": backend,
                   "case": case, "mode": mode, "round": round_index + 1,
                   "censored": result.get("censored", False),
                   "wall_seconds": result["wall_seconds"],
                   "cache_policy": cache_policy,
                   "compile_strategy": compile_strategy,
                   "cache_initially_empty": cache_metadata["cache_initially_empty"],
                   "cache_reused": cache_metadata["cache_reused"],
                   "cache_seeded": cache_metadata["cache_seeded"],
                   "cache_root": cache_metadata["cache_root"],
                   "compile_threads": compile_threads,
                   "cache_files_before": cache_metadata["cache_files_before"],
                   "cache_files_after": result["cache_files_after"],
                   "log": result["log"], "error": result.get("error")}
        failure_path = output.with_suffix(".failure.json")
        atomic_json(failure_path, failure)
        result["failure_file"] = str(failure_path)
        if not output.exists():
            atomic_json(output, failure)
    else:
        try:
            worker_report = json.loads(output.read_text())
            if isinstance(worker_report, dict):
                worker_report["orchestration"] = {key: result[key] for key in (
                    "cache_policy", "cache_directory", "cache_initially_empty", "cache_reused",
                    "cache_root", "compile_threads", "compile_strategy", "cache_seeded",
                    "cache_files_before", "cache_files_after", "cache_reuse_definition",
                    "first_call_interpretation")}
                atomic_json(output, worker_report)
        except (OSError, json.JSONDecodeError):
            result.update(status="invalid-report", error="worker JSON could not be annotated with cache state")
    return result


def read_worker_report(arm: dict[str, object]) -> dict[str, object] | None:
    if arm.get("status") != "ok":
        return None
    try:
        report = json.loads(Path(str(arm["file"])).read_text())
        return report if isinstance(report, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def compare_pair(root: Path, left: dict[str, object], right: dict[str, object],
                 case: str, mode: str, round_index: int, timeout: int) -> dict[str, object]:
    left_report, right_report = read_worker_report(left), read_worker_report(right)
    pair = f"{case}-{mode}-round-{round_index + 1:02d}-{left['backend']}-vs-{right['backend']}"
    gate_path = root / "gates" / f"{pair}.json"
    if left_report is None or right_report is None:
        state = {"status": "unavailable", "pass": False,
                 "reason": "one or both worker reports are unavailable",
                 "left_status": left.get("status"), "right_status": right.get("status")}
        atomic_json(gate_path, state)
        return {"status": "unavailable", "pass": False, "file": str(gate_path)}
    command = ["tools/runtime/run.sh", "torch", "python", "-u", "-m",
               "benchmarks.torch_compile_speed", "--compare",
               str(Path(str(left["file"])).with_suffix("")),
               str(Path(str(right["file"])).with_suffix("")),
               "--comparison-output", str(gate_path)]
    gate_path.unlink(missing_ok=True)
    result = run_child(command, gate_path.with_suffix(".log"), timeout)
    if result["status"] != "ok":
        failure = {"status": result["status"], "pass": False, "censored": result.get("censored", False),
                   "wall_seconds": result["wall_seconds"], "log": str(gate_path.with_suffix(".log"))}
        if gate_path.exists():
            try:
                existing = json.loads(gate_path.read_text())
                if isinstance(existing, dict):
                    existing["pass"] = False
                    existing["status"] = result["status"]
                    existing["comparison_process_status"] = result["status"]
                    existing["comparison_log"] = failure["log"]
                    atomic_json(gate_path, existing)
                    atomic_json(gate_path.with_suffix(".failure.json"), failure)
                    return existing
            except (OSError, json.JSONDecodeError):
                pass
        atomic_json(gate_path, failure)
        return failure
    try:
        gate = json.loads(gate_path.read_text())
        if not isinstance(gate, dict) or "pass" not in gate:
            raise ValueError("comparison output lacks pass field")
        gate.setdefault("status", "pass" if gate["pass"] else "failed")
        return gate
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        failure = {"status": "invalid-comparison-output", "pass": False, "error": str(exc)}
        atomic_json(gate_path, failure)
        return failure


def median_latency(report: dict[str, object]) -> float | None:
    values = report.get("times_ms")
    if not isinstance(values, list) or not values:
        return None
    try:
        nums = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0 for value in nums):
        return None
    return statistics.median(nums)


def summarize_cell(cell: dict[str, object], selected_backends: tuple[str, ...] | None = None) -> dict[str, object]:
    rounds = cell["rounds"]
    selected_backends = selected_backends or tuple(cell.get("backends", BACKENDS))
    summaries: dict[str, object] = {}
    for candidate in (backend for backend in selected_backends if backend != "torch-eager"):
        ratios: list[float] = []
        eager_observations: list[float] = []
        candidate_observations: list[float] = []
        candidate_cold: list[float] = []
        eager_cold: list[float] = []
        candidate_warm_cache: list[float] = []
        eager_warm_cache: list[float] = []
        candidate_seeded: list[float] = []
        eager_seeded: list[float] = []
        accepted_gates = 0
        failed_gates = 0
        for row in rounds:  # type: ignore[assignment]
            arms = row.get("arms", {})
            gate = row.get("comparisons", {}).get(f"{candidate}-vs-torch-eager", {})
            if gate.get("pass") is True:
                accepted_gates += 1
                a = read_worker_report(arms.get("torch-eager", {}))
                b = read_worker_report(arms.get(candidate, {}))
                if a is not None and b is not None:
                    ta, tb = median_latency(a), median_latency(b)
                    if ta is not None and tb is not None:
                        ratios.append(ta / tb)
                        eager_observations.extend(float(value) for value in a["times_ms"])
                        candidate_observations.extend(float(value) for value in b["times_ms"])
                    for report, dest in ((a, eager_cold), (b, candidate_cold)):
                        try:
                            v = float(report["cold_first_call_seconds"])
                            if math.isfinite(v) and v >= 0:
                                orchestration = report.get("orchestration", {})
                                warm_destination = (eager_warm_cache if report is a else candidate_warm_cache)
                                seeded_destination = eager_seeded if report is a else candidate_seeded
                                if orchestration.get("cache_seeded"):
                                    seeded_destination.append(v)
                                elif orchestration.get("cache_reused"):
                                    warm_destination.append(v)
                                elif orchestration.get("cache_initially_empty"):
                                    dest.append(v)
                        except (KeyError, TypeError, ValueError):
                            pass
            elif gate:
                failed_gates += 1
        summary: dict[str, object] = {
            "comparison": f"torch-eager / {candidate}",
            "speedup_ratio_definition": "median eager milliseconds / median candidate milliseconds; >1 favors candidate",
            "valid_pairs": len(ratios), "requested_pairs": len(rounds),
            "passed_numerical_gates": accepted_gates, "failed_or_unavailable_gates": failed_gates,
            "valid_for_speed_claim": len(ratios) == len(rounds) and len(rounds) > 0,
            "paired_speedup_median": statistics.median(ratios) if ratios else None,
            "paired_speedup_min": min(ratios) if ratios else None,
            "paired_speedup_max": max(ratios) if ratios else None,
            "paired_speedups": ratios,
            "pooled_median_eager_ms": statistics.median(eager_observations) if eager_observations else None,
            "pooled_median_candidate_ms": statistics.median(candidate_observations) if candidate_observations else None,
            "pooled_median_speedup": (statistics.median(eager_observations) /
                                       statistics.median(candidate_observations)
                                       if eager_observations and candidate_observations else None),
            "eager_first_uncached_call_median_seconds": statistics.median(eager_cold) if eager_cold else None,
            "candidate_first_uncached_call_median_seconds": statistics.median(candidate_cold) if candidate_cold else None,
            "eager_warm_cache_first_call_median_seconds": statistics.median(eager_warm_cache) if eager_warm_cache else None,
            "candidate_warm_cache_first_call_median_seconds": statistics.median(candidate_warm_cache) if candidate_warm_cache else None,
            "eager_uncached_first_call_count": len(eager_cold),
            "candidate_uncached_first_call_count": len(candidate_cold),
            "eager_warm_cache_first_call_count": len(eager_warm_cache),
            "candidate_warm_cache_first_call_count": len(candidate_warm_cache),
            "eager_seeded_first_call_median_seconds": statistics.median(eager_seeded) if eager_seeded else None,
            "candidate_seeded_first_call_median_seconds": statistics.median(candidate_seeded) if candidate_seeded else None,
            "eager_seeded_first_call_count": len(eager_seeded),
            "candidate_seeded_first_call_count": len(candidate_seeded),
        }
        summaries[candidate] = summary
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="report directory")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS),
                        help="selected arms; comparisons always use torch-eager as the reference")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--grid-points", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-policy", choices=("isolated", "shared-after-first"),
                        default="isolated",
                        help="isolated cold caches, or reuse each backend/case/mode cache after round one")
    parser.add_argument("--cache-root", type=Path,
                        help="compiler cache root (defaults to OUTPUT/caches; use node-local scratch for measurements)")
    parser.add_argument("--seed-cache-root", type=Path,
                        help="copy compiler caches from this prior OUTPUT/caches tree into a fresh cache root")
    parser.add_argument("--compile-threads", type=int, default=1,
                        help="Torch Inductor compiler threads (positive and no greater than Slurm CPU allocation)")
    parser.add_argument("--compile-strategy", choices=("fullgraph",), default="fullgraph")
    parser.add_argument("--timeout", type=int, default=900,
                        help="per worker/comparison timeout in seconds")
    args = parser.parse_args()
    if args.rounds < 1 or args.timeout < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("rounds/timeout/iterations must be positive and warmup nonnegative")
    if len(set(args.backends)) != len(args.backends):
        parser.error("--backends entries must be distinct")
    if "torch-eager" not in args.backends or len(args.backends) < 2:
        parser.error("--backends requires torch-eager and at least one distinct comparison backend")
    selected_backends = tuple(args.backends)
    try:
        compile_threads = validate_compile_threads(args.compile_threads)
    except ValueError as exc:
        parser.error(str(exc))
    if args.seed_cache_root and args.cache_policy != "shared-after-first":
        parser.error("--seed-cache-root requires --cache-policy shared-after-first")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() or (root / "register.json").exists():
        parser.error("Report directory already contains a run; use a fresh directory")
    cache_root = (args.cache_root or (root / "caches")).resolve()
    seed_manifest: dict[str, object] = {"status": "not-requested", "destination": str(cache_root)}
    if args.seed_cache_root:
        try:
            seed_manifest = seed_compiler_caches(args.seed_cache_root, cache_root,
                                                 args.cases, args.modes, selected_backends)
        except (OSError, ValueError, shutil.Error) as exc:
            state = {"status": "cache-seeding-failed", "cache_root": str(cache_root),
                     "seed_cache_root": str(args.seed_cache_root.resolve()),
                     "compile_threads": compile_threads,
                     "compile_strategy": args.compile_strategy,
                     "cache_seeding": {"status": "failed", "error": str(exc)}}
            atomic_json(root / "register.json", state)
            print(f"Cache seeding failed: {exc}", file=sys.stderr)
            return 1
    else:
        if cache_root.exists() and any(cache_root.iterdir()):
            parser.error(f"--cache-root must be fresh or empty when no seed is requested: {cache_root}")
        cache_root.mkdir(parents=True, exist_ok=True)
    register_path = root / "register.json"
    configuration = {"backends": list(selected_backends), "cases": args.cases, "modes": args.modes,
                     "rounds": args.rounds, "warmup": args.warmup,
                     "iterations": args.iterations, "timeout_seconds": args.timeout,
                     "batch_size": args.batch_size, "grid_points": args.grid_points,
                     "cache_policy": args.cache_policy,
                     "cache_root": str(cache_root),
                     "seed_cache_root": str(args.seed_cache_root.resolve()) if args.seed_cache_root else None,
                     "compile_threads": compile_threads,
                     "compile_strategy": args.compile_strategy,
                     "pair_order": "rotated backend order; alternating forward/reverse rotation",
                     "comparison_runtime": "qualified Torch runtime via tools/runtime/run.sh",
                     "precision_gate": {"rtol": 1e-8, "atol": 1e-8},
                     "first_call_labels": ({"round_1": "seeded-cache continuation first call; not cold",
                                            "rounds_2_plus": "warm-cache startup first call"}
                                           if args.seed_cache_root else
                                           {"round_1": "uncached first call",
                                            "rounds_2_plus": "uncached first call"}
                                           if args.cache_policy == "isolated" else
                                           {"round_1": "uncached first call with initially empty compiler caches",
                                            "rounds_2_plus": "fresh-process startup first call with persistent caches available"}),
                     "cache_policy_details": ("Seeded compiler cache provenance applies to round one; first calls are continuations, not cold compiles."
                                               if args.seed_cache_root else
                                               "Every arm uses a fresh backend/case/round/mode cache; its first call is uncached and Torch FX graph cache is disabled."
                                               if args.cache_policy == "isolated" else
                                               "Round one starts with an empty backend/case/mode cache; later rounds start fresh processes with persistent Torch FX/Inductor/Triton/CUDA and JAX caches available. The first call in later rounds is a warm-cache startup call, not a cold compile. Torch FX graph cache is enabled.")}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    atomic_json(root / "manifest.json", {"configuration": configuration,
                                         "configuration_sha256": config_hash})
    state: dict[str, object] = {
        "protocol": configuration, "configuration_sha256": config_hash,
        "cache_root": str(cache_root), "compile_threads": compile_threads,
        "cache_seeding": seed_manifest,
        "cases": args.cases, "modes": args.modes, "backends": list(selected_backends),
        "status": "running", "cells": [],
    }
    write_register(register_path, state)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    generator = ["tools/runtime/run.sh", "torch", "python", "-u", "-m",
                 "benchmarks.torch_compile_speed", "--generate", str(root / "fixtures"),
                 "--cases", *args.cases, "--grid-points", str(args.grid_points),
                 "--batch-size", str(args.batch_size)]
    state["fixture_generation"] = {"status": "running", "log": str(root / "fixture-generation.log")}
    write_register(register_path, state)
    generated = run_child(generator, root / "fixture-generation.log", args.timeout)
    state["fixture_generation"] = generated
    write_register(register_path, state)
    if generated["status"] != "ok":
        state["status"] = "fixture-generation-failed"
        write_register(register_path, state)
        return 1

    cells: list[dict[str, object]] = state["cells"]  # type: ignore[assignment]
    aborted = False
    for case in args.cases:
        for mode in args.modes:
            cell: dict[str, object] = {"case": case, "mode": mode,
                                       "backends": list(selected_backends),
                                       "status": "running", "rounds": []}
            cells.append(cell)
            write_register(register_path, state)
            compile_failed_first = False
            rows: list[dict[str, object]] = cell["rounds"]  # type: ignore[assignment]
            for round_index in range(args.rounds):
                row: dict[str, object] = {"round": round_index + 1,
                                          "order": list(arm_order(round_index, selected_backends)),
                                          "arms": {}, "comparisons": {}}
                rows.append(row)
                arms: dict[str, dict[str, object]] = row["arms"]  # type: ignore[assignment]
                write_register(register_path, state)
                for order_index, backend in enumerate(arm_order(round_index, selected_backends)):
                    if STOP_SIGNAL is not None:
                        row["interrupted_by_signal"] = STOP_SIGNAL
                        cell["status"] = "interrupted"
                        aborted = True
                        break
                    if backend == "torch-compile" and compile_failed_first:
                        result: dict[str, object] = {
                            "status": "skipped-after-first-compile-failure",
                            "backend": backend, "round": round_index + 1,
                            "reason": "identical cell/config was not retried after its first compile failure",
                            "cache_policy": args.cache_policy,
                            "cache_root": str(cache_root), "compile_threads": compile_threads,
                            "compile_strategy": args.compile_strategy,
                            "cache_initially_empty": False,
                            "cache_reused": False,
                            "cache_seeded": False,
                            "file": str(arm_record_path(root, case, mode, round_index, backend)),
                        }
                        atomic_json(Path(str(result["file"])), result)
                    else:
                        print(f"ARM START case={case} mode={mode} round={round_index + 1} backend={backend}",
                              flush=True)
                        result = invoke_arm(root, case, mode, round_index, backend,
                                            args.warmup, args.iterations, args.timeout,
                                            args.cache_policy, cache_root, compile_threads,
                                            bool(args.seed_cache_root and round_index == 0 and
                                                 seed_manifest.get("entries", {}).get(
                                                     f"{case}/{backend}/{mode}", {}).get("seeded")),
                                            args.compile_strategy)
                        result["order_index"] = order_index
                        print(f"ARM END case={case} mode={mode} round={round_index + 1} backend={backend} "
                              f"status={result['status']} wall={result['wall_seconds']:.1f}s", flush=True)
                    arms[backend] = result
                    if backend == "torch-compile" and result["status"] != "ok":
                        compile_failed_first = True
                    write_register(register_path, state)
                if aborted:
                    break
                for candidate in (backend for backend in selected_backends if backend != "torch-eager"):
                    if candidate in arms and "torch-eager" in arms:
                        print(f"GATE START case={case} mode={mode} round={round_index + 1} "
                              f"backend={candidate}", flush=True)
                        gate = compare_pair(root, arms["torch-eager"], arms[candidate],
                                            case, mode, round_index, args.timeout)
                        row["comparisons"][f"{candidate}-vs-torch-eager"] = gate  # type: ignore[index]
                        print(f"GATE END case={case} mode={mode} round={round_index + 1} "
                              f"backend={candidate} status={gate.get('status')}", flush=True)
                        write_register(register_path, state)
                if STOP_SIGNAL is not None:
                    row["interrupted_by_signal"] = STOP_SIGNAL
                    cell["status"] = "interrupted"
                    aborted = True
                row["status"] = ("complete" if len(arms) == len(selected_backends) and
                                  all(arm.get("status") == "ok" for arm in arms.values()) and
                                  all(gate.get("pass") is True for gate in row["comparisons"].values()) and
                                  len(row["comparisons"]) == len(selected_backends) - 1 and not aborted
                                  else "partial")
                write_register(register_path, state)
            cell["summary"] = summarize_cell(cell, selected_backends)
            cell["status"] = "interrupted" if aborted else (
                "complete" if all(row.get("status") == "complete" for row in rows) else "partial")
            write_register(register_path, state)
            if aborted:
                break
        if aborted:
            break
    state["status"] = "interrupted" if aborted else (
        "complete" if all(cell.get("status") == "complete" for cell in cells) else "partial")
    state["completed_at_unix"] = time.time()
    write_register(register_path, state)
    return 128 + STOP_SIGNAL if STOP_SIGNAL is not None else (0 if state["status"] == "complete" else 1)


if __name__ == "__main__":
    sys.exit(main())
