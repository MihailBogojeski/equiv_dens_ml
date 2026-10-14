#!/usr/bin/env python3
"""Pause one verified Slurm training PID while a bounded GPU benchmark runs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

# Direct execution sets sys.path[0] to tools/performance, not the checkout
# root. Bootstrap the project root before importing the shared validator.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.performance.three_backend_speed import validate_compile_threads

EXPECTED_TRAINING_SCRIPT = Path(
    "/home/johannes/atomistic-deq-ff/src/atomistic_deq_ff/scripts/train_deq_modular.py")
MODE_CHOICES = ("ef", "calculator", "density", "density_update", "energy_force_update")
BACKEND_CHOICES = ("torch-eager", "torch-compile", "jax")
STOP_SIGNAL: int | None = None


class SafetyError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _signal_handler(signum: int, _frame: object) -> None:
    global STOP_SIGNAL
    STOP_SIGNAL = signum


class System:
    """Small OS boundary so all pause/scheduler/GPU behavior can be mocked."""

    def process_identity(self, pid: int) -> dict[str, Any]:
        proc = Path(f"/proc/{pid}")
        status: dict[str, str] = {}
        for line in (proc / "status").read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
        uid_fields = status.get("Uid", "").split()
        if not uid_fields:
            raise SafetyError(f"Cannot read UID for PID {pid}")
        command = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
        stat = (proc / "stat").read_text()
        close = stat.rfind(")")
        if close < 0:
            raise SafetyError(f"Cannot parse /proc/{pid}/stat")
        fields_after_comm = stat[close + 1:].split()
        if len(fields_after_comm) <= 19:
            raise SafetyError(f"Missing start tick in /proc/{pid}/stat")
        return {"pid": pid, "uid": int(uid_fields[0]), "state": fields_after_comm[0],
                "start_ticks": int(fields_after_comm[19]), "command": command,
                "command_line": " ".join(command)}

    def batch_pids(self, job_id: str) -> list[int]:
        result = subprocess.run(["scontrol", "listpids", job_id], check=True,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        rows: list[int] = []
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                pid = int(fields[0])
            except ValueError:
                continue
            row_job = fields[1].split(".", 1)[0]
            step = fields[2].lower()
            if row_job == str(job_id) and step == "batch":
                rows.append(pid)
        return rows

    def gpu_snapshot(self) -> dict[str, Any]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible:
            raise SafetyError("CUDA_VISIBLE_DEVICES is empty")
        device = subprocess.run(
            ["nvidia-smi", "-i", visible, "--query-gpu=uuid,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip().splitlines()
        if len(device) != 1:
            raise SafetyError(f"Expected exactly one visible GPU, got {len(device)}")
        columns = next(csv.reader([device[0]], skipinitialspace=True))
        if len(columns) != 3:
            raise SafetyError(f"Unexpected nvidia-smi GPU row: {device[0]}")
        apps_result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pids: list[int] = []
        for line in apps_result.stdout.splitlines():
            fields = next(csv.reader([line], skipinitialspace=True))
            if len(fields) < 2 or fields[1].strip() != columns[0].strip():
                continue
            try:
                pids.append(int(fields[0]))
            except ValueError:
                continue
        return {"uuid": columns[0].strip(), "memory_free_mib": int(columns[1].strip()),
                "utilization_pct": float(columns[2].strip().rstrip("% ")),
                "compute_pids": sorted(set(pids))}

    def send_signal(self, pid: int, signum: int) -> None:
        os.kill(pid, signum)

    def popen(self, command: list[str], **kwargs: Any) -> Any:
        return subprocess.Popen(command, **kwargs)

    def kill_child_group(self, pid: int, signum: int) -> None:
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            pass

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def verify_identity(identity: dict[str, Any], pid: int, uid: int,
                    expected_ticks: int) -> None:
    if identity["pid"] != pid or identity["uid"] != uid:
        raise SafetyError(f"PID {pid} is not owned by current UID {uid}")
    if identity["start_ticks"] != expected_ticks:
        raise SafetyError(f"PID {pid} start tick changed: expected {expected_ticks}, got {identity['start_ticks']}")
    command = identity["command"]
    if EXPECTED_TRAINING_SCRIPT.as_posix() not in command:
        raise SafetyError("PID command does not contain the exact expected train_deq_modular.py path")
    if not EXPECTED_TRAINING_SCRIPT.is_file():
        raise SafetyError(f"Expected training script is absent: {EXPECTED_TRAINING_SCRIPT}")
    if identity["state"] in ("T", "t"):
        raise SafetyError(f"PID {pid} is already stopped; refusing to claim or resume another pause")


def _append_log(log_path: Path, message: str) -> None:
    with log_path.open("a") as stream:
        stream.write(f"{utc_now()} {message}\n")
        stream.flush()


def _record(state_path: Path, state: dict[str, Any], log_path: Path,
            event: str, **details: Any) -> None:
    event_data = {"time": utc_now(), "event": event, **details}
    state.setdefault("events", []).append(event_data)
    state["updated_at"] = event_data["time"]
    atomic_json(state_path, state)
    _append_log(log_path, json.dumps(event_data, sort_keys=True))


def _terminate_child(system: System, child: Any, grace_seconds: float = 5.) -> None:
    if child.poll() is not None:
        return
    system.kill_child_group(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        system.kill_child_group(child.pid, signal.SIGKILL)
        child.wait()


def _wait_for_pause(system: System, args: argparse.Namespace,
                    state_path: Path, state: dict[str, Any], log_path: Path,
                    timeout_seconds: float = 30.) -> dict[str, Any]:
    deadline = system.monotonic() + timeout_seconds
    last_gpu: dict[str, Any] = {}
    while system.monotonic() <= deadline:
        if STOP_SIGNAL is not None:
            raise SafetyError(f"Wrapper interrupted by signal {STOP_SIGNAL} during pause verification")
        identity = system.process_identity(args.training_pid)
        if identity["start_ticks"] != args.expected_training_start_ticks:
            raise SafetyError("Training PID identity changed while waiting for it to stop")
        last_gpu = system.gpu_snapshot()
        if last_gpu["uuid"] != args.expected_gpu_uuid:
            raise SafetyError("Visible GPU UUID changed while waiting for training to stop")
        unexpected = sorted(set(last_gpu["compute_pids"]) - {args.training_pid})
        if unexpected:
            raise SafetyError(f"Unexpected visible GPU compute PIDs after pause: {unexpected}")
        if identity["state"] in ("T", "t") and last_gpu["utilization_pct"] == 0:
            _record(state_path, state, log_path, "pause_confirmed",
                    process=identity, gpu=last_gpu)
            return {"process": identity, "gpu": last_gpu}
        system.sleep(.25)
    raise SafetyError(f"Training did not reach stopped state and GPU idle within {timeout_seconds}s; last GPU={last_gpu}")


def _resume_owned_pause(system: System, args: argparse.Namespace,
                        state_path: Path, state: dict[str, Any], log_path: Path,
                        paused_by_us: bool) -> bool:
    if not paused_by_us:
        return False
    try:
        current = system.process_identity(args.training_pid)
    except (OSError, SafetyError) as exc:
        _record(state_path, state, log_path, "resume_skipped_process_unavailable", error=str(exc))
        return False
    if (current["uid"] != os.getuid() or
            current["start_ticks"] != args.expected_training_start_ticks or
            EXPECTED_TRAINING_SCRIPT.as_posix() not in current["command"]):
        _record(state_path, state, log_path, "resume_skipped_identity_changed", process=current)
        return False
    if current["state"] not in ("T", "t"):
        _record(state_path, state, log_path, "resume_skipped_already_running", process=current)
        return False
    try:
        system.send_signal(args.training_pid, signal.SIGCONT)
    except OSError as exc:
        _record(state_path, state, log_path, "resume_signal_failed", error=str(exc),
                process=current, pid=args.training_pid)
        return False
    _record(state_path, state, log_path, "resume_sent", process_before=current,
            signal="SIGCONT", pid=args.training_pid)
    deadline = system.monotonic() + 2.
    while True:
        try:
            after = system.process_identity(args.training_pid)
            if (after["uid"] != os.getuid() or
                    after["start_ticks"] != args.expected_training_start_ticks or
                    EXPECTED_TRAINING_SCRIPT.as_posix() not in after["command"]):
                _record(state_path, state, log_path, "resume_confirmation_identity_changed", process=after)
                return False
            if after["state"] not in ("T", "t"):
                _record(state_path, state, log_path, "resume_confirmed", process=after)
                return True
            if system.monotonic() >= deadline:
                _record(state_path, state, log_path, "resume_not_confirmed", process=after)
                return False
            system.sleep(.1)
        except (OSError, SafetyError) as exc:
            _record(state_path, state, log_path, "resume_confirmation_unavailable", error=str(exc))
            return False


def run_wrapper(args: argparse.Namespace, system: System | None = None) -> int:
    global STOP_SIGNAL
    STOP_SIGNAL = None
    system = system or System()
    output = args.output.resolve()
    cache_root = Path(getattr(args, "cache_root", None) or (output / "caches")).resolve()
    seed_cache_root = getattr(args, "seed_cache_root", None)
    compile_strategy = getattr(args, "compile_strategy", "fullgraph")
    compile_threads = int(getattr(args, "compile_threads", 1))
    selected_backends = list(getattr(args, "backends", BACKEND_CHOICES))
    if output.exists() and any(output.iterdir()):
        raise SafetyError(f"Output directory is not fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "pause-state.json"
    log_path = output / "wrapper.log"
    state: dict[str, Any] = {"status": "preflight", "job_id": args.job_id,
                             "training_pid": args.training_pid,
                             "expected_training_start_ticks": args.expected_training_start_ticks,
                             "expected_gpu_uuid": args.expected_gpu_uuid,
                             "source_root": str(args.source_root.resolve()),
                             "output": str(output), "events": [], "started_at": utc_now(),
                             "cache_root": str(cache_root),
                             "seed_cache_root": str(seed_cache_root.resolve()) if seed_cache_root else None,
                             "compile_threads": compile_threads,
                             "compile_strategy": compile_strategy}
    state["backends"] = selected_backends
    atomic_json(state_path, state)
    paused_by_us = False
    child: Any = None
    exit_code = 1
    try:
        source_root = args.source_root.resolve(strict=True)
        benchmark_script = source_root / "tools/performance/three_backend_speed.sbatch"
        if not benchmark_script.is_file():
            raise SafetyError(f"Benchmark Slurm script missing from frozen root: {benchmark_script}")
        if os.environ.get("SLURM_JOB_ID") not in (None, str(args.job_id)):
            raise SafetyError(f"Current Slurm job {os.environ.get('SLURM_JOB_ID')} != requested job {args.job_id}")
        if STOP_SIGNAL is not None:
            raise SafetyError(f"Wrapper interrupted by signal {STOP_SIGNAL} before pausing training")
        identity = system.process_identity(args.training_pid)
        state["training_identity_initial"] = identity
        verify_identity(identity, args.training_pid, os.getuid(), args.expected_training_start_ticks)
        _record(state_path, state, log_path, "process_identity_verified", process=identity)
        batch_pids = system.batch_pids(args.job_id)
        if args.training_pid not in batch_pids:
            raise SafetyError(f"PID {args.training_pid} is absent from batch step process list for job {args.job_id}")
        _record(state_path, state, log_path, "batch_step_identity_verified", batch_pids=batch_pids)
        gpu = system.gpu_snapshot()
        if gpu["uuid"] != args.expected_gpu_uuid:
            raise SafetyError(f"Visible GPU UUID mismatch: expected {args.expected_gpu_uuid}, got {gpu['uuid']}")
        unexpected = sorted(set(gpu["compute_pids"]) - {args.training_pid})
        if unexpected:
            raise SafetyError(f"Other visible GPU compute PIDs are present: {unexpected}")
        if gpu["memory_free_mib"] < 10 * 1024:
            raise SafetyError(f"Visible GPU has less than 10 GiB free: {gpu['memory_free_mib']} MiB")
        _record(state_path, state, log_path, "gpu_identity_and_capacity_verified", gpu=gpu)

        identity = system.process_identity(args.training_pid)
        verify_identity(identity, args.training_pid, os.getuid(), args.expected_training_start_ticks)
        system.send_signal(args.training_pid, signal.SIGSTOP)
        paused_by_us = True
        state["pause_owned_by_wrapper"] = True
        _record(state_path, state, log_path, "pause_signal_sent", pid=args.training_pid,
                signal="SIGSTOP", process_before=identity)
        pause_confirmation = _wait_for_pause(system, args, state_path, state, log_path)

        command = ["bash", "tools/performance/three_backend_speed.sbatch", str(output),
                   "--cases", "ethanol-paper", "--backends", *selected_backends,
                   "--modes", *args.modes,
                   "--rounds", "5", "--warmup", "2", "--iterations", "10",
                   "--cache-policy", "shared-after-first", "--timeout", "900",
                   "--cache-root", str(cache_root),
                   "--compile-threads", str(compile_threads),
                   "--compile-strategy", compile_strategy]
        if seed_cache_root is not None:
            command.extend(["--seed-cache-root", str(seed_cache_root.resolve())])
        child_env = os.environ.copy()
        child_env.update({"EQUIV_DENS_PROJECT_ROOT": str(source_root),
                          "DENSNET_ALLOWED_PAUSED_GPU_PIDS": str(args.training_pid),
                          "DENSNET_MEASUREMENT_LOCK_SCOPE": "allocated-gpu"})
        if STOP_SIGNAL is not None:
            raise SafetyError(f"Wrapper interrupted by signal {STOP_SIGNAL} before benchmark launch")
        with (output / "benchmark.log").open("wb") as benchmark_log:
            _record(state_path, state, log_path, "benchmark_started", command=command,
                    pause_confirmation=pause_confirmation, log=str(output / "benchmark.log"),
                    max_seconds=args.max_seconds)
            child = system.popen(command, cwd=str(source_root), env=child_env,
                                 stdin=subprocess.DEVNULL, stdout=benchmark_log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
            deadline = system.monotonic() + args.max_seconds
            timed_out = False
            while True:
                if STOP_SIGNAL is not None:
                    _terminate_child(system, child)
                    exit_code = 128 + STOP_SIGNAL
                    state["status"] = "interrupted"
                    break
                remaining = deadline - system.monotonic()
                if remaining <= 0:
                    _terminate_child(system, child)
                    timed_out = True
                    exit_code = 124
                    state["status"] = "timeout"
                    break
                try:
                    returncode = child.wait(timeout=min(1., remaining))
                    exit_code = int(returncode)
                    state["status"] = "complete" if exit_code == 0 else "benchmark-failed"
                    break
                except subprocess.TimeoutExpired:
                    continue
        state["benchmark_returncode"] = getattr(child, "returncode", exit_code)
        _record(state_path, state, log_path, "benchmark_finished",
                returncode=state["benchmark_returncode"], timed_out=timed_out,
                interrupted_by_signal=STOP_SIGNAL)
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        _record(state_path, state, log_path, "wrapper_error", error=state["error"])
        exit_code = 1
    finally:
        try:
            if child is not None and child.poll() is None:
                _terminate_child(system, child)
        except BaseException as exc:
            state["child_cleanup_error"] = f"{type(exc).__name__}: {exc}"
            state["status"] = "child-cleanup-failed"
            exit_code = 1
            try:
                _record(state_path, state, log_path, "child_cleanup_error",
                        error=state["child_cleanup_error"])
            except BaseException:
                pass
        finally:
            try:
                resumed = _resume_owned_pause(system, args, state_path, state, log_path, paused_by_us)
            except BaseException as exc:
                resumed = False
                state["resume_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        state["training_resumed_by_wrapper"] = resumed
        if paused_by_us and not resumed and state.get("status") == "complete":
            state["status"] = "resume-not-confirmed"
            exit_code = 1
        state["finished_at"] = utc_now()
        atomic_json(state_path, state)
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--expected-training-start-ticks", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path,
                        help="node-local compiler cache root; defaults to OUTPUT/caches")
    parser.add_argument("--seed-cache-root", type=Path,
                        help="prior compiler cache tree to copy into a fresh cache root")
    parser.add_argument("--compile-threads", type=int, default=1)
    parser.add_argument("--compile-strategy", choices=("fullgraph", "regional-density"), default="fullgraph")
    parser.add_argument("--backends", nargs="+", choices=BACKEND_CHOICES, default=list(BACKEND_CHOICES))
    parser.add_argument("--max-seconds", type=int, default=5400)
    parser.add_argument("--modes", nargs="+", choices=MODE_CHOICES, required=True)
    args = parser.parse_args(argv)
    if args.training_pid <= 0 or args.expected_training_start_ticks <= 0 or args.max_seconds <= 0:
        parser.error("training PID, start ticks, and max-seconds must be positive")
    try:
        args.compile_threads = validate_compile_threads(args.compile_threads)
    except ValueError as exc:
        parser.error(str(exc))
    if args.cache_root is None:
        args.cache_root = args.output / "caches"
    if args.compile_strategy == "regional-density" and args.modes != ["density_update"]:
        parser.error("regional-density compile strategy requires --modes density_update only")
    if len(set(args.backends)) != len(args.backends):
        parser.error("--backends entries must be distinct")
    if "torch-eager" not in args.backends or len(args.backends) < 2:
        parser.error("--backends requires torch-eager and at least one distinct comparison backend")
    if not args.job_id.isdecimal():
        parser.error("job-id must be numeric")
    return args


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    args = parse_args(argv)
    try:
        return run_wrapper(args)
    except SafetyError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
