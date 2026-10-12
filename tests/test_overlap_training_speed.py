"""Safety checks for pausing only the verified training process."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.performance import overlap_training_speed as wrapper


class FakeChild:
    pid = 90001

    def __init__(self, system):
        self.system = system
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        self.system.clock += float(timeout or 0.)
        raise subprocess.TimeoutExpired("fake benchmark", timeout)


class FakeSystem:
    def __init__(self, pid, ticks, gpu_uuid, script, state="R"):
        self.pid = pid
        self.ticks = ticks
        self.gpu_uuid = gpu_uuid
        self.script = script
        self.process_state = state
        self.signals = []
        self.child_signals = []
        self.clock = 0.
        self.child = None
        self.command = None
        self.environment = None
        self.fail_child_termination = False

    def process_identity(self, pid):
        return {"pid": pid, "uid": os.getuid(), "state": self.process_state,
                "start_ticks": self.ticks, "command": ["python", self.script],
                "command_line": f"python {self.script}"}

    def batch_pids(self, job_id):
        assert job_id == "5053839"
        return [self.pid]

    def gpu_snapshot(self):
        return {"uuid": self.gpu_uuid, "memory_free_mib": 40000,
                "utilization_pct": 0., "compute_pids": [self.pid]}

    def send_signal(self, pid, signum):
        self.signals.append((pid, signum))
        if signum == signal.SIGSTOP:
            self.process_state = "T"
        elif signum == signal.SIGCONT:
            self.process_state = "R"

    def popen(self, command, **kwargs):
        self.command = command
        self.environment = kwargs["env"]
        assert kwargs["start_new_session"] is True
        self.child = FakeChild(self)
        return self.child

    def kill_child_group(self, pid, signum):
        self.child_signals.append((pid, signum))
        if self.fail_child_termination:
            raise RuntimeError("mock child-group termination failure")
        self.child.returncode = -signum

    def monotonic(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds


def make_args(tmp_path, pid=1635119, ticks=817608690, gpu_uuid="GPU-test", max_seconds=2):
    source = tmp_path / "frozen-source"
    benchmark = source / "tools/performance/three_backend_speed.sbatch"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("# fixture for wrapper preflight\n")
    output = tmp_path / "report"
    return SimpleNamespace(job_id="5053839", training_pid=pid,
                           expected_training_start_ticks=ticks,
                           expected_gpu_uuid=gpu_uuid, source_root=source,
                           output=output, max_seconds=max_seconds,
                           modes=["ef", "calculator"])


def test_identity_mismatch_refuses_before_any_signal(tmp_path, monkeypatch):
    script = tmp_path / "train_deq_modular.py"
    script.write_text("# expected script\n")
    monkeypatch.setattr(wrapper, "EXPECTED_TRAINING_SCRIPT", script)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = make_args(tmp_path)
    system = FakeSystem(args.training_pid, args.expected_training_start_ticks + 1,
                        args.expected_gpu_uuid, str(script))

    result = wrapper.run_wrapper(args, system)

    assert result == 1
    assert system.signals == []
    state = json.loads((args.output / "pause-state.json").read_text())
    assert state["status"] == "failed"
    assert "start tick changed" in state["error"]


def test_initially_stopped_process_is_never_claimed_or_resumed(tmp_path, monkeypatch):
    script = tmp_path / "train_deq_modular.py"
    script.write_text("# expected script\n")
    monkeypatch.setattr(wrapper, "EXPECTED_TRAINING_SCRIPT", script)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = make_args(tmp_path)
    system = FakeSystem(args.training_pid, args.expected_training_start_ticks,
                        args.expected_gpu_uuid, str(script), state="T")

    result = wrapper.run_wrapper(args, system)

    assert result == 1
    assert system.signals == []
    state = json.loads((args.output / "pause-state.json").read_text())
    assert "already stopped" in state["error"]
    assert state.get("training_resumed_by_wrapper") is False


def test_watchdog_timeout_terminates_only_child_and_resumes_owned_training_pid(tmp_path, monkeypatch):
    script = tmp_path / "train_deq_modular.py"
    script.write_text("# expected script\n")
    monkeypatch.setattr(wrapper, "EXPECTED_TRAINING_SCRIPT", script)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = make_args(tmp_path, max_seconds=2)
    args.cache_root = tmp_path / "node-local-cache"
    args.seed_cache_root = tmp_path / "seed-caches"
    args.compile_threads = 3
    args.compile_strategy = "regional-density"
    args.backends = ["torch-eager", "jax"]
    system = FakeSystem(args.training_pid, args.expected_training_start_ticks,
                        args.expected_gpu_uuid, str(script))

    result = wrapper.run_wrapper(args, system)

    assert result == 124
    assert system.signals == [(args.training_pid, signal.SIGSTOP),
                              (args.training_pid, signal.SIGCONT)]
    assert system.child_signals == [(system.child.pid, signal.SIGTERM)]
    assert system.command[:3] == ["bash", "tools/performance/three_backend_speed.sbatch",
                                  str(args.output.resolve())]
    assert "--cache-root" in system.command
    assert system.command[system.command.index("--cache-root") + 1] == str(args.cache_root.resolve())
    assert system.command[system.command.index("--seed-cache-root") + 1] == str(args.seed_cache_root.resolve())
    assert system.command[system.command.index("--compile-threads") + 1] == "3"
    assert system.command[system.command.index("--compile-strategy") + 1] == "regional-density"
    assert system.command[system.command.index("--backends") + 1:system.command.index("--modes")] == ["torch-eager", "jax"]
    assert system.environment["DENSNET_ALLOWED_PAUSED_GPU_PIDS"] == str(args.training_pid)
    assert system.environment["DENSNET_MEASUREMENT_LOCK_SCOPE"] == "allocated-gpu"
    state = json.loads((args.output / "pause-state.json").read_text())
    events = [event["event"] for event in state["events"]]
    assert "pause_confirmed" in events
    assert "benchmark_finished" in events
    assert "resume_confirmed" in events
    assert state["status"] == "timeout"
    assert state["training_resumed_by_wrapper"] is True
    assert state["cache_root"] == str(args.cache_root.resolve())
    assert state["compile_threads"] == 3
    assert state["backends"] == ["torch-eager", "jax"]


def test_cleanup_does_not_resume_pid_after_identity_changes(tmp_path, monkeypatch):
    script = tmp_path / "train_deq_modular.py"
    script.write_text("# expected script\n")
    monkeypatch.setattr(wrapper, "EXPECTED_TRAINING_SCRIPT", script)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = make_args(tmp_path, max_seconds=2)
    system = FakeSystem(args.training_pid, args.expected_training_start_ticks,
                        args.expected_gpu_uuid, str(script))
    original_popen = system.popen

    def popen_and_reuse_pid(command, **kwargs):
        child = original_popen(command, **kwargs)
        system.ticks += 1
        return child

    system.popen = popen_and_reuse_pid
    result = wrapper.run_wrapper(args, system)

    assert result == 124
    assert system.signals == [(args.training_pid, signal.SIGSTOP)]
    state = json.loads((args.output / "pause-state.json").read_text())
    assert state["training_resumed_by_wrapper"] is False
    assert any(event["event"] == "resume_skipped_identity_changed" for event in state["events"])


def test_child_termination_error_still_runs_training_resume_cleanup(tmp_path, monkeypatch):
    script = tmp_path / "train_deq_modular.py"
    script.write_text("# expected script\n")
    monkeypatch.setattr(wrapper, "EXPECTED_TRAINING_SCRIPT", script)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = make_args(tmp_path, max_seconds=1)
    system = FakeSystem(args.training_pid, args.expected_training_start_ticks,
                        args.expected_gpu_uuid, str(script))
    system.fail_child_termination = True

    result = wrapper.run_wrapper(args, system)

    assert result == 1
    assert system.signals == [(args.training_pid, signal.SIGSTOP),
                              (args.training_pid, signal.SIGCONT)]
    state = json.loads((args.output / "pause-state.json").read_text())
    assert state["status"] == "child-cleanup-failed"
    assert state["training_resumed_by_wrapper"] is True
    assert state["child_cleanup_error"] == "RuntimeError: mock child-group termination failure"


def test_cli_compiler_threads_respect_slurm_allocation(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    base = ["--job-id", "5053839", "--training-pid", "1635119",
            "--expected-training-start-ticks", "817608690", "--expected-gpu-uuid", "GPU-test",
            "--source-root", str(tmp_path), "--output", str(tmp_path / "report"), "--modes", "ef"]
    args = wrapper.parse_args([*base, "--compile-threads", "2"])
    assert args.compile_threads == 2
    assert args.cache_root == args.output / "caches"
    try:
        wrapper.parse_args([*base, "--compile-threads", "3"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("compiler thread request above SLURM allocation was accepted")


def test_absolute_script_help_imports_without_pythonpath(tmp_path):
    script = Path(wrapper.__file__).resolve()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            env=environment, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--compile-threads" in result.stdout


def test_regional_density_cli_rejects_non_update_modes_before_pause(monkeypatch, tmp_path):
    base = ["--job-id", "5053839", "--training-pid", "1635119",
            "--expected-training-start-ticks", "817608690", "--expected-gpu-uuid", "GPU-test",
            "--source-root", str(tmp_path), "--output", str(tmp_path / "report"),
            "--modes", "ef", "--compile-strategy", "regional-density"]
    try:
        wrapper.parse_args(base)
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("regional-density accepted non-density_update modes")


def test_wrapper_requires_distinct_eager_comparison_backend(tmp_path):
    base = ["--job-id", "5053839", "--training-pid", "1635119",
            "--expected-training-start-ticks", "817608690", "--expected-gpu-uuid", "GPU-test",
            "--source-root", str(tmp_path), "--output", str(tmp_path / "report"), "--modes", "ef"]
    for backend_args in (("jax",), ("torch-eager",), ("torch-eager", "jax", "jax")):
        try:
            wrapper.parse_args([*base, "--backends", *backend_args])
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError(f"invalid backend selection accepted: {backend_args}")
