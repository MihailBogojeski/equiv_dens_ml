"""Measurement acceptance must fail closed after failed or reused runs."""
import json
import os
from pathlib import Path
import sys

import pytest

from tools.performance import three_backend_speed as runner


def test_isolated_caches_are_separate_and_reuse_is_rejected(tmp_path):
    first, first_meta = runner.cache_environment(
        tmp_path, "ethanol-small", "torch-compile", 0, "ef", "isolated")
    second, second_meta = runner.cache_environment(
        tmp_path, "ethanol-paper", "torch-compile", 0, "ef", "isolated")
    assert first_meta["cache_initially_empty"] is True
    assert first_meta["cache_reused"] is False
    assert next(v for v in first if v.startswith("TORCHINDUCTOR_CACHE_DIR=")) != next(
        v for v in second if v.startswith("TORCHINDUCTOR_CACHE_DIR="))
    _, later_round = runner.cache_environment(
        tmp_path, "ethanol-small", "torch-compile", 1, "ef", "isolated")
    assert first_meta["cache_directory"] != later_round["cache_directory"]
    with pytest.raises(FileExistsError):
        runner.cache_environment(tmp_path, "ethanol-small", "torch-compile", 0, "ef", "isolated")


def test_shared_after_first_cache_is_reused_by_a_later_fresh_process(tmp_path):
    first_env, first_meta = runner.cache_environment(
        tmp_path, "ethanol-small", "torch-compile", 0, "ef", "shared-after-first")
    assert first_meta["cache_initially_empty"] is True
    assert first_meta["cache_reused"] is False
    assert first_meta["cache_files_before"] == 0
    first_cache = Path(str(first_meta["cache_directory"]))
    assignments = dict(item.split("=", 1) for item in first_env[1:])
    first_result = tmp_path / "first-process.json"
    second_result = tmp_path / "second-process.json"
    child_code = (
        "import json,os,pathlib; "
        "cache=pathlib.Path(os.environ['TORCHINDUCTOR_CACHE_DIR']); "
        "marker=cache/'compiled-artifact'; "
        "before=marker.exists(); marker.write_text(str(os.getpid())); "
        "pathlib.Path(os.environ['TEST_PROCESS_RESULT']).write_text(json.dumps({'pid':os.getpid(),'saw_cache':before}))"
    )
    first_child_env = {**os.environ, **assignments, "TEST_PROCESS_RESULT": str(first_result)}
    first_child = runner.run_child([sys.executable, "-c", child_code],
                                   tmp_path / "first-process.log", 10, first_child_env)
    assert first_child["status"] == "ok"
    assert json.loads(first_result.read_text())["saw_cache"] is False
    assert (first_cache / "inductor" / "compiled-artifact").is_file()

    later_env, later_meta = runner.cache_environment(
        tmp_path, "ethanol-small", "torch-compile", 1, "ef", "shared-after-first")
    assert later_meta["cache_directory"] == first_meta["cache_directory"]
    assert later_meta["cache_initially_empty"] is False
    assert later_meta["cache_reused"] is True
    assert later_meta["cache_files_before"] >= 1
    later_assignments = dict(item.split("=", 1) for item in later_env[1:])
    assert later_assignments["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert later_assignments["JAX_ENABLE_COMPILATION_CACHE"] == "true"
    second_child_env = {**os.environ, **later_assignments, "TEST_PROCESS_RESULT": str(second_result)}
    second_child = runner.run_child([sys.executable, "-c", child_code],
                                    tmp_path / "second-process.log", 10, second_child_env)
    assert second_child["status"] == "ok"
    observed = json.loads(second_result.read_text())
    assert observed["saw_cache"] is True
    assert observed["pid"] != json.loads(first_result.read_text())["pid"]


def test_explicit_cache_root_and_compiler_threads_are_forwarded(tmp_path):
    cache_root = tmp_path / "node-local" / "compiler-cache"
    env, metadata = runner.cache_environment(
        tmp_path / "report", "ethanol-small", "torch-compile", 0, "density",
        "shared-after-first", cache_root, compile_threads=3)
    assignments = dict(item.split("=", 1) for item in env[1:])
    assert metadata["cache_root"] == str(cache_root.resolve())
    assert metadata["compile_threads"] == 3
    assert assignments["TORCHINDUCTOR_COMPILE_THREADS"] == "3"
    assert metadata["cache_initially_empty"] is True

    report_root = tmp_path / "run-report"
    def fake_child(command, log, timeout):
        assert "--compile-strategy" in command
        assert command[command.index("--compile-strategy") + 1] == "fullgraph"
        assert "TORCHINDUCTOR_COMPILE_THREADS=3" in command
        worker_output = Path(command[command.index("--output") + 1])
        worker_output.write_text(json.dumps({"times_ms": [1.], "cold_first_call_seconds": 2.}))
        return {"status": "ok", "wall_seconds": 1.}
    original = runner.run_child
    runner.run_child = fake_child
    try:
        result = runner.invoke_arm(report_root, "ethanol-small", "density_update", 0,
                                   "torch-compile", 0, 1, 10, "shared-after-first",
                                   cache_root, 3, False, "fullgraph")
    finally:
        runner.run_child = original
    annotated = json.loads(Path(result["file"]).read_text())
    assert annotated["orchestration"]["compile_threads"] == 3
    assert annotated["orchestration"]["cache_root"] == str(cache_root.resolve())


def test_seeded_cache_provenance_is_not_reported_as_cold_and_dirty_dest_refused(tmp_path):
    seed_root = tmp_path / "prior-report" / "caches"
    source = seed_root / "ethanol-small" / "torch-compile" / "density_update"
    (source / "inductor").mkdir(parents=True)
    (source / "inductor" / "compiled.bin").write_bytes(b"cache")
    (source / "densnet").mkdir()
    (source / "densnet" / "model-output.bin").write_bytes(b"not a compiler cache")
    destination = tmp_path / "scratch" / "run-cache"
    seeded = runner.seed_compiler_caches(seed_root, destination,
                                         ["ethanol-small"], ["density_update"])
    entry = seeded["entries"]["ethanol-small/torch-compile/density_update"]
    assert entry["seeded"] is True
    assert entry["compiler_cache_directories_copied"] == ["inductor"]
    assert (destination / "ethanol-small/torch-compile/density_update/inductor/compiled.bin").read_bytes() == b"cache"
    assert not (destination / "ethanol-small/torch-compile/density_update/densnet").exists()

    _, metadata = runner.cache_environment(
        tmp_path / "report", "ethanol-small", "torch-compile", 0, "density_update",
        "shared-after-first", destination,
        seeded=True)
    assert metadata["cache_seeded"] is True
    assert metadata["cache_initially_empty"] is False
    assert "not cold" in metadata["first_call_interpretation"]

    with pytest.raises(FileExistsError, match="fresh or empty"):
        runner.seed_compiler_caches(seed_root, destination, ["ethanol-small"], ["density_update"])


def test_seeded_startup_summary_has_separate_category(tmp_path):
    rows = []
    for index in range(5):
        is_seeded = index == 0
        arms = {}
        eager_seconds = 8. if is_seeded else 3. + index
        compiled_seconds = 20. if is_seeded else 8. + 2 * index
        for backend, seconds in (("torch-eager", eager_seconds),
                                 ("torch-compile", compiled_seconds)):
            report_path = tmp_path / f"seed-{index}-{backend}.json"
            report_path.write_text(json.dumps({
                "times_ms": [2.], "cold_first_call_seconds": seconds,
                "orchestration": {"cache_seeded": is_seeded,
                                  "cache_initially_empty": False,
                                  "cache_reused": not is_seeded},
            }))
            arms[backend] = {"status": "ok", "file": str(report_path)}
        rows.append({"arms": arms,
                     "comparisons": {"torch-compile-vs-torch-eager": {"pass": True}}})
    summary = runner.summarize_cell({"rounds": rows})["torch-compile"]
    assert summary["eager_seeded_first_call_count"] == 1
    assert summary["candidate_seeded_first_call_median_seconds"] == 20.
    assert summary["eager_first_uncached_call_median_seconds"] is None
    assert summary["candidate_first_uncached_call_median_seconds"] is None
    assert summary["eager_warm_cache_first_call_median_seconds"] == 5.5
    assert summary["candidate_warm_cache_first_call_median_seconds"] == 13.
    assert summary["eager_warm_cache_first_call_count"] == 4
    assert summary["candidate_warm_cache_first_call_count"] == 4


def test_compile_thread_validation_default_and_slurm_limit():
    assert runner.validate_compile_threads(1, {}) == 1
    assert runner.validate_compile_threads(3, {"SLURM_CPUS_PER_TASK": "4"}) == 3
    with pytest.raises(ValueError, match="positive"):
        runner.validate_compile_threads(0, {})
    with pytest.raises(ValueError, match="exceeds SLURM_CPUS_PER_TASK"):
        runner.validate_compile_threads(5, {"SLURM_CPUS_PER_TASK": "4"})


def test_backend_selection_scopes_rotated_order_and_jax_only_summary(tmp_path):
    selected = ("torch-eager", "jax")
    assert runner.arm_order(0, selected) == ("torch-eager", "jax")
    assert runner.arm_order(1, selected) == ("jax", "torch-eager")
    assert runner.arm_order(0) == runner.BACKENDS
    rows = []
    for index in range(5):
        arms = {}
        for backend, latency in (("torch-eager", 20.), ("jax", 10.)):
            report = tmp_path / f"selected-{index}-{backend}.json"
            report.write_text(json.dumps({"times_ms": [latency], "cold_first_call_seconds": 1.}))
            arms[backend] = {"status": "ok", "file": str(report)}
        rows.append({"arms": arms, "comparisons": {"jax-vs-torch-eager": {"pass": True}}})
    cell = {"backends": list(selected), "rounds": rows}
    summary = runner.summarize_cell(cell)
    assert list(summary) == ["jax"]
    assert summary["jax"]["valid_pairs"] == 5
    assert summary["jax"]["valid_for_speed_claim"] is True


@pytest.mark.parametrize("backends", [["jax"], ["torch-eager"],
                                      ["torch-eager", "jax", "jax"]])
def test_backend_selection_requires_eager_two_distinct(backends, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["three_backend_speed", "--output", str(tmp_path),
                                      "--backends", *backends])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_default_backend_configuration_still_selects_all_three(tmp_path, monkeypatch):
    report = tmp_path / "default-report"
    monkeypatch.setattr(sys, "argv", ["three_backend_speed", "--output", str(report)])
    monkeypatch.setattr(runner, "run_child", lambda *args, **kwargs: {"status": "exit", "returncode": 1})
    assert runner.main() == 1
    manifest = json.loads((report / "manifest.json").read_text())
    register = json.loads((report / "register.json").read_text())
    assert manifest["configuration"]["backends"] == list(runner.BACKENDS)
    assert register["backends"] == list(runner.BACKENDS)


@pytest.mark.parametrize("writes_gate", [False, True])
def test_failed_comparison_never_accepts_passing_gate(tmp_path, monkeypatch, writes_gate):
    arms = []
    for backend in ("torch-eager", "torch-compile"):
        report = tmp_path / (backend + ".json")
        report.write_text(json.dumps({"times_ms": [1.], "cold_first_call_seconds": 2.}))
        arms.append({"backend": backend, "file": str(report), "status": "ok"})
    gate_path = tmp_path / "gates/ethanol-paper-ef-round-01-torch-eager-vs-torch-compile.json"
    gate_path.parent.mkdir()
    gate_path.write_text(json.dumps({"pass": True, "rows": [{"field": "stale"}]}))
    def failed_child(command, log, timeout):
        assert not gate_path.exists(), "An earlier gate must be removed before the new comparison"
        if writes_gate:
            gate_path.write_text(json.dumps({"pass": True, "rows": [{"field": "current"}]}))
        return {"status": "timeout", "censored": True, "wall_seconds": 10.}
    monkeypatch.setattr(runner, "run_child", failed_child)
    result = runner.compare_pair(tmp_path, *arms, "ethanol-paper", "ef", 0, 10)
    assert result["pass"] is False
    assert json.loads(gate_path.read_text())["pass"] is False
    if writes_gate:
        assert result["rows"][0]["field"] == "current"


def test_existing_run_refused_before_launch(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr("sys.argv", ["three_backend_speed", "--output", str(tmp_path)])
    monkeypatch.setattr(runner, "run_child", lambda *a, **k: pytest.fail("Existing run was launched"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_partial_numerical_qualification_cannot_become_headline_speed(tmp_path):
    rows = []
    for index, passed in enumerate((True, False)):
        arms = {}
        for backend, latency in (("torch-eager", 20.), ("torch-compile", 2.)):
            report = tmp_path / f"{index}-{backend}.json"
            report.write_text(json.dumps({"times_ms": [latency], "cold_first_call_seconds": 1.}))
            arms[backend] = {"status": "ok", "file": str(report)}
        rows.append({"arms": arms, "comparisons": {"torch-compile-vs-torch-eager": {"pass": passed}}})
    summary = runner.summarize_cell({"rounds": rows})["torch-compile"]
    assert summary["valid_pairs"] == 1
    assert summary["valid_for_speed_claim"] is False


def test_summary_separates_uncached_from_cache_assisted_first_calls(tmp_path):
    rows = []
    call_seconds = {
        0: {"torch-eager": 10., "torch-compile": 30.},
        1: {"torch-eager": 4., "torch-compile": 12.},
        2: {"torch-eager": 6., "torch-compile": 14.},
    }
    for round_index, values in call_seconds.items():
        arms = {}
        for backend, cold_seconds in values.items():
            report = tmp_path / f"round-{round_index}-{backend}.json"
            report.write_text(json.dumps({
                "times_ms": [2., 2.],
                "cold_first_call_seconds": cold_seconds,
                "orchestration": {"cache_initially_empty": round_index == 0,
                                  "cache_reused": round_index > 0},
            }))
            arms[backend] = {"status": "ok", "file": str(report)}
        rows.append({"arms": arms,
                     "comparisons": {"torch-compile-vs-torch-eager": {"pass": True}}})

    summary = runner.summarize_cell({"rounds": rows})["torch-compile"]
    assert summary["eager_first_uncached_call_median_seconds"] == 10.
    assert summary["candidate_first_uncached_call_median_seconds"] == 30.
    assert summary["eager_uncached_first_call_count"] == 1
    assert summary["candidate_uncached_first_call_count"] == 1
    assert summary["eager_warm_cache_first_call_median_seconds"] == 5.
    assert summary["candidate_warm_cache_first_call_median_seconds"] == 13.
    assert summary["eager_warm_cache_first_call_count"] == 2
    assert summary["candidate_warm_cache_first_call_count"] == 2
