"""Control-flow tests for native staged handoff and checkpoint metadata."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "tools/training/train_native_staged.py"
SPEC = importlib.util.spec_from_file_location("train_native_staged", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@dataclass(frozen=True)
class FakeOptimizer:
    learning_rate: float = 0.1
    ema_decay: float | None = 0.9


class FakeLoss:
    def __init__(self, loss_weights, **_kwargs):
        self.loss_weights = dict(loss_weights)

    def configuration(self):
        return {"loss_weights": self.loss_weights}


class FakePipeline:
    def __init__(self, params=None, metadata=None):
        self.state = {"params": params or {"density_repr_model.w": np.array([1.]),
                                           "property_models.energy.w": np.array([5.])},
                      "buffers": {}}
        self.metadata = metadata or {}

    @classmethod
    def load(cls, _path):
        return cls()


class FakeTrainer:
    def __init__(self, pipeline, loss, optimizer, *, stage, seed=0, jit=True,
                 state=None, scheduler=None, **_kwargs):
        self.pipeline, self.loss, self.optimizer, self.stage = pipeline, loss, optimizer, stage
        params = dict(pipeline.state["params"])
        self.state = state or {"params": params, "ema": {k: v.copy() for k, v in params.items()},
                               "step": 0, "optimizer": {"count": 0}}
        self.loop = {"epoch": 0, "cursor": 0, "order": [], "batch_count": 0,
                     "rng": np.random.default_rng(seed).bit_generator.state,
                     "learning_rate": optimizer.learning_rate,
                     "schedule_calls": 0,
                     "loss_weights": dict(loss.loss_weights),
                     "validation_loss_weights": dict(loss.loss_weights)}
        self.seed, self.scheduler = seed, scheduler
        self.scheduler_enabled = scheduler is not None
        self.convert_inputs = False
        self.grid_scaling_start = 10000
        self.grid_scaling_annealing = 1.0
        self.rng = np.random.default_rng(seed)

    def step(self, batches, *, shuffle=True):
        del shuffle
        self.state["step"] += 1
        if self.stage == "density":
            self.state["params"]["density_repr_model.w"] += 1
            self.state["ema"]["density_repr_model.w"] = np.array([12.])
        else:
            self.state["params"]["property_models.energy.w"] += 1
        self.loop["epoch"] += 1
        return {"loss": 3., "energy_rmse": 2.}, True

    def validate(self, batches):
        del batches
        return {"loss": 1. if self.stage == "density" else .5, "energy_rmse": .2}

    def schedule(self, _metric):
        self.loop["schedule_calls"] += 1
        self.loop["learning_rate"] *= .5


FakeError = FakeLoss


def _config():
    return {"seed": 31, "shuffle": True, "initial_validation": True,
            "checkpoint_retained_archives": 2,
            "data": {"train": "train.npz", "validation": "validation.npz"},
            "density": {"epochs": 1, "validate_every": 1, "patience": 2,
                        "optimizer": {"learning_rate": .1},
                        "scheduler": {"patience": 1, "factor": .5},
                        "loss": {"loss_weights": {"density": 1.}}},
            "energy": {"epochs": 1, "validate_every": 1, "patience": 2,
                       "optimizer": {"learning_rate": .2},
                       "scheduler": {"patience": 1, "factor": .5},
                       "loss": {"loss_weights": {"energy": 1.}}}}


def _factory():
    return FakePipeline, FakeTrainer, FakeOptimizer, FakeError, None


def _loader(_path):
    return [{"fake": np.array([1])}]


def _prepare_paths(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.npz").write_bytes(b"train")
    (data / "validation.npz").write_bytes(b"validation")
    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_text("portable")
    return model, data


def test_ema_best_snapshot_matches_pipeline_and_training_params(monkeypatch, tmp_path):
    params = {"density_repr_model.w": np.array([2.]),
              "property_models.energy.w": np.array([3.])}
    pipeline = FakePipeline(params)
    trainer = FakeTrainer(pipeline, FakeLoss({"density": 1.}), FakeOptimizer(), stage="density")
    trainer.state["params"]["density_repr_model.w"] = np.array([8.])
    trainer.state["ema"]["density_repr_model.w"] = np.array([11.])
    trainer.pipeline.state["params"] = trainer.state["params"]
    captured = {}

    def save_checkpoint(path, saved_pipeline, state, **kwargs):
        captured["params"] = saved_pipeline.state["params"]
        captured["training_params"] = state["params"]

    import equiv_dens_jax.training as training
    monkeypatch.setattr(training, "save_checkpoint", save_checkpoint)
    runner._snapshot(tmp_path / "best", trainer, metadata={}, use_ema=True)
    assert captured["params"]["density_repr_model.w"][0] == 11.
    assert captured["training_params"]["density_repr_model.w"][0] == 11.
    assert trainer.pipeline.state["params"]["density_repr_model.w"][0] == 8.


def test_bundle_archive_retention_keeps_manifest_target_and_recent_predecessor(tmp_path):
    bundle = tmp_path / "checkpoint"
    bundle.mkdir()
    archives = []
    for index in range(4):
        archive = bundle / f"arrays-{index}.npz"
        archive.write_bytes(str(index).encode())
        os.utime(archive, ns=(index + 1, index + 1))
        archives.append(archive)
    (bundle / "manifest.json").write_text(json.dumps({"arrays": archives[3].name}))
    runner._prune_bundle_archives(bundle, retain=2)
    assert {item.name for item in bundle.glob("arrays-*.npz")} == {"arrays-2.npz", "arrays-3.npz"}


def test_best_ema_density_transfers_then_energy_resets_optimizer_and_freezes(tmp_path, monkeypatch):
    model, data = _prepare_paths(tmp_path)
    snapshots = {}

    def snapshot(path, trainer, *, metadata, use_ema=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "manifest.json").write_text("fake")
        params = {k: v.copy() for k, v in trainer.state["params"].items()}
        ema = {k: v.copy() for k, v in trainer.state["ema"].items()}
        if use_ema:
            params = ema
            ema = {k: v.copy() for k, v in params.items()}
        snapshots[path.resolve()] = (FakePipeline(params, metadata),
            {"params": params, "ema": ema, "step": trainer.state["step"],
             "optimizer": {"count": trainer.state["optimizer"]["count"]},
             "loop": dict(trainer.loop)}, trainer.stage, trainer.optimizer, trainer.loss)

    def load_checkpoint(path):
        return snapshots[Path(path).resolve()]

    monkeypatch.setattr(runner, "_snapshot", snapshot)
    output = tmp_path / "run"
    outcome = runner.run_staged(model_path=model, data_root=data, config=_config(),
        output=output, stage="both", trainer_factory=(*_factory()[:4], load_checkpoint),
        batch_loader=_loader)
    assert outcome == "completed"
    latest = snapshots[(output / "latest").resolve()]
    assert latest[2] == "energy"
    # The transfer uses validated EMA density weights (12), and starts a new energy optimizer.
    assert latest[1]["params"]["density_repr_model.w"][0] == 12.
    assert latest[1]["optimizer"]["count"] == 0
    # Only energy changed during energy training; density checksum stays equal to transfer checksum.
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    transition = next(row for row in records if row["event"] == "stage_change")
    validation = next(row for row in records if row["event"] == "validation" and row["stage"] == "energy")
    baselines = [row for row in records if row["event"] == "initial_validation"]
    assert transition["optimizer_reset"] is True
    assert validation["density_sha256"] == transition["density_sha256"]
    assert len(baselines) == 2 and all(row["selected"] is False for row in baselines)
    assert latest[0].metadata["native_staged_runner"]["best_validation"] == {"density": 1., "energy": .5}
    best_density = snapshots[(output / "best_density").resolve()]
    best_energy = snapshots[(output / "best_energy").resolve()]
    assert best_density[1]["loop"]["schedule_calls"] == 0
    assert best_energy[1]["loop"]["schedule_calls"] == 0
    assert latest[1]["loop"]["schedule_calls"] == 1


def test_resume_preserves_best_scores_and_terminal_failure_is_written(tmp_path, monkeypatch):
    model, data = _prepare_paths(tmp_path)
    output = tmp_path / "run"
    snapshots = {}

    def snapshot(path, trainer, *, metadata, use_ema=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "manifest.json").write_text("fake")
        params = {k: v.copy() for k, v in trainer.state["params"].items()}
        snapshots[path.resolve()] = (FakePipeline(params, metadata),
            {"params": params, "ema": trainer.state["ema"], "step": trainer.state["step"],
             "optimizer": trainer.state["optimizer"], "loop": dict(trainer.loop)},
            trainer.stage, trainer.optimizer, trainer.loss)

    def load_checkpoint(path):
        return snapshots[Path(path).resolve()]

    factory = (*_factory()[:4], load_checkpoint)
    monkeypatch.setattr(runner, "_snapshot", snapshot)
    runner.run_staged(model_path=model, data_root=data, config=_config(), output=output,
        stage="both", trainer_factory=factory, batch_loader=_loader)
    previous_best = snapshots[(output / "latest").resolve()][0].metadata["native_staged_runner"]["best_validation"]
    resumed_output = tmp_path / "resumed"
    runner.run_staged(model_path=output / "latest", data_root=data, config=_config(), output=resumed_output,
        stage="both", resume=True, trainer_factory=factory, batch_loader=_loader)
    resumed_best = snapshots[(output / "latest").resolve()][0].metadata["native_staged_runner"]["best_validation"]
    assert resumed_best == previous_best == {"density": 1., "energy": .5}
    assert json.loads((resumed_output / "terminal.json").read_text())["reason"] == "completed"
    assert (resumed_output / "best_density/manifest.json").exists()
    assert (resumed_output / "best_energy/manifest.json").exists()

    class FailingTrainer(FakeTrainer):
        def step(self, *_args, **_kwargs):
            raise FloatingPointError("nonfinite fake step")

    failing_output = tmp_path / "failed"
    with np.testing.assert_raises(RuntimeError):
        runner.run_staged(model_path=model, data_root=data, config=_config(), output=failing_output,
            stage="density", trainer_factory=(FakePipeline, FailingTrainer, FakeOptimizer, FakeError, load_checkpoint),
            batch_loader=_loader)
    terminal = json.loads((failing_output / "terminal.json").read_text())
    assert terminal["reason"] == "failed"
    assert "nonfinite fake step" in terminal["exception"]


def test_cli_preserves_detailed_terminal_from_run_staged(tmp_path, monkeypatch):
    output = tmp_path / "run"
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    expected = {"reason": "failed", "stage": "energy", "elapsed_seconds": 12.5,
                "exception": "full traceback retained"}

    def failing_run(**_kwargs):
        output.mkdir(parents=True, exist_ok=True)
        (output / "terminal.json").write_text(json.dumps(expected))
        (output / "metrics.jsonl").write_text('{"event":"error"}\n{"event":"terminal"}\n')
        raise RuntimeError("wrapper error")

    monkeypatch.setattr(sys, "argv", ["runner", "--model", "model", "--data", "data",
        "--config", str(config_path), "--output", str(output), "--stage", "energy"])
    monkeypatch.setattr(runner, "run_staged", failing_run)
    assert runner.main() == 1
    assert json.loads((output / "terminal.json").read_text()) == expected
    assert (output / "metrics.jsonl").read_text().count('"event":"terminal"') == 1
