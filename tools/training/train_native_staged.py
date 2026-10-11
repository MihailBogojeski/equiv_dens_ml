#!/usr/bin/env python3
"""Generic two-stage runner for the committed native JAX Pipeline/Trainer API."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback


_STOP = False


def _request_stop(_signum, _frame):
    global _STOP
    _STOP = True


def _canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _path_hash(path):
    """Hash input files/directories deterministically, including bundle manifests."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files under input path: {path}")
    for file in files:
        relative = file.name if path.is_file() else file.relative_to(path).as_posix()
        digest.update(relative.encode() + b"\0")
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _density_checksum(parameters):
    """Stable checksum for every parameter except the energy head."""
    digest = hashlib.sha256()
    selected = [(name, value) for name, value in parameters.items()
                if not name.startswith("property_models.energy.")]
    if not selected:
        raise ValueError("No density or shared parameters found in model state")
    import numpy as np
    for name, value in sorted(selected):
        array = np.ascontiguousarray(value)
        digest.update(name.encode() + b"\0" + array.dtype.str.encode() + b"\0")
        digest.update(json.dumps(array.shape).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _resolve_data(config, data_root, train_override, validation_override):
    data = config.get("data", {})
    root = Path(data_root).resolve()
    train = train_override or data.get("train")
    validation = validation_override or data.get("validation")
    if train is None:
        candidates = [root / "train", root / "training", root]
        train = next((path for path in candidates if path.exists() and
                      (path.is_file() or (path / "manifest.json").exists())), None)
    else:
        train = Path(train)
        if not train.is_absolute():
            train = root / train
    if validation is not None:
        validation = Path(validation)
        if not validation.is_absolute():
            validation = root / validation
    else:
        validation = next((root / name for name in ("validation", "valid", "val")
                           if (root / name).exists()), None)
    if train is None or not Path(train).exists():
        raise ValueError("Training data not found; set data.train in config or pass --train")
    if validation is None or not Path(validation).exists():
        raise ValueError("Validation data is required for best-checkpoint selection; set data.validation or pass --validation")
    return Path(train).resolve(), Path(validation).resolve()


def _write_jsonl(stream, record):
    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def _prune_bundle_archives(path, retain=2):
    """Keep the manifest archive and recent predecessors in runner-owned bundles.

    Readers follow the manifest's content hash. Two archives let a reader that
    already opened the previous manifest finish while repeated epoch saves stay
    bounded; this applies only to this runner's checkpoint directories.
    """
    if retain < 1:
        raise ValueError("checkpoint_retained_archives must be at least one")
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return
    current = json.loads(manifest_path.read_text()).get("arrays")
    archives = sorted(path.glob("arrays-*.npz"), key=lambda item: item.stat().st_mtime_ns, reverse=True)
    keep = {item.name for item in archives[:retain]}
    if current:
        keep.add(current)
    for archive in archives:
        if archive.name not in keep:
            archive.unlink()


def _snapshot(path, trainer, *, metadata, use_ema=False):
    """Write a native checkpoint; optional EMA selection becomes its params."""
    from equiv_dens_jax.training import save_checkpoint
    state = dict(trainer.state, loop=trainer.loop)
    if use_ema:
        state = dict(state, params=dict(state["ema"]), ema=dict(state["ema"]))
    native_loop = {"scheduler": trainer.scheduler if trainer.scheduler_enabled else None,
                   "seed": trainer.seed, "convert_inputs": trainer.convert_inputs,
                   "grid_scaling_start": trainer.grid_scaling_start,
                   "grid_scaling_annealing": trainer.grid_scaling_annealing,
                   "jit_validation": getattr(trainer, "jit_validation", False)}
    original = trainer.pipeline.state
    if use_ema:
        trainer.pipeline.state = {**original, "params": state["params"]}
    try:
        save_checkpoint(path, trainer.pipeline, state, stage=trainer.stage,
                        optimizer=trainer.optimizer, loss=trainer.loss,
                        metadata={**metadata, "native_loop": native_loop})
    finally:
        trainer.pipeline.state = original
    runner_metadata = metadata.get("native_staged_runner", {})
    _prune_bundle_archives(path, int(runner_metadata.get("checkpoint_retained_archives", 2)))


def run_staged(*, model_path, data_root, config, output, stage="both", train_path=None,
               validation_path=None, resume=False, jit=True, trainer_factory=None,
               batch_loader=None):
    """Run density, energy, or density→energy, supporting exact native resume."""
    if stage not in {"density", "energy", "both"}:
        raise ValueError("stage must be density, energy or both")
    retained_archives = int(config.get("checkpoint_retained_archives", 2))
    if retained_archives < 1:
        raise ValueError("checkpoint_retained_archives must be at least one")
    for name in ("density", "energy"):
        if name not in config:
            raise ValueError(f"Config is missing the {name!r} stage")
        item = config[name]
        if int(item.get("epochs", 0)) <= 0 or int(item.get("validate_every", 1)) <= 0:
            raise ValueError(f"{name} epochs and validate_every must be positive")
        if int(item.get("patience", 0)) < 0:
            raise ValueError(f"{name} patience cannot be negative")
        if "optimizer" not in item or "loss" not in item:
            raise ValueError(f"{name} stage requires optimizer and loss configuration")
    train_path, validation_path = _resolve_data(config, data_root, train_path, validation_path)
    if batch_loader is None:
        from equiv_dens_jax.data import load_batches
        batch_loader = load_batches
    if trainer_factory is None:
        from equiv_dens_jax.losses import ErrorDict
        from equiv_dens_jax.pipeline import Pipeline
        from equiv_dens_jax.trainer import Trainer
        from equiv_dens_jax.training import Optimizer, load_checkpoint
        trainer_factory = (Pipeline, Trainer, Optimizer, ErrorDict, load_checkpoint)
    Pipeline, Trainer, Optimizer, ErrorDict, load_checkpoint = trainer_factory
    train_batches = batch_loader(train_path)
    validation_batches = batch_loader(validation_path)
    if not train_batches or not validation_batches:
        raise ValueError("Training and validation data must both contain batches")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.jsonl").touch(exist_ok=True)
    config_hash = _canonical_hash(config)
    data_hashes = {"train": _path_hash(train_path), "validation": _path_hash(validation_path)}
    original_model_hash = _path_hash(model_path)
    seed = int(config.get("seed", 0))
    shuffle = bool(config.get("shuffle", True))
    runner_state = {"best_validation": {}, "source_model_sha256": original_model_hash,
                    "config_sha256": config_hash, "data_sha256": data_hashes,
                    "initial_density_sha256": None, "bad_validations": 0,
                    "best_checkpoints": {},
                    "initial_validation_stages": [],
                    "early_stopped": False}
    if resume:
        pipeline, state, saved_stage, optimizer, loss = load_checkpoint(model_path)
        stored = pipeline.metadata.get("native_staged_runner")
        if not stored:
            raise ValueError("Resume checkpoint lacks native staged-runner metadata")
        if stored.get("config_sha256") != config_hash:
            raise ValueError("Resume config differs from the checkpoint; stage optimizer/loss and training budget cannot be replaced")
        if stored.get("data_sha256") != data_hashes:
            raise ValueError("Resume train/validation data hashes differ from checkpoint")
        runner_state.update({"best_validation": stored.get("best_validation", {}),
                             "source_model_sha256": stored.get("source_model_sha256"),
                             "initial_density_sha256": stored.get("initial_density_sha256"),
                             "bad_validations": int(stored.get("bad_validations", 0)),
                             "best_checkpoints": stored.get("best_checkpoints", {}),
                             "initial_validation_stages": stored.get("initial_validation_stages", [])})
        for best_stage, old_path in list(runner_state["best_checkpoints"].items()):
            destination = output / f"best_{best_stage}"
            source = Path(old_path)
            if not (destination / "manifest.json").exists() and (source / "manifest.json").exists():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            runner_state["best_checkpoints"][best_stage] = str(destination)
        stage_order = ["density", "energy"] if stage == "both" else [stage]
        if saved_stage not in stage_order:
            raise ValueError(f"Checkpoint stage {saved_stage!r} is outside requested stage path {stage_order}")
        current = saved_stage
        stage_config = config[current]
        if optimizer != Optimizer(**stage_config["optimizer"]) or loss.configuration() != ErrorDict(**stage_config["loss"]).configuration():
            raise ValueError("Resume checkpoint optimizer/loss differs from effective stage config")
        native_loop = pipeline.metadata.get("native_loop", {})
        trainer = Trainer(pipeline, loss, optimizer, stage=current, state=state,
                          seed=native_loop.get("seed", seed), jit=jit,
                          convert_inputs=native_loop.get("convert_inputs", False),
                          scheduler=native_loop.get("scheduler"),
                          grid_scaling_start=native_loop.get("grid_scaling_start",10000),
                          grid_scaling_annealing=native_loop.get("grid_scaling_annealing",1.0),
                          jit_validation=native_loop.get("jit_validation",config.get("jit_validation",False)))
    else:
        pipeline = Pipeline.load(model_path)
        stage_order = ["density", "energy"] if stage == "both" else [stage]
        current = stage_order[0]
        stage_config = config[current]
        optimizer = Optimizer(**stage_config["optimizer"])
        loss = ErrorDict(**stage_config["loss"])
        trainer = Trainer(pipeline, loss, optimizer, stage=current, seed=seed, jit=jit,
                          scheduler=stage_config.get("scheduler"),
                          jit_validation=bool(config.get("jit_validation",False)))
        state = None
    if current == "energy" and stage == "both" and runner_state["initial_density_sha256"] is None:
        raise ValueError("Energy-stage resume is missing selected density checksum provenance")
    if current == "energy" and stage == "both":
        runner_state["initial_density_sha256"] = runner_state["initial_density_sha256"] or _density_checksum(trainer.state["params"])
    best_score = runner_state["best_validation"].get(current, math.inf)
    bad_validations = int(runner_state.get("bad_validations", 0))
    start_time = time.monotonic()
    metrics_path = output / "metrics.jsonl"

    def metadata():
        return {"native_staged_runner": {
            "schema": 1, "config_sha256": config_hash, "data_sha256": data_hashes,
            "source_model_sha256": runner_state["source_model_sha256"],
            "best_validation": runner_state["best_validation"],
            "best_checkpoints": runner_state["best_checkpoints"],
            "initial_density_sha256": runner_state["initial_density_sha256"],
            "bad_validations": bad_validations,
            "initial_validation_stages": runner_state["initial_validation_stages"],
            "checkpoint_retained_archives": retained_archives,
            "stage2_shuffle_policy": "reset NumPy shuffle generator to seed+1 and epoch cursor/order to zero after density best transfer",
        }}

    terminal_reason = "completed"
    exception_text = None
    with metrics_path.open("a", buffering=1) as metrics:
        _write_jsonl(metrics, {"event": "run_start", "stage": current,
                               "config_sha256": config_hash, "model_sha256": runner_state["source_model_sha256"],
                               "data_sha256": data_hashes, "effective_config": config,
                               "train_path": str(train_path), "validation_path": str(validation_path),
                               "stage_path": stage, "elapsed_seconds": 0.0})
        try:
            while True:
                stage_config = config[current]
                if current == "energy" and stage == "both":
                    checksum = _density_checksum(trainer.state["params"])
                    if checksum != runner_state["initial_density_sha256"]:
                        raise RuntimeError("Frozen density parameter checksum changed before energy-stage update")
                if (bool(config.get("initial_validation", False)) and
                        int(trainer.loop["epoch"]) == 0 and
                        current not in runner_state["initial_validation_stages"]):
                    initial_density = (_density_checksum(trainer.state["params"])
                                       if current == "energy" and stage == "both" else None)
                    baseline = trainer.validate(validation_batches)
                    if not math.isfinite(float(baseline["loss"])):
                        raise FloatingPointError("Nonfinite initial validation loss")
                    if current == "energy" and stage == "both" and _density_checksum(trainer.state["params"]) != initial_density:
                        raise RuntimeError("Frozen density parameter checksum changed during initial energy validation")
                    runner_state["initial_validation_stages"].append(current)
                    _write_jsonl(metrics, {"event": "initial_validation", "stage": current,
                        "epoch": 0, "step": int(trainer.state["step"]), "metrics": baseline,
                        "selected": False, "elapsed_seconds": time.monotonic() - start_time,
                        "density_sha256": initial_density})
                    _snapshot(output / "latest", trainer, metadata=metadata())
                target_epoch = int(stage_config["epochs"])
                while int(trainer.loop["epoch"]) < target_epoch:
                    if _STOP:
                        terminal_reason = "preempted"
                        _snapshot(output / "latest", trainer, metadata=metadata())
                        break
                    step_start = time.monotonic()
                    step_metrics, completed_epoch = trainer.step(train_batches, shuffle=shuffle)
                    record = {"event": "train_step", "stage": current,
                              "epoch": int(trainer.loop["epoch"]), "step": int(trainer.state["step"]),
                              "elapsed_seconds": time.monotonic() - start_time,
                              "step_seconds": time.monotonic() - step_start}
                    record.update({key: float(value) for key, value in step_metrics.items()})
                    _write_jsonl(metrics, record)
                    if completed_epoch:
                        epoch = int(trainer.loop["epoch"])
                        if epoch % int(stage_config["validate_every"]) == 0 or epoch == target_epoch:
                            if current == "energy" and stage == "both":
                                before = _density_checksum(trainer.state["params"])
                                if before != runner_state["initial_density_sha256"]:
                                    raise RuntimeError("Frozen density parameter checksum changed before validation")
                            validation = trainer.validate(validation_batches)
                            score = float(validation["loss"])
                            if not math.isfinite(score):
                                raise FloatingPointError("Nonfinite validation loss")
                            if current == "energy" and stage == "both":
                                after = _density_checksum(trainer.state["params"])
                                if after != runner_state["initial_density_sha256"] or after != before:
                                    raise RuntimeError("Frozen density parameter checksum changed during energy validation")
                            improved = score < best_score
                            runner_state["bad_validations"] = 0 if improved else bad_validations + 1
                            if improved:
                                best_score = score
                                bad_validations = 0
                                runner_state["best_validation"][current] = score
                                runner_state["best_checkpoints"][current] = str(output / f"best_{current}")
                                _snapshot(output / f"best_{current}", trainer,
                                          metadata=metadata(), use_ema=trainer.optimizer.ema_decay is not None)
                            else:
                                bad_validations += 1
                            runner_state["bad_validations"] = bad_validations
                            _write_jsonl(metrics, {"event": "validation", "stage": current,
                                "epoch": epoch, "step": int(trainer.state["step"]),
                                "elapsed_seconds": time.monotonic() - start_time,
                                "metrics": validation, "selected": improved,
                                "selected_weights": "ema" if trainer.optimizer.ema_decay is not None else "params",
                                "density_sha256": _density_checksum(trainer.state["params"])
                                    if current == "energy" and stage == "both" else None})
                            # Match native validation control: select first, then advance
                            # the scheduler, then checkpoint the resumable latest state.
                            trainer.schedule(score)
                            _snapshot(output / "latest", trainer, metadata=metadata())
                            patience = int(stage_config["patience"])
                            if patience and bad_validations >= patience:
                                runner_state["early_stopped"] = True
                                _write_jsonl(metrics, {"event": "early_stop", "stage": current,
                                    "epoch": epoch, "best_validation": best_score,
                                    "patience": patience, "elapsed_seconds": time.monotonic() - start_time})
                                break
                        elif epoch % int(stage_config["validate_every"]) != 0:
                            _snapshot(output / "latest", trainer, metadata=metadata())
                    if _STOP:
                        terminal_reason = "preempted"
                        _snapshot(output / "latest", trainer, metadata=metadata())
                        break
                if terminal_reason == "preempted":
                    break
                stopped = bool(stage_config["patience"] and bad_validations >= int(stage_config["patience"]))
                if current == "density" and stage == "both":
                    stage1_terminal = {"epoch": int(trainer.loop["epoch"]),
                                       "step": int(trainer.state["step"]),
                                       "reason": "early_stopped" if stopped else "completed"}
                    best_path = output / "best_density"
                    if not (best_path / "manifest.json").exists():
                        raise RuntimeError("Density stage ended without a validated best checkpoint")
                    pipeline, state, _, _, _ = load_checkpoint(best_path)
                    optimizer = Optimizer(**config["energy"]["optimizer"])
                    loss = ErrorDict(**config["energy"]["loss"])
                    trainer = Trainer(pipeline, loss, optimizer, stage="energy", seed=seed, jit=jit,
                                      scheduler=config["energy"].get("scheduler"),
                                      jit_validation=bool(config.get("jit_validation",False)))
                    # The density-best bundle stores its selected EMA (when enabled) as params.
                    runner_state["initial_density_sha256"] = _density_checksum(trainer.state["params"])
                    import numpy as np
                    trainer.loop.update(epoch=0, cursor=0, order=[], batch_count=0,
                                        rng=np.random.default_rng(seed + 1).bit_generator.state,
                                        best=math.inf, bad_epochs=0,
                                        learning_rate=optimizer.learning_rate,
                                        loss_weights=dict(loss.loss_weights),
                                        validation_loss_weights=dict(loss.loss_weights))
                    trainer.rng = np.random.default_rng()
                    trainer.rng.bit_generator.state = trainer.loop["rng"]
                    current = "energy"
                    best_score = runner_state["best_validation"].get("energy", math.inf)
                    bad_validations = 0
                    runner_state["bad_validations"] = 0
                    _write_jsonl(metrics, {"event": "stage_change", "from": "density", "to": "energy",
                        "selected_density_validation": runner_state["best_validation"]["density"],
                        "density_sha256": runner_state["initial_density_sha256"],
                        "stage1_terminal": stage1_terminal,
                        "optimizer_reset": True, "epoch_cursor": 0,
                        "shuffle_seed": seed + 1, "elapsed_seconds": time.monotonic() - start_time})
                    _snapshot(output / "latest", trainer, metadata=metadata())
                    continue
                if stopped:
                    terminal_reason = "early_stopped"
                break
        except BaseException as error:
            terminal_reason = "failed"
            exception_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            try:
                if "trainer" in locals():
                    _snapshot(output / "latest", trainer, metadata=metadata())
            except Exception:
                pass
            _write_jsonl(metrics, {"event": "error", "stage": current,
                "elapsed_seconds": time.monotonic() - start_time,
                "exception": f"{type(error).__name__}: {error}"})
        finally:
            terminal = {"reason": terminal_reason, "stage": current,
                        "elapsed_seconds": time.monotonic() - start_time,
                        "exception": exception_text}
            (output / "terminal.json").write_text(json.dumps(terminal, indent=2) + "\n")
            _write_jsonl(metrics, {"event": "terminal", **terminal})
    if terminal_reason == "failed":
        raise RuntimeError(exception_text)
    return terminal_reason


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Portable Pipeline bundle or native checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Prepared data directory; config data paths or conventional train/validation bundles")
    parser.add_argument("--train", type=Path, help="Explicit training bundle, overrides config data.train")
    parser.add_argument("--validation", type=Path, help="Explicit validation bundle, overrides config data.validation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("density", "energy", "both"), default="both")
    parser.add_argument("--resume", type=Path, help="Resume exact native state from this checkpoint")
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    global _STOP
    _STOP = False
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    terminal_path = args.output / "terminal.json"
    prior_terminal = terminal_path.read_bytes() if terminal_path.exists() else None
    metrics_path = args.output / "metrics.jsonl"
    prior_metrics_size = metrics_path.stat().st_size if metrics_path.exists() else 0
    try:
        config = json.loads(args.config.read_text())
        reason = run_staged(model_path=args.resume or args.model, data_root=args.data,
            config=config, output=args.output, stage=args.stage, train_path=args.train,
            validation_path=args.validation, resume=args.resume is not None, jit=args.jit)
        return 0 if reason in {"completed", "early_stopped", "preempted"} else 1
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        current_terminal = terminal_path.read_bytes() if terminal_path.exists() else None
        if current_terminal is None or current_terminal == prior_terminal:
            terminal = {"reason": "failed", "stage": args.stage,
                        "elapsed_seconds": 0.0,
                        "exception": f"{type(error).__name__}: {error}"}
            terminal_path.write_text(json.dumps(terminal, indent=2) + "\n")
            with metrics_path.open("a") as stream:
                if not metrics_path.exists() or metrics_path.stat().st_size == prior_metrics_size:
                    _write_jsonl(stream, {"event": "terminal", **terminal})
        print(f"native staged runner failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
