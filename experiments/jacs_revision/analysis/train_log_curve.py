"""Extract density-stage learning curves from appended campaign train logs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_STEP = re.compile(r"^(\d+)/(\d+)\s+epoch:")
_PERC = re.compile(r"train density_perc_mae:\s+([0-9.]+)")
_VALID_MAE = re.compile(r"valid 0 mae:\s+([0-9.]+)")
_VALID_LOSS = re.compile(r"valid 0 loss:\s+([0-9.]+)")
_RESTORE_PHASE = re.compile(r"^restore before training (\S+)")
_FINISHED_PHASE = re.compile(r"^finished trainer run of phase (\S+)")
_LR_STOP = re.compile(r"Learning rate is smaller than ([0-9.eE+-]+)! Training stopped\.")

# Unconstrained density uses stop_at_learning_rate=1e-5; fine-tune uses 1e-6.
_DENSITY_LR_STOP = 1e-5


@dataclass(frozen=True)
class DensityCurveResult:
    """Stitched unconstrained density-phase validation curve."""

    points: list[dict[str, float]]
    stop_reason: str  # lr | max_steps | unfinished


def parse_density_curve_result(log_path: Path) -> DensityCurveResult:
    """Stitch Slurm-resumed logs into one unconstrained ``density`` timeline.

    ``valid 0 loss`` is the relative perc. MAE the trainer optimizes. Points
    from ``df`` and ``density_fine_tuning`` are ignored. After the first
    genuine density completion (LR below 1e-5 or max_steps), later watchdog
    re-entries are dropped.
    """
    phase: str | None = None
    pending_step: int | None = None
    pending_max_steps: int | None = None
    pending_in_density_block = False
    fragments: list[list[dict[str, float]]] = [[]]
    density_complete = False
    pending_lr_stop_threshold: float | None = None
    stop_reason = "unfinished"

    def flush_point() -> None:
        nonlocal pending_step, pending_in_density_block
        pending_step = None
        pending_in_density_block = False

    lines = Path(log_path).read_text(errors="replace").splitlines()
    for i, raw in enumerate(lines):
        restore = _RESTORE_PHASE.match(raw.strip())
        if restore:
            phase = restore.group(1)
            flush_point()
            continue

        finished = _FINISHED_PHASE.match(raw.strip())
        if finished:
            finished_phase = finished.group(1)
            if (
                finished_phase == "density"
                and pending_lr_stop_threshold is not None
                and pending_lr_stop_threshold <= _DENSITY_LR_STOP * 1.01
            ):
                density_complete = True
                stop_reason = "lr"
            phase = finished_phase
            pending_lr_stop_threshold = None
            flush_point()
            continue

        lr_stop = _LR_STOP.search(raw)
        if lr_stop and phase == "density" and not density_complete:
            pending_lr_stop_threshold = float(lr_stop.group(1))
            continue

        if density_complete:
            continue

        if phase != "density":
            flush_point()
            continue

        step_match = _STEP.match(raw)
        if step_match:
            pending_step = int(step_match.group(1))
            pending_max_steps = int(step_match.group(2))
            pending_in_density_block = False
            continue

        if raw.strip() == "density:":
            pending_in_density_block = True
            continue

        if pending_step is None or not pending_in_density_block:
            continue

        perc = _PERC.search(raw)
        valid_mae = _VALID_MAE.search(raw)
        valid_loss = _VALID_LOSS.search(raw)
        if perc is None or valid_mae is None:
            continue

        valid_perc = float(valid_loss.group(1)) if valid_loss else float(valid_mae.group(1))
        if valid_perc >= 1.0:
            flush_point()
            continue

        step_val = pending_step
        point = {
            "step": float(step_val),
            "train_perc_mae": float(perc.group(1)),
            "valid_mae": float(valid_mae.group(1)),
            "valid_perc_mae": valid_perc,
        }
        frag = fragments[-1]
        if frag and point["step"] < frag[-1]["step"]:
            fragments.append([])
            frag = fragments[-1]
        frag.append(point)
        if pending_max_steps is not None and step_val >= pending_max_steps:
            density_complete = True
            stop_reason = "max_steps"
        flush_point()

    cleaned: list[list[dict[str, float]]] = []
    for frag in fragments:
        if len(frag) < 2:
            continue
        if any(p["valid_perc_mae"] >= 1.0 for p in frag):
            continue
        cleaned.append(frag)

    merged: dict[float, dict[str, float]] = {}
    for frag in cleaned:
        for p in frag:
            merged[p["step"]] = p

    points = [merged[k] for k in sorted(merged)]
    if len(points) < 2:
        return DensityCurveResult(points=[], stop_reason=stop_reason)

    if stop_reason == "unfinished" and points:
        last_step = points[-1]["step"]
        if pending_max_steps and last_step >= pending_max_steps:
            stop_reason = "max_steps"

    return DensityCurveResult(points=points, stop_reason=stop_reason)


def parse_density_curve(log_path: Path) -> list[dict[str, float]]:
    """Return stitched density points only (backward compatible for plots)."""
    return parse_density_curve_result(log_path).points
