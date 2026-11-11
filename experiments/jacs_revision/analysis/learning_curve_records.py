"""Shared schema for water learning-curve JSON (N_train and step curves)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCORE_STEPS = (5000, 10000, 25000, 50000, 75000, 100000)
VALID_SPLITS = (
    "water_id_test",
    "water_ood_size",
    "water_ood_order",
    "water_ood_density",
)
VALID_ARMS = ("sad", "direct")
N_TRAIN_SIZES = (50, 125, 250, 500, 1250)


def _require_split_arm(split: str, arm: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {VALID_SPLITS}")
    if arm not in VALID_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {VALID_ARMS}")


def ntrain_record(n_train: int, split: str, afe_mean: float, arm: str) -> dict[str, Any]:
    """One data-efficiency point."""
    _require_split_arm(split, arm)
    return {
        "n_train": int(n_train),
        "split": split,
        "afe_mean": float(afe_mean),
        "arm": arm,
    }


def step_record(step: int, split: str, afe_mean: float, arm: str) -> dict[str, Any]:
    """One OOD-vs-step point."""
    _require_split_arm(split, arm)
    return {
        "step": int(step),
        "split": split,
        "afe_mean": float(afe_mean),
        "arm": arm,
    }


def validate_curve_record(rec: dict[str, Any], kind: str) -> None:
    """Raise if *rec* is missing the fields of *kind* (``ntrain`` or ``step``)."""
    if kind == "ntrain":
        required = ("n_train", "split", "afe_mean", "arm")
    elif kind == "step":
        required = ("step", "split", "afe_mean", "arm")
    else:
        raise ValueError(f"unknown kind {kind!r}")
    missing = [key for key in required if key not in rec]
    if missing:
        raise ValueError(f"record missing {missing}: {rec}")
    _require_split_arm(str(rec["split"]), str(rec["arm"]))


def collect_curve_records(paths: Sequence[Path], kind: str) -> dict[str, list[dict[str, Any]]]:
    """Read per-eval JSON files that each store one ``curve`` record."""
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        rec = payload.get("curve")
        if rec is None and "summary" in payload:
            raise ValueError(f"{path} has summary but no curve record")
        validate_curve_record(rec, kind)
        records.append(rec)
    records.sort(key=lambda r: (r.get("arm"), r.get("split"), r.get("n_train", r.get("step"))))
    return {"kind": kind, "records": records}


def series_for_figure(
    records: Iterable[dict[str, Any]],
    x_key: str,
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """Group ``(x, afe_mean)`` by ``(arm, split)``, sorted by *x*."""
    series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for rec in records:
        key = (str(rec["arm"]), str(rec["split"]))
        series.setdefault(key, []).append((int(rec[x_key]), float(rec["afe_mean"])))
    for key in series:
        series[key].sort()
    return series
