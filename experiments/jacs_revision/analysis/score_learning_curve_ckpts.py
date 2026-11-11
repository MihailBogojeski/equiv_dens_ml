#!/usr/bin/env python3
"""Score kept water-curve checkpoints on the reviewer test splits.

csh_evaluate.py loads latest_checkpoint.pth for the run ID, then best_{ID}.pth
from the run root. Intermediate numbered checkpoints are never that file, so
this wraps one checkpoint as both latest and best_* before calling evaluate.

Usage (one checkpoint × one split):
  python scripts/revision/analysis/score_learning_curve_ckpts.py score \\
    --run-dir results/revision/water_pbe_d4_avdz_curve/<stamp> \\
    --step 5000 --split water_id_test --arm sad --theory pbe_d4_avdz

Collect:
  python scripts/revision/analysis/score_learning_curve_ckpts.py collect \\
    --kind step --glob 'results/revision/eval/learning_curves/step_*.json'
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/score_learning_curve_ckpts.py'


import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
_CSH_DIR = _SCRIPT_DIR.parent / "csh"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import repo_root  # noqa: E402
from experiments.jacs_revision.analysis.learning_curve_records import (  # noqa: E402
    VALID_ARMS,
    VALID_SPLITS,
    collect_curve_records,
    ntrain_record,
    step_record,
)

_REPO_ROOT = repo_root()


def find_checkpoint(run_dir: Path, step: int | None) -> Path:
    """Numbered ``checkpoint_{step:010d}*.pth``, or ``latest_checkpoint.pth``."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if step is None:
        latest = ckpt_dir / "latest_checkpoint.pth"
        if not latest.is_file():
            raise FileNotFoundError(f"no latest_checkpoint.pth under {ckpt_dir}")
        return latest
    padded = f"{int(step):010d}"
    exact = ckpt_dir / f"checkpoint_{padded}.pth"
    if exact.is_file():
        return exact
    density = ckpt_dir / f"checkpoint_{padded}_density.pth"
    if density.is_file():
        return density
    matches = sorted(ckpt_dir.glob(f"checkpoint_{padded}*.pth"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"no checkpoint for step {step} under {ckpt_dir}")


def _checkpoint_id(checkpoint: Path) -> str:
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and payload.get("ID"):
            return str(payload["ID"])
    except Exception:
        pass
    return "curve"


def prepare_eval_run_dir(run_dir: Path, checkpoint: Path, dest: Path) -> Path:
    """Copy args.txt and expose *checkpoint* as latest and best_{ID}."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "checkpoints").mkdir(parents=True)
    src_args = Path(run_dir) / "args.txt"
    if src_args.is_file():
        shutil.copy2(src_args, dest / "args.txt")
    target = Path(checkpoint).resolve()
    latest = dest / "checkpoints" / "latest_checkpoint.pth"
    latest.symlink_to(target)
    model_id = _checkpoint_id(target)
    best = dest / f"best_{model_id}.pth"
    best.symlink_to(target)
    return dest


def _labels(theory: str, split: str) -> tuple[Path, Path]:
    root = _REPO_ROOT / "datasets" / "revision" / "water_labels"
    return root / f"{split}_{theory}_base.npy", root / f"{split}_{theory}.npy"


def score_one(
    run_dir: Path,
    split: str,
    arm: str,
    theory: str,
    out: Path,
    step: int | None = None,
    n_train: int | None = None,
    checkpoint: Path | None = None,
) -> dict:
    """Score one checkpoint on one split and attach a ``curve`` record."""
    ckpt = Path(checkpoint) if checkpoint is not None else find_checkpoint(run_dir, step)
    base, dens = _labels(theory, split)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_CSH_DIR / "csh_evaluate.py"),
        "--args-file",
        str(Path(run_dir) / "args.txt"),
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        str(ckpt),
        "--np-dataset",
        str(base),
        "--dens-dataset",
        str(dens),
        "--elements",
        "1,8",
        "--label",
        split,
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    payload = json.loads(out.read_text())
    afe = float(payload["summary"]["afe_mean"])
    if n_train is not None:
        payload["curve"] = ntrain_record(n_train, split, afe, arm)
    else:
        resolved_step = int(step) if step is not None else _step_from_name(ckpt)
        payload["curve"] = step_record(resolved_step, split, afe, arm)
    out.write_text(json.dumps(payload, indent=2))
    return payload


def checkpoint_step(checkpoint: Path) -> int:
    """Step encoded in the filename, or the ``step`` field of a trainer ckpt."""
    stem = checkpoint.stem
    if stem.startswith("checkpoint_"):
        digits = stem.replace("checkpoint_", "").split("_")[0]
        if digits.isdigit():
            return int(digits)
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "step" in payload:
            return int(payload["step"])
    except Exception:
        pass
    return -1


def _step_from_name(checkpoint: Path) -> int:
    return checkpoint_step(checkpoint)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    score = sub.add_parser("score", help="score one checkpoint on one split")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--split", choices=VALID_SPLITS, required=True)
    score.add_argument("--arm", choices=VALID_ARMS, required=True)
    score.add_argument("--theory", default="pbe_d4_avdz")
    score.add_argument("--step", type=int, default=None)
    score.add_argument("--n-train", type=int, default=None)
    score.add_argument("--checkpoint", type=Path, default=None)
    score.add_argument("--out", type=Path, required=True)

    collect = sub.add_parser("collect", help="merge per-eval JSON into one curve file")
    collect.add_argument("--kind", choices=("ntrain", "step"), required=True)
    collect.add_argument("--glob", required=True)
    collect.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "score":
        payload = score_one(
            run_dir=args.run_dir,
            split=args.split,
            arm=args.arm,
            theory=args.theory,
            out=args.out,
            step=args.step,
            n_train=args.n_train,
            checkpoint=args.checkpoint,
        )
        print(json.dumps(payload["curve"], indent=2))
        return 0
    pattern = Path(args.glob)
    if pattern.is_absolute():
        paths = sorted(pattern.parent.glob(pattern.name))
    else:
        paths = sorted(_REPO_ROOT.glob(args.glob))
        if not paths:
            paths = sorted(Path().glob(args.glob))
    collected = collect_curve_records(paths, kind=args.kind)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(collected, indent=2))
    print(f"wrote {args.out} ({len(collected['records'])} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
