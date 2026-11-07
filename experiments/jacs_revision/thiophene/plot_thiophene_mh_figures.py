#!/usr/bin/env python3
"""SO3krates-style figures: train (blue) vs MH minima (green) in (θ12, θ23)."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/thiophene/plot_thiophene_mh_figures.py'


import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)
from experiments.jacs_revision.thiophene.thiophene_collective_variables import fold_cv_pair  # noqa: E402

_REPO_ROOT = repo_root()


def plot_cv_map(
    train_cv: np.ndarray,
    mh_cv: np.ndarray,
    selected: list[dict] | None,
    dest: Path,
    *,
    fold: bool = True,
) -> Path:
    import matplotlib.pyplot as plt

    train_cv = np.asarray(train_cv, dtype=float).reshape(-1, 2)
    mh_cv = np.asarray(mh_cv, dtype=float).reshape(-1, 2)
    if fold:
        if len(train_cv):
            train_cv = fold_cv_pair(train_cv)
        if len(mh_cv):
            mh_cv = fold_cv_pair(mh_cv)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    if len(train_cv):
        ax.scatter(
            train_cv[:, 0],
            train_cv[:, 1],
            s=10,
            c="#1f77b4",
            alpha=0.45,
            label="train",
            linewidths=0,
        )
    if len(mh_cv):
        ax.scatter(
            mh_cv[:, 0],
            mh_cv[:, 1],
            s=8,
            c="#2ca02c",
            alpha=0.55,
            label="MH minima",
            linewidths=0,
        )
    if selected:
        for row in selected:
            point = np.array([row["theta12"], row["theta23"]], dtype=float)
            if fold:
                point = fold_cv_pair(point)
            ax.scatter(
                [point[0]],
                [point[1]],
                s=42,
                facecolors="none",
                edgecolors="k",
                linewidths=1.2,
            )
            ax.annotate(row["label"], (point[0], point[1]), fontsize=9)
    if fold:
        ax.set_xlabel(r"$|\theta_{12}|$ / deg")
        ax.set_ylabel(r"$|\theta_{23}|$ / deg")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 180)
        ax.set_title("Terthiophene PES topology (folded)")
    else:
        ax.set_xlabel(r"$\theta_{12}$ / deg")
        ax.set_ylabel(r"$\theta_{23}$ / deg")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_title("Terthiophene PES topology")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=180)
    plt.close(fig)
    return dest


def plot_energy_bars(rows: list[dict], dest: Path) -> Path | None:
    import matplotlib.pyplot as plt

    usable = [r for r in rows if r.get("energy_dft_sp_ev") is not None and r.get("energy_ml_ev") is not None]
    if not usable:
        return None
    labels = [r["label"] for r in usable]
    ml = np.asarray([r["energy_ml_ev"] for r in usable], dtype=float)
    dft = np.asarray([r["energy_dft_sp_ev"] for r in usable], dtype=float)
    ml = (ml - ml.min()) * 1000.0
    dft = (dft - dft.min()) * 1000.0
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(x - 0.18, ml, width=0.36, label="DenSNet", color="#2ca02c")
    ax.bar(x + 0.18, dft, width=0.36, label="DFT", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"relative energy / meV")
    ax.legend(frameon=False)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=180)
    plt.close(fig)
    return dest


def plot_available(out_dir: Path | None = None) -> list[Path]:
    """Write figures if revision JSON products exist; skip quietly otherwise."""
    out_dir = out_dir or (_REPO_ROOT / "results/revision/figures/response")
    root = _REPO_ROOT / "results/revision/thiophene_mh"
    train_path = root / "train_cv.json"
    minima_path = root / "minima.json"
    selected_path = root / "selected_for_dft.json"
    dft_path = root / "dft_validation.json"
    train = np.zeros((0, 2))
    if train_path.is_file():
        payload = json.loads(train_path.read_text())
        train = np.column_stack([payload["theta12"], payload["theta23"]])
    mh = np.zeros((0, 2))
    if minima_path.is_file():
        payload = json.loads(minima_path.read_text())
        mh = np.array(
            [
                [row.get("theta12", 0.0), row.get("theta23", 0.0)]
                for row in payload.get("minima", [])
            ],
            dtype=float,
        )
    selected = []
    if selected_path.is_file():
        selected = json.loads(selected_path.read_text()).get("selected", [])
    written: list[Path] = []
    if len(train) or len(mh):
        written.append(plot_cv_map(train, mh, selected, out_dir / "thiophene_mh_cv_map.png"))
    if dft_path.is_file():
        rows = json.loads(dft_path.read_text()).get("rows", [])
        dest = plot_energy_bars(rows, out_dir / "thiophene_mh_dft_energy.png")
        if dest is not None:
            written.append(dest)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cv", type=Path, default=None)
    parser.add_argument("--minima", type=Path, default=None)
    parser.add_argument("--selected", type=Path, default=None)
    parser.add_argument("--dft", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results/revision/figures/response",
    )
    args = parser.parse_args()

    train = np.zeros((0, 2))
    if args.train_cv and args.train_cv.is_file():
        payload = json.loads(args.train_cv.read_text())
        train = np.column_stack([payload["theta12"], payload["theta23"]])
    mh = np.zeros((0, 2))
    if args.minima and args.minima.is_file():
        payload = json.loads(args.minima.read_text())
        mh = np.array(
            [[row.get("theta12", 0.0), row.get("theta23", 0.0)] for row in payload.get("minima", [])],
            dtype=float,
        )
    selected = []
    if args.selected and args.selected.is_file():
        selected = json.loads(args.selected.read_text()).get("selected", [])
    dest = args.out_dir / "thiophene_mh_cv_map.png"
    plot_cv_map(train, mh, selected, dest)
    if args.dft and args.dft.is_file():
        rows = json.loads(args.dft.read_text()).get("rows", [])
        plot_energy_bars(rows, args.out_dir / "thiophene_mh_dft_energy.png")
    print(f"wrote figures under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
