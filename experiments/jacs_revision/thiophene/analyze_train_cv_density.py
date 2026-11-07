#!/usr/bin/env python3
"""2D training histogram of terthiophene (θ12, θ23) before minima hopping.

Marks empty bins that can serve as the SO3krates unsampled (gray) box.
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/thiophene/analyze_train_cv_density.py'


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
from experiments.jacs_revision.thiophene.thiophene_collective_variables import (  # noqa: E402
    cv_histogram_2d,
    fold_abs_deg,
    frames_to_cv,
    load_npy_frames,
    low_density_bins,
)

_REPO_ROOT = repo_root()


def train_cv_density(
    theta12: np.ndarray,
    theta23: np.ndarray,
    bins: int = 18,
    *,
    fold: bool = True,
) -> dict:
    if fold:
        theta12 = fold_abs_deg(theta12)
        theta23 = fold_abs_deg(theta23)
        range_deg = (0.0, 180.0)
    else:
        range_deg = (-180.0, 180.0)
    hist, edges = cv_histogram_2d(theta12, theta23, bins=bins, range_deg=range_deg)
    empty = low_density_bins(hist, edges, max_count=0)
    return {
        "hist": hist,
        "xedges": edges[0],
        "yedges": edges[1],
        "empty_bins": empty,
        "n_empty": len(empty),
        "n_frames": int(len(theta12)),
        "folded": fold,
        "theta12": theta12,
        "theta23": theta23,
    }


def _save_figure(hist: np.ndarray, edges: tuple, dest: Path, *, folded: bool = True) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    xedges, yedges = edges
    mesh = ax.pcolormesh(xedges, yedges, hist.T, cmap="Blues", shading="auto")
    fig.colorbar(mesh, ax=ax, label="train frames")
    if folded:
        ax.set_xlabel(r"$|\theta_{12}|$ / deg")
        ax.set_ylabel(r"$|\theta_{23}|$ / deg")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 180)
        ax.set_title("Terthiophene training CV density (folded)")
        ax.text(12, 12, "syn–syn", fontsize=8, color="0.35")
        ax.text(118, 168, "anti–anti", fontsize=8, color="0.35")
    else:
        ax.set_xlabel(r"$\theta_{12}$ / deg")
        ax.set_ylabel(r"$\theta_{23}$ / deg")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_title("Terthiophene training CV density")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npy",
        type=Path,
        default=_REPO_ROOT / "datasets/thiophene_all_train_d4.npy",
    )
    parser.add_argument("--n-atoms", type=int, default=23)
    parser.add_argument("--bins", type=int, default=18)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "results/revision/thiophene_mh/train_cv_density.npz",
    )
    args = parser.parse_args()
    frames = load_npy_frames(args.npy, n_atoms=args.n_atoms)
    cv = frames_to_cv(frames)
    payload = train_cv_density(cv[:, 0], cv[:, 1], bins=args.bins, fold=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        hist=payload["hist"],
        xedges=payload["xedges"],
        yedges=payload["yedges"],
        theta12=cv[:, 0],
        theta23=cv[:, 1],
        theta12_folded=payload["theta12"],
        theta23_folded=payload["theta23"],
    )
    fig_path = args.out.with_suffix(".png")
    _save_figure(
        payload["hist"],
        (payload["xedges"], payload["yedges"]),
        fig_path,
        folded=True,
    )
    signed = train_cv_density(cv[:, 0], cv[:, 1], bins=args.bins, fold=False)
    _save_figure(
        signed["hist"],
        (signed["xedges"], signed["yedges"]),
        fig_path.with_name("train_cv_density_signed.png"),
        folded=False,
    )
    summary = args.out.with_name("train_cv_density.json")
    summary.write_text(
        json.dumps(
            {
                "npy": str(args.npy),
                "n_frames": payload["n_frames"],
                "n_empty": payload["n_empty"],
                "folded": True,
                "empty_bins": payload["empty_bins"][:40],
                "figure": str(fig_path),
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} and {fig_path} ({payload['n_empty']} empty bins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
