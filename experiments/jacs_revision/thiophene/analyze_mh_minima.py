#!/usr/bin/env python3
"""Overlay MH minima on the training (θ12, θ23) cloud and pick DFT targets.

High-density MH bins become A–D. The unsampled box is the MH point whose
training-bin count is lowest (SO3krates gray-box analogue).
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/thiophene/analyze_mh_minima.py'


import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
    frames_to_cv,
    load_npy_frames,
    nearest_train_cv_distance,
    terthiophene_theta12_theta23,
)

_REPO_ROOT = repo_root()


def _bin_index(value: float, edges: np.ndarray) -> int:
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    return int(np.clip(idx, 0, len(edges) - 2))


def select_regions_for_dft(
    train_cv: np.ndarray,
    mh_cv: np.ndarray,
    energies: np.ndarray,
    n_dense: int = 4,
    bins: int = 18,
) -> list[dict[str, Any]]:
    """Return labelled DFT targets (A–D plus one unsampled MH point)."""
    train_cv = np.asarray(train_cv, dtype=float).reshape(-1, 2)
    mh_cv = np.asarray(mh_cv, dtype=float).reshape(-1, 2)
    energies = np.asarray(energies, dtype=float).reshape(-1)
    hist_mh, edges = cv_histogram_2d(mh_cv[:, 0], mh_cv[:, 1], bins=bins)
    hist_train, _ = cv_histogram_2d(train_cv[:, 0], train_cv[:, 1], bins=bins)
    xedges, yedges = edges
    ny = hist_mh.shape[1]
    order = np.argsort(-hist_mh.ravel())
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    letters = "ABCD"
    n_picked = 0
    for ravel_i in order:
        if n_picked >= n_dense:
            break
        if hist_mh.ravel()[int(ravel_i)] <= 0:
            break
        ix = int(ravel_i) // ny
        iy = int(ravel_i) % ny
        in_bin = [
            i
            for i, (t12, t23) in enumerate(mh_cv)
            if _bin_index(t12, xedges) == ix and _bin_index(t23, yedges) == iy
        ]
        if not in_bin:
            continue
        pick = min(in_bin, key=lambda i: float(energies[i]))
        if pick in used:
            continue
        used.add(pick)
        selected.append(
            {
                "label": letters[n_picked],
                "index": int(pick),
                "theta12": float(mh_cv[pick, 0]),
                "theta23": float(mh_cv[pick, 1]),
                "energy": float(energies[pick]),
                "train_bin_count": int(hist_train[ix, iy]),
                "mh_bin_count": int(hist_mh[ix, iy]),
                "kind": "dense",
            }
        )
        n_picked += 1

    train_counts = []
    for t12, t23 in mh_cv:
        ix = _bin_index(t12, xedges)
        iy = _bin_index(t23, yedges)
        train_counts.append(int(hist_train[ix, iy]))
    train_counts_arr = np.asarray(train_counts)
    # Prefer empty train bins; break ties by distance from the train cloud.
    distances = np.asarray(
        [nearest_train_cv_distance(row, train_cv) for row in mh_cv],
        dtype=float,
    )
    rank = np.lexsort((-distances, train_counts_arr))
    unsampled_idx = int(rank[0])
    ix = _bin_index(mh_cv[unsampled_idx, 0], xedges)
    iy = _bin_index(mh_cv[unsampled_idx, 1], yedges)
    selected.append(
        {
            "label": "unsampled",
            "index": unsampled_idx,
            "theta12": float(mh_cv[unsampled_idx, 0]),
            "theta23": float(mh_cv[unsampled_idx, 1]),
            "energy": float(energies[unsampled_idx]),
            "train_bin_count": int(hist_train[ix, iy]),
            "mh_bin_count": int(hist_mh[ix, iy]),
            "kind": "unsampled",
            "cv_distance_to_train": float(distances[unsampled_idx]),
        }
    )
    return selected


def _minima_from_json(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    rows = payload["minima"]
    cvs = []
    energies = []
    for row in rows:
        symbols = row["symbols"]
        pos = np.asarray(row["positions"], dtype=float)
        t12, t23 = terthiophene_theta12_theta23(pos, symbols)
        cvs.append((t12, t23))
        energies.append(row["energy"])
        row["theta12"] = t12
        row["theta23"] = t23
    return np.asarray(cvs), np.asarray(energies), rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minima", type=Path, required=True)
    parser.add_argument(
        "--train-npy",
        type=Path,
        default=_REPO_ROOT / "datasets/thiophene_all_train_d4.npy",
    )
    parser.add_argument("--n-atoms", type=int, default=23)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "results/revision/thiophene_mh/selected_for_dft.json",
    )
    parser.add_argument("--bins", type=int, default=18)
    args = parser.parse_args()

    mh_cv, energies, minima = _minima_from_json(args.minima)
    if args.train_npy.is_file():
        train_cv = frames_to_cv(load_npy_frames(args.train_npy, n_atoms=args.n_atoms))
    else:
        train_cv = np.zeros((0, 2))
    selected = select_regions_for_dft(train_cv, mh_cv, energies, bins=args.bins)
    for row in selected:
        src = minima[row["index"]]
        row["symbols"] = src["symbols"]
        row["positions"] = src["positions"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "minima": str(args.minima),
                "train_npy": str(args.train_npy),
                "n_train": int(len(train_cv)),
                "n_minima": int(len(mh_cv)),
                "selected": selected,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} ({len(selected)} targets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
