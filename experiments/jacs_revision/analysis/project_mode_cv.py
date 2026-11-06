#!/usr/bin/env python3
"""Project same-N frames onto extreme-mode coordinates.

The Hessian atom count is required. 1-6-mer thiophene train frames are
not projected onto 12-mer modes.

Usage:
  python scripts/revision/analysis/project_mode_cv.py \\
    --modes datasets/revision/ood_modes/ethanol/modes.npz \\
    --train datasets/ethanol_train_10.xyz \\
    --test datasets/revision/ood_modes/ethanol/ood_modes.xyz \\
    --out results/revision/mode_cv/ethanol.json
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/project_mode_cv.py'


import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR, _SCRIPT_DIR.parent / "water")
from experiments.jacs_revision.water.generate_ood_distance_scale import read_xyz_symbol_frames  # noqa: E402
from experiments.jacs_revision.lib.normal_modes import align_kabsch, nearest_mode_distance, project_modes  # noqa: E402

_REPO_ROOT = repo_root()


def load_modes(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "equilibrium": np.asarray(data["equilibrium"], dtype=float),
        "modes": np.asarray(data["modes"], dtype=float),
        "masses": np.asarray(data["masses"], dtype=float),
        "frequencies": np.asarray(data["frequencies"], dtype=float),
        "symbols": [str(s) for s in data["symbols"]],
    }


def iter_same_n_positions(path: Path, n_atoms: int, max_frames: int | None = None) -> list[np.ndarray]:
    path = Path(path)
    frames: list[np.ndarray] = []
    if path.suffix.lower() == ".xyz":
        for _sym, coords, _c in read_xyz_symbol_frames(path):
            if len(coords) == n_atoms:
                frames.append(np.asarray(coords, dtype=float))
            if max_frames is not None and len(frames) >= max_frames:
                break
        return frames

    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item() if data.ndim == 0 else data
    if not isinstance(data, dict):
        raise ValueError(f"unsupported npy in {path}")
    pos = np.asarray(data["positions"], dtype=float)
    z = np.asarray(data["atom_numbers"])
    if pos.ndim == 2:
        pos = pos[None, ...]
    if z.ndim == 1:
        z = np.broadcast_to(z, (pos.shape[0], len(z)))
    for i in range(len(pos)):
        mask = np.asarray(z[i]) > 0
        if int(mask.sum()) != n_atoms:
            continue
        frames.append(pos[i, mask])
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def project_cloud(
    positions_list: list[np.ndarray],
    equilibrium: np.ndarray,
    modes: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    rows = []
    for pos in positions_list:
        aligned = align_kabsch(pos, equilibrium)
        rows.append(project_modes(aligned - equilibrium, modes, masses))
    if not rows:
        return np.zeros((0, modes.shape[1]))
    return np.asarray(rows, dtype=float)


def build_mode_cv_report(
    modes_path: Path,
    train_path: Path | None,
    test_path: Path | None,
    max_train: int = 400,
) -> dict:
    bundle = load_modes(modes_path)
    eq = bundle["equilibrium"]
    n_atoms = len(eq)
    train_pos = iter_same_n_positions(train_path, n_atoms, max_frames=max_train) if train_path else [eq]
    test_pos = iter_same_n_positions(test_path, n_atoms) if test_path else []
    train_q = project_cloud(train_pos, eq, bundle["modes"], bundle["masses"])
    test_q = project_cloud(test_pos, eq, bundle["modes"], bundle["masses"])
    distances = [nearest_mode_distance(q, train_q) for q in test_q]
    return {
        "modes": str(modes_path),
        "train": str(train_path) if train_path else None,
        "test": str(test_path) if test_path else None,
        "n_atoms": n_atoms,
        "n_modes": int(bundle["modes"].shape[1]),
        "n_train": int(len(train_q)),
        "n_test": int(len(test_q)),
        "train_q": train_q.tolist(),
        "test_q": test_q.tolist(),
        "test_nn_distance": distances,
        "train_q_rms": [float(np.sqrt(np.mean(train_q[:, k] ** 2))) for k in range(train_q.shape[1])]
        if len(train_q)
        else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=Path, required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--max-train", type=int, default=400)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_mode_cv_report(args.modes, args.train, args.test, max_train=args.max_train)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("n_atoms", "n_modes", "n_train", "n_test")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
