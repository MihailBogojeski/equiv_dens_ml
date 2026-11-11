#!/usr/bin/env python3
"""Stratified, nested water-train subsets for data-efficiency curves.

The trainer's --num_train flag draws a seeded random subset of the train npy.
At N=50 that can be almost all dimers. This writes one npy pair per N_train
with the same number of frames from every n_water stratum, and with nested
index sets (N=50 ⊂ N=125 ⊂ …) so the curve is not five independent draws.

Usage:
  python scripts/revision/water/make_learning_curve_splits.py \\
    --base datasets/revision/water_labels/water_train_pbe_d4_avdz_base.npy \\
    --dens datasets/revision/water_labels/water_train_pbe_d4_avdz.npy \\
    --out datasets/revision/water_labels/learning_curves
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/make_learning_curve_splits.py'


import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import repo_root  # noqa: E402

_REPO_ROOT = repo_root()

DEFAULT_SIZES = (50, 125, 250, 500, 1250)
DEFAULT_PER_STRATUM = {50: 10, 125: 25, 250: 50, 500: 100, 1250: 250}
DEFAULT_SEED = 20260906


def n_water_per_frame(atom_numbers: np.ndarray) -> np.ndarray:
    """Waters per frame: count of oxygen (Z=8), ignoring padded zeros."""
    z = np.asarray(atom_numbers)
    if z.ndim == 1:
        z = z[None, :]
    return np.sum(z == 8, axis=-1).astype(int)


def nested_stratified_indices(
    n_water: np.ndarray,
    sizes: Sequence[int] = DEFAULT_SIZES,
    seed: int = DEFAULT_SEED,
    per_stratum: Mapping[int, int] | None = None,
) -> dict[int, list[int]]:
    """Return nested index lists, ``sizes[0] ⊂ sizes[1] ⊂ …``.

    Each stratum (distinct n_water) is shuffled once. Size *N* keeps the first
    ``per_stratum[N]`` frames of every stratum.
    """
    counts = np.asarray(n_water)
    quota = dict(per_stratum) if per_stratum is not None else dict(DEFAULT_PER_STRATUM)
    groups: dict[int, list[int]] = {}
    for i, n in enumerate(counts.tolist()):
        groups.setdefault(int(n), []).append(int(i))
    rng = np.random.default_rng(seed)
    shuffled: dict[int, np.ndarray] = {}
    for key in sorted(groups):
        idxs = np.asarray(groups[key], dtype=int)
        rng.shuffle(idxs)
        shuffled[key] = idxs
    out: dict[int, list[int]] = {}
    for n_total in sizes:
        keep = int(quota[int(n_total)])
        chosen: list[int] = []
        for key in sorted(shuffled):
            pool = shuffled[key]
            if len(pool) < keep:
                raise ValueError(
                    f"stratum n_water={key} has {len(pool)} frames, need {keep} for N={n_total}"
                )
            chosen.extend(int(i) for i in pool[:keep])
        if len(chosen) != int(n_total):
            raise ValueError(
                f"N={n_total} expected {n_total} frames, got {len(chosen)} "
                f"from {len(shuffled)} strata × {keep}"
            )
        chosen.sort()
        out[int(n_total)] = chosen
    return out


def slice_dataset(
    base: dict,
    dens: np.ndarray,
    indices: Sequence[int],
) -> tuple[dict, np.ndarray]:
    """Keep frame-aligned arrays; leave per-dataset metadata untouched."""
    idx = np.asarray(list(indices), dtype=int)
    n_frames = int(np.asarray(dens).shape[0])
    sliced: dict = {}
    for key, value in base.items():
        arr = np.asarray(value, dtype=object) if isinstance(value, list) else np.asarray(value)
        if arr.shape[:1] == (n_frames,):
            sliced[key] = np.asarray(value)[idx]
        else:
            sliced[key] = value
    return sliced, np.asarray(dens)[idx]


def write_learning_curve_splits(
    base_npy: Path,
    dens_npy: Path,
    out_dir: Path,
    sizes: Sequence[int] = DEFAULT_SIZES,
    seed: int = DEFAULT_SEED,
    theory: str = "pbe_d4_avdz",
    per_stratum: Mapping[int, int] | None = None,
) -> dict:
    """Write ``splits.json`` and one train npy pair per *N*."""
    base = np.load(base_npy, allow_pickle=True).item()
    dens = np.load(dens_npy, allow_pickle=True)
    n_water = n_water_per_frame(base["atom_numbers"])
    splits = nested_stratified_indices(n_water, sizes=sizes, seed=seed, per_stratum=per_stratum)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sizes": [int(s) for s in sizes],
        "seed": int(seed),
        "theory": theory,
        "per_stratum": {str(k): int(v) for k, v in (per_stratum or DEFAULT_PER_STRATUM).items()},
        "n_source": int(len(n_water)),
        "strata": {str(k): int(v) for k, v in zip(*np.unique(n_water, return_counts=True))},
        "indices": {str(n): idxs for n, idxs in splits.items()},
        "files": {},
    }
    for n_total, idxs in splits.items():
        sliced_base, sliced_dens = slice_dataset(base, dens, idxs)
        stem = f"water_train_n{n_total}_{theory}"
        base_out = out_dir / f"{stem}_base.npy"
        dens_out = out_dir / f"{stem}.npy"
        np.save(base_out, sliced_base, allow_pickle=True)
        np.save(dens_out, sliced_dens, allow_pickle=True)
        manifest["files"][str(n_total)] = {
            "n_train": int(n_total),
            "base": str(base_out),
            "dens": str(dens_out),
        }
    (out_dir / "splits.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=_REPO_ROOT / "datasets/revision/water_labels/water_train_pbe_d4_avdz_base.npy",
    )
    parser.add_argument(
        "--dens",
        type=Path,
        default=_REPO_ROOT / "datasets/revision/water_labels/water_train_pbe_d4_avdz.npy",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "datasets/revision/water_labels/learning_curves",
    )
    parser.add_argument("--theory", default="pbe_d4_avdz")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if not args.base.is_file() or not args.dens.is_file():
        print(f"missing train npy: {args.base} or {args.dens}", file=sys.stderr)
        return 1
    manifest = write_learning_curve_splits(
        args.base,
        args.dens,
        args.out,
        sizes=DEFAULT_SIZES,
        seed=args.seed,
        theory=args.theory,
    )
    print(json.dumps({k: manifest[k] for k in ("sizes", "strata", "files")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
