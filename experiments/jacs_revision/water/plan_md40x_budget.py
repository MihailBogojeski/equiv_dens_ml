#!/usr/bin/env python3
"""Count, keep-fraction, and ORCA core-hour estimate for the md40x expansion.

Reads the finished 3k XYZ manifests, applies ``n_keep`` and a post-dedup
keep-fraction, and prints expected frames plus COST_MODEL core hours. The
90k–120k band is the gate: if the printout is outside it, change n_keep or
add Phase B parents before launching GFN2.

Usage:
  python scripts/revision/water/plan_md40x_budget.py
  python scripts/revision/water/plan_md40x_budget.py --n-keep 40 --keep-fraction 0.85
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/plan_md40x_budget.py'


import argparse
import json
import sys
from pathlib import Path


_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)

from experiments.jacs_revision.lib.build_shards_from_xyz import predicted_cost_s  # noqa: E402

_REPO_ROOT = repo_root()

DEFAULT_KEEP_FRACTION = 0.85
TARGET_BAND = (90_000, 120_000)
THEORIES = ("pbe_d4_avdz", "wb97mv_def2tzvpd")
LARGE_ATOM_CUTOFF = 33
SMALL_CORES = 8
LARGE_CORES = 16

# Typical atoms per water / malon split when a manifest has no by_n table.
_DEFAULT_N_ATOMS = {
    "water_train": 12,  # n=2–6, 4 waters on average
    "water_val": 12,
    "water_id_test": 12,
    "water_ood_size": 30,  # n=8,10,12
    "water_ood_order": 54,  # ice n=12–24
    "water_ood_density": 60,  # droplets n=16–24
    "malon_train": 9,
    "malon_val": 9,
    "malon_ood_proton_transfer": 9,
}

_MANIFEST_SPLITS = (
    ("datasets/revision/water_clusters/manifest.json", "water", ("train", "val", "id_test", "ood_size")),
    ("datasets/revision/water_ood/manifest.json", "water", ("ood_order", "ood_density")),
    ("datasets/revision/malonaldehyde/manifest.json", "malon", ("train", "val", "ood_proton_transfer")),
)


def parent_counts_from_manifests(repo: Path) -> dict[str, int]:
    """Return ``{water_train: 1250, ...}`` from the finished 3k manifests."""
    counts: dict[str, int] = {}
    for rel, prefix, splits in _MANIFEST_SPLITS:
        path = repo / rel
        payload = json.loads(path.read_text())
        block = payload.get("splits", {})
        for split in splits:
            entry = block.get(split)
            if entry is None:
                raise KeyError(f"{path} has no splits.{split}")
            counts[f"{prefix}_{split}"] = int(entry["n_frames"])
    return counts


def _mean_atoms_from_by_n(by_n: dict) -> int | None:
    if not by_n:
        return None
    total_frames = 0
    total_atoms = 0
    for n_str, n_frames in by_n.items():
        n_waters = int(n_str)
        n_frames_i = int(n_frames)
        total_frames += n_frames_i
        total_atoms += 3 * n_waters * n_frames_i
    if total_frames == 0:
        return None
    return max(1, int(round(total_atoms / total_frames)))


def n_atoms_from_manifests(repo: Path) -> dict[str, int]:
    """Typical atom count per split, from ``by_n`` when present."""
    atoms = dict(_DEFAULT_N_ATOMS)
    water = json.loads((repo / "datasets/revision/water_clusters/manifest.json").read_text())
    for split, entry in water.get("splits", {}).items():
        mean = _mean_atoms_from_by_n(entry.get("by_n") or {})
        if mean is not None:
            atoms[f"water_{split}"] = mean
    return atoms


def estimate_budget(
    parent_counts: dict[str, int],
    n_keep: int,
    keep_fraction: float,
    n_atoms_by_split: dict[str, int] | None = None,
    cores_small: int = SMALL_CORES,
    cores_large: int = LARGE_CORES,
    large_atom_cutoff: int = LARGE_ATOM_CUTOFF,
) -> dict:
    """Expected frames and per-theory core hours after thicken + dedup."""
    if n_keep < 1:
        raise ValueError(f"n_keep must be >= 1, got {n_keep}")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    n_atoms_by_split = n_atoms_by_split or dict(_DEFAULT_N_ATOMS)
    n_parents = int(sum(parent_counts.values()))
    n_expected = int(round(n_parents * n_keep * keep_fraction))
    per_split = {
        name: int(round(int(n_par) * n_keep * keep_fraction)) for name, n_par in parent_counts.items()
    }
    theories: dict[str, dict] = {}
    for theory in THEORIES:
        frame_cost: dict[str, float] = {}
        core_seconds = 0.0
        for name, n_frames in per_split.items():
            n_atoms = int(n_atoms_by_split.get(name, 12))
            cost_s = float(predicted_cost_s(n_atoms, theory))
            frame_cost[name] = cost_s
            cores = cores_large if n_atoms >= large_atom_cutoff else cores_small
            core_seconds += n_frames * cost_s * cores
        theories[theory] = {
            "frame_cost_s": frame_cost,
            "core_hours": core_seconds / 3600.0,
        }
    return {
        "n_parents": n_parents,
        "n_keep": n_keep,
        "keep_fraction": keep_fraction,
        "n_expected_frames": n_expected,
        "per_split": per_split,
        "in_target_band": TARGET_BAND[0] <= n_expected <= TARGET_BAND[1],
        "target_band": list(TARGET_BAND),
        "theories": theories,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=_REPO_ROOT)
    parser.add_argument("--n-keep", type=int, default=40)
    parser.add_argument("--keep-fraction", type=float, default=DEFAULT_KEEP_FRACTION)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    counts = parent_counts_from_manifests(args.repo)
    atoms = n_atoms_from_manifests(args.repo)
    plan = estimate_budget(counts, args.n_keep, args.keep_fraction, n_atoms_by_split=atoms)
    plan["parent_counts"] = counts
    print(f"parents: {plan['n_parents']}")
    print(f"n_keep: {plan['n_keep']}  keep_fraction: {plan['keep_fraction']}")
    print(f"expected frames: {plan['n_expected_frames']}  band: {TARGET_BAND}")
    print(f"in_target_band: {plan['in_target_band']}")
    for name, n_frames in plan["per_split"].items():
        print(f"  {name}: {n_frames}")
    for theory, rec in plan["theories"].items():
        print(f"{theory}: {rec['core_hours']:.1f} core-hours")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(plan, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
