#!/usr/bin/env python3
"""Import Rackers/Rakshit water-cluster XYZ as extra parents and a hold-out.

n=3–6 go to train, n=8/10/12 to size-OOD, n≥15 to ``ood_rackers`` (never
train). Existing campaign parents are RMSD-deduped so a published isomer
wins over an analytic clone. Relabel at our theories; do not load their
PBE0/CCSD JFIT coefficients.

Usage:
  python scripts/revision/water/import_rackers_parents.py \\
    --xyz datasets/revision/external/rackers_water/raw.xyz \\
    --water-out datasets/revision/water_clusters_rackers \\
    --ood-out datasets/revision/water_ood_rackers
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/import_rackers_parents.py'


import argparse
import json
import sys
from collections import Counter
from pathlib import Path


_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)

import numpy as np

from experiments.jacs_revision.water.md_sampling import Frame, heavy_atom_rmsd, read_xyz_symbol_frames, write_xyz_frames  # noqa: E402

_REPO_ROOT = repo_root()


def n_waters(symbols: list[str]) -> int:
    """Number of water molecules from oxygen count."""
    return sum(1 for s in symbols if s.upper() == "O")


def assign_split(n_water: int) -> str:
    """Train for n≤6, size-OOD for 7–12, external eval for n≥15."""
    if n_water <= 6:
        return "train"
    if n_water <= 12:
        return "ood_size"
    return "ood_rackers"


def _is_duplicate(frame: Frame, existing: list[Frame], rmsd_cutoff: float) -> bool:
    for other in existing:
        if len(other.symbols) != len(frame.symbols):
            continue
        if heavy_atom_rmsd(frame.coords, other.coords, frame.symbols) < rmsd_cutoff:
            return True
    return False


def import_rackers_xyz(
    src: Path,
    existing: list[Frame] | None = None,
    rmsd_cutoff: float = 0.20,
) -> tuple[list[Frame], list[Frame]]:
    """Read XYZ; return (kept, rejected-as-duplicates)."""
    existing = list(existing or [])
    kept: list[Frame] = []
    rejected: list[Frame] = []
    for symbols, coords, comment in read_xyz_symbol_frames(src):
        n_w = n_waters(symbols)
        split = assign_split(n_w)
        frame = Frame(
            symbols,
            np.asarray(coords, dtype=float),
            comment or f"rackers n={n_w}",
            {"n": n_w, "source": "rackers", "split": split},
        )
        pool = existing + kept
        if _is_duplicate(frame, pool, rmsd_cutoff):
            rejected.append(frame)
            continue
        kept.append(frame)
    return kept, rejected


def write_imported_trees(
    src: Path,
    water_out: Path,
    ood_out: Path,
    existing: list[Frame] | None = None,
    rmsd_cutoff: float = 0.20,
) -> dict:
    kept, rejected = import_rackers_xyz(src, existing=existing, rmsd_cutoff=rmsd_cutoff)
    by_split: dict[str, list[Frame]] = {}
    for frame in kept:
        by_split.setdefault(str(frame.meta["split"]), []).append(frame)

    water_out.mkdir(parents=True, exist_ok=True)
    ood_out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "source": str(src),
        "n_kept": len(kept),
        "n_rejected_dedup": len(rejected),
        "splits": {},
    }
    water_splits = ("train", "val", "id_test", "ood_size")
    for split, frames in by_split.items():
        dest_dir = ood_out if split == "ood_rackers" else water_out
        dest = dest_dir / f"{split}.xyz"
        write_xyz_frames(dest, frames)
        by_n = Counter(str(int(f.meta["n"])) for f in frames)
        manifest["splits"][split] = {
            "xyz": str(dest),
            "n_frames": len(frames),
            "by_n": dict(sorted(by_n.items(), key=lambda kv: int(kv[0]))),
        }
    for split in water_splits:
        manifest["splits"].setdefault(split, {"xyz": None, "n_frames": 0, "by_n": {}})
    manifest["splits"].setdefault("ood_rackers", {"xyz": None, "n_frames": 0, "by_n": {}})
    (water_out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (ood_out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _load_existing_xyz(paths: list[Path]) -> list[Frame]:
    frames: list[Frame] = []
    for path in paths:
        if not path.is_file():
            continue
        for symbols, coords, comment in read_xyz_symbol_frames(path):
            frames.append(Frame(symbols, coords, comment, {"source": "existing"}))
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument(
        "--water-out",
        type=Path,
        default=_REPO_ROOT / "datasets/revision/water_clusters_rackers",
    )
    parser.add_argument(
        "--ood-out",
        type=Path,
        default=_REPO_ROOT / "datasets/revision/water_ood_rackers",
    )
    parser.add_argument("--existing-xyz", type=Path, nargs="*", default=[])
    parser.add_argument("--rmsd-cutoff", type=float, default=0.20)
    args = parser.parse_args(argv)
    if not args.xyz.is_file():
        raise SystemExit(f"missing Rackers XYZ: {args.xyz}")
    existing = _load_existing_xyz(list(args.existing_xyz))
    manifest = write_imported_trees(
        args.xyz, args.water_out, args.ood_out, existing=existing, rmsd_cutoff=args.rmsd_cutoff
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}, indent=2))
    for split, rec in manifest["splits"].items():
        if rec["n_frames"]:
            print(f"  {split}: {rec['n_frames']}  by_n={rec['by_n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
