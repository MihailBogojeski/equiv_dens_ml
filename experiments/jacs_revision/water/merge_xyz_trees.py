#!/usr/bin/env python3
"""Concatenate geometry trees and write a by_n manifest.

Used to merge md40x + newmotifs + Rackers parents into one campaign tree
without touching the finished 3k XYZ files.

Usage:
  python scripts/revision/water/merge_xyz_trees.py \\
    --src datasets/revision/water_clusters_md40x \\
         datasets/revision/water_clusters_newmotifs_md40x \\
    --dest datasets/revision/water_clusters_md40x_merged
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/merge_xyz_trees.py'


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

from experiments.jacs_revision.water.md_sampling import Frame, heavy_atom_rmsd, read_xyz_symbol_frames, write_xyz_frames  # noqa: E402

_REPO_ROOT = repo_root()

DEFAULT_SPLITS = (
    "train",
    "val",
    "id_test",
    "ood_size",
    "ood_order",
    "ood_density",
    "ood_proton_transfer",
    "ood_rackers",
)


def write_xyz_comment(comment: str) -> str:
    """Identity helper so tests can pin that comments are not rewritten."""
    return comment


def _n_waters(symbols: list[str]) -> int:
    n_o = sum(1 for s in symbols if s.upper() == "O")
    return n_o if n_o else max(1, len(symbols) // 3)


def _dedup(frames: list[Frame], rmsd_cutoff: float | None) -> list[Frame]:
    if rmsd_cutoff is None:
        return frames
    kept: list[Frame] = []
    for frame in frames:
        dup = False
        for other in kept:
            if len(other.symbols) != len(frame.symbols):
                continue
            if heavy_atom_rmsd(frame.coords, other.coords, frame.symbols) < rmsd_cutoff:
                dup = True
                break
        if not dup:
            kept.append(frame)
    return kept


def merge_trees(
    sources: list[Path],
    dest: Path,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    rmsd_cutoff: float | None = None,
) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"sources": [str(p) for p in sources], "splits": {}}
    for split in splits:
        frames: list[Frame] = []
        for src in sources:
            path = src / f"{split}.xyz"
            if not path.is_file():
                continue
            for symbols, coords, comment in read_xyz_symbol_frames(path):
                frames.append(Frame(symbols, coords, write_xyz_comment(comment), {"split": split}))
        frames = _dedup(frames, rmsd_cutoff)
        if not frames:
            continue
        out = dest / f"{split}.xyz"
        write_xyz_frames(out, frames)
        by_n = Counter(str(_n_waters(f.symbols)) for f in frames)
        manifest["splits"][split] = {
            "xyz": str(out),
            "n_frames": len(frames),
            "by_n": dict(sorted(by_n.items(), key=lambda kv: int(kv[0]))),
        }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, nargs="+", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--rmsd-cutoff", type=float, default=None)
    args = parser.parse_args(argv)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    manifest = merge_trees(args.src, args.dest, splits=splits, rmsd_cutoff=args.rmsd_cutoff)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
