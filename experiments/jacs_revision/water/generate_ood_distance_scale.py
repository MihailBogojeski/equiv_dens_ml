#!/usr/bin/env python3
"""OMol25-style intermolecular distance scaling for water clusters.

Keeps each H2O rigid and scales monomer COM-to-cluster-COM vectors. The
default lambda grid matches OMol25 Appendix H.6 (2 compressive, 10
short-range expansive, 10 long-range expansive). Frames that violate the
covalent-overlap cut are tagged, not dropped.

Usage:
  python scripts/revision/water/generate_ood_distance_scale.py
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/generate_ood_distance_scale.py'


import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)
import numpy as np
from experiments.jacs_revision.lib.rigid_scale import (  # noqa: E402
    covalent_overlap_pairs,
    min_intermolecular_oo,
    molecules_to_water_frame,
    omol25_scale_factors,
    scale_rigid_coms,
    split_water_molecules,
    sr_lr_label,
)

from experiments.jacs_revision.water.generate_ood_water import write_xyz  # noqa: E402

_REPO_ROOT = repo_root()

DEFAULT_PARENTS = (
    "n2_dimer.xyz",
    "n4_cyclic.xyz",
    "n6_prism.xyz",
    "n6_cage.xyz",
    "n6_book.xyz",
    "n6_cyclic.xyz",
    "n12_hex_prism.xyz",
)


def read_xyz_symbol_frames(path: Path) -> list[tuple[list[str], np.ndarray, str]]:
    lines = Path(path).read_text().splitlines()
    frames: list[tuple[list[str], np.ndarray, str]] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        nat = int(lines[i].split()[0])
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        symbols: list[str] = []
        coords: list[list[float]] = []
        for row in lines[i + 2 : i + 2 + nat]:
            parts = row.split()
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        frames.append((symbols, np.asarray(coords, dtype=float), comment.strip()))
        i += 2 + nat
    return frames


def scan_parent(
    symbols: list[str],
    coords: np.ndarray,
    parent_label: str,
    scale_factors: list[float],
) -> tuple[list[tuple[list[str], np.ndarray]], list[str], list[dict]]:
    mols0 = split_water_molecules(symbols, coords)
    frames = []
    comments = []
    records = []
    for lam in scale_factors:
        mols = scale_rigid_coms(mols0, lam)
        sym, xyz = molecules_to_water_frame(mols)
        r_oo = min_intermolecular_oo(mols)
        clash, pairs = covalent_overlap_pairs(sym, xyz)
        frames.append((sym, xyz))
        comments.append(
            f"distance_scale parent={parent_label} lambda={lam:.6f} "
            f"r_oo_min={r_oo:.6f} regime={sr_lr_label(r_oo)}"
        )
        records.append(
            {
                "parent": parent_label,
                "n_water": len(mols0),
                "lambda": float(lam),
                "r_oo_min": r_oo,
                "regime": sr_lr_label(r_oo),
                "clash": clash,
                "clash_pairs": [{"i": i, "j": j, "dist": d} for i, j, d in pairs],
            }
        )
    return frames, comments, records


def build_distance_scale_split(
    parent_paths: list[Path],
    out_dir: Path,
    parent_labels: list[str] | None = None,
    scale_factors: list[float] | None = None,
) -> dict:
    if scale_factors is None:
        scale_factors = omol25_scale_factors()
    if parent_labels is None:
        parent_labels = [p.stem for p in parent_paths]
    if len(parent_labels) != len(parent_paths):
        raise ValueError("parent_labels must match parent_paths")

    frames: list[tuple[list[str], np.ndarray]] = []
    comments: list[str] = []
    records: list[dict] = []
    for path, label in zip(parent_paths, parent_labels):
        loaded = read_xyz_symbol_frames(path)
        if not loaded:
            raise ValueError(f"no frames in {path}")
        symbols, coords, _ = loaded[0]
        f, c, r = scan_parent(symbols, coords, label, scale_factors)
        frames.extend(f)
        comments.extend(c)
        records.extend(r)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = out_dir / "ood_distance_scale.xyz"
    manifest_path = out_dir / "manifest.json"
    write_xyz(xyz_path, frames, comments)
    payload = {
        "xyz": str(xyz_path),
        "n_frames": len(frames),
        "n_parents": len(parent_paths),
        "scale_factors": [float(x) for x in scale_factors],
        "records": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    payload["manifest"] = str(manifest_path)
    return payload


def default_parent_paths(minima_dir: Path) -> list[Path]:
    paths = [minima_dir / name for name in DEFAULT_PARENTS]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing water minima: " + ", ".join(str(p) for p in missing))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minima-dir",
        type=Path,
        default=_REPO_ROOT / "datasets/revision/water_clusters/minima",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "datasets/revision/ood_scale",
    )
    args = parser.parse_args()
    result = build_distance_scale_split(default_parent_paths(args.minima_dir), args.output_dir)
    print(json.dumps({k: result[k] for k in ("xyz", "manifest", "n_frames", "n_parents")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
