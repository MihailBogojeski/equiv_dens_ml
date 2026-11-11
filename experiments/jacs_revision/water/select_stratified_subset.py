#!/usr/bin/env python3
"""Stratified subsample of an XYZ tree for the hybrid DFT wave.

Wave 1 labels every Phase A frame at PBE-D4. Wave 2 needs only enough ωB97M-V
to rebuild the PBE-versus-hybrid table: a 10% draw, stratified by split and
atom count (and by the malonaldehyde δ bin when the comment carries one).

Usage:
  python scripts/revision/water/select_stratified_subset.py \\
    --src-water datasets/revision/water_clusters_md10x \\
    --src-ood datasets/revision/water_ood_md10x \\
    --src-malon datasets/revision/malonaldehyde_md10x \\
    --out-water datasets/revision/water_clusters_md10x_hybrid10 \\
    --frac 0.1
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/select_stratified_subset.py'


import argparse
import json
import re
import sys
from collections import defaultdict
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

from experiments.jacs_revision.water.md_sampling import Frame, read_xyz_symbol_frames, write_xyz_frames  # noqa: E402

_REPO_ROOT = repo_root()
_DELTA_RE = re.compile(r"delta=([+-]?\d+(?:\.\d+)?)")


def bin_key(symbols: list[str], comment: str, n_delta_bins: int = 6) -> tuple:
    """Stratum: atom count, plus a δ bin when the comment records one."""
    n_atoms = len(symbols)
    match = _DELTA_RE.search(comment)
    if match is None:
        return (n_atoms,)
    delta = float(match.group(1))
    # Scan spans roughly [-1.5, +0.8]; six equal bins cover that without
    # emptying the far-transfer tail.
    edges = np.linspace(-1.6, 0.9, n_delta_bins + 1)
    bin_id = int(np.clip(np.searchsorted(edges, delta, side="right") - 1, 0, n_delta_bins - 1))
    return (n_atoms, bin_id)


def stratified_sample(frames: list[Frame], frac: float, seed: int) -> list[Frame]:
    """Keep ``ceil(frac * n)`` frames from every stratum, at least one if n>0."""
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, frame in enumerate(frames):
        groups[bin_key(frame.symbols, frame.comment)].append(i)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for key in sorted(groups):
        idxs = np.asarray(groups[key], dtype=int)
        rng.shuffle(idxs)
        n_keep = max(1, int(np.ceil(frac * len(idxs))))
        chosen.extend(int(i) for i in idxs[:n_keep])
    chosen.sort()
    return [frames[i] for i in chosen]


def subset_xyz(src: Path, dest: Path, frac: float, seed: int) -> dict:
    if not src.is_file():
        return {"src": str(src), "skipped": True}
    raw = read_xyz_symbol_frames(src)
    frames = [Frame(s, c, comment, {}) for s, c, comment in raw]
    kept = stratified_sample(frames, frac=frac, seed=seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_xyz_frames(dest, kept)
    return {"src": str(src), "dest": str(dest), "n_src": len(frames), "n_kept": len(kept), "frac": frac}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-water", type=Path, default=_REPO_ROOT / "datasets/revision/water_clusters_md10x")
    parser.add_argument("--src-ood", type=Path, default=_REPO_ROOT / "datasets/revision/water_ood_md10x")
    parser.add_argument("--src-malon", type=Path, default=_REPO_ROOT / "datasets/revision/malonaldehyde_md10x")
    parser.add_argument("--out-water", type=Path, default=_REPO_ROOT / "datasets/revision/water_clusters_md10x_hybrid10")
    parser.add_argument("--out-ood", type=Path, default=_REPO_ROOT / "datasets/revision/water_ood_md10x_hybrid10")
    parser.add_argument("--out-malon", type=Path, default=_REPO_ROOT / "datasets/revision/malonaldehyde_md10x_hybrid10")
    parser.add_argument("--frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    jobs = [
        (args.src_water / "train.xyz", args.out_water / "train.xyz"),
        (args.src_water / "val.xyz", args.out_water / "val.xyz"),
        (args.src_water / "id_test.xyz", args.out_water / "id_test.xyz"),
        (args.src_water / "ood_size.xyz", args.out_water / "ood_size.xyz"),
        (args.src_ood / "ood_order.xyz", args.out_ood / "ood_order.xyz"),
        (args.src_ood / "ood_density.xyz", args.out_ood / "ood_density.xyz"),
        (args.src_malon / "train.xyz", args.out_malon / "train.xyz"),
        (args.src_malon / "val.xyz", args.out_malon / "val.xyz"),
        (args.src_malon / "ood_proton_transfer.xyz", args.out_malon / "ood_proton_transfer.xyz"),
    ]
    manifest = {"frac": args.frac, "seed": args.seed, "files": []}
    for src, dest in jobs:
        rec = subset_xyz(src, dest, args.frac, args.seed)
        manifest["files"].append(rec)
        if rec.get("skipped"):
            print(f"skip missing {src}")
        else:
            print(f"{src.name}: {rec['n_src']} -> {rec['n_kept']}")
    for out_dir in (args.out_water, args.out_ood, args.out_malon):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
