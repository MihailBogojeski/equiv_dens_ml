#!/usr/bin/env python3
"""Turn multi-frame XYZ files into ORCA/PySCF work shards.

qm7x_build_shards.py only reads the QM7-X npy/jsonl layout. The water, ice,
droplet and malonaldehyde splits are all plain XYZ, so they get their own
builder rather than a QM7-X-shaped detour through npy.

Shards are sized by cost, not by frame count: a shard holds frames until their
predicted cost fills the walltime budget, using the scaling measured by
calibrate_theory.py. Frames are ordered smallest-first so an array that gets
cut short still leaves a contiguous, useful block of finished small clusters.

Usage:
  python scripts/revision/build_shards_from_xyz.py \\
    --xyz datasets/revision/water_clusters/train.xyz \\
    --split water_train --theory wb97mv_def2tzvpd \\
    --outdir datasets/revision/shards
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from qm7x_orca_common import is_closed_shell, write_shard  # noqa: E402
from theory_levels import DEFAULT_LEVEL, get_level, level_keys  # noqa: E402

# Wall-clock seconds for one frame, as a(n_atoms)^b fitted to the calibration
# runs. Only used to pack shards, so being within a factor of two is enough;
# the worker's per-frame resume makes an over-full shard recoverable anyway.
COST_MODEL = {
    "wb97mv_def2tzvpd": (0.55, 2.6),
    "pbe0_avdz": (0.30, 2.4),
    "pbe0_d4_avdz": (0.30, 2.4),
    "pbe_d4_avdz": (0.20, 2.4),
}


def read_xyz_frames(path: Path) -> list[dict]:
    from ase.data import atomic_numbers

    lines = Path(path).read_text().splitlines()
    frames: list[dict] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        nat = int(lines[i].strip())
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        z, xyz = [], []
        for row in lines[i + 2 : i + 2 + nat]:
            parts = row.split()
            sym = parts[0]
            num = int(sym) if sym.isdigit() else atomic_numbers[sym[0].upper() + sym[1:].lower()]
            z.append(num)
            xyz.append([float(v) for v in parts[1:4]])
        frames.append(
            {
                "index": len(frames),
                "atom_numbers": z,
                "positions": xyz,
                "comment": comment.strip(),
            }
        )
        i += 2 + nat
    return frames


def predicted_cost_s(n_atoms: int, theory: str) -> float:
    a, b = COST_MODEL.get(theory, COST_MODEL["pbe0_avdz"])
    return a * float(n_atoms) ** b


def pack_shards(frames: list[dict], theory: str, budget_s: float, max_frames: int) -> list[list[dict]]:
    """Greedy smallest-first packing under a per-shard time budget."""
    ordered = sorted(frames, key=lambda f: (len(f["atom_numbers"]), f["index"]))
    shards: list[list[dict]] = []
    current: list[dict] = []
    accumulated = 0.0
    for frame in ordered:
        cost = predicted_cost_s(len(frame["atom_numbers"]), theory)
        too_long = current and (accumulated + cost > budget_s)
        too_many = len(current) >= max_frames
        if too_long or too_many:
            shards.append(current)
            current, accumulated = [], 0.0
        current.append(frame)
        accumulated += cost
    if current:
        shards.append(current)
    return shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", required=True, type=Path, nargs="+")
    parser.add_argument("--split", required=True)
    parser.add_argument("--theory", default=DEFAULT_LEVEL, choices=level_keys())
    parser.add_argument("--outdir", type=Path, default=_REPO_ROOT / "datasets/revision/shards")
    parser.add_argument("--walltime-s", type=float, default=4 * 3600.0)
    parser.add_argument("--safety", type=float, default=0.7)
    parser.add_argument("--max-frames-per-shard", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-atoms", type=int, default=0)
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=0,
        help="carve a size tier out of the input so it can be given its own walltime and core count",
    )
    parser.add_argument("--include-open-shell", action="store_true")
    args = parser.parse_args()

    theory = get_level(args.theory)

    frames: list[dict] = []
    for path in args.xyz:
        for frame in read_xyz_frames(path):
            frame = dict(frame)
            frame["index"] = len(frames)
            frame["source"] = str(path)
            frames.append(frame)

    if args.min_atoms or args.max_atoms:
        upper = args.max_atoms or 10**9
        frames = [f for f in frames if args.min_atoms <= len(f["atom_numbers"]) <= upper]

    skipped = []
    if not args.include_open_shell:
        keep = []
        for frame in frames:
            if is_closed_shell(frame["atom_numbers"]):
                keep.append(frame)
            else:
                skipped.append(frame["index"])
        frames = keep
    if args.max_frames:
        frames = frames[: args.max_frames]

    budget = args.walltime_s * args.safety
    shards = pack_shards(frames, args.theory, budget, args.max_frames_per_shard)

    shard_dir = args.outdir / args.split / args.theory
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("shard_*.json"):
        old.unlink()

    for shard_id, group in enumerate(shards):
        write_shard(shard_dir / f"shard_{shard_id:04d}.json", shard_id, args.split, group)

    manifest = {
        "split": args.split,
        "theory": args.theory,
        "theory_label": theory.label,
        "engines": list(theory.engines),
        "sources": [str(p) for p in args.xyz],
        "n_frames": len(frames),
        "n_shards": len(shards),
        "min_atoms": args.min_atoms,
        "max_atoms": args.max_atoms,
        "atom_counts": sorted({len(f["atom_numbers"]) for f in frames}),
        "walltime_budget_s": budget,
        "max_frames_per_shard": args.max_frames_per_shard,
        "skipped_open_shell": skipped,
        "shard_sizes": [len(g) for g in shards],
        "predicted_shard_cost_s": [
            round(sum(predicted_cost_s(len(f["atom_numbers"]), args.theory) for f in g), 1)
            for g in shards
        ],
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"{args.split} @ {args.theory}: {len(frames)} frames -> {len(shards)} shards "
        f"in {shard_dir} (array 0-{max(0, len(shards) - 1)})"
    )
    if skipped:
        print(f"  skipped {len(skipped)} open-shell frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
