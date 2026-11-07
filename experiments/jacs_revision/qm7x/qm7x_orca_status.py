#!/usr/bin/env python3
"""Report QM7-X ORCA labelling progress and build Slurm array specs.

Handles the cluster MaxArraySize=10000 limit by expressing shard indices >= 10000
as (ARRAY_OFFSET, local_array_task_id) pairs.
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/qm7x/qm7x_orca_status.py'


import argparse
import json
import sys
from pathlib import Path


def repo_root() -> Path:
    return _LEGACY_FILE.resolve().parents[3]


def shard_results_path(outdir: Path, shard_id: int) -> Path:
    return outdir / f"shard_{shard_id:04d}" / "results.npy"


def is_done(outdir: Path, shard_id: int) -> bool:
    return shard_results_path(outdir, shard_id).is_file()


def compress_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    indices = sorted(indices)
    ranges: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))
    return ranges


def slurm_array_jobs(
    missing: list[int],
    *,
    max_width: int = 2500,
    max_index: int = 9999,
) -> list[dict[str, int | str]]:
    """Group missing shard ids into sbatch --array jobs with optional offset."""
    jobs: list[dict[str, int | str]] = []
    if not missing:
        return jobs

    # Process each MaxArraySize window separately.
    windows: dict[int, list[int]] = {}
    for shard_id in missing:
        offset = (shard_id // (max_index + 1)) * (max_index + 1)
        local = shard_id - offset
        windows.setdefault(offset, []).append(local)

    for offset in sorted(windows):
        locals_missing = sorted(windows[offset])
        for chunk_start in range(0, len(locals_missing), max_width):
            chunk = locals_missing[chunk_start : chunk_start + max_width]
            for lo, hi in compress_ranges(chunk):
                jobs.append(
                    {
                        "offset": offset,
                        "array": f"{lo}-{hi}",
                        "n_tasks": hi - lo + 1,
                    }
                )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "valid", "smoke"))
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--summary-and-jobs", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    outdir = args.out_root or root / "results/revision/qm7x_orca" / args.split
    manifest = root / "datasets/revision/qm7x/shards" / args.split / "manifest.json"
    if manifest.is_file():
        n_shards = json.loads(manifest.read_text())["n_shards"]
    elif args.split == "train":
        n_shards = 20000
    elif args.split == "valid":
        n_shards = 2500
    else:
        n_shards = 100

    done = sum(1 for i in range(n_shards) if is_done(outdir, i))
    missing = [i for i in range(n_shards) if not is_done(outdir, i)]
    jobs = slurm_array_jobs(missing)

    payload = {
        "split": args.split,
        "n_shards": n_shards,
        "n_done": done,
        "n_missing": len(missing),
        "jobs": jobs,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    if args.summary_and_jobs:
        print(f"{done}/{n_shards} done missing={len(missing)}", end="")
        if jobs:
            print(" jobs=" + ";".join(f"off{j['offset']}:{j['array']}" for j in jobs))
        else:
            print(" jobs=")
        return 0

    print(f"{args.split}: {done}/{n_shards} labelled ({len(missing)} missing)")
    for job in jobs:
        print(f"  offset={job['offset']} array={job['array']} ({job['n_tasks']} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
