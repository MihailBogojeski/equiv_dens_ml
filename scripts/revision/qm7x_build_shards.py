#!/usr/bin/env python3
"""Build JSON shard manifests for the QM7-X ORCA CPU array.

Default source is the existing DensNet split npy
(datasets/qm7x_{train,valid,test}_dft_augccpvdz_base.npy). Official JSONL from
qm7x_extract_geoms.py is also accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from qm7x_orca_common import (  # noqa: E402
    electron_count,
    is_closed_shell,
    iter_base_frames,
    load_base_npy,
    read_jsonl,
    write_shard,
)

DEFAULT_NPY = {
    "train": _REPO_ROOT / "datasets/qm7x_train_dft_augccpvdz_base.npy",
    "valid": _REPO_ROOT / "datasets/qm7x_valid_dft_augccpvdz_base.npy",
    "test": _REPO_ROOT / "datasets/qm7x_test_dft_augccpvdz_base.npy",
}


def frames_from_npy(path: Path, max_frames: int, offset: int) -> list[dict]:
    frames = list(iter_base_frames(load_base_npy(path)))
    if offset:
        frames = frames[offset:]
    if max_frames:
        frames = frames[:max_frames]
    return frames


def frames_from_jsonl(path: Path, max_frames: int, offset: int) -> list[dict]:
    rows = read_jsonl(path)
    frames = []
    for i, row in enumerate(rows):
        frames.append(
            {
                "index": i,
                "qm7x_id": row.get("qm7x_id"),
                "atom_numbers": row["atom_numbers"],
                "positions": row["positions"],
                "ePBE0_MBD_eV": row.get("ePBE0_MBD_eV"),
            }
        )
    if offset:
        frames = frames[offset:]
    if max_frames:
        frames = frames[:max_frames]
    return frames


def annotate_shell(frames: list[dict]) -> tuple[list[dict], list[dict]]:
    kept, skipped = [], []
    for fr in frames:
        rec = dict(fr)
        rec["n_elec"] = electron_count(fr["atom_numbers"])
        rec["closed_shell"] = is_closed_shell(fr["atom_numbers"])
        if rec["closed_shell"]:
            kept.append(rec)
        else:
            skipped.append(rec)
    return kept, skipped


def shard_frames(frames: list[dict], frames_per_shard: int) -> list[list[dict]]:
    if frames_per_shard < 1:
        raise ValueError("frames_per_shard must be >= 1")
    return [frames[i : i + frames_per_shard] for i in range(0, len(frames), frames_per_shard)]


def slurm_array_ranges(n_tasks: int, max_array: int = 10000) -> list[tuple[int, int]]:
    """Split task indices into Greene-legal Slurm array ranges (max 10,000)."""
    if n_tasks < 1:
        return []
    if max_array < 1:
        raise ValueError("max_array must be >= 1")
    ranges = []
    start = 0
    while start < n_tasks:
        end = min(start + max_array, n_tasks) - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "valid", "test", "smoke"), default="smoke")
    parser.add_argument("--source", choices=("npy", "jsonl"), default="npy")
    parser.add_argument("--npy", default=None)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--frames-per-shard", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--include-open-shell", action="store_true")
    args = parser.parse_args()

    split = "train" if args.split == "smoke" else args.split
    if args.source == "npy":
        npy_path = Path(args.npy) if args.npy else DEFAULT_NPY[split]
        frames = frames_from_npy(npy_path, args.max_frames, args.offset)
        source_label = str(npy_path)
    else:
        jsonl_path = Path(args.jsonl) if args.jsonl else _REPO_ROOT / "datasets/revision/qm7x/qm7x_official.jsonl"
        frames = frames_from_jsonl(jsonl_path, args.max_frames, args.offset)
        source_label = str(jsonl_path)

    kept, skipped = annotate_shell(frames)
    if args.include_open_shell:
        used = frames
        skipped = []
    else:
        used = kept

    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "datasets/revision/qm7x/shards" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = shard_frames(used, args.frames_per_shard)
    paths = []
    for i, chunk in enumerate(shards):
        dest = out_dir / f"shard_{i:04d}.json"
        write_shard(dest, i, args.split, chunk)
        paths.append(str(dest))

    manifest = {
        "split": args.split,
        "source": source_label,
        "n_input": len(frames),
        "n_closed_shell": len(kept),
        "n_open_shell_skipped": len(skipped),
        "n_queued": len(used),
        "frames_per_shard": args.frames_per_shard,
        "n_shards": len(shards),
        "shards": paths,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if skipped:
        (out_dir / "open_shell_skipped.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in skipped)
        )
    print(
        f"{args.split}: queued {len(used)}/{len(frames)} frames in {len(shards)} shards "
        f"(skipped {len(skipped)} open-shell) -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
