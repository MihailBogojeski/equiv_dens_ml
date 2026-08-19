#!/usr/bin/env python3
"""Report which shards of a labelling campaign are still outstanding.

The watchdog needs two things that are awkward in bash: a shard is finished when
`shard_XXXX/shard.done` exists, and a shard held by a *live* worker must not be
re-submitted while the claim is fresh. Both are per-shard filesystem facts, so
this reads them once and prints either a human summary or a Slurm array spec
covering only the shards that still need work.

Re-submitting every index would also work -- the worker exits immediately on a
finished shard -- but on a 484-task campaign that is hundreds of scheduler slots
spent on tasks that do nothing, which slows down the shards that do need to run.

Usage:
  campaign_status.py --shard-dir DIR --outdir DIR [--array-spec] [--stale-s S]
  campaign_status.py --root datasets/revision/shards --out-root results/... --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# Imported rather than re-implemented: this decides what to re-submit and
# acquire_claim decides what a worker will accept, so the two answering the same
# question differently is a livelock, not a discrepancy.
from shard_claim import claim_task_id, live_task_ids  # noqa: E402

#: Matches the large tier's --claim-stale-s. Only used for claims with no Slurm
#: id recorded; when there is one, the scheduler is asked directly.
DEFAULT_STALE_S = 86400.0


def shard_indices(shard_dir: Path) -> list[int]:
    out = []
    for path in sorted(shard_dir.glob("shard_*.json")):
        try:
            out.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return out


def shard_state(outdir: Path, index: int, stale_s: float, live: set[str] | None) -> str:
    """One of ``done``, ``claimed``, ``partial``, ``pending``."""
    d = outdir / f"shard_{index:04d}"
    if (d / "shard.done").exists():
        return "done"
    claim = d / "shard.claim"
    try:
        fresh = time.time() - claim.stat().st_mtime <= stale_s
    except FileNotFoundError:
        fresh = False
    if fresh:
        owner = claim_task_id(claim)
        # A claim with a known-dead owner is dead now, not when it ages out.
        if owner is None or live is None or owner in live:
            return "claimed"
    return "partial" if (d / "status.jsonl").exists() else "pending"


def survey(shard_dir: Path, outdir: Path, stale_s: float, live: set[str] | None = None) -> dict:
    indices = shard_indices(shard_dir)
    states = {i: shard_state(outdir, i, stale_s, live) for i in indices}
    counts: dict[str, int] = {}
    for state in states.values():
        counts[state] = counts.get(state, 0) + 1
    outstanding = [i for i, s in states.items() if s in ("pending", "partial")]
    return {
        "shard_dir": str(shard_dir),
        "outdir": str(outdir),
        "n_shards": len(indices),
        "counts": counts,
        "n_done": counts.get("done", 0),
        "outstanding": outstanding,
    }


def array_spec(indices: list[int]) -> str:
    """Collapse indices into the range syntax Slurm accepts (``0-3,7,9-11``)."""
    if not indices:
        return ""
    parts = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = i
    parts.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(parts)


def discover(root: Path, out_root: Path) -> list[tuple[str, str, Path, Path]]:
    """Every (split, theory) pair that has shards built under `root`."""
    pairs = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for theory_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if not any(theory_dir.glob("shard_*.json")):
                continue
            outdir = out_root / split_dir.name / theory_dir.name
            pairs.append((split_dir.name, theory_dir.name, theory_dir, outdir))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--root", type=Path, help="scan every split/theory under this root")
    parser.add_argument("--out-root", type=Path, default=Path("results/revision/water_orca"))
    parser.add_argument("--stale-s", type=float, default=DEFAULT_STALE_S)
    parser.add_argument("--array-spec", action="store_true", help="print only the Slurm spec")
    parser.add_argument(
        "--summary-and-spec",
        action="store_true",
        help="one whitespace-free summary token and the Slurm spec, on one line, "
        "so the watchdog gets both from a single scheduler query",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    live = live_task_ids()

    if args.shard_dir is not None:
        if args.outdir is None:
            parser.error("--outdir is required with --shard-dir")
        info = survey(args.shard_dir, args.outdir, args.stale_s, live)
        if args.summary_and_spec:
            bits = ",".join(f"{k}={v}" for k, v in sorted(info["counts"].items()))
            summary = f"{info['n_done']}/{info['n_shards']}_done_[{bits}]"
            print(f"{summary} {array_spec(info['outstanding'])}")
        elif args.array_spec:
            print(array_spec(info["outstanding"]))
        elif args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"{info['n_done']}/{info['n_shards']} done  {info['counts']}")
        return 0

    if args.root is None:
        parser.error("pass either --shard-dir/--outdir or --root")

    rows = []
    for split, theory, shard_dir, outdir in discover(args.root, args.out_root):
        info = survey(shard_dir, outdir, args.stale_s, live)
        info["split"], info["theory"] = split, theory
        rows.append(info)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    total_done = sum(r["n_done"] for r in rows)
    total = sum(r["n_shards"] for r in rows)
    for r in rows:
        bits = " ".join(f"{k}={v}" for k, v in sorted(r["counts"].items()))
        print(f"{r['split']:38s} {r['theory']:18s} {r['n_done']:4d}/{r['n_shards']:<4d}  {bits}")
    print(f"{'TOTAL':38s} {'':18s} {total_done:4d}/{total:<4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
