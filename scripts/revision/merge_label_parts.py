#!/usr/bin/env python3
"""Concatenate the per-array-task label parts back into one dataset pair.

The GPU array splits a trajectory into contiguous index ranges, one output part
per task. This stitches the parts back together in index order and refuses to
write a short file silently: a gap in the middle of a trajectory would quietly
shift every later frame's label onto the wrong geometry.

Usage:
  python scripts/revision/merge_label_parts.py \\
    --dir datasets/revision/water_clusters --prefix water_train \\
    --dens-tag augccpvdz_pbe --chunk 50 --expect 1250
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_PART_RE = re.compile(r"_part(\d+)_pyscf_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--dens-tag", required=True, help="e.g. augccpvdz_pbe")
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--expect", type=int, default=0, help="total frames; 0 = do not check")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    pattern = f"{args.prefix}_part*_pyscf_{args.dens_tag}.npy"
    parts = sorted(args.dir.glob(pattern))
    if not parts:
        raise SystemExit(f"no parts matching {args.dir / pattern}")

    dens: list = []
    positions: list = []
    numbers: list = []
    energies: list = []
    forces: list = []
    dipoles: list = []
    gaps: list[str] = []
    expected_start = 0

    for part in parts:
        match = _PART_RE.search(part.name)
        if match is None:
            raise SystemExit(f"cannot read a part index from {part.name}")
        task = int(match.group(1))
        start = task * args.chunk
        if start != expected_start:
            gaps.append(f"part {task:04d} starts at {start}, expected {expected_start}")

        rows = list(np.load(part, allow_pickle=True))
        dens.extend(rows)

        base = part.with_name(f"{args.prefix}_part{task:04d}_npy.npy")
        if base.exists():
            raw = np.load(base, allow_pickle=True)
            geo = raw.item() if getattr(raw, "shape", ()) == () else raw
            positions.extend(list(geo["positions"]))
            numbers.extend(list(geo["atom_numbers"]))
            if "energy" in geo:
                energies.extend(list(geo["energy"]))
                forces.extend(list(geo["forces"]))
            if "dipole_moment" in geo:
                dipoles.extend(list(geo["dipole_moment"]))
        expected_start = start + len(rows)

    if gaps and not args.allow_incomplete:
        raise SystemExit(
            "refusing to merge: label parts are not contiguous, which would pair "
            "labels with the wrong geometries:\n  " + "\n  ".join(gaps)
        )
    if args.expect and len(dens) != args.expect and not args.allow_incomplete:
        raise SystemExit(
            f"refusing to merge: {len(dens)} frames but --expect {args.expect}; "
            "pass --allow-incomplete to write a partial dataset anyway"
        )

    dens_out = args.dir / f"{args.prefix}_pyscf_{args.dens_tag}.npy"
    np.save(dens_out, np.array(dens, dtype=object), allow_pickle=True)
    print(f"wrote {dens_out} ({len(dens)} frames from {len(parts)} parts)")

    if positions:
        geo_out = args.dir / f"{args.prefix}_npy.npy"
        payload = {
            "positions": np.asarray(positions, dtype=object),
            "atom_numbers": np.asarray(numbers, dtype=object),
        }
        if energies:
            payload["energy"] = np.asarray(energies)
            payload["forces"] = np.asarray(forces, dtype=object)
        if dipoles:
            payload["dipole_moment"] = np.asarray(dipoles)
        np.save(geo_out, payload, allow_pickle=True)
        print(f"wrote {geo_out}")
    if gaps:
        print(f"WARNING: merged with {len(gaps)} contiguity gaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
