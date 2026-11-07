#!/usr/bin/env python3
"""Index the OMol_CSH HDF5 files and select the trainable subset.

Selection is driven by what the unmodified DenSNet code supports: the shipped
orbital-basis file `datasets/augccpvqzjkfit_orbital_basis_df.npy` defines the ML
channel layout and only covers H, C, N, O, S, and neutral systems keep the
integral constraint and the neutral-atom SAD prior valid without touching the
network.

Entries are emitted smallest first so a usable training set exists early: the
density-fitting cost grows roughly as nao^2 * naux, so the large structures
dominate the wall clock.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

# Multiplicity is 1 throughout (CSH is closed shell), which anchors the parse:
# names end either in _{charge}_1 or _{charge}_1_{frame}.
_CHARGE_TAIL = re.compile(r"_(-?\d+)_1$")
_CHARGE_TAIL_FRAME = re.compile(r"_(-?\d+)_1_\d+$")


def parse_charge(leaf: str, parent: str | None) -> int | None:
    """Charge from the group name, falling back to the parent.

    The `omol` subset names its leaves `step0`, `step1`, ... and carries the
    charge on the parent group instead.
    """
    for name in (leaf, parent):
        if name is None:
            continue
        m = _CHARGE_TAIL.search(name) or _CHARGE_TAIL_FRAME.search(name)
        if m:
            return int(m.group(1))
    return None


def triangle_dim(length: int) -> int:
    return int(round((np.sqrt(8.0 * length + 1.0) - 1.0) / 2.0))


def iter_entries(handle):
    stack = [(handle, None, None)]
    while stack:
        group, parent, top = stack.pop()
        keys = list(group)
        if "fock" in keys and "elements" in keys:
            yield group, parent, top
            continue
        for key in keys:
            child = group[key]
            if isinstance(child, h5py.Group):
                stack.append((child, group.name.split("/")[-1], top or key))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5")
    parser.add_argument("--split", required=True, help="train | test_common | test_all")
    parser.add_argument("--elements", default="1,6,7,8,16")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--any-charge", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    allowed = set(int(x) for x in args.elements.split(","))
    total = 0
    unparsed = 0
    charge_hist: Counter = Counter()
    subset_hist: Counter = Counter()
    unparsed_hist: Counter = Counter()
    unparsed_in_elements = 0
    kept = []

    with h5py.File(args.h5, "r") as fh:
        for group, parent, top in iter_entries(fh):
            total += 1
            leaf = group.name.split("/")[-1]
            charge = parse_charge(leaf, parent)
            if charge is None:
                # Entries whose name carries no charge are the `omol/metal_organics`
                # job-id groups and the noble-gas set. Counting how many of those
                # fall inside the allowed elements shows whether anything is
                # actually being lost, rather than leaving a bare drop count.
                unparsed += 1
                unparsed_hist[top] += 1
                zset = set(int(z) for z in np.asarray(group["elements"][()]))
                if zset <= allowed:
                    unparsed_in_elements += 1
                continue
            charge_hist[charge] += 1
            elements = np.asarray(group["elements"][()])
            zset = set(int(z) for z in elements)
            if not zset <= allowed:
                continue
            if not args.any_charge and charge != args.charge:
                continue
            nao = triangle_dim(group["fock"].shape[0])
            subset_hist[top] += 1
            kept.append(
                {
                    "path": group.name,
                    "subset": top,
                    "natoms": int(len(elements)),
                    "nao": int(nao),
                    "charge": int(charge),
                    "elements": sorted(zset),
                }
            )

    kept.sort(key=lambda r: (r["nao"], r["path"]))

    print(f"file      : {args.h5}")
    print(f"split     : {args.split}")
    print(f"total     : {total}   unparsed charge: {unparsed}")
    if unparsed:
        print(f"  unparsed by subset: {dict(unparsed_hist.most_common(6))}")
        print(f"  of which inside allowed elements: {unparsed_in_elements} (lost data)")
    print(f"selected  : {len(kept)} ({100.0 * len(kept) / max(total, 1):.1f}%)")
    print(f"charge histogram (all entries): {dict(sorted(charge_hist.items()))}")

    if kept:
        natoms = np.array([r["natoms"] for r in kept])
        nao = np.array([r["nao"] for r in kept])
        print(f"natoms med/mean/max: {np.median(natoms):.0f} {natoms.mean():.0f} {natoms.max()}")
        print(f"nao    med/mean/max: {np.median(nao):.0f} {nao.mean():.0f} {nao.max()}")
        for cap in (30, 50, 70, 100, 150):
            n = int((natoms <= cap).sum())
            print(f"  tier natoms<={cap:>3}: {n:>6} ({100.0 * n / len(kept):.0f}%)")
        # Relative cost proxy; the density fit dominates and scales as nao^2*naux,
        # with naux itself proportional to system size.
        cost = (nao.astype(float) ** 3) / (nao.astype(float).min() ** 3)
        print(f"  relative cost of full set vs smallest structure: {cost.sum():.3g}x")
        print(f"  subsets: {dict(subset_hist.most_common(12))}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "source": args.h5,
                "split": args.split,
                "elements_allowed": sorted(allowed),
                "charge_filter": None if args.any_charge else args.charge,
                "total_entries": total,
                "selected": len(kept),
                "entries": kept,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
