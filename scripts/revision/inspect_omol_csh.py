#!/usr/bin/env python3
"""Print the layout of an OMol_CSH HDF5 file.

We need to know how the Fock/overlap/density matrices are stored (and in which
orbital convention) before any of it can be turned into DenSNet density-fitting
labels, so this walks the first few entries and dumps names/shapes/dtypes.

Usage:
  python scripts/revision/inspect_omol_csh.py datasets/revision/omol_csh/omol_csh_1k_test_common.h5
"""

from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np


def describe(obj, indent: int) -> str:
    pad = "  " * indent
    if isinstance(obj, h5py.Dataset):
        preview = ""
        if obj.size and obj.size <= 12:
            preview = f" = {np.asarray(obj[()]).ravel()[:12]}"
        elif obj.size:
            flat = np.asarray(obj[()]).ravel()
            preview = f" first={flat[:4]} min={flat.min():.6g} max={flat.max():.6g}"
        return f"{pad}{obj.name.split('/')[-1]}: shape={obj.shape} dtype={obj.dtype}{preview}"
    return f"{pad}{obj.name.split('/')[-1]}/ (group, {len(obj)} keys)"


def walk(node, indent: int, max_children: int, depth: int) -> None:
    if depth < 0:
        return
    for i, key in enumerate(node.keys()):
        if i >= max_children:
            print("  " * indent + f"... {len(node) - max_children} more")
            break
        child = node[key]
        print(describe(child, indent))
        if isinstance(child, h5py.Group):
            walk(child, indent + 1, max_children, depth - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--entries", type=int, default=2, help="top-level entries to expand")
    parser.add_argument("--max-children", type=int, default=40)
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()

    with h5py.File(args.path, "r") as fh:
        print(f"file: {args.path}")
        print(f"root attrs: {dict(fh.attrs)}")
        keys = list(fh.keys())
        print(f"top-level entries: {len(keys)}")
        print(f"first keys: {keys[:8]}")

        for key in keys[: args.entries]:
            print(f"\n=== entry {key} ===")
            node = fh[key]
            if isinstance(node, h5py.Group):
                print(f"attrs: {dict(node.attrs)}")
                walk(node, 1, args.max_children, args.depth)
            else:
                print(describe(node, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
