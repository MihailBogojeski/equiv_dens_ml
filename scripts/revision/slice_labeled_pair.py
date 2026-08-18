#!/usr/bin/env python3
"""Align geometry NPY length to a (possibly partial) DFT density NPY.

generate_dft_labels.py writes the dens file incrementally but only rewrites
the geometry NPY at the end of a split. Eval / train need matching lengths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dens", type=Path, required=True)
    parser.add_argument("--geom", type=Path, required=True)
    parser.add_argument("--out-dens", type=Path, required=True)
    parser.add_argument("--out-geom", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    dens = np.load(args.dens, allow_pickle=True)
    n = len(dens)
    if args.max_frames:
        n = min(n, args.max_frames)
    dens = dens[:n]

    geom = np.load(args.geom, allow_pickle=True)
    if isinstance(geom, np.ndarray) and geom.dtype == object and geom.ndim == 0:
        geom = geom.item()
    if not isinstance(geom, dict):
        raise ValueError(f"Expected geometry dict in {args.geom}")

    out = {}
    for key, value in geom.items():
        arr = np.asarray(value, dtype=object) if key in ("positions", "atom_numbers", "forces") else np.asarray(value)
        if arr.shape and arr.shape[0] >= n:
            out[key] = arr[:n]
        else:
            out[key] = value
    if dens.size and isinstance(dens[0], (list, tuple, np.ndarray)) and len(dens[0]) == 2:
        calc0 = dens[0][1]
        if isinstance(calc0, dict):
            if "energy" in calc0 and "energy" not in out:
                out["energy"] = np.array([row[1]["energy"] for row in dens])
            if "forces" in calc0 and "forces" not in out:
                out["forces"] = np.asarray([row[1]["forces"] for row in dens], dtype=object)
            if "dipole" in calc0 and "dipole_moment" not in out:
                out["dipole_moment"] = np.array([row[1]["dipole"] for row in dens])

    args.out_dens.parent.mkdir(parents=True, exist_ok=True)
    args.out_geom.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dens, np.asarray(dens, dtype=object), allow_pickle=True)
    np.save(args.out_geom, out, allow_pickle=True)
    print(f"wrote {args.out_geom} and {args.out_dens} frames={n}")


if __name__ == "__main__":
    main()
