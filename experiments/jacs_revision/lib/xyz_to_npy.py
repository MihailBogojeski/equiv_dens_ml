#!/usr/bin/env python3
"""Write positions/atom_numbers NPY from a multi-frame XYZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import iread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames = list(iread(str(args.xyz)))
    if not frames:
        raise ValueError(f"No frames in {args.xyz}")
    n_atoms = max(len(a) for a in frames)
    positions = np.zeros((len(frames), n_atoms, 3), dtype=np.float64)
    numbers = np.zeros((len(frames), n_atoms), dtype=np.int64)
    for i, atoms in enumerate(frames):
        positions[i, : len(atoms)] = atoms.get_positions()
        numbers[i, : len(atoms)] = atoms.get_atomic_numbers()
    if np.all(numbers == numbers[0]):
        numbers = numbers[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, {"positions": positions, "atom_numbers": numbers}, allow_pickle=True)
    print(f"wrote {args.out} frames={len(frames)} n_atoms={n_atoms}")


if __name__ == "__main__":
    main()
