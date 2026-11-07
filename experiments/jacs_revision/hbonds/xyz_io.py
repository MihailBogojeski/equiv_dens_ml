"""Read and write multi-frame XYZ for the H-bond campaign."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_xyz(
    path: Path,
    frames: list[tuple[list[str], np.ndarray]],
    comments: list[str],
) -> None:
    """Write a multi-frame XYZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for (symbols, coords), comment in zip(frames, comments):
            fh.write(f"{len(symbols)}\n{comment}\n")
            for sym, xyz in zip(symbols, coords):
                fh.write(f"{sym:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}\n")


def read_xyz(path: Path) -> list[tuple[list[str], np.ndarray]]:
    """Load all frames from a multi-frame XYZ."""
    from ase.io import iread

    frames: list[tuple[list[str], np.ndarray]] = []
    for atoms in iread(str(path)):
        frames.append((list(atoms.get_chemical_symbols()), atoms.get_positions().copy()))
    return frames


def frame_natoms(frames: list[tuple[list[str], np.ndarray]]) -> int:
    """Return atom count; all frames must match."""
    if not frames:
        raise ValueError("empty frame list")
    n = len(frames[0][0])
    for symbols, _ in frames:
        if len(symbols) != n:
            raise ValueError("inconsistent atom counts across frames")
    return n
