#!/usr/bin/env python3
"""Turn Rackers Zenodo energy/force pickles into a multi-frame XYZ.

Only n=3–30 (oxygen count). n=53 is left out of the ORCA wave on purpose.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_Z_TO_SYM = {1: "H", 8: "O"}


def frames_from_pkl(path: Path) -> list[tuple[list[str], np.ndarray, str]]:
    rows = pickle.load(path.open("rb"))
    if isinstance(rows, dict):
        rows = list(rows.values())
    out = []
    for i, rec in enumerate(rows):
        z = np.asarray(rec["type"]).reshape(-1)
        pos = np.asarray(rec["pos"], dtype=float).reshape(-1, 3)
        symbols = [_Z_TO_SYM.get(int(v), str(int(v))) for v in z]
        n_w = sum(1 for s in symbols if s == "O")
        out.append((symbols, pos, f"rackers {path.stem} i={i} n={n_w}"))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--xyz-out", type=Path, required=True)
    parser.add_argument("--max-n", type=int, default=30)
    args = parser.parse_args(argv)
    pkls = sorted(args.src.glob("w*_energy_force.pkl"))
    pkls = [p for p in pkls if "ccsd" not in p.name]
    frames = []
    for path in pkls:
        for symbols, pos, comment in frames_from_pkl(path):
            n_w = sum(1 for s in symbols if s == "O")
            if 3 <= n_w <= args.max_n:
                frames.append((symbols, pos, comment))
    args.xyz_out.parent.mkdir(parents=True, exist_ok=True)
    with args.xyz_out.open("w") as handle:
        for symbols, pos, comment in frames:
            handle.write(f"{len(symbols)}\n{comment}\n")
            for sym, xyz in zip(symbols, pos):
                handle.write(f"{sym} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}\n")
    print(f"wrote {args.xyz_out} ({len(frames)} frames from {len(pkls)} pkls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
