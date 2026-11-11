#!/usr/bin/env python3
"""Build a small PBE0 proof-of-principle XYZ (R3.1 fallback without OMol25).

Takes ethanol_train_10 plus the first N water dimers/trimers from the
water-cluster train set. Paper ethanol DFT (~400 frames) is not on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def read_xyz(path: Path):
    text = path.read_text().strip().splitlines()
    frames = []
    i = 0
    while i < len(text):
        n = int(text[i].split()[0])
        comment = text[i + 1] if i + 1 < len(text) else ""
        atoms = []
        for line in text[i + 2 : i + 2 + n]:
            parts = line.split()
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        frames.append((comment, atoms))
        i += 2 + n
    return frames


def write_xyz(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for comment, atoms in frames:
            fh.write(f"{len(atoms)}\n{comment}\n")
            for s, x, y, z in atoms:
                fh.write(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ethanol", type=Path, default=Path("datasets/ethanol_train_10.xyz"))
    parser.add_argument(
        "--water-train",
        type=Path,
        default=Path("datasets/revision/water_clusters/train.xyz"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/revision/pbe0/ethanol_water_pbe0_subset.xyz"),
    )
    parser.add_argument("--n-water-dimer", type=int, default=40)
    parser.add_argument("--n-water-trimer", type=int, default=20)
    args = parser.parse_args()

    out = []
    for comment, atoms in read_xyz(args.ethanol):
        out.append((comment or "ethanol_train_10", atoms))

    n2 = n3 = 0
    for comment, atoms in read_xyz(args.water_train):
        n_o = sum(1 for s, *_ in atoms if s == "O")
        if n_o == 2 and n2 < args.n_water_dimer:
            out.append((f"water_dimer {comment}", atoms))
            n2 += 1
        elif n_o == 3 and n3 < args.n_water_trimer:
            out.append((f"water_trimer {comment}", atoms))
            n3 += 1
        if n2 >= args.n_water_dimer and n3 >= args.n_water_trimer:
            break

    write_xyz(args.out, out)
    print(f"wrote {args.out} n={len(out)} ethanol={len(read_xyz(args.ethanol))} n2={n2} n3={n3}")


if __name__ == "__main__":
    main()
