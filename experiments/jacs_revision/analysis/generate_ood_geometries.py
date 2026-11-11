#!/usr/bin/env python3
"""Generate OOD geometries from an existing XYZ (stdlib only).

Modes:
  - affine strain about the center of mass (default 0.90 and 1.10)
  - high-T-like Gaussian Cartesian noise (default 0.12 and 0.20 Angstrom)
"""

from __future__ import annotations

import argparse
import json
import random
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


def write_xyz(path: Path, frames, default_comment):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for comment, atoms in frames:
            fh.write(f"{len(atoms)}\n")
            fh.write(f"{comment or default_comment}\n")
            for s, x, y, z in atoms:
                fh.write(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def com(atoms):
    sx = sy = sz = 0.0
    for _, x, y, z in atoms:
        sx += x
        sy += y
        sz += z
    n = float(len(atoms))
    return sx / n, sy / n, sz / n


def strain_frame(atoms, scale):
    cx, cy, cz = com(atoms)
    out = []
    for s, x, y, z in atoms:
        out.append((s, cx + scale * (x - cx), cy + scale * (y - cy), cz + scale * (z - cz)))
    return out


def noise_frame(atoms, sigma, rng):
    return [
        (s, x + rng.gauss(0.0, sigma), y + rng.gauss(0.0, sigma), z + rng.gauss(0.0, sigma))
        for s, x, y, z in atoms
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--molecule", type=str, default="molecule")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/revision/ood"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--scales", type=str, default="0.90,1.10")
    parser.add_argument("--sigmas", type=str, default="0.12,0.20")
    parser.add_argument("--noise-replicas", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    parents = read_xyz(args.input)
    scales = [float(x) for x in args.scales.split(",") if x]
    sigmas = [float(x) for x in args.sigmas.split(",") if x]

    frames = []
    records = []
    for pi, (comment, atoms) in enumerate(parents):
        frames.append((f"{args.molecule} parent {pi} {comment}", atoms))
        records.append({"parent": pi, "mode": "parent"})
        for scale in scales:
            frames.append((f"{args.molecule} strain {scale} parent={pi}", strain_frame(atoms, scale)))
            records.append({"parent": pi, "mode": "strain", "scale": scale})
        for sigma in sigmas:
            for _ in range(args.noise_replicas):
                frames.append(
                    (f"{args.molecule} noise sigma={sigma} parent={pi}", noise_frame(atoms, sigma, rng))
                )
                records.append({"parent": pi, "mode": "highT_noise", "sigma": sigma})

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    xyz = out / f"{args.molecule}_ood.xyz"
    write_xyz(xyz, frames, f"{args.molecule} OOD")
    manifest = {
        "input": str(args.input),
        "molecule": args.molecule,
        "n_parent": len(parents),
        "n_frames": len(frames),
        "xyz": str(xyz),
        "records": records,
    }
    (out / f"{args.molecule}_ood_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("xyz", "n_parent", "n_frames")}, indent=2))


if __name__ == "__main__":
    main()
