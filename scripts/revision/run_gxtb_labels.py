#!/usr/bin/env python3
"""Single-point g-xTB energies on a multi-frame XYZ (CPU only, no GPU)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
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


def write_xyz(path: Path, atoms, comment=""):
    with path.open("w") as fh:
        fh.write(f"{len(atoms)}\n{comment}\n")
        for s, x, y, z in atoms:
            fh.write(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def parse_energy(stdout: str, workdir: Path) -> float | None:
    energy_path = workdir / "energy"
    if energy_path.exists():
        for line in energy_path.read_text().splitlines():
            if line.strip().startswith("$"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    for pat in (
        r"total energy\s*[:=]\s*([-\d.]+)",
        r"Etot\s*[:=]\s*([-\d.]+)",
        r"SCF energy\s*[:=]\s*([-\d.]+)",
    ):
        m = re.search(pat, stdout, re.I)
        if m:
            return float(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gxtb", type=Path, default=Path("g-xtb/binary/gxtb"))
    parser.add_argument("--gxtbhome", type=Path, default=Path("g-xtb/parameters"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    frames = read_xyz(args.xyz)
    if args.max_frames:
        frames = frames[args.offset : args.offset + args.max_frames]
    else:
        frames = frames[args.offset :]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GXTBHOME"] = str(args.gxtbhome.resolve())
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    gxtb = str(args.gxtb.resolve())
    env["PATH"] = str(Path(gxtb).parent) + os.pathsep + env.get("PATH", "")

    records = []
    if args.out.exists():
        records = [json.loads(l) for l in args.out.read_text().splitlines() if l.strip()]
    done = {r["index"] for r in records}

    with tempfile.TemporaryDirectory(prefix="gxtb_rev_") as td:
        work = Path(td)
        for i, (comment, atoms) in enumerate(frames, start=args.offset):
            if i in done:
                continue
            coord = work / "coord.xyz"
            write_xyz(coord, atoms, comment)
            result = subprocess.run(
                [gxtb, "-c", "coord.xyz"],
                cwd=str(work),
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            energy = parse_energy((result.stdout or "") + (result.stderr or ""), work)
            rec = {
                "index": i,
                "comment": comment,
                "n_atoms": len(atoms),
                "energy_hartree": energy,
                "returncode": result.returncode,
            }
            with args.out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if result.returncode != 0 and energy is None:
                tail = (result.stdout or result.stderr or "")[-400:]
                print(f"gxtb failed frame {i}: {tail}", flush=True)


if __name__ == "__main__":
    main()
