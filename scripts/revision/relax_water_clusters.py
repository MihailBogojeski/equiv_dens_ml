#!/usr/bin/env python3
"""Relax water clusters with a trained DenSNet and write the geometries (R1.1).

Reviewer 1 asked for hydrogen-bond statistics on clusters larger than anything
in training. A density model does not predict geometries directly, so the
comparable quantity is the structure the model's own energy surface relaxes to:
minimise each reference cluster under DenSNet, then histogram the O-O and O-H
distances of the result against the same histogram of the DFT geometries
(``evaluate_hbond_metrics.py --pred-xyz``).

Each frame is relaxed independently and failures are recorded rather than
raised, because one non-converging 24-water cluster should not discard the
other 299. The written XYZ holds only the frames that converged, and the
sidecar JSON says which those were -- the histogram script warns when the two
frame counts differ, so a silently truncated ensemble cannot be mistaken for
agreement.

Relaxation starts from the reference geometry rather than from a perturbed one
on purpose: the question is whether the model's minimum sits where DFT's does,
so any displacement in the output is the model's doing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "md"))


def read_xyz(path: Path):
    lines = path.read_text().strip().splitlines()
    frames, i = [], 0
    while i < len(lines):
        n = int(lines[i].split()[0])
        symbols, positions = [], []
        for line in lines[i + 2 : i + 2 + n]:
            p = line.split()
            symbols.append(p[0])
            positions.append([float(p[1]), float(p[2]), float(p[3])])
        frames.append((symbols, np.asarray(positions, dtype=float)))
        i += 2 + n
    return frames


def write_xyz(path: Path, frames, comments=None):
    out = []
    for k, (symbols, positions) in enumerate(frames):
        out.append(str(len(symbols)))
        out.append(comments[k] if comments else "")
        for s, (x, y, z) in zip(symbols, positions):
            out.append(f"{s} {x:.8f} {y:.8f} {z:.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")


def rmsd(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xyz", type=Path, required=True, help="reference geometries to relax")
    ap.add_argument("--run-dir", required=True,
                    help="trained DenSNet run directory (holds args.txt and checkpoints)")
    ap.add_argument("--args-file", default=None)
    ap.add_argument("--out-xyz", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--np-dataset", default=None,
                    help="the training set this model was fitted on. Needed because the "
                         "output scaling is the standard deviation of its forces; without "
                         "it predictions come back in normalised units and the relaxation "
                         "barely moves the geometry")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="relax only the first N frames (0 = all)")
    ap.add_argument("--use-gpu", action="store_true")
    cli = ap.parse_args()

    frames = read_xyz(cli.xyz)
    if cli.limit:
        frames = frames[: cli.limit]
    print(f"relaxing {len(frames)} frames from {cli.xyz}")

    from equiv_dens.md.dft_network_calculator import load_densnet_calculator

    calc = load_densnet_calculator(cli.run_dir, args_file=cli.args_file,
                                   np_dataset=cli.np_dataset, use_gpu=cli.use_gpu)

    relaxed, comments, records = [], [], []
    t0 = time.time()
    for idx, (symbols, positions) in enumerate(frames):
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc
        rec = {"index": idx, "n_atoms": len(symbols)}
        try:
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=cli.fmax, steps=cli.max_steps)
            new_pos = atoms.get_positions()
            rec["converged"] = bool(opt.converged())
            rec["n_steps"] = int(opt.get_number_of_steps())
            rec["rmsd_from_reference"] = rmsd(new_pos, positions)
            relaxed.append((symbols, new_pos))
            comments.append(
                f"densnet relaxed index={idx} rmsd={rec['rmsd_from_reference']:.4f} "
                f"converged={rec['converged']}"
            )
        except Exception as exc:
            rec["converged"] = False
            rec["error"] = str(exc)
        records.append(rec)
        if (idx + 1) % 10 == 0 or idx + 1 == len(frames):
            ok = sum(1 for r in records if r.get("converged"))
            print(f"  [{idx + 1}/{len(frames)}] converged={ok} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    write_xyz(cli.out_xyz, relaxed, comments)
    ok = [r for r in records if r.get("converged")]
    moved = [r["rmsd_from_reference"] for r in records if "rmsd_from_reference" in r]
    summary = {
        "reference_xyz": str(cli.xyz),
        "run_dir": cli.run_dir,
        "out_xyz": str(cli.out_xyz),
        "n_frames": len(frames),
        "n_written": len(relaxed),
        "n_converged": len(ok),
        "fmax": cli.fmax,
        "max_steps": cli.max_steps,
        "mean_rmsd_from_reference": float(np.mean(moved)) if moved else None,
        "median_rmsd_from_reference": float(np.median(moved)) if moved else None,
        "max_rmsd_from_reference": float(np.max(moved)) if moved else None,
        "wall_seconds": time.time() - t0,
        "records": records,
    }
    cli.out_json.parent.mkdir(parents=True, exist_ok=True)
    cli.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {cli.out_xyz} ({len(relaxed)} frames) and {cli.out_json}")
    print(f"converged {len(ok)}/{len(frames)}; "
          f"median RMSD from reference {summary['median_rmsd_from_reference']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
