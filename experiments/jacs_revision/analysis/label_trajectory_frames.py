#!/usr/bin/env python3
"""Full-SCF reference labels for consecutive frames of an archived ML-MD trajectory.

Writes a label pair in the format of the revision label sets (a base .npy with
geometries and a density .npy of (mol.pack(), calc) pairs), so that
density_error_cubes.py can compare DenSNet with DFT frame by frame. Sharded:
each shard writes <out>_part<k>.npy; --merge concatenates them.

    .venv/bin/python -B experiments/jacs_revision/analysis/label_trajectory_frames.py \
        --traj paper/trajectories/ethanethiol/simulation_-ethanethiol_cluster_all_001_compressed_0.npy \
        --start 20000 --stride 2 --n 240 --level pbe_d4_avdz --shard 0 --n-shards 4 \
        --out datasets/revision/density_movie/ethanethiol_traj
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(REPO), str(REPO / "src")]


def load_traj(path: Path):
    d = np.load(path, allow_pickle=True)
    d = d.item() if d.dtype == object and d.shape == () else d
    z = np.asarray(d["atom_numbers"])
    return (z if z.ndim == 1 else z[0]), np.asarray(d["positions"], float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--level", default="pbe_d4_avdz")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    ap.add_argument("--merge", action="store_true", help="concatenate the shards into <out>_base.npy and <out>.npy")
    opt = ap.parse_args()

    if opt.merge:
        parts = sorted(opt.out.parent.glob(opt.out.name + "_part*.npy"), key=lambda p: int(p.stem.split("_part")[-1]))
        recs = [r for p in parts for r in np.load(p, allow_pickle=True).item()["records"]]
        recs.sort(key=lambda r: r["frame"])
        z = recs[0]["atom_numbers"]
        base = {
            "positions": np.stack([r["positions"] for r in recs]),
            "atom_numbers": np.repeat(z[None], len(recs), 0),
            "atom_types": sorted(set(int(a) for a in z)),
            "energy": np.array([r["calc"]["energy"] for r in recs]),
            "forces": np.zeros((len(recs), len(z), 3)),
            "dipole_moment": np.stack([r["calc"]["dipole"] for r in recs]),
            "source_index": np.array([r["frame"] for r in recs]),
        }
        dens = np.empty((len(recs), 2), dtype=object)
        for i, r in enumerate(recs):
            dens[i, 0], dens[i, 1] = r["mol"], r["calc"]
        np.save(f"{opt.out}_base.npy", base, allow_pickle=True)
        np.save(f"{opt.out}.npy", dens, allow_pickle=True)
        print(f"merged {len(recs)} frames from {len(parts)} shards -> {opt.out}_base.npy, {opt.out}.npy")
        return 0

    from experiments.jacs_revision.lib.pyscf_labeler import label_frame

    z, pos = load_traj(opt.traj)
    frames = [opt.start + i * opt.stride for i in range(opt.n)][opt.shard::opt.n_shards]
    out = Path(f"{opt.out}_part{opt.shard}.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    records = np.load(out, allow_pickle=True).item()["records"] if out.is_file() else []
    done = {r["frame"] for r in records}
    for k, f in enumerate(frames):
        if f in done:
            continue
        mol, calc, diag = label_frame(z, pos[f], opt.level, use_gpu=False, fit_df=True, with_forces=False)
        if not diag["converged"]:
            print(f"frame {f}: SCF not converged, skipped", flush=True)
            continue
        records.append({"frame": int(f), "atom_numbers": z, "positions": pos[f], "mol": mol.pack(), "calc": calc})
        np.save(out, {"records": records, "traj": str(opt.traj), "level": opt.level}, allow_pickle=True)
        print(f"[{k + 1}/{len(frames)}] frame {f}: {diag['t_scf_s']:.1f}s scf, {diag['scf_cycles']} cycles", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
