#!/usr/bin/env python3
"""Quantify train/test (or train/OOD) geometry overlap (R1.5, R1.2).

Reports:
  - Hungarian-aligned RMSD of each test frame to its nearest train frame
  - pairwise-distance vector Euclidean distance (SOAP-like cheap descriptor)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _load_positions(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".xyz":
        from ase.io import iread

        frames = [np.asarray(atoms.get_positions()) for atoms in iread(str(path))]
        n_atoms = max(len(p) for p in frames)
        out = np.zeros((len(frames), n_atoms, 3), dtype=np.float64)
        for i, p in enumerate(frames):
            out[i, : len(p)] = p
        return out
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item() if data.ndim == 0 else data
    if isinstance(data, dict):
        pos = np.asarray(data["positions"])
        if pos.ndim == 2:
            pos = pos[None, ...]
        return pos
    raise ValueError(f"Unsupported dataset format in {path}")


def _load_numbers(path: Path) -> np.ndarray | None:
    if path.suffix.lower() == ".xyz":
        from ase.io import iread

        frames = [np.asarray(atoms.get_atomic_numbers()) for atoms in iread(str(path))]
        n_atoms = max(len(z) for z in frames)
        out = np.zeros((len(frames), n_atoms), dtype=np.int64)
        for i, z in enumerate(frames):
            out[i, : len(z)] = z
        return out
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item() if data.ndim == 0 else data
    if isinstance(data, dict) and "atom_numbers" in data:
        anum = np.asarray(data["atom_numbers"])
        if anum.ndim == 1:
            return np.broadcast_to(anum, (len(_load_positions(path)), len(anum))).copy()
        return anum
    return None


def _kabsch_rmsd(p, q):
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    cost = cdist(p, q)
    ri, ci = linear_sum_assignment(cost)
    p = p[ri]
    q = q[ci]
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    p_rot = p @ r.T
    return float(np.sqrt(np.mean(np.sum((p_rot - q) ** 2, axis=1))))


def _pair_desc(pos):
    d = cdist(pos, pos)
    iu = np.triu_indices(len(pos), k=1)
    return np.sort(d[iu])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/revision/overlap.json"))
    parser.add_argument("--max-train", type=int, default=400)
    parser.add_argument("--max-test", type=int, default=400)
    args = parser.parse_args()

    train = _load_positions(args.train)[: args.max_train]
    test = _load_positions(args.test)[: args.max_test]
    train_z = _load_numbers(args.train)
    test_z = _load_numbers(args.test)
    if train_z is not None:
        train_z = train_z[: args.max_train]
    if test_z is not None:
        test_z = test_z[: args.max_test]

    groups = {("all", train.shape[1]): (train, test)}
    if train.shape[1] != test.shape[1] or (
        train_z is not None and test_z is not None and train_z.shape[1] != test_z.shape[1]
    ):
        if train_z is None or test_z is None:
            raise ValueError("Train and test atom counts differ; compare per-molecule splits.")
        groups = {}
        for n in sorted(set(int(np.sum(z > 0)) for z in train_z)):
            tr = train[np.array([int(np.sum(z > 0)) == n for z in train_z])]
            te = test[np.array([int(np.sum(z > 0)) == n for z in test_z])]
            if len(tr) and len(te):
                groups[(f"n_atoms={n}", n)] = (tr[:, :n], te[:, :n])
        if not groups:
            raise ValueError("No overlapping atom-count groups between train and test.")

    rmsds = []
    desc_dists = []
    per_group = {}
    for key, (tr, te) in groups.items():
        label = key[0]
        train_desc = np.stack([_pair_desc(p) for p in tr])
        g_rmsds = []
        g_desc = []
        for q in te:
            g_rmsds.append(min(_kabsch_rmsd(p, q) for p in tr))
            g_desc.append(float(np.min(np.linalg.norm(train_desc - _pair_desc(q), axis=1))))
        rmsds.extend(g_rmsds)
        desc_dists.extend(g_desc)
        g_rmsds = np.asarray(g_rmsds)
        per_group[label] = {
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "rmsd_min_mean": float(g_rmsds.mean()),
            "rmsd_min_median": float(np.median(g_rmsds)),
        }

    rmsds = np.asarray(rmsds)
    desc_dists = np.asarray(desc_dists)
    summary = {
        "train": str(args.train),
        "test": str(args.test),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "rmsd_min_mean": float(rmsds.mean()),
        "rmsd_min_median": float(np.median(rmsds)),
        "rmsd_min_p10": float(np.percentile(rmsds, 10)),
        "rmsd_min_p90": float(np.percentile(rmsds, 90)),
        "pairdesc_min_mean": float(desc_dists.mean()),
        "pairdesc_min_median": float(np.median(desc_dists)),
        "n_test_rmsd_below_0.2A": int(np.sum(rmsds < 0.2)),
        "per_group": per_group,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    np.savez(args.out.with_suffix(".npz"), rmsds=rmsds, pairdesc=desc_dists)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
