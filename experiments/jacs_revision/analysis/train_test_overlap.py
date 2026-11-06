#!/usr/bin/env python3
"""Train/test geometric overlap with a train-train baseline (R1.5).

For every test structure we report its distance to the nearest training
structure, and for a sample of training structures the distance to their
nearest *other* training structure. A test set drawn from the training
distribution sits at the same distance from the training set as training
frames sit from each other; a genuinely held-out set sits further out.

Two distances are used.

RMSD after optimal superposition, minimised over rotations and over
permutations of chemically identical atoms (element-aware). Rotation is fitted
first, then atoms are matched by the Hungarian algorithm within each element on
the aligned coordinates, and the two steps alternate until the matching stops
changing. Several starting orientations are tried (file order on all atoms,
file order on heavy atoms, principal axes with the four proper sign choices),
and the file-order Kabsch RMSD is always a candidate, so the result is never
worse than the plain fixed-order value. The nearest-neighbour search is exact
with respect to a lower bound: after centring, |R p_i - q_j| >= | |p_i| - |q_j| |,
so matching sorted centroid distances per element bounds the RMSD from below.
Candidates are visited in order of that bound and the search stops once the
bound exceeds the best RMSD found.

SOAP distance: Euclidean distance between L2-normalised, structure-averaged
SOAP vectors (dscribe), which is invariant to rotation and atom order and
compares structures of different size.

Only structures with the same composition are compared by RMSD; a padded
multi-size training set (polythiophene) is split by composition first.

Usage:
  PYTHONPATH=tools/pydeps-soap python experiments/jacs_revision/analysis/train_test_overlap.py \\
    --system ethanethiol --train datasets/ethanethiol_combo_Aidx-1000.npy \\
    --test datasets/ethanethiol_rand-1000.npy --out results/revision/overlap_v2/ethanethiol.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())


# ----------------------------------------------------------------------------- loading
def load_frames(path: str, max_frames: int = 0, seed: int = 0, indices=None):
    """List of (Z, R) per structure with padding removed. Accepts .npy dicts, .npz (MD17), .xyz."""
    path = str(path)
    if path.endswith(".xyz"):
        from ase.io import iread

        frames = [(a.get_atomic_numbers().astype(int), a.get_positions().astype(float)) for a in iread(path)]
    elif path.endswith(".npz"):
        data = np.load(path)
        z = np.asarray(data["z"], dtype=int)
        pos = data["R"]
        idx = np.arange(len(pos)) if indices is None else np.asarray(indices)
        frames = [(z, np.asarray(pos[i], dtype=float)) for i in idx]
        indices = None
    else:
        data = np.load(path, allow_pickle=True)
        data = data.item() if data.dtype == object and data.ndim == 0 else data
        pos = data["positions"]
        zs = data["atom_numbers"]
        frames = []
        shared = np.ndim(zs) == 1 and np.asarray(zs).dtype != object  # one Z list for all frames
        for i in range(len(pos)):
            z = np.asarray(zs if shared else zs[i], dtype=int)
            r = np.asarray(pos[i], dtype=float)
            mask = z > 0
            frames.append((z[mask], r[: len(z)][mask]))
    if indices is not None:
        frames = [frames[i] for i in indices]
    if max_frames and len(frames) > max_frames:
        pick = np.sort(np.random.default_rng(seed).choice(len(frames), max_frames, replace=False))
        frames = [frames[i] for i in pick]
    return frames


def composition(z: np.ndarray) -> tuple:
    return tuple(sorted(np.asarray(z).tolist()))


# ----------------------------------------------------------------------------- RMSD
def kabsch(p: np.ndarray, q: np.ndarray):
    """Rotation R (proper) minimising |p R^T - q| for centred p, q, and the RMSD."""
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    diff = p @ r.T - q
    return r, float(np.sqrt((diff * diff).sum() / len(p)))


def _principal_frames(x: np.ndarray):
    """Rotations taking x to its principal axes, four proper sign choices."""
    _, vecs = np.linalg.eigh(x.T @ x)
    out = []
    for s in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        v = vecs * np.asarray(s)
        if np.linalg.det(v) < 0:
            v[:, 0] *= -1
        out.append(v)
    return out


class Structure:
    """Centred coordinates plus per-element index groups and the sorted-radius key."""

    def __init__(self, z: np.ndarray, r: np.ndarray, elements: list[int]):
        self.z = np.asarray(z, dtype=int)
        self.x = r - r.mean(axis=0)
        self.groups = [np.flatnonzero(self.z == e) for e in elements if (self.z == e).any()]
        self.heavy = np.flatnonzero(self.z > 1)
        radii = np.linalg.norm(self.x, axis=1)
        self.key = np.concatenate([np.sort(radii[g]) for g in self.groups])


def match(p: Structure, q_x: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Permutation of q's atoms onto p's order, per element, for p rotated by rot."""
    aligned = p.x @ rot.T
    perm = np.empty(len(p.z), dtype=int)
    for g in p.groups:
        cost = cdist(aligned[g], q_x[g], "sqeuclidean")
        rows, cols = linear_sum_assignment(cost)
        perm[g[rows]] = g[cols]
    return perm


def perm_rmsd(p: Structure, q: Structure, max_iter: int = 8) -> float:
    """Element-aware, permutation- and rotation-minimised RMSD (align, then match, iterate)."""
    _, best = kabsch(p.x, q.x)  # fixed file order is always a candidate
    starts = [kabsch(p.x, q.x)[0]]
    if len(p.heavy) >= 3:
        starts.append(kabsch(p.x[p.heavy], q.x[q.heavy])[0])
    vq = _principal_frames(q.x)[0]
    for vp in _principal_frames(p.x):
        starts.append(vq @ vp.T)
    for rot in starts:
        perm = None
        for _ in range(max_iter):
            new = match(p, q.x, rot)
            if perm is not None and np.array_equal(new, perm):
                break
            perm = new
            rot, value = kabsch(p.x, q.x[perm])
            best = min(best, value)
    return best


def nearest_rmsd(queries: list[Structure], refs: list[Structure], exclude_self: bool = False):
    """Exact-under-bound nearest-reference RMSD for each query (same composition only)."""
    keys = np.stack([s.key for s in refs])
    out = np.full(len(queries), np.nan)
    evaluated = 0
    for i, p in enumerate(queries):
        bound = np.linalg.norm(keys - p.key, axis=1) / np.sqrt(len(p.key))
        order = np.argsort(bound)
        best = np.inf
        for j in order:
            if exclude_self and refs[j] is p:
                continue
            if bound[j] >= best:
                break
            best = min(best, perm_rmsd(p, refs[j]))
            evaluated += 1
        out[i] = best
    return out, evaluated


# ----------------------------------------------------------------------------- SOAP
def soap_vectors(frames, species, r_cut=5.0, n_max=8, l_max=6):
    from ase import Atoms
    from dscribe.descriptors import SOAP

    soap = SOAP(species=species, r_cut=r_cut, n_max=n_max, l_max=l_max, average="inner", periodic=False)
    vecs = np.stack([soap.create(Atoms(numbers=z, positions=r)) for z, r in frames])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def nearest_soap(queries: np.ndarray, refs: np.ndarray, exclude_self: bool = False):
    d = cdist(queries, refs)
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


# ----------------------------------------------------------------------------- report
def describe(values: np.ndarray, thresholds=(0.1, 0.2, 0.5)) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out = {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }
    for t in thresholds:
        out[f"frac_below_{t}"] = float(np.mean(v < t))
    return out


def analyse(system, train_frames, test_frames, n_baseline, seed, soap_species, note=""):
    t0 = time.time()
    elements = sorted({int(e) for z, _ in train_frames + test_frames for e in z})
    report = {"system": system, "note": note, "n_train": len(train_frames), "n_test": len(test_frames)}
    rng = np.random.default_rng(seed)

    # RMSD: compare within the test set's composition(s) only.
    test_comps = {composition(z) for z, _ in test_frames}
    train_by_comp = {}
    for z, r in train_frames:
        train_by_comp.setdefault(composition(z), []).append(Structure(z, r, elements))
    rmsd_test, rmsd_base, n_eval = [], [], 0
    for comp in sorted(test_comps):
        refs = train_by_comp.get(comp, [])
        queries = [Structure(z, r, elements) for z, r in test_frames if composition(z) == comp]
        if not refs:
            continue
        values, ev = nearest_rmsd(queries, refs)
        rmsd_test.append(values)
        n_eval += ev
        k = min(n_baseline, len(refs))
        base_q = [refs[i] for i in rng.choice(len(refs), k, replace=False)]
        values, ev = nearest_rmsd(base_q, refs, exclude_self=True)
        rmsd_base.append(values)
        n_eval += ev
    if rmsd_test:
        rt, rb = np.concatenate(rmsd_test), np.concatenate(rmsd_base)
        report["rmsd_test_to_train"] = describe(rt)
        report["rmsd_train_to_train"] = describe(rb)
        report["rmsd_median_ratio"] = float(np.median(rt) / np.median(rb))
        from scipy.stats import ks_2samp

        ks = ks_2samp(rt, rb)
        report["rmsd_ks"] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}
        report["rmsd_pairs_evaluated"] = int(n_eval)
        report["_rmsd_test"], report["_rmsd_base"] = rt, rb
    else:
        report["rmsd_test_to_train"] = None
        report["rmsd_note"] = "no training structure with the test composition (size extrapolation)"

    # SOAP: all training structures, any size.
    species = soap_species or [int(e) for e in elements]
    s_train = soap_vectors(train_frames, species)
    s_test = soap_vectors(test_frames, species)
    st = nearest_soap(s_test, s_train)
    k = min(n_baseline, len(train_frames))
    pick = rng.choice(len(train_frames), k, replace=False)
    d = cdist(s_train[pick], s_train)
    d[np.arange(k), pick] = np.inf
    sb = d.min(axis=1)
    report["soap_test_to_train"] = describe(st, thresholds=())
    report["soap_train_to_train"] = describe(sb, thresholds=())
    report["soap_median_ratio"] = float(np.median(st) / np.median(sb))
    report["soap_settings"] = {"r_cut": 5.0, "n_max": 8, "l_max": 6, "average": "inner", "species": species}
    report["_soap_test"], report["_soap_base"] = st, sb
    report["wall_s"] = time.time() - t0
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train-indices", default=None, help=".npy of frame indices into --train (e.g. MD17)")
    ap.add_argument("--test-indices", default=None)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--baseline", type=int, default=300, help="training frames used for the train-train baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True, type=Path)
    cli = ap.parse_args()

    tr_idx = np.load(cli.train_indices) if cli.train_indices else None
    te_idx = np.load(cli.test_indices) if cli.test_indices else None
    train = load_frames(cli.train, indices=tr_idx)
    test = load_frames(cli.test, max_frames=cli.max_test, seed=cli.seed, indices=te_idx)
    report = analyse(cli.system, train, test, cli.baseline, cli.seed, None, cli.note)
    report["train"], report["test"] = cli.train, cli.test
    arrays = {k[1:]: v for k, v in report.items() if k.startswith("_")}
    report = {k: v for k, v in report.items() if not k.startswith("_")}
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=1) + "\n")
    np.savez(cli.out.with_suffix(".npz"), **arrays)
    print(json.dumps({k: report[k] for k in report if k.startswith(("system", "n_", "rmsd_test", "rmsd_train", "rmsd_median", "soap_test", "soap_train", "soap_median"))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
