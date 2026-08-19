#!/usr/bin/env python3
"""Quantify how far each test split sits from the training distribution.

The manuscript currently asserts that test structures are out of distribution;
Reviewer 1 pointed out that coming from a separate trajectory does not
establish that, and the co-author's note asks for train-test separation to be
quantified rather than claimed. This produces the number.

Two complementary measures, because either alone is easy to argue with:

- Bhattacharyya overlap coefficient, BC = integral sqrt(p q), per collective
  variable and for the (order, density) pair jointly. BC = 1 means the
  distributions are identical, 0 means they share no support. Being a
  distribution-level measure, it cannot be satisfied by a handful of outliers.
- Nearest-neighbour distance in standardised collective-variable space, from
  each test frame to the closest *training* frame, reported against the spread
  of the training set's own internal nearest-neighbour distances. This is a
  per-frame measure, so it catches a split that is broadly separated but has a
  few frames sitting inside the training cloud.

Usage:
  python scripts/revision/ood_overlap_report.py \\
    --train results/revision/cv/train.json \\
    --test results/revision/cv/ood_order.json results/revision/cv/ood_density.json \\
    --out results/revision/ood_overlap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

# Deliberately excludes n_water: size alone is the weak notion of
# out-of-distribution the reviewers already rejected, and including it would let
# a split look separated purely by being bigger.
DEFAULT_CVS = ("q_tetrahedral", "n_hbond", "ring_6_frac", "local_density", "surface_frac", "mean_nn_oo")

OVERLAP_THRESHOLD = 0.05


def load_cvs(path: Path, names: tuple[str, ...]) -> tuple[np.ndarray, str]:
    """Raw CV matrix, NaNs preserved.

    Tetrahedral order is undefined for a molecule without four neighbours, so
    it is missing for the n=2-4 training clusters. Dropping whole rows on that
    basis would throw away most of the training set, so NaNs are kept here and
    each analysis masks only the columns it actually uses.
    """
    payload = json.loads(Path(path).read_text())
    rows = payload["frames"]
    matrix = np.array([[float(r.get(n, np.nan)) for n in names] for r in rows], dtype=float)
    return matrix, payload.get("label", Path(path).stem)


def bhattacharyya_1d(a: np.ndarray, b: np.ndarray, grid_points: int = 512) -> float:
    """BC = integral sqrt(p q) for two 1-D samples, via Gaussian KDE."""
    from scipy.stats import gaussian_kde

    if len(a) < 3 or len(b) < 3:
        return float("nan")
    if np.std(a) < 1e-12 and np.std(b) < 1e-12:
        return 1.0 if abs(np.mean(a) - np.mean(b)) < 1e-9 else 0.0
    # A CV that is constant across a split (every ice frame having no
    # five-membered rings, say) has no KDE bandwidth. Widening it to a small
    # fraction of the other sample's spread keeps such a split comparable
    # instead of dropping it as NaN, which would quietly hide a hard separation.
    spread = max(np.std(a), np.std(b))
    rng = np.random.default_rng(0)
    if np.std(a) < 1e-12:
        a = a + rng.normal(0.0, 1e-3 * spread, size=len(a))
    if np.std(b) < 1e-12:
        b = b + rng.normal(0.0, 1e-3 * spread, size=len(b))

    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    pad = 0.1 * (hi - lo) + 1e-9
    grid = np.linspace(lo - pad, hi + pad, grid_points)
    try:
        pa = gaussian_kde(a)(grid)
        pb = gaussian_kde(b)(grid)
    except np.linalg.LinAlgError:
        return float("nan")
    step = grid[1] - grid[0]
    pa /= max(pa.sum() * step, 1e-300)
    pb /= max(pb.sum() * step, 1e-300)
    return float(np.sqrt(pa * pb).sum() * step)


def bhattacharyya_2d(a: np.ndarray, b: np.ndarray, grid_points: int = 96) -> float:
    """BC for two 2-D samples on a shared grid."""
    from scipy.stats import gaussian_kde

    if len(a) < 5 or len(b) < 5:
        return float("nan")
    both = np.vstack([a, b])
    lo = both.min(axis=0)
    hi = both.max(axis=0)
    pad = 0.1 * (hi - lo) + 1e-9
    gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid_points)
    gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid_points)
    mesh = np.vstack([m.ravel() for m in np.meshgrid(gx, gy)])
    try:
        pa = gaussian_kde(a.T)(mesh)
        pb = gaussian_kde(b.T)(mesh)
    except np.linalg.LinAlgError:
        return float("nan")
    cell = (gx[1] - gx[0]) * (gy[1] - gy[0])
    pa /= max(pa.sum() * cell, 1e-300)
    pb /= max(pb.sum() * cell, 1e-300)
    return float(np.sqrt(pa * pb).sum() * cell)


def fraction_outside_train_range(train: np.ndarray, test: np.ndarray, low_pct=1.0, high_pct=99.0) -> float:
    """Fraction of test values outside the training set's central 98 percent.

    Sharper and easier to read than an overlap coefficient: "94 percent of ice
    frames are more tetrahedral than 99 percent of the training set" is a claim
    a reader can check against the histogram, and unlike a KDE it does not
    depend on a bandwidth choice.
    """
    a = train[np.isfinite(train)]
    b = test[np.isfinite(test)]
    if len(a) < 2 or len(b) == 0:
        return float("nan")
    lo, hi = np.percentile(a, low_pct), np.percentile(a, high_pct)
    return float(((b < lo) | (b > hi)).mean())


def nearest_neighbour_distances(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Distance from each test row to the closest training row, standardised."""
    from sklearn.neighbors import NearestNeighbors

    centre = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    model = NearestNeighbors(n_neighbors=1).fit((train - centre) / scale)
    dist, _ = model.kneighbors((test - centre) / scale)
    return dist.ravel()


def train_internal_scale(train: np.ndarray) -> float:
    """Typical nearest-neighbour distance *within* the training set.

    Without this, a raw distance is meaningless: it answers 'far compared to
    what?'. Test distances are reported as multiples of this.
    """
    from sklearn.neighbors import NearestNeighbors

    centre = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normed = (train - centre) / scale
    model = NearestNeighbors(n_neighbors=2).fit(normed)
    dist, _ = model.kneighbors(normed)
    return float(np.median(dist[:, 1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path, nargs="+")
    parser.add_argument("--cvs", default=",".join(DEFAULT_CVS))
    parser.add_argument("--joint", default="n_hbond,local_density")
    parser.add_argument("--threshold", type=float, default=OVERLAP_THRESHOLD)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    names = tuple(v.strip() for v in args.cvs.split(",") if v.strip())
    joint = tuple(v.strip() for v in args.joint.split(",") if v.strip())
    joint_idx = [names.index(v) for v in joint if v in names]

    train, _ = load_cvs(args.train, names)
    # Columns used for the multivariate measures have to be finite in both sets,
    # so restrict them to the CVs that every frame actually defines.
    dense_cols = [i for i in range(len(names)) if np.isfinite(train[:, i]).all()]
    train_dense = train[:, dense_cols]
    if len(train_dense) == 0:
        raise SystemExit(f"no training rows in {args.train}")
    reference = train_internal_scale(train_dense)

    report: dict = {
        "train": str(args.train),
        "n_train": int(len(train)),
        "cvs": list(names),
        "distance_cvs": [names[i] for i in dense_cols],
        "joint_cvs": list(joint),
        "threshold": args.threshold,
        "train_internal_nn_distance": reference,
        "splits": {},
    }

    for path in args.test:
        test, label = load_cvs(path, names)
        if len(test) == 0:
            report["splits"][label] = {"error": "no rows"}
            continue

        per_cv = {}
        outside = {}
        for i, name in enumerate(names):
            a = train[:, i][np.isfinite(train[:, i])]
            b = test[:, i][np.isfinite(test[:, i])]
            per_cv[name] = bhattacharyya_1d(a, b)
            outside[name] = fraction_outside_train_range(a, b)

        joint_bc = float("nan")
        if len(joint_idx) == 2:
            a = train[:, joint_idx]
            b = test[:, joint_idx]
            a = a[np.isfinite(a).all(axis=1)]
            b = b[np.isfinite(b).all(axis=1)]
            joint_bc = bhattacharyya_2d(a, b)

        test_dense = test[:, dense_cols]
        test_dense = test_dense[np.isfinite(test_dense).all(axis=1)]
        dist = nearest_neighbour_distances(train_dense, test_dense)
        finite = [v for v in per_cv.values() if np.isfinite(v)]

        entry = {
            "path": str(path),
            "n_test": int(len(test)),
            "bhattacharyya_per_cv": per_cv,
            "frac_outside_train_p01_p99": outside,
            "bhattacharyya_joint": joint_bc,
            "max_bhattacharyya": float(max(finite)) if finite else float("nan"),
            "min_bhattacharyya": float(min(finite)) if finite else float("nan"),
            "nn_distance": {
                "median": float(np.median(dist)),
                "p05": float(np.percentile(dist, 5)),
                "min": float(dist.min()),
                "median_in_train_units": float(np.median(dist) / reference) if reference > 0 else float("nan"),
                "frac_within_train_scale": float((dist <= reference).mean()),
            },
            "verdict_joint_below_threshold": bool(np.isfinite(joint_bc) and joint_bc <= args.threshold),
        }
        report["splits"][label] = entry

        print(f"\n{label}  (n={len(test)})")
        print(f"  {'collective variable':18s} {'overlap':>8s}  {'outside train p1-p99':>20s}")
        for name, value in per_cv.items():
            print(f"  {name:18s} {value:8.4f}  {outside[name] * 100:19.1f}%")
        print(f"  BC joint({','.join(joint)}) = {joint_bc:.4f}")
        print(
            f"  nearest-training distance: median {np.median(dist):.2f} "
            f"({entry['nn_distance']['median_in_train_units']:.1f}x the training set's own spacing); "
            f"{entry['nn_distance']['frac_within_train_scale'] * 100:.1f}% of frames sit within it"
        )
        print(f"  {'SEPARATED' if entry['verdict_joint_below_threshold'] else 'OVERLAPPING'} at BC <= {args.threshold}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
