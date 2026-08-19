#!/usr/bin/env python3
"""Density error as a function of distance from the training distribution.

This is the figure the co-author's note asks for and the one that answers
Reviewer 1's objection directly. The manuscript's current claim is binary --
these structures are out of distribution, the model still works -- and a binary
claim invites the reply that the split was not really out of distribution. A
curve does not: it shows the error growing (or not) as structures move away from
what was trained on, and it puts every tier on one axis so they can be compared.

Two x-axes are produced, because each is weak where the other is strong:

- Descriptor distance: nearest-neighbour distance in standardised
  collective-variable space to the closest training frame, in units of the
  training set's own internal spacing. General enough to place ice, droplets and
  larger clusters on one axis, but it is still a descriptor, and a reader can
  reasonably ask what a distance of 4 means physically.
- The malonaldehyde proton-transfer coordinate |delta|. Only defined for that
  one system, but it is a real reaction coordinate: the training set covers one
  enol basin, and the scan walks through the transition state into the other. No
  descriptor choice is being trusted, so it is the harder test of the two.

Errors come from csh_evaluate.py, which reports the paper's absolute fractional
error per structure, so the y-axis needs no further definition here.

Usage:
  python scripts/revision/error_vs_distance.py \\
    --train-cv results/revision/cv/water_train.json \\
    --tier id_test:results/revision/cv/water_id_test.json:results/revision/eval/id_test.json \\
    --tier ood_size:results/revision/cv/water_ood_size.json:results/revision/eval/ood_size.json \\
    --out results/revision/error_vs_distance.json
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

from ood_overlap_report import (  # noqa: E402
    DEFAULT_CVS,
    load_cvs,
    nearest_neighbour_distances,
    train_internal_scale,
)


def train_extent_scale(train: np.ndarray, max_rows: int = 400, seed: int = 0) -> float:
    """Median pairwise distance within the training set, in standardised units.

    Used as the x-axis unit in preference to the nearest-neighbour spacing that
    ood_overlap_report.py reports. The two answer different questions and differ
    by a factor of ~45 here: with 1250 frames drawn from a handful of motifs, the
    training set is sampled very finely relative to its own extent, so distances
    in units of its spacing run into the hundreds. "This frame is 6.5 training
    widths away" is a statement a reader can picture; "294 times the training
    spacing" is not, and it overstates a separation that is real but not
    astronomical. The spacing remains the right denominator for the separate
    question of whether *any* training frame is nearby, which is why the overlap
    report keeps it.
    """
    rng = np.random.default_rng(seed)
    rows = train
    if len(rows) > max_rows:
        rows = rows[rng.choice(len(rows), max_rows, replace=False)]
    centre = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normed = (rows - centre) / scale
    dists = np.linalg.norm(normed[:, None, :] - normed[None, :, :], axis=-1)
    upper = dists[np.triu_indices(len(normed), k=1)]
    median = float(np.median(upper)) if upper.size else 0.0
    return median if median > 1e-12 else 1.0


def load_errors(path: Path) -> dict[int, float]:
    """Frame index -> absolute fractional error, from a csh_evaluate.py report."""
    payload = json.loads(Path(path).read_text())
    return {int(r["index"]): float(r["afe"]) for r in payload["records"]}


def bin_curve(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> list[dict]:
    """Mean error per x-bin, with the spread and count needed to read it honestly.

    A mean alone would hide that the far bins often hold only a handful of
    frames, which is exactly where a reader should be most sceptical.
    """
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if not mask.any():
            continue
        vals = y[mask]
        rows.append(
            {
                "x_lo": float(lo),
                "x_hi": float(hi),
                "x_mean": float(x[mask].mean()),
                "n": int(mask.sum()),
                "afe_mean": float(vals.mean()),
                "afe_median": float(np.median(vals)),
                "afe_std": float(vals.std(ddof=1)) if mask.sum() > 1 else 0.0,
                "afe_sem": float(vals.std(ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else 0.0,
            }
        )
    return rows


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, so 'error grows with distance' is one number.

    Rank rather than Pearson because the growth is not expected to be linear and
    a single distant outlier should not be able to manufacture the trend.
    """
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def parse_tier(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--tier wants NAME:CV_JSON:EVAL_JSON, got {spec!r}")
    return parts[0], Path(parts[1]), Path(parts[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cv", required=True, type=Path)
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        required=True,
        metavar="NAME:CV_JSON:EVAL_JSON",
        help="repeat once per OOD tier",
    )
    parser.add_argument("--cvs", default=",".join(DEFAULT_CVS))
    parser.add_argument(
        "--physical-axis",
        default=None,
        metavar="NAME:CV_JSON:EVAL_JSON:CV_NAME",
        help="a second panel against a real reaction coordinate, e.g. malonaldehyde abs_delta_pt",
    )
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    names = tuple(v.strip() for v in args.cvs.split(",") if v.strip())
    train, _ = load_cvs(args.train_cv, names)
    # Multivariate distance needs columns every frame defines; tetrahedral order
    # is undefined below four neighbours, so it drops out for small clusters.
    dense_cols = [i for i in range(len(names)) if np.isfinite(train[:, i]).all()]
    if not dense_cols:
        raise SystemExit(f"no collective variable is finite for every frame of {args.train_cv}")
    train_dense = train[:, dense_cols]
    unit = train_extent_scale(train_dense)
    spacing = train_internal_scale(train_dense)

    report: dict = {
        "train_cv": str(args.train_cv),
        "cvs_used": [names[i] for i in dense_cols],
        "distance_unit": "median pairwise distance within the training set",
        "train_extent": unit,
        "train_internal_nn_distance": spacing,
        "n_train": int(len(train_dense)),
        "tiers": {},
    }

    series = []
    for spec in args.tier:
        tier, cv_path, eval_path = parse_tier(spec)
        if not cv_path.exists() or not eval_path.exists():
            print(f"{tier}: missing {'CVs' if not cv_path.exists() else 'errors'}, skipping")
            continue
        test, _ = load_cvs(cv_path, names)
        errors = load_errors(eval_path)

        keep = [i for i in range(len(test)) if i in errors and np.isfinite(test[i, dense_cols]).all()]
        if not keep:
            print(f"{tier}: no frame has both a finite descriptor and an error, skipping")
            continue
        dist = nearest_neighbour_distances(train_dense, test[np.array(keep)][:, dense_cols]) / unit
        afe = np.array([errors[i] for i in keep], dtype=float)

        edges = np.linspace(dist.min(), dist.max() + 1e-12, args.n_bins + 1)
        report["tiers"][tier] = {
            "n": len(keep),
            "distance_median": float(np.median(dist)),
            "distance_max": float(dist.max()),
            "distance_median_in_train_spacings": float(np.median(dist) * unit / spacing),
            "afe_mean": float(afe.mean()),
            "afe_median": float(np.median(afe)),
            "spearman_afe_vs_distance": spearman(dist, afe),
            "curve": bin_curve(dist, afe, edges),
        }
        series.append((tier, dist, afe))
        print(
            f"{tier:16s} n={len(keep):4d}  distance {np.median(dist):5.2f} training widths  "
            f"AFE {afe.mean():.5f}  rho(AFE, distance) = {spearman(dist, afe):+.2f}"
        )

    if args.physical_axis:
        parts = args.physical_axis.split(":")
        if len(parts) != 4:
            raise SystemExit("--physical-axis wants NAME:CV_JSON:EVAL_JSON:CV_NAME")
        tier, cv_path, eval_path, cv_name = parts[0], Path(parts[1]), Path(parts[2]), parts[3]
        if cv_path.exists() and eval_path.exists():
            payload = json.loads(cv_path.read_text())
            errors = load_errors(eval_path)
            xs, ys = [], []
            for row in payload["frames"]:
                idx = int(row["index"])
                value = float(row.get(cv_name, np.nan))
                if idx in errors and np.isfinite(value):
                    xs.append(value)
                    ys.append(errors[idx])
            if xs:
                x = np.array(xs)
                y = np.array(ys)
                edges = np.linspace(x.min(), x.max() + 1e-12, args.n_bins + 1)
                report["physical_axis"] = {
                    "tier": tier,
                    "coordinate": cv_name,
                    "n": len(x),
                    "spearman_afe_vs_coordinate": spearman(x, y),
                    "curve": bin_curve(x, y, edges),
                }
                print(
                    f"{tier:16s} n={len(x):4d}  vs {cv_name}: "
                    f"rho = {spearman(x, y):+.2f}"
                )
        else:
            print(f"{args.physical_axis}: inputs missing, skipping physical axis")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")

    if args.figure and series:
        make_figure(series, report, args.figure)
        print(f"wrote {args.figure}")
    return 0


def make_figure(series, report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_physical = "physical_axis" in report
    fig, axes = plt.subplots(1, 2 if has_physical else 1, figsize=(11 if has_physical else 6, 4.2))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for tier, dist, afe in series:
        ax.scatter(dist, afe, s=8, alpha=0.25, label=None)
        curve = report["tiers"][tier]["curve"]
        if curve:
            ax.errorbar(
                [c["x_mean"] for c in curve],
                [c["afe_mean"] for c in curve],
                yerr=[c["afe_sem"] for c in curve],
                marker="o",
                capsize=3,
                label=tier,
            )
    ax.set_xlabel(
        "distance from training set\n"
        "(nearest training frame, in units of the training set's own width)"
    )
    ax.set_ylabel("absolute fractional density error")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Error versus distance from training")

    if has_physical:
        ax = axes[1]
        entry = report["physical_axis"]
        curve = entry["curve"]
        ax.errorbar(
            [c["x_mean"] for c in curve],
            [c["afe_mean"] for c in curve],
            yerr=[c["afe_sem"] for c in curve],
            marker="o",
            capsize=3,
            color="crimson",
        )
        ax.set_xlabel(f"{entry['coordinate']} (A)")
        ax.set_ylabel("absolute fractional density error")
        ax.set_yscale("log")
        ax.set_title(f"{entry['tier']}: error along the reaction coordinate")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
