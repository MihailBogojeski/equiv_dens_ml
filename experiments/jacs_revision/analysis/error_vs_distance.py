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
- The signed malonaldehyde proton-transfer coordinate delta. Only defined for
  that one system, but it is a real reaction coordinate: the training set covers
  one enol basin, and the scan walks through the transition state into the
  other. No descriptor choice is being trusted, so it is the harder test of the
  two.

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

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/error_vs_distance.py'


import argparse
import json
import sys
from pathlib import Path


_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402
ensure_revision_imports(_SCRIPT_DIR)
import numpy as np

_REPO_ROOT = repo_root()
sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402
from experiments.jacs_revision.analysis.mpl_classic_tex import apply as apply_classic_tex, save_response_figure  # noqa: E402
from experiments.jacs_revision.analysis.ood_overlap_report import (  # noqa: E402
    DEFAULT_CVS,
    load_cvs,
    nearest_neighbour_distances,
    train_internal_scale,
)


#: Axis labels for the physical-coordinate panel. The whole point of that panel
#: is that a chemist can read the x-axis, which "abs_delta_pt (A)" defeats.
AXIS_LABELS = {
    "abs_delta_pt": r"$|\delta|$ / $\mathrm{\AA}$",
    "delta_pt": r"$\delta$ / $\mathrm{\AA}$",
    "r_oo": r"$r_{\mathrm{OO}}$ / $\mathrm{\AA}$",
    "co_alternation": r"$\Delta r_{\mathrm{CO}}$ / $\mathrm{\AA}$",
}

# Relative L1 density error, \int|\Delta\rho| dV / N_e.
EPSILON_RHO = r"$\varepsilon_\rho$"
EPSILON_RHO_PERCENT = r"$\varepsilon_\rho$ / \%"


def style_epsilon_rho_ticks(ax) -> None:
    """Show ε_ρ in ×10^{-3} scientific notation."""
    ax.set_ylabel(EPSILON_RHO)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))
    ax.yaxis.set_major_formatter(formatter)


def style_dw_error_axis(ax) -> None:
    """Linear epsilon_rho versus d/w, with the distance axis starting at zero."""
    ax.set_xlabel(r"$d/w$")
    ax.set_yscale("linear")
    ax.set_xlim(left=0.0)
    style_epsilon_rho_ticks(ax)


MALON_TRAIN_DELTA_MAX = -0.40

TIER_COLORS = {
    "id_test": "#4c72b0",
    "ood_size": "#55a868",
    "ood_order": "#c44e52",
    "ood_density": "#8172b2",
}
TIER_LABELS = {
    "id_test": "ID test",
    "ood_size": "Size OOD",
    "ood_order": "Ice",
    "ood_density": "Droplets",
}

SATURATION_LINEWIDTH = 3.2
SATURATION_LINECOLOR = "0.12"


def exp_kernel(x, length_scale: float = 1.0):
    """K(x) = exp(-x/ℓ). Unit length recovers e^{-x}."""
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    x = np.asarray(x, dtype=float)
    return np.exp(-x / float(length_scale))


def saturation_curve(x, sigma2: float, length_scale: float = 1.0, floor: float = 0.0):
    """y = ε0 + σ² (1 - K(x)), the exponential generalisation curve."""
    return floor + sigma2 * (1.0 - exp_kernel(x, length_scale))


def fit_saturation_curve(
    x: np.ndarray,
    y: np.ndarray,
    length_scales: np.ndarray | None = None,
) -> dict[str, float]:
    """Least-squares fit of y = floor + sigma2 * (1 - exp(-x/ℓ)).

    For each candidate length the remaining two parameters are linear, so the
    search is a 1-D sweep over ℓ rather than a 3-D nonlinear solve. Negative
    σ² is rejected: that would be a curve that falls with distance. Negative
    ε₀ is rejected as well: the density error cannot be negative, and a tiny
    ℓ with ε₀ ≪ 0 is an unidentifiable step already saturated at the first
    datum.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 3:
        raise ValueError("need at least three points to fit a saturation curve")
    if length_scales is None:
        length_scales = np.geomspace(0.05, 12.0, 160)
    best: tuple[float, dict[str, float]] | None = None
    for ell in length_scales:
        feature = 1.0 - exp_kernel(x, float(ell))
        if float(np.ptp(feature)) < 0.05:
            continue
        design = np.column_stack((np.ones_like(feature), feature))
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        floor, sigma2 = float(coef[0]), float(coef[1])
        if floor < 0.0 or sigma2 < 0.0:
            continue
        residual = float(np.sum(np.square(y - (floor + sigma2 * feature))))
        if best is None or residual < best[0]:
            best = (
                residual,
                {"floor": floor, "sigma2": sigma2, "length_scale": float(ell)},
            )
    if best is None:
        raise ValueError("no length scale produced a rising saturation curve")
    return best[1]


def saturation_fit_xy(series) -> tuple[np.ndarray, np.ndarray]:
    """Points the saturation line is fitted to.

    All-point least squares is dominated by the ID-test pile-up at d/w ≈ 0, and
    the length scale then collapses to a step. With three or more tiers, the
    per-tier medians are the same numbers as the letter table and leave a
    visible rise before the plateau.
    """
    if len(series) >= 3:
        x = np.array([float(np.median(dist)) for _, dist, _ in series], dtype=float)
        y = np.array([float(np.median(afe)) for _, _, afe in series], dtype=float)
        return x, y
    x = np.concatenate([np.asarray(dist, dtype=float) for _, dist, _ in series])
    y = np.concatenate([np.asarray(afe, dtype=float) for _, _, afe in series])
    return x, y


def fit_series_saturation(series) -> dict[str, float]:
    """Fit y = ε0 + σ² (1 - K(x)) to a water-tier series."""
    x, y = saturation_fit_xy(series)
    return fit_saturation_curve(x, y)


def draw_saturation_fit(ax, series, *, color: str = SATURATION_LINECOLOR, lw: float = SATURATION_LINEWIDTH) -> dict[str, float] | None:
    """Thick guide-to-the-eye through every water tier on the panel."""
    if not series:
        return None
    try:
        params = fit_series_saturation(series)
    except ValueError:
        return None
    x_max = max(float(np.max(dist)) for _, dist, _ in series)
    grid = np.linspace(0.0, max(x_max, 1.0), 400)
    ax.plot(
        grid,
        saturation_curve(grid, params["sigma2"], params["length_scale"], params["floor"]),
        color=color,
        lw=lw,
        solid_capstyle="round",
        zorder=4,
        label=r"$\varepsilon_0+\sigma^2(1-K)$",
    )
    return params


def draw_water_error_panel(ax, series, *, marker_size: float = 16) -> None:
    """One scatter cloud per water tier, plus the saturating generalisation curve."""
    for i, (tier, dist, afe) in enumerate(series):
        color = TIER_COLORS.get(tier, f"C{i}")
        ax.scatter(
            dist,
            afe,
            s=marker_size,
            alpha=0.55,
            color=color,
            label=TIER_LABELS.get(tier, tier),
            zorder=3,
        )
    draw_saturation_fit(ax, series)
    style_dw_error_axis(ax)
    ax.legend()


def draw_malon_error_panel(ax, x, y, train_delta_max: float = MALON_TRAIN_DELTA_MAX) -> None:
    """Signed proton-transfer coordinate with the training well shaded."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    left = float(min(np.min(x), train_delta_max) - 0.05)
    right = float(max(np.max(x), 0.0) + 0.05)
    ax.axvspan(left, train_delta_max, color="#4c72b0", alpha=0.12, zorder=0)
    ax.axvline(train_delta_max, color="#4c72b0", ls="--", lw=1.0)
    ax.axvline(0.0, color="0.35", ls=":", lw=1.0)
    in_train = x <= train_delta_max
    if np.any(in_train):
        ax.scatter(x[in_train], y[in_train], s=16, alpha=0.7, color="#4c72b0", label=r"train well")
    if np.any(~in_train):
        ax.scatter(x[~in_train], y[~in_train], s=16, alpha=0.7, color="#c44e52", label=r"held-out")
    ax.set_xlabel(r"$\delta$ / $\mathrm{\AA}$")
    ax.set_yscale("linear")
    ax.set_xlim(left, right)
    style_epsilon_rho_ticks(ax)
    ax.legend()


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
    """Frame index -> absolute fractional error, from a csh_evaluate.py report.

    Keyed on `source_index` -- the frame's number in the geometry file -- rather
    than its row in the evaluated dataset. The two differ whenever a frame failed
    its ORCA calculation and was dropped during assembly, and since the
    collective variables are numbered over the geometry file, joining on the row
    number would pair each error after the first gap with the wrong structure.
    That would not fail; it would just move points along the x-axis of the
    headline figure.

    Reports without the key are older and are read positionally, which is what
    they meant at the time.
    """
    payload = json.loads(Path(path).read_text())
    records = payload["records"]
    if any("source_index" not in r for r in records):
        return {int(r["index"]): float(r["afe"]) for r in records}
    return {int(r["source_index"]): float(r["afe"]) for r in records}


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
        help="a second panel against a real reaction coordinate, e.g. malonaldehyde delta_pt",
    )
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument(
        "--figure-water",
        type=Path,
        default=None,
        help="single-panel water ε_ρ versus d/w figure",
    )
    parser.add_argument(
        "--figure-physical",
        type=Path,
        default=None,
        help="single-panel physical-axis figure (e.g. malonaldehyde δ)",
    )
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

    if series:
        try:
            fit = fit_series_saturation(series)
        except ValueError:
            fit = None
        if fit is not None:
            report["saturation_fit"] = fit
            print(
                f"{'saturation':16s}  ε0={fit['floor']:.5f}  σ²={fit['sigma2']:.5f}  "
                f"ℓ={fit['length_scale']:.3f}"
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
                    "x": x.tolist(),
                    "y": y.tolist(),
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
    if args.figure_water and series:
        make_water_figure(series, args.figure_water)
        print(f"wrote {args.figure_water}")
    if args.figure_physical and "physical_axis" in report:
        make_physical_figure(report, args.figure_physical)
        print(f"wrote {args.figure_physical}")
    return 0


def _enlarge_water_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=15)
    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(17)
    ax.yaxis.get_offset_text().set_size(15)
    legend = ax.get_legend()
    if legend is not None:
        plt.setp(legend.get_texts(), fontsize=13)
        for handle in legend.legend_handles:
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([48])


def make_water_figure(series, path: Path) -> None:
    """Water-only ε_ρ versus d/w, for the Response 1.2 water-cluster block."""
    apply_classic_tex()
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.8))
    draw_water_error_panel(ax, series, marker_size=28)
    _enlarge_water_axis(ax)
    fig.tight_layout()
    save_response_figure(fig, path)


def make_water_pair_figure(series_left, series_right, path: Path) -> None:
    """PBE-D4 (left) and ωB97M-V (right) water ε_ρ versus d/w."""
    apply_classic_tex()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    draw_water_error_panel(axes[0], series_left, marker_size=22)
    draw_water_error_panel(axes[1], series_right, marker_size=22)
    for ax in axes:
        _enlarge_water_axis(ax)
    fig.tight_layout()
    save_response_figure(fig, path)


def series_from_eval(train_cv: Path, eval_dir: Path, cv_dir: Path | None = None) -> list:
    """Rebuild the water scatter series from on-disk CV and eval reports."""
    names = tuple(DEFAULT_CVS)
    train, _ = load_cvs(train_cv, names)
    dense_cols = [i for i in range(len(names)) if np.isfinite(train[:, i]).all()]
    if not dense_cols:
        raise SystemExit(f"no collective variable is finite for every frame of {train_cv}")
    train_dense = train[:, dense_cols]
    unit = train_extent_scale(train_dense)
    cv_dir = cv_dir or train_cv.parent
    specs = (
        ("id_test", cv_dir / "water_id_test.json", eval_dir / "water_id_test.json"),
        ("ood_size", cv_dir / "water_ood_size.json", eval_dir / "water_ood_size.json"),
        ("ood_order", cv_dir / "ood_order.json", eval_dir / "water_ood_order.json"),
        ("ood_density", cv_dir / "ood_density.json", eval_dir / "water_ood_density.json"),
    )
    series = []
    for tier, cv_path, eval_path in specs:
        if not cv_path.exists() or not eval_path.exists():
            continue
        test, _ = load_cvs(cv_path, names)
        errors = load_errors(eval_path)
        keep = [i for i in range(len(test)) if i in errors and np.isfinite(test[i, dense_cols]).all()]
        if not keep:
            continue
        dist = nearest_neighbour_distances(train_dense, test[np.array(keep)][:, dense_cols]) / unit
        afe = np.array([errors[i] for i in keep], dtype=float)
        series.append((tier, dist, afe))
    return series


def make_physical_figure(report: dict, path: Path) -> None:
    """Physical-axis panel only (malonaldehyde δ). Unused by the letter."""
    apply_classic_tex()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.2))
    entry = report["physical_axis"]
    draw_malon_error_panel(ax, entry["x"], entry["y"])
    fig.tight_layout()
    save_response_figure(fig, path)


def make_figure(series, report: dict, path: Path) -> None:
    apply_classic_tex()
    has_physical = "physical_axis" in report
    fig, axes = plt.subplots(1, 2 if has_physical else 1, figsize=(11 if has_physical else 6, 4.2))
    axes = np.atleast_1d(axes)

    draw_water_error_panel(axes[0], series)

    if has_physical:
        entry = report["physical_axis"]
        draw_malon_error_panel(axes[1], entry["x"], entry["y"])

    fig.tight_layout()
    save_response_figure(fig, path)


if __name__ == "__main__":
    raise SystemExit(main())
