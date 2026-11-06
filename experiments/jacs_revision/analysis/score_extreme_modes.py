#!/usr/bin/env python3
"""Join DenSNet density AFE onto extreme-mode frames and report error vs |q|.

csh_evaluate.py writes one record per assembled label. Assembly drops frames
ORCA failed on, so the evaluator's row numbers are not the XYZ indices. The
join key is ``source_index`` -- the frame number in the mode scan -- which is
what the manifest is numbered over.

Usage:
  python scripts/revision/analysis/score_extreme_modes.py \\
    --pair ethanol:datasets/revision/ood_modes/ethanol/manifest.json:results/revision/eval/ood_modes_ethanol.json \\
    --out results/revision/eval/ood_extreme_modes.json \\
    --figure results/revision/figures/response/ood_extreme_mode_error.pdf
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/score_extreme_modes.py'


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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from experiments.jacs_revision.analysis.error_vs_distance import (  # noqa: E402
    EPSILON_RHO,
    SATURATION_LINEWIDTH,
    fit_saturation_curve,
    saturation_curve,
    spearman,
)
from experiments.jacs_revision.analysis.mpl_classic_tex import apply as apply_classic_tex, save_response_figure  # noqa: E402

_REPO_ROOT = repo_root()


def join_mode_eval(manifest_records: list[dict], eval_records: list[dict]) -> list[dict]:
    """Attach AFE to each mode-scan frame that has a labelled source_index."""
    errors: dict[int, float] = {}
    for rec in eval_records:
        if "source_index" not in rec:
            continue
        errors[int(rec["source_index"])] = float(rec["afe"])

    joined: list[dict] = []
    for source_index, rec in enumerate(manifest_records):
        if source_index not in errors:
            continue
        q_k = float(rec["q_k"])
        rmsd = float(rec["rmsd"])
        joined.append(
            {
                "source_index": source_index,
                "mode_index": int(rec.get("mode_index", -1)),
                "q_k": q_k,
                "abs_q": abs(q_k),
                "rmsd": rmsd,
                "rmsd_target": float(rec.get("rmsd_target", rmsd)),
                "bond_stretched": bool(rec.get("bond_stretched", False)),
                "afe": errors[source_index],
            }
        )
    return joined


def summarize_mode_errors(joined: list[dict]) -> dict:
    """Mean AFE and rank correlation against |q_k| and Cartesian RMSD."""
    if not joined:
        return {
            "n": 0,
            "afe_mean": float("nan"),
            "afe_median": float("nan"),
            "spearman_afe_vs_abs_q": float("nan"),
            "spearman_afe_vs_rmsd": float("nan"),
        }
    afe = np.array([row["afe"] for row in joined], dtype=float)
    abs_q = np.array([row["abs_q"] for row in joined], dtype=float)
    rmsd = np.array([row["rmsd"] for row in joined], dtype=float)
    return {
        "n": len(joined),
        "afe_mean": float(afe.mean()),
        "afe_median": float(np.median(afe)),
        "afe_max": float(afe.max()),
        "spearman_afe_vs_abs_q": spearman(abs_q, afe),
        "spearman_afe_vs_rmsd": spearman(rmsd, afe),
        "n_bond_stretched": int(sum(1 for row in joined if row["bond_stretched"])),
    }


def parse_pair(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--pair wants NAME:MANIFEST:EVAL_JSON, got {spec!r}")
    return parts[0], Path(parts[1]), Path(parts[2])


# Letter table systems only. Resorcinol / thiophene are incomplete and
# contain AFE outliers that collapse every other series on a log axis.
AFE_MAX = 0.5

SERIES_STYLE = {
    "water_n2_pbe": {
        "color": "#4c72b0",
        "marker": "o",
        "label": "Water dimer, PBE-D4",
    },
    "water_n2_hybrid": {
        "color": "#4c72b0",
        "marker": "s",
        "label": "Water dimer, ωB97M-V",
    },
    "water_n6_pbe": {
        "color": "#55a868",
        "marker": "o",
        "label": "Water hexamer, PBE-D4",
    },
    "water_n6_hybrid": {
        "color": "#55a868",
        "marker": "s",
        "label": "Water hexamer, ωB97M-V",
    },
    "malonaldehyde_pbe": {
        "color": "#c44e52",
        "marker": "o",
        "label": "Malonaldehyde, PBE-D4",
    },
    "malonaldehyde_hybrid": {
        "color": "#c44e52",
        "marker": "s",
        "label": "Malonaldehyde, ωB97M-V",
    },
    "ethanol_pbe": {
        "color": "#8172b2",
        "marker": "o",
        "label": "Ethanol, PBE-D4",
    },
    "ethanethiol_pbe": {
        "color": "#ccb974",
        "marker": "o",
        "label": "Ethanethiol, PBE-D4",
    },
}


def _finite_points(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    abs_q, rmsd, afe = [], [], []
    for row in records:
        q = float(row["abs_q"])
        r = float(row["rmsd"])
        e = float(row["afe"])
        if not (np.isfinite(q) and np.isfinite(r) and np.isfinite(e)):
            continue
        if e <= 0.0 or e > AFE_MAX:
            continue
        abs_q.append(q)
        rmsd.append(r)
        afe.append(e)
    return np.asarray(abs_q), np.asarray(rmsd), np.asarray(afe)


MOLECULE_PANELS = (
    (r"$(\mathrm{H_2O})_2$", ("water_n2_pbe", "water_n2_hybrid")),
    (r"$(\mathrm{H_2O})_6$", ("water_n6_pbe", "water_n6_hybrid")),
    ("Malonaldehyde", ("malonaldehyde_pbe", "malonaldehyde_hybrid")),
    ("Ethanol", ("ethanol_pbe",)),
    ("Ethanethiol", ("ethanethiol_pbe",)),
)

THEORY_STYLE = {
    "pbe": {"marker": "o", "color": "#4c72b0", "label": "PBE-D4"},
    "hybrid": {"marker": "s", "color": "#c44e52", "label": r"$\omega$B97M-V"},
}


def _theory_key(series_name: str) -> str:
    return "hybrid" if series_name.endswith("_hybrid") else "pbe"


def _rung_medians(rmsd: np.ndarray, afe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One (median RMSD, median AFE) per amplitude rung."""
    keys = np.round(np.asarray(rmsd, dtype=float), decimals=2)
    xs, ys = [], []
    for key in np.unique(keys):
        mask = keys == key
        xs.append(float(np.median(rmsd[mask])))
        ys.append(float(np.median(afe[mask])))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def fit_mode_saturation(
    rmsd: np.ndarray,
    afe: np.ndarray,
    rmsd_target: np.ndarray | None = None,
) -> dict[str, float]:
    """ε_ρ = ε0 + σ² (1 - e^{-RMSD/ℓ}) on the amplitude rungs."""
    rmsd = np.asarray(rmsd, dtype=float).ravel()
    afe = np.asarray(afe, dtype=float).ravel()
    mask = np.isfinite(rmsd) & np.isfinite(afe) & (afe > 0.0) & (afe <= AFE_MAX)
    rmsd, afe = rmsd[mask], afe[mask]
    if rmsd_target is not None:
        keys = np.round(np.asarray(rmsd_target, dtype=float).ravel()[mask], decimals=2)
        xs, ys = [], []
        for key in np.unique(keys):
            xs.append(float(key))
            ys.append(float(np.median(afe[keys == key])))
        x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    else:
        x, y = _rung_medians(rmsd, afe)
    if x.size < 3:
        x, y = rmsd, afe
    return fit_saturation_curve(x, y)


def draw_mode_error_figure(systems: dict[str, dict]):
    """One panel per molecule; circles PBE-D4, squares ωB97M-V."""
    apply_classic_tex()
    # cmr10 has no space glyph; titles and the legend must use a Unicode serif.
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "STIXGeneral"]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), sharex=True, sharey=True)
    axes_flat = list(axes.ravel())
    legend_handles = {}
    for ax, (title, names) in zip(axes_flat, MOLECULE_PANELS):
        ax.set_title(title, fontsize=12)
        for name in names:
            entry = systems.get(name)
            if not entry:
                continue
            _abs_q, rmsd, afe = _finite_points(entry.get("records", []))
            if rmsd.size == 0:
                continue
            theory = _theory_key(name)
            style = THEORY_STYLE[theory]
            h = ax.scatter(
                rmsd,
                afe,
                s=22,
                alpha=0.75,
                color=style["color"],
                marker=style["marker"],
                linewidths=0.4,
                edgecolors="0.15",
                label=style["label"],
                zorder=3,
            )
            legend_handles.setdefault(theory, h)
            try:
                params = fit_mode_saturation(rmsd, afe)
            except ValueError:
                continue
            grid = np.linspace(0.0, 11.0, 400)
            ax.plot(
                grid,
                saturation_curve(grid, params["sigma2"], params["length_scale"], params["floor"]),
                color=style["color"],
                lw=SATURATION_LINEWIDTH,
                solid_capstyle="round",
                zorder=4,
            )
        ax.set_xlim(0.0, 11.0)
        ax.set_ylim(0.0, None)
        ax.set_yscale("linear")
    for ax in axes_flat[len(MOLECULE_PANELS):]:
        ax.set_visible(False)
    for ax in axes[1, :2]:
        ax.set_xlabel(r"$\mathrm{RMSD}$ / $\mathrm{\AA}$")
    axes[1, 0].set_xlabel(r"$\mathrm{RMSD}$ / $\mathrm{\AA}$")
    axes[0, 0].set_ylabel(EPSILON_RHO)
    axes[1, 0].set_ylabel(EPSILON_RHO)
    if legend_handles:
        fig.legend(
            [legend_handles[k] for k in ("pbe", "hybrid") if k in legend_handles],
            [THEORY_STYLE[k]["label"] for k in ("pbe", "hybrid") if k in legend_handles],
            loc="lower right",
            bbox_to_anchor=(0.97, 0.08),
        )
    fig.tight_layout()
    return fig


def make_figure(systems: dict[str, dict], path: Path):
    fig = draw_mode_error_figure(systems)
    save_response_figure(fig, path)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        required=True,
        metavar="NAME:MANIFEST:EVAL_JSON",
        help="repeat once per molecule / theory",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    report: dict = {"systems": {}}
    for spec in args.pair:
        name, manifest_path, eval_path = parse_pair(spec)
        if not manifest_path.is_file() or not eval_path.is_file():
            print(f"{name}: missing {'manifest' if not manifest_path.is_file() else 'eval'}, skipping")
            continue
        manifest = json.loads(manifest_path.read_text())
        payload = json.loads(eval_path.read_text())
        joined = join_mode_eval(manifest.get("records", []), payload.get("records", []))
        summary = summarize_mode_errors(joined)
        rmsd = np.array([row["rmsd"] for row in joined], dtype=float)
        afe = np.array([row["afe"] for row in joined], dtype=float)
        rmsd_target = np.array([row.get("rmsd_target", row["rmsd"]) for row in joined], dtype=float)
        try:
            fit = fit_mode_saturation(rmsd, afe, rmsd_target=rmsd_target)
        except ValueError:
            fit = None
        report["systems"][name] = {
            **summary,
            "manifest": str(manifest_path),
            "eval": str(eval_path),
            "records": joined,
            "saturation_fit": fit,
        }
        print(
            f"{name:24s} n={summary['n']:4d}  AFE {summary['afe_mean']:.5f}  "
            f"rho(|q|)={summary['spearman_afe_vs_abs_q']:+.2f}  "
            f"rho(RMSD)={summary['spearman_afe_vs_rmsd']:+.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    if args.figure and report["systems"]:
        make_figure(report["systems"], args.figure)
        print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
