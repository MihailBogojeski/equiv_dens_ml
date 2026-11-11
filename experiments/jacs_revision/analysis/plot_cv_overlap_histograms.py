#!/usr/bin/env python3
"""Overlay train vs test histograms for each OOD collective variable.

Reviewer 1 asked for evidence that test splits are not merely drawn from a
different trajectory but actually sit in a different region of configuration
space. Bhattacharyya overlap in ood_overlap_report.py answers that at the
distribution level; these figures make the same claim visible on the histograms
a reader can inspect directly.

Usage:
  python scripts/revision/plot_cv_overlap_histograms.py \\
    --overlap-json results/revision/ood_overlap.json \\
    --out-dir results/revision/figures/cv_separation_water
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/plot_cv_overlap_histograms.py'


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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(_SCRIPT_DIR))

from experiments.jacs_revision.analysis.mpl_classic_tex import apply as apply_classic_tex, save_response_figure
from experiments.jacs_revision.analysis.ood_overlap_report import (
    DEFAULT_JOINTS,
    load_cvs,
    load_q_local_density_pairs,
    load_q_per_molecule,
)

apply_classic_tex()

CV_AXIS = {
    "q3_tetrahedral": r"$q_3$",
    "q_tetrahedral": r"$q$",
    "n_hbond": r"$n_{\mathrm{HB}}$",
    "ring_6_frac": r"$f_6$",
    "local_density": r"$\rho_{\mathrm{loc}}$ / $\mathrm{\AA}^{-3}$",
    "surface_frac": r"$f_{\mathrm{surf}}$",
    "mean_nn_oo": r"$\bar r_{\mathrm{OO}}$ / $\mathrm{\AA}$",
    "delta_pt": r"$\delta$ / $\mathrm{\AA}$",
    "r_oo": r"$r_{\mathrm{OO}}$ / $\mathrm{\AA}$",
    "co_alternation": r"$\Delta r_{\mathrm{CO}}$ / $\mathrm{\AA}$",
    "r_donor_h": r"$r(\mathrm{O_d{-}H})$ / $\mathrm{\AA}$",
    "r_acceptor_h": r"$r(\mathrm{H{-}O_a})$ / $\mathrm{\AA}$",
}


def _cv_label(name: str) -> str:
    return CV_AXIS.get(name, name)


def _values(matrix: np.ndarray, col: int) -> np.ndarray:
    return matrix[:, col][np.isfinite(matrix[:, col])]


def plot_1d(
    train: np.ndarray,
    test: np.ndarray,
    names: tuple[str, ...],
    label: str,
    out_dir: Path,
    *,
    train_q: np.ndarray | None = None,
    test_q: np.ndarray | None = None,
) -> None:
    n = len(names)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()
    for i, name in enumerate(names):
        ax = axes[i]
        if name in ("q_tetrahedral", "q3_tetrahedral") and train_q is not None and test_q is not None:
            a, b = train_q, test_q
        else:
            a, b = _values(train, i), _values(test, i)
        if len(a) == 0 or len(b) == 0:
            ax.set_xlabel(_cv_label(name))
            ax.set_ylabel(r"$H$ (hist.)")
            continue
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        pad = 0.05 * (hi - lo) + 1e-9
        bins = np.linspace(lo - pad, hi + pad, 40)
        ax.hist(a, bins=bins, density=True, alpha=0.55, label=f"train (n={len(a)})", color="#1f77b4")
        ax.hist(b, bins=bins, density=True, alpha=0.55, label=f"{label} (n={len(b)})", color="#d62728")
        p01, p99 = np.percentile(a, [1, 99])
        ax.axvspan(p01, p99, color="#1f77b4", alpha=0.08)
        ax.set_xlabel(_cv_label(name))
        ax.set_ylabel(r"$H$ (hist.)")
        ax.legend()
    for j in range(n, len(axes)):
        axes[j].axis("off")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_response_figure(fig, out_dir / f"{label}_1d.png")


def plot_joint(
    train: np.ndarray,
    test: np.ndarray,
    plane: tuple[str, ...],
    names: tuple[str, ...],
    label: str,
    out_dir: Path,
    *,
    train_path: Path | None = None,
    test_path: Path | None = None,
) -> None:
    if plane[0] in ("q_tetrahedral", "q3_tetrahedral") and train_path and test_path:
        field = "q3_per_molecule" if plane[0] == "q3_tetrahedral" else "q_per_molecule"
        aq, ad = load_q_local_density_pairs(train_path, field=field)
        bq, bd = load_q_local_density_pairs(test_path, field=field)
        if len(aq) < 2 or len(bq) < 2:
            return
        a = np.column_stack([aq, ad])
        b = np.column_stack([bq, bd])
    else:
        ix = [names.index(v) for v in plane]
        a = train[:, ix]
        b = test[:, ix]
        a = a[np.isfinite(a).all(axis=1)]
        b = b[np.isfinite(b).all(axis=1)]
    if len(a) < 2 or len(b) < 2:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    ax.hist2d(a[:, 0], a[:, 1], bins=32, density=True, cmap="Blues", alpha=0.85, label="train")
    ax.scatter(b[:, 0], b[:, 1], s=12, c="#d62728", alpha=0.65, edgecolors="none", label=label)
    ax.set_xlabel(_cv_label(plane[0]))
    ax.set_ylabel(_cv_label(plane[1]))
    ax.legend(loc="best")
    tag = "_".join(plane)
    save_response_figure(fig, out_dir / f"{label}_joint_{tag}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.overlap_json.read_text())
    names = tuple(report["cvs"])
    joints = [tuple(j) for j in report.get("joint_cvs", DEFAULT_JOINTS)]
    train_path = Path(report["train"])
    train, _ = load_cvs(train_path, names)
    train_q = load_q_per_molecule(train_path, field="q3_per_molecule")

    for split_label, entry in report["splits"].items():
        if "error" in entry:
            continue
        test_path = Path(entry["path"])
        test, _ = load_cvs(test_path, names)
        test_q = load_q_per_molecule(test_path, field="q3_per_molecule")
        plot_1d(
            train, test, names, split_label, args.out_dir,
            train_q=train_q, test_q=test_q,
        )
        for plane in joints:
            if all(v in names for v in plane):
                plot_joint(
                    train, test, plane, names, split_label, args.out_dir,
                    train_path=train_path, test_path=test_path,
                )
        print(
            f"wrote histograms for {split_label} "
            f"(q molecules: train={len(train_q)}, test={len(test_q)}) -> {args.out_dir}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
