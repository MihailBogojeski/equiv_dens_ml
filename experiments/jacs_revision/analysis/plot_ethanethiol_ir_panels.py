#!/usr/bin/env python3
"""Ethanethiol IR from DenSNet and ML force fields against NIST, one panel per method (R2.2, R3.2).

Reads the recovery runs (1 fs dipole sampling, 300 K Langevin) written by
run_mlip_ir_md.py. Each spectrum is normalized to its maximum between
450 cm^-1 (the start of the NIST record) and 4000 cm^-1. Band positions are
the maxima inside the C-S, S-H and C-H windows; the windows are wide enough
that no method's band sits on an edge.

Usage:
  python experiments/jacs_revision/analysis/plot_ethanethiol_ir_panels.py --window-ps 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.jacs_revision.analysis.mpl_classic_tex import apply as apply_classic_tex  # noqa: E402

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
RUNS = REPO / "results/revision/jacs_recovery_20260927/ethanethiol_ir"
NIST = REPO / "datasets/revision/external/ethanethiol_nist_gas_ir.csv"
# DenSNet: the paper's archived NVE trajectory (same model), dipoles thinned to
# 1 fs and passed through the same MESA routine (paper_densnet/ir_*ps.csv).
METHODS = [("paper_densnet", "DenSNet"), ("maceoff", "MACE-OFF (DenSNet dipole)"),
           ("macepolar", "MACE-POLAR-1-M"), ("so3lr", "SO3LR")]
BANDS = {"C–S stretch": (550.0, 800.0), "Fingerprint": (1150.0, 1350.0), "S–H stretch": (2450.0, 2780.0),
         "C–H stretch": (2800.0, 3350.0)}
LO, HI = 450.0, 4000.0
# Drawn at one journal column (3.4 in) so text is 7-9 pt at natural size.
FIGSIZE = (3.4, 4.3)
RC = {"font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5}


def read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def normalized(freq: np.ndarray, inten: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(freq)
    y = np.interp(grid, freq[order], inten[order], left=0.0, right=0.0)
    return y / y.max()


def band_peaks(grid: np.ndarray, y: np.ndarray) -> dict[str, float]:
    out = {}
    for name, (a, b) in BANDS.items():
        m = (grid >= a) & (grid <= b)
        out[name] = float(grid[m][np.argmax(y[m])])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-ps", type=int, default=50)
    args = ap.parse_args()

    apply_classic_tex()
    matplotlib.rcParams.update(RC)
    grid = np.arange(LO, HI + 0.5, 2.0)
    nist = normalized(*read(NIST), grid)
    metrics = {"window_ps": args.window_ps, "bands_cm1": {k: list(v) for k, v in BANDS.items()},
               "NIST": {"peaks_cm1": band_peaks(grid, nist)}}

    fig, axes = plt.subplots(len(METHODS), 1, figsize=FIGSIZE, sharex=True)
    for ax, (key, label) in zip(axes, METHODS):
        path = RUNS / key / f"ir_{args.window_ps}ps.csv"
        ax.plot(grid, nist, color="0.45", lw=0.8, ls="--", label="NIST (gas phase)")
        if path.is_file():
            y = normalized(*read(path), grid)
            ax.plot(grid, y, color="#1f5fa8" if "densnet" in key and key != "maceoff" else "#c44e52", lw=0.9, label=label)
            peaks = band_peaks(grid, y)
            metrics[label] = {"peaks_cm1": peaks,
                              "shift_vs_nist_cm1": {k: peaks[k] - metrics["NIST"]["peaks_cm1"][k] for k in BANDS},
                              }
        ax.text(0.03, 0.9, label, transform=ax.transAxes, ha="left", va="top")
        ax.set_yticks([])
        ax.set_ylim(0, 1.25)
    for name, x in (("C–S", 670), ("S–H", 2600), ("C–H", 3050)):
        axes[0].text(x, 1.3, name, ha="center", va="bottom", fontsize=8)
    axes[0].legend(loc="upper right", handles=axes[0].get_lines()[:1], frameon=False)
    axes[-1].set_xlim(LO, 3600)
    axes[-1].set_xlabel(r"Wavenumber / cm$^{-1}$")
    fig.supylabel("Normalized IR intensity", fontsize=matplotlib.rcParams["axes.labelsize"])
    fig.tight_layout(h_pad=0.4)

    stem = RUNS / f"ir_panels_{args.window_ps}ps"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    stem.with_suffix(".json").write_text(json.dumps(metrics, indent=1) + "\n")
    print(json.dumps(metrics, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
