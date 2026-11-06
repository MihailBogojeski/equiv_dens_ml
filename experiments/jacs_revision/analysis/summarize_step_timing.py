#!/usr/bin/env python3
"""Tables and figures for benchmark_step_timing.py (R1.6, R2.2, R3.2, R3.4).

Usage:
  python experiments/jacs_revision/analysis/summarize_step_timing.py results/revision/timing_v2/dedicated_gpu
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SYSTEMS = ["ethanol", "resorcinol", "water_n6", "water_n12", "thiophene_6mer", "water_n24", "thiophene_12mer",
           "water_n30", "water_4x30"]
LABEL = {"densnet": "DenSNet (E+F)", "densnet_dipole": "DenSNet (E+F+μ)", "densnet_direct": "DenSNet direct (E+F)",
         "so3lr": "SO3LR", "maceoff": "MACE-OFF", "macepolar": "MACE-POLAR-1-M", "aimnet2": "AIMNet2",
         "gfn2xtb": "GFN2-xTB", "gxtb": "g-xTB", "dftb": "DFTB3/3ob", "dft_gpu": "DFT (GPU4PySCF)",
         "aimd_paper": "paper AIMD calculator"}
ORDER = ["densnet_dipole", "densnet", "densnet_direct", "so3lr", "maceoff", "macepolar", "aimnet2",
         "gfn2xtb", "gxtb", "dftb", "dft_gpu", "aimd_paper"]
# energy_forces is the combined stage of runs made before energy_forward and
# force_backward were timed separately.
STAGES = ["inputs", "sad_prior", "density_network", "energy_forward", "force_backward", "energy_forces",
          "dipole", "output"]


def fmt(s):
    if s is None:
        return "—"
    if s < 1e-2:
        return f"{s * 1e3:.2f} ms"
    if s < 1:
        return f"{s * 1e3:.0f} ms"
    return f"{s:.2f} s"


def main() -> int:
    root = Path(sys.argv[1])
    runs = {}
    for f in sorted((root / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        runs[(r["system"], r["method"], r["device"])] = r
    natoms = {s: next((r["n_atoms"] for (sy, _, _), r in runs.items() if sy == s and "n_atoms" in r), None) for s in SYSTEMS}

    md = [f"# Per-step timing ({root.name})", ""]
    for node in sorted(root.glob("node*.txt")):
        md += [f"`{node.name}`", "```", node.read_text().strip(), "```", ""]
    for device in ("cuda", "cpu"):
        header = "| Method | " + " | ".join(f"{s} ({natoms[s]})" for s in SYSTEMS) + " |"
        md += [f"## {'GPU' if device == 'cuda' else 'CPU'}: median wall time per energy+force call (IQR in the JSON)",
               "", header, "|---" * (len(SYSTEMS) + 1) + "|"]
        for m in ORDER:
            cells = []
            for s in SYSTEMS:
                r = runs.get((s, m, device))
                if r is None:
                    cells.append("—")
                elif "error" in r:
                    cells.append("failed")
                else:
                    cells.append(f"{fmt(r['median_s'])} (n={r['n_timed']})")
            if any(c != "—" for c in cells):
                md.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
        md.append("")

    for device in ("cuda", "cpu"):
        md += [f"## DenSNet step broken down ({'GPU' if device == 'cuda' else 'CPU'}, median per stage)", "",
               "| System | " + " | ".join(STAGES) + " | sum | full step |", "|---" * (len(STAGES) + 3) + "|"]
        for s in SYSTEMS:
            for m in ("densnet_parts", "densnet_direct_parts"):
                r = runs.get((s, m, device))
                if not r or "parts" not in r:
                    continue
                full = runs.get((s, "densnet_dipole" if m == "densnet_parts" else "densnet_direct", device), {})
                cells = [fmt(r["parts"][k]["median_s"]) if k in r["parts"] else "—" for k in STAGES]
                md.append(f"| {s} ({natoms[s]}){' (direct)' if 'direct' in m else ''} | " + " | ".join(cells) +
                          f" | {fmt(r['parts_sum_median_s'])} | {fmt(full.get('median_s'))} |")
        md.append("")

    grid_stages = ["grid", "sad_on_grid", "density_network", "density_head"]
    md += ["## Real-space density on the full level-2 grid (GPU, median per stage)", "",
           "| System | grid points | " + " | ".join(grid_stages) + " | total |", "|---" * (len(grid_stages) + 3) + "|"]
    for s in SYSTEMS:
        for m in ("density_grid", "density_grid_direct"):
            r = runs.get((s, m, "cuda"))
            if not r or "parts" not in r:
                continue
            md.append(f"| {s}{' (direct)' if 'direct' in m else ''} | {r.get('grid_info', {}).get('n_grid', '—')} | "
                      + " | ".join(fmt(r["parts"][k]["median_s"]) for k in grid_stages) + f" | {fmt(r['median_s'])} |")
    md.append("")

    # Speed-up of DenSNet over the DFT reference on the same GPU.
    md += ["## DenSNet (E+F+μ) against DFT on the same GPU", "", "| System | DenSNet | DFT (GPU4PySCF) | speed-up | paper AIMD calculator |", "|---|---|---|---|---|"]
    for s in SYSTEMS:
        d = runs.get((s, "densnet_dipole", "cuda"), {})
        q = runs.get((s, "dft_gpu", "cuda"), {})
        a = runs.get((s, "aimd_paper", "cuda"), {})
        if "median_s" in d and "median_s" in q:
            md.append(f"| {s} ({natoms[s]}) | {fmt(d['median_s'])} | {fmt(q['median_s'])} | "
                      f"{q['median_s'] / d['median_s']:.2g}× | {fmt(a.get('median_s'))} |")
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True, sharey=True)
    styles = {
        "densnet": dict(color="black", lw=2.4, marker="o", label="DenSNet, energy + forces"),
        "densnet_dipole": dict(color="black", lw=1.4, ls="--", marker="o", label="DenSNet, + dipole (IR step)"),
        "densnet_direct": dict(color="#e67e22", lw=1.4, marker="s", label="DenSNet direct, energy + forces"),
        "so3lr": dict(color="#2ca02c", lw=1.2, marker="o", label="SO3LR"),
        "maceoff": dict(color="#d62728", lw=1.2, marker="o", label="MACE-OFF"),
        "macepolar": dict(color="#9467bd", lw=1.2, marker="o", label="MACE-POLAR-1-M"),
        "aimnet2": dict(color="#8c564b", lw=1.2, marker="o", label="AIMNet2"),
        "gfn2xtb": dict(color="#7f7f7f", lw=1.2, marker="^", label="GFN2-xTB"),
        "gxtb": dict(color="#bcbd22", lw=1.2, marker="^", label="g-xTB"),
        "dftb": dict(color="#17becf", lw=1.2, marker="v", label="DFTB3/3ob"),
        "dft_gpu": dict(color="#1f77b4", lw=2.0, marker="D", label="DFT, PBE/aug-cc-pVDZ (GPU4PySCF)"),
        "aimd_paper": dict(color="#1f77b4", lw=1.0, ls=":", marker="x", label="paper AIMD calculator"),
    }
    grid_dipole = {s for s in SYSTEMS if runs.get((s, "densnet_dipole", "cuda"), {}).get("dipole_head") is False}
    for ax, device in zip(axes, ("cuda", "cpu")):
        for m, st in styles.items():
            pts = [(natoms[s], runs[(s, m, device)]) for s in SYSTEMS
                   if (s, m, device) in runs and "median_s" in runs[(s, m, device)]]
            if not pts:
                continue
            pts.sort(key=lambda t: t[0])
            x = np.array([p[0] for p in pts])
            y = np.array([p[1]["median_s"] for p in pts])
            err = [y - np.array([p[1]["p25_s"] for p in pts]), np.array([p[1]["p75_s"] for p in pts]) - y]
            ax.errorbar(x, y, yerr=err, ms=4, capsize=2, **st)
            if m == "densnet_dipole":
                hollow = [i for i, s in enumerate(sorted([s for s in SYSTEMS if (s, m, device) in runs and "median_s" in runs[(s, m, device)]], key=lambda s: natoms[s])) if s in grid_dipole]
                ax.plot(x[hollow], y[hollow], "o", ms=8, mfc="white", mec="black", zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("atoms")
        ax.set_xticks([9, 14, 18, 36, 44, 86], ["9", "14", "18", "36", "44", "86"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title("GPU: one A100, 16 pinned cores" if device == "cuda" else "CPU: 16 pinned cores", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("wall time per step (s), median ± IQR")
    handles, labels = axes[0].get_legend_handles_labels()
    extra_h, extra_l = axes[1].get_legend_handles_labels()
    for h, l in zip(extra_h, extra_l):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc="outside right upper", fontsize=7, frameon=False)
    if grid_dipole:
        fig.text(0.815, 0.12, "open circles: dipole from the\ndensity on the grid (ethanol and\nresorcinol checkpoints have no\nanalytic dipole head)", fontsize=6.5, va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(root / f"timing_scan.{ext}", dpi=200)
    print(f"wrote {root / 'summary.md'}, {root / 'timing_scan.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
