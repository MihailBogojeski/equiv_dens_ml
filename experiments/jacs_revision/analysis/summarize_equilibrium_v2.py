#!/usr/bin/env python3
"""Tables and figures for equilibrium_frequencies.py (R2.11).

Writes results/revision/equilibrium_v2/summary.{md,json} and
results/revision/equilibrium_v2/{frequency_parity,ir_harmonic}.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DIR = REPO / "results" / "revision" / "equilibrium_v2"
MOLECULES = ["ethanol", "ethanethiol", "resorcinol"]


def lorentzian(grid, centres, heights, fwhm=10.0):
    g = fwhm / 2
    out = np.zeros_like(grid)
    for c, h in zip(centres, heights):
        out += h * g ** 2 / ((grid - c) ** 2 + g ** 2)
    return out


def main() -> int:
    reports = {m: json.loads((DIR / m / "report.json").read_text()) for m in MOLECULES if (DIR / m / "report.json").is_file()}
    lines = ["| Molecule | Conformer | DFT / DenSNet steps | RMSD heavy / all (Å) | bond MAE / max (Å) | angle MAE / max (°) | "
             "X–H torsion DFT / DenSNet (°) | DenSNet max\\|F\\| at DFT min (eV/Å) | DFT max\\|F\\| at DenSNet min (eV/Å) | "
             "freq MAE / max (cm⁻¹) | MAE < 2000 / ≥ 2000 (cm⁻¹) | min mode overlap |",
             "|---" * 12 + "|"]
    energy_lines = ["| Molecule | Conformer | ΔE DFT (kcal/mol) | ΔE DenSNet (kcal/mol) |", "|---|---|---|---|"]
    for mol, rep in reports.items():
        for conf, e in rep["conformers"].items():
            g, f = e["geometry"], e["frequencies"]
            tors = "; ".join(f"{t['a']:.1f} / {t['b']:.1f}" for t in g["torsions_deg"])
            hi = f["mae_above_2000_cm1"]
            lines.append(
                f"| {mol} | {conf} | {e['dft_opt']['steps']} / {e['densnet_opt']['steps']} | "
                f"{g['rmsd_heavy_A']:.4f} / {g['rmsd_all_A']:.4f} | {g['bond_mae_A']:.4f} / {g['bond_max_A']:.4f} | "
                f"{g['angle_mae_deg']:.2f} / {g['angle_max_deg']:.2f} | {tors} | "
                f"{e['densnet_fmax_at_dft_min_eV_A']:.3f} | {e['dft_fmax_at_densnet_min_eV_A']:.3f} | "
                f"{f['mae_cm1']:.1f} / {f['max_abs_cm1']:.1f} | {f['mae_below_2000_cm1']:.1f} / "
                f"{'—' if hi is None else f'{hi:.1f}'} | {f['min_overlap']:.2f} |")
        for conf, v in rep.get("conformer_energies_kcal_mol", {}).items():
            energy_lines.append(f"| {mol} | {conf} | {v['dft']:.2f} | {v['densnet']:.2f} |")
    notes = []
    for mol, rep in reports.items():
        if rep.get("force_scale_fitted"):
            s = rep["force_scale_fitted"]
            notes.append(f"{mol}: checkpoint lacks its force scale; factor {s['scale']:.3f} fitted to DFT forces "
                         f"(Pearson r {s['pearson_r']:.5f}, {s['n_frames']} frames) and applied to DenSNet energies and forces.")
        notes.append(f"{mol}: {rep['level']}, model {rep['model']}, fmax {rep['fmax_eV_A']} eV/Å, "
                     f"finite-difference step {rep['fd_delta_A']} Å; conformer ordering agrees: {rep.get('ordering_agrees')}.")
    vals = [json.loads((DIR / f"validate_{m}.json").read_text()) for m in MOLECULES if (DIR / f"validate_{m}.json").is_file()]
    val_lines = ["| Molecule | reference | frames | force MAE (kcal/mol/Å) | fit slope | Pearson r |", "|---|---|---|---|---|---|"]
    for v in vals:
        mae = v.get("force_mae_after_fit_kcal_mol_A") if v.get("missing_keys") else v["force_mae_kcal_mol_A"]
        val_lines.append(f"| {v['molecule']} | {v['reference']} | {v['n_frames']} | {mae:.3f} | "
                         f"{v['fit_slope']:.4f}{' (scale fitted)' if v.get('missing_keys') else ''} | {v['pearson_r']:.5f} |")
    md = ["# DenSNet vs DFT minima, conformers and harmonic spectra", "", "## Force validation", *val_lines, "",
          "## Minima and frequencies", *lines, "", "## Conformer energies (relative to each method's lowest)",
          *energy_lines, "", *[f"- {n}" for n in notes], ""]
    (DIR / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.4, 4.2), constrained_layout=True)
    markers = {"ethanol": "o", "ethanethiol": "s", "resorcinol": "^"}
    for mol, rep in reports.items():
        for conf, e in rep["conformers"].items():
            f = e["frequencies"]
            ax.plot(f["dft_cm1"], f["densnet_cm1_paired"], markers[mol], ms=3.5, mfc="none", label=f"{mol} {conf}")
    lim = [0, 3900]
    ax.plot(lim, lim, color="0.4", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("DFT harmonic frequency (cm⁻¹)")
    ax.set_ylabel("DenSNet harmonic frequency (cm⁻¹)")
    ax.legend(fontsize=6, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(DIR / f"frequency_parity.{ext}", dpi=200)

    fig, axes = plt.subplots(len(reports), 1, figsize=(7, 2.3 * len(reports)), constrained_layout=True, squeeze=False)
    grid = np.linspace(0, 4000, 4001)
    for ax, (mol, rep) in zip(axes[:, 0], reports.items()):
        conf = next(iter(rep["conformers"]))
        f = rep["conformers"][conf]["frequencies"]
        sd = lorentzian(grid, f["dft_cm1"], f["dft_ir_km_mol"])
        sm = lorentzian(grid, f["densnet_cm1_paired"], f["densnet_ir_km_mol_paired"])
        norm = max(sd.max(), sm.max())
        ax.plot(grid, sd / norm, color="black", lw=1.2, label="DFT")
        ax.plot(grid, sm / norm, color="#1f5fa8", lw=1.0, ls="--", label="DenSNet")
        ax.set_title(f"{mol} ({conf}): harmonic IR, 10 cm⁻¹ Lorentzian", fontsize=9)
        ax.set_yticks([])
        ax.legend(fontsize=7, frameon=False)
    axes[-1, 0].set_xlabel("wavenumber (cm⁻¹)")
    for ext in ("pdf", "png"):
        fig.savefig(DIR / f"ir_harmonic.{ext}", dpi=200)
    (DIR / "summary.json").write_text(json.dumps({"validation": vals, "reports": reports}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
