#!/usr/bin/env python3
"""Every energy/force number the response letter quotes, from the same score sets as the figures.

Reads results/revision/energy_forces_v3/<name><suffix>/per_frame.csv (score_energy_forces.py,
score_malon_energy.py) and prints, per model: tier means of ε_E (per water / per molecule) and
ε_F; the water errors in bins of local coordination (other waters within the cutoff of a water;
training clusters reach at most 5); malonaldehyde errors inside the training well, near the
transition state and in the opposite enol; and extreme-mode medians per amplitude rung.
Writes the same as JSON next to the figures.

    .venv/bin/python -B experiments/jacs_revision/analysis/energy_letter_numbers.py [--suffix _interim]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
ENERGY = REPO / "results/revision/energy_forces_v3"
THEORIES = ("pbe_d4_avdz", "wb97mv_def2tzvpd")


def rows(name, suffix):
    path = ENERGY / f"{name}{suffix}" / "per_frame.csv"
    if not path.is_file():
        return None
    out = list(csv.DictReader(open(path)))
    for r in out:
        for k, v in r.items():
            if k != "split":
                r[k] = float(v)
    return out


def stats(group, per_water=True):
    if not group:
        return None
    e = np.array([abs(r["energy_error_kcal_mol"]) / (r["n_water"] if per_water else 1.0) for r in group])
    signed = np.array([r["energy_error_kcal_mol"] / (r["n_water"] if per_water else 1.0) for r in group])
    f = np.array([r["force_mae_kcal_mol_A"] for r in group])
    out = {"n": len(group), "eps_E_mean": float(e.mean()), "eps_E_median": float(np.median(e)),
           "eps_E_signed_mean": float(signed.mean()), "eps_E_bias_removed": float(np.abs(signed - signed.mean()).mean()),
           "eps_F_mean": float(f.mean()), "eps_F_median": float(np.median(f))}
    if all("force_abs_sum" in r and "n_atoms" in r for r in group):
        # MAE over every Cartesian component of the tier: the convention of summary.json and the paper
        out["eps_F_pooled"] = float(sum(r["force_abs_sum"] for r in group) / sum(3 * r["n_atoms"] for r in group))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="_interim")
    opt = ap.parse_args()
    from experiments.jacs_revision.analysis.energy_training_dashboard import water_neighbours

    out = {"suffix": opt.suffix, "water": {}, "coordination": {}, "malon": {}, "modes": {}}
    for th in THEORIES:
        for pre in ("", "direct_"):
            rr = rows(f"water_{pre}{th}", opt.suffix)
            if not rr:
                continue
            key = f"{pre}{th}"
            out["water"][key] = {s: stats([r for r in rr if r["split"] == s]) for s in ("id_test", "ood_size", "ood_order", "ood_density")}
            nb = np.array([water_neighbours(r["split"], th, int(r["index"]), 6.0)[0] for r in rr])
            bins = {"<=5 (training range)": nb <= 5.0 + 1e-9, "5-10": (nb > 5.0 + 1e-9) & (nb <= 10), ">10": nb > 10}
            out["coordination"][key] = {lab: stats([r for r, m in zip(rr, mask) if m]) for lab, mask in bins.items()}
            ok = np.isfinite(nb)
            out["coordination"][key]["corr_F_vs_nb"] = float(np.corrcoef(nb[ok], [r["force_mae_kcal_mol_A"] for r in rr])[0, 1])
        mr = rows(f"malon_{th}", opt.suffix)
        if mr:
            cv = json.loads((REPO / "results/revision/cv/malon_ood_proton_transfer_pt.json").read_text())["frames"]
            delta = {int(f.get("index", i)): float(f["delta_pt"]) for i, f in enumerate(cv)}
            pt = [r for r in mr if r["split"] == "ood_proton_transfer" and int(r["source_index"]) in delta]
            d = np.array([delta[int(r["source_index"])] for r in pt])
            out["malon"][th] = {"val": stats([r for r in mr if r["split"] == "val"], False),
                                "pt_all": stats(pt, False),
                                "pt_training_well(d<=-0.40)": stats([r for r, x in zip(pt, d) if x <= -0.40], False),
                                "pt_transition(|d|<0.1)": stats([r for r, x in zip(pt, d) if abs(x) < 0.1], False),
                                "pt_opposite_enol(d>0.4)": stats([r for r, x in zip(pt, d) if x > 0.4], False)}
            modes = [r for r in mr if r["split"] == "ood_modes_malonaldehyde"]
            if modes:
                out["modes"][f"malonaldehyde/{th}"] = modes
        for pre in ("", "direct_"):
            mw = rows(f"modes_water_{pre}{th}", opt.suffix)
            for system in ("water_n2", "water_n6"):
                sel = [r for r in mw or [] if r["split"] == f"ood_modes_{system}"]
                if sel:
                    out["modes"][f"{system}/{pre}{th}"] = sel
    # extreme modes: medians per RMSD rung
    for key, sel in list(out["modes"].items()):
        system = key.split("/")[0]
        manifest = json.loads((REPO / "datasets/revision/ood_modes" / system / "manifest.json").read_text())
        rmsd = np.array([manifest["records"][int(r["source_index"])]["rmsd"] for r in sel])
        e = np.array([abs(r["energy_error_kcal_mol"]) for r in sel])
        f = np.array([r["force_mae_kcal_mol_A"] for r in sel])
        bad = ~(np.isfinite(e) & np.isfinite(f))  # diverged head (NaN prediction): an infinite error, as in the figure
        e, f = np.where(np.isfinite(e), e, np.inf), np.where(np.isfinite(f), f, np.inf)
        rungs = np.unique(np.round(rmsd, 1))
        out["modes"][key] = {f"{g:g}": {"n": int((np.round(rmsd, 1) == g).sum()),
                                        "n_nonfinite": int(bad[np.round(rmsd, 1) == g].sum()),
                                        "eps_E_median": float(np.median(e[np.round(rmsd, 1) == g])),
                                        "eps_F_median": float(np.median(f[np.round(rmsd, 1) == g]))} for g in rungs}
    text = json.dumps(out, indent=1)
    (REPO / "results/revision/figures/response_v2" / f"energy_letter_numbers{opt.suffix}.json").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
