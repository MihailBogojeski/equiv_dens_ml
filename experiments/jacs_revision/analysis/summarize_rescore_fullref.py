#!/usr/bin/env python3
"""Tables and figures from the full-density re-score (rescore_full_density.py).

Reads results/revision/rescore_fullref/eval and writes, next to it,
summary.md / summary.json and the figures under figures/:

* tier table: epsilon_rho against the full SCF density, the draft-letter
  metric (against the DF projection) and the DF floor, PBE-D4 and wB97M-V
  (production seed plus the three extra seeds), malonaldehyde PT scan;
* paired PBE - wB97M-V difference by source_index;
* delta (SAD) versus direct learning, and density phase versus fine-tuning;
* cutoff sweep and N_train learning curves;
* error versus d/w and the malonaldehyde delta axis (error_vs_distance.py),
  extreme-mode displacements (score_extreme_modes.py), both re-run on the new
  evaluations.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
ROOT = REPO / "results" / "revision" / "rescore_fullref"
EVAL = ROOT / "eval"
FIG = ROOT / "figures"
CV = REPO / "results" / "revision" / "cv"
TIERS = [("water_id_test", "ID test (n=2-6)"), ("water_ood_size", "Size OOD (n=8,10,12)"),
         ("water_ood_order", "Ice cutouts"), ("water_ood_density", "Droplets")]
MALON = "malonaldehyde_ood_proton_transfer"
SEEDS = ["", "_seed7", "_seed123", "_seed20260821"]
PY = sys.executable
ENV = {**__import__("os").environ, "PYTHONPATH": f"{REPO}:{REPO / 'src'}", "MPLBACKEND": "Agg"}


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def records(payload) -> dict[int, dict]:
    return {int(r.get("source_index", r["index"])): r for r in payload["records"]}


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.3f}"


def mean_of(payload, key="afe", stat="mean"):
    """Mean (or median) over structures with a finite value; non-finite labels are skipped."""
    if payload is None:
        return None
    vals = np.array([r[key] for r in payload["records"] if key in r], dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None
    return float(np.median(vals) if stat == "median" else vals.mean())


def n_nonfinite(payload, key="afe"):
    return 0 if payload is None else int(sum(1 for r in payload["records"] if key in r and not np.isfinite(r[key])))


def tier_table() -> tuple[str, dict]:
    out = {}
    lines = ["| Tier | n | PBE-D4 ε_ρ | ωB97M-V ε_ρ (4 seeds) | PBE vs DF (draft) | ωB97M-V vs DF (draft) | DF floor PBE | DF floor ωB97M-V |",
             "|---|---|---|---|---|---|---|---|"]
    for tier, label in TIERS + [(MALON, "Malonaldehyde PT scan")]:
        pbe = load(EVAL / "pbe_d4_avdz" / f"{tier}.json")
        hyb = [load(EVAL / f"wb97mv_def2tzvpd{s}" / f"{tier}.json") for s in SEEDS]
        hyb_means = [mean_of(h) for h in hyb if h is not None]
        row = {
            "n": len(pbe["records"]) if pbe else None,
            "pbe": mean_of(pbe), "pbe_vs_df": mean_of(pbe, "afe_vs_df"), "pbe_df_floor": mean_of(pbe, "df_floor"),
            "wb97mv": mean_of(hyb[0]), "wb97mv_seed_means": hyb_means,
            "wb97mv_seed_std": float(np.std(hyb_means, ddof=1)) if len(hyb_means) > 1 else None,
            "wb97mv_vs_df": mean_of(hyb[0], "afe_vs_df"), "wb97mv_df_floor": mean_of(hyb[0], "df_floor"),
        }
        if pbe and hyb[0]:
            a, b = records(pbe), records(hyb[0])
            common = sorted(set(a) & set(b))
            d = np.array([b[k]["afe"] - a[k]["afe"] for k in common])
            row["paired"] = {"n": len(common), "mean_diff_wb97mv_minus_pbe": float(d.mean()),
                             "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                             "frac_wb97mv_lower": float(np.mean(d < 0))}
        out[tier] = row
        spread = (f"{pct(row['wb97mv'])} ({pct(min(hyb_means))}–{pct(max(hyb_means))})"
                  if len(hyb_means) > 1 else pct(row["wb97mv"]))
        lines.append(f"| {label} | {row['n']} | {pct(row['pbe'])} | {spread} | {pct(row['pbe_vs_df'])} | "
                     f"{pct(row['wb97mv_vs_df'])} | {pct(row['pbe_df_floor'])} | {pct(row['wb97mv_df_floor'])} |")
    note = ("ε_ρ = ∫|ρ_pred − ρ_ref| dV / ∫ρ_ref dV in %, full level-2 grid. 'vs DF (draft)' is the metric the "
            "draft letter reported (reference = DF projection of the SCF density). DF floor = ∫|ρ_DF − ρ_SCF| / N_e.")
    return note + "\n\n" + "\n".join(lines), out


def paired_table(rows: dict) -> str:
    lines = ["| Tier | n paired | ε(ωB97M-V) − ε(PBE-D4), mean ± s.e. (%) | ωB97M-V lower on |",
             "|---|---|---|---|"]
    for tier, label in TIERS + [(MALON, "Malonaldehyde PT scan")]:
        p = rows[tier].get("paired")
        if p:
            lines.append(f"| {label} | {p['n']} | {100 * p['mean_diff_wb97mv_minus_pbe']:+.3f} ± {100 * p['se']:.3f} | "
                         f"{100 * p['frac_wb97mv_lower']:.0f}% of frames |")
    return "\n".join(lines)


def arms_table() -> tuple[str, dict]:
    out = {}
    lines = ["| Tier | Theory | Δ (SAD), production | Δ (SAD), density phase | direct, density phase | direct, fine-tuned |",
             "|---|---|---|---|---|---|"]
    for theory, short in (("pbe_d4_avdz", "PBE-D4"), ("wb97mv_def2tzvpd", "ωB97M-V")):
        for tier, label in TIERS:
            vals = {
                "sad_prod": mean_of(load(EVAL / theory / f"{tier}.json")),
                "sad_density": mean_of(load(EVAL / f"{theory}_densityphase" / f"{tier}.json")),
                "direct_density": mean_of(load(EVAL / f"direct_{theory}_densityphase" / f"{tier}.json")),
                "direct_ft": mean_of(load(EVAL / f"direct_{theory}" / f"{tier}.json")),
            }
            out[f"{theory}/{tier}"] = vals
            ft = vals["direct_ft"]
            ft_s = "diverged (ε ≈ {:.1f})".format(ft) if ft is not None and ft > 1 else pct(ft)
            lines.append(f"| {label} | {short} | {pct(vals['sad_prod'])} | {pct(vals['sad_density'])} | "
                         f"{pct(vals['direct_density'])} | {ft_s} |")
    return "\n".join(lines), out


def cutoff_table() -> tuple[str, dict]:
    out = {}
    lines = ["| Cutoff (Å) | ID test ε_ρ (%) | Size OOD ε_ρ (%) | ID vs DF (draft) | Size OOD vs DF (draft) |", "|---|---|---|---|---|"]
    for cut in (4, 5, 6, 8):
        a = load(EVAL / "cutoff_sweep" / f"cutoff_{cut}_water_id_test.json")
        b = load(EVAL / "cutoff_sweep" / f"cutoff_{cut}_water_ood_size.json")
        out[cut] = {"id": mean_of(a), "ood_size": mean_of(b), "id_vs_df": mean_of(a, "afe_vs_df"),
                    "ood_size_vs_df": mean_of(b, "afe_vs_df")}
        lines.append(f"| {cut} | {pct(out[cut]['id'])} | {pct(out[cut]['ood_size'])} | "
                     f"{pct(out[cut]['id_vs_df'])} | {pct(out[cut]['ood_size_vs_df'])} |")
    return "\n".join(lines), out


def learning_curves() -> tuple[str, dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {}
    sizes = [50, 125, 250, 500, 1250]
    for arm in ("sad", "direct"):
        for tier, _ in TIERS:
            ys = []
            for n in sizes:
                if n == 1250:
                    path = EVAL / ("pbe_d4_avdz_densityphase" if arm == "sad" else "direct_pbe_d4_avdz_densityphase") / f"{tier}.json"
                else:
                    path = EVAL / "learning_curves" / f"ntrain_{n}_{arm}_{tier}.json"
                ys.append(mean_of(load(path)))
            out[f"{arm}/{tier}"] = dict(zip(sizes, ys))
    lines = ["| N_train | " + " | ".join(f"Δ {l}" for _, l in TIERS) + " | " + " | ".join(f"direct {l}" for _, l in TIERS) + " |",
             "|---" * (1 + 2 * len(TIERS)) + "|"]
    for n in sizes:
        cells = [pct(out[f"sad/{t}"][n]) for t, _ in TIERS] + [pct(out[f"direct/{t}"][n]) for t, _ in TIERS]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    fig, axes = plt.subplots(1, len(TIERS), figsize=(11, 3.0), sharey=True, constrained_layout=True)
    for ax, (tier, label) in zip(axes, TIERS):
        for arm, color, name in (("sad", "#1f5fa8", "Δ (SAD prior)"), ("direct", "#c0392b", "direct")):
            xs = [n for n in sizes if out[f"{arm}/{tier}"][n] is not None]
            ax.plot(xs, [100 * out[f"{arm}/{tier}"][n] for n in xs], "o-", color=color, label=name)
        floor = mean_of(load(EVAL / "pbe_d4_avdz" / f"{tier}.json"), "df_floor")
        if floor:
            ax.axhline(100 * floor, color="0.5", ls="--", lw=1, label="DF floor")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("N_train")
    axes[0].set_ylabel("ε_ρ vs full density (%)")
    axes[0].legend(frameon=False, fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"learning_curves_ntrain_fullref.{ext}", dpi=200)
    note = ("Water PBE-D4, full-density ε_ρ (%). N_train = 1250 is the full-data run at the end of its density phase "
            "(the N_train runs have no fine-tuning phase).")
    return note + "\n\n" + "\n".join(lines), out


def malon_split() -> tuple[str, dict]:
    """In-well (delta <= -0.40 A, the training range) versus held-out frames of the PT scan."""
    cv = load(CV / "malon_ood_proton_transfer_pt.json")
    if cv is None:
        return "no malonaldehyde CVs", {}
    frames = cv.get("frames", cv.get("records", []))
    delta = {}
    for i, f in enumerate(frames):
        key = int(f.get("source_index", f.get("index", i)))
        value = f.get("delta_pt", f.get("cvs", {}).get("delta_pt") if isinstance(f.get("cvs"), dict) else None)
        if value is not None:
            delta[key] = float(value)
    out = {}
    lines = ["| Theory | frames in training well (δ ≤ −0.40 Å) | held out (δ > −0.40 Å) | of which δ > 0 (opposite enol) | DF floor |",
             "|---|---|---|---|---|"]
    for theory, short in (("pbe_d4_avdz", "PBE-D4"), ("wb97mv_def2tzvpd", "ωB97M-V")):
        payload = load(EVAL / theory / f"{MALON}.json")
        if payload is None:
            continue
        recs = records(payload)
        groups = {"in_well": [], "held_out": [], "opposite": []}
        for k, r in recs.items():
            if k not in delta:
                continue
            if delta[k] <= -0.40:
                groups["in_well"].append(r["afe"])
            else:
                groups["held_out"].append(r["afe"])
                if delta[k] > 0:
                    groups["opposite"].append(r["afe"])
        out[theory] = {g: {"n": len(v), "mean": float(np.mean(v)) if v else None} for g, v in groups.items()}
        out[theory]["df_floor"] = mean_of(payload, "df_floor")
        o = out[theory]
        lines.append(f"| {short} | {pct(o['in_well']['mean'])} (n={o['in_well']['n']}) | {pct(o['held_out']['mean'])} "
                     f"(n={o['held_out']['n']}) | {pct(o['opposite']['mean'])} (n={o['opposite']['n']}) | {pct(o['df_floor'])} |")
    return "\n".join(lines), out


def run_error_vs_distance() -> dict:
    out = {}
    for theory in ("pbe_d4_avdz", "wb97mv_def2tzvpd"):
        e = EVAL / theory
        cmd = [PY, "-B", str(REPO / "experiments/jacs_revision/analysis/error_vs_distance.py"),
               "--train-cv", str(CV / "water_train.json"),
               "--tier", f"id_test:{CV / 'water_id_test.json'}:{e / 'water_id_test.json'}",
               "--tier", f"ood_size:{CV / 'water_ood_size.json'}:{e / 'water_ood_size.json'}",
               "--tier", f"ood_order:{CV / 'ood_order.json'}:{e / 'water_ood_order.json'}",
               "--tier", f"ood_density:{CV / 'ood_density.json'}:{e / 'water_ood_density.json'}",
               "--physical-axis", f"malonaldehyde:{CV / 'malon_ood_proton_transfer_pt.json'}:{e / (MALON + '.json')}:delta_pt",
               "--out", str(ROOT / f"error_vs_distance_{theory}.json"),
               "--figure", str(FIG / f"error_vs_distance_{theory}.pdf"),
               "--figure-water", str(FIG / f"error_vs_distance_water_{theory}.pdf"),
               "--figure-physical", str(FIG / f"malon_error_vs_delta_{theory}.pdf")]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=ENV)
        out[theory] = {"returncode": proc.returncode, "stderr_tail": proc.stderr[-800:]}
        if proc.returncode == 0:
            out[theory]["report"] = load(ROOT / f"error_vs_distance_{theory}.json")
    return out


def run_extreme_modes() -> dict:
    pairs = []
    for name, system in [("water_n2_pbe", "water_n2"), ("water_n2_hybrid", "water_n2"), ("water_n6_pbe", "water_n6"),
                         ("water_n6_hybrid", "water_n6"), ("malonaldehyde_pbe", "malonaldehyde"),
                         ("malonaldehyde_hybrid", "malonaldehyde"), ("ethanol_pbe", "ethanol"),
                         ("ethanethiol_pbe", "ethanethiol"), ("resorcinol_pbe", "resorcinol"),
                         ("thiophene_2mer_pbe", "thiophene_2mer")]:
        pairs += ["--pair", f"{name}:{REPO / 'datasets/revision/ood_modes' / system / 'manifest.json'}:{EVAL / 'ood_geom' / (name + '.json')}"]
    # score_extreme_modes averages per amplitude bin; drop frames whose labels are
    # non-finite (a few resorcinol SCFs) into a filtered copy and count them.
    filtered = ROOT / "eval_filtered" / "ood_geom"
    filtered.mkdir(parents=True, exist_ok=True)
    dropped = {}
    for f in sorted((EVAL / "ood_geom").glob("*.json")):
        payload = load(f)
        keep = [r for r in payload["records"] if np.isfinite(r["afe"])]
        dropped[f.stem] = len(payload["records"]) - len(keep)
        (filtered / f.name).write_text(json.dumps({**payload, "records": keep}) + "\n")
    pairs = [x.replace(str(EVAL / "ood_geom"), str(filtered)) for x in pairs]
    cmd = [PY, "-B", str(REPO / "experiments/jacs_revision/analysis/score_extreme_modes.py"), *pairs,
           "--out", str(ROOT / "ood_extreme_modes.json"), "--figure", str(FIG / "ood_extreme_mode_error.pdf")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=ENV)
    rep = load(ROOT / "ood_extreme_modes.json") if proc.returncode == 0 else None
    # DF floor per system from the evaluations themselves.
    floors = {}
    for f in sorted((EVAL / "ood_geom").glob("*.json")):
        payload = load(f)
        vals = np.array([r["afe"] for r in payload["records"]], dtype=float)
        floors[f.stem] = {"afe_mean": mean_of(payload), "afe_median": mean_of(payload, stat="median"),
                          "n_afe_above_1": int(np.sum(vals[np.isfinite(vals)] > 1)),
                          "n_nonfinite": dropped.get(f.stem, 0), "n": len(vals),
                          "df_floor_mean": mean_of(payload, "df_floor"),
                          "afe_vs_df_median": mean_of(payload, "afe_vs_df", stat="median")}
    lines = ["| System | n | ε_ρ median (%) | ε_ρ mean (%) | frames ε_ρ > 100% | non-finite labels | DF floor (%) |",
             "|---|---|---|---|---|---|---|"]
    for name, v in floors.items():
        lines.append(f"| {name} | {v['n']} | {pct(v['afe_median'])} | {pct(v['afe_mean'])} | {v['n_afe_above_1']} | "
                     f"{v['n_nonfinite']} | {pct(v['df_floor_mean'])} |")
    return {"returncode": proc.returncode, "stderr_tail": proc.stderr[-800:], "report": rep, "per_system": floors,
            "table": "\n".join(lines)}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    tier_md, tiers = tier_table()
    arms_md, arms = arms_table()
    cut_md, cuts = cutoff_table()
    lc_md, lcs = learning_curves()
    malon_md, malon = malon_split()
    evd = run_error_vs_distance()
    modes = run_extreme_modes()
    summary = {"tiers": tiers, "arms": arms, "cutoffs": cuts, "learning_curves": lcs, "malon_split": malon,
               "error_vs_distance": evd, "extreme_modes": modes}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    md = ["# Full-density re-score of the response-letter density errors", "",
          "## Tiers", tier_md, "", "## Paired PBE-D4 vs ωB97M-V (same geometries)", paired_table(tiers), "",
          "## Δ-learning (SAD prior) vs direct learning", arms_md, "",
          "## Cutoff sweep (PBE-D4)", cut_md, "", "## N_train learning curves", lc_md, "",
          "## Malonaldehyde proton-transfer scan", malon_md, ""]
    md += ["## Extreme-amplitude one-mode displacements (all amplitudes pooled)", modes.get("table", ""), ""]
    for theory, res in evd.items():
        md.append(f"error_vs_distance ({theory}): return code {res['returncode']}")
    md.append(f"extreme modes: return code {modes['returncode']}")
    (ROOT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
