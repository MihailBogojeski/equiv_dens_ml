"""Energy-head cutoff sweep (water PBE-D4, 40k-step v3 heads at 4/5/6 A): per-split E/F errors side by side,
with the production 6 A head (100k steps, results/revision/energy_forces_v2/water_pbe_d4_avdz_v3_slurm) as reference."""
import json, os, sys
O = "results/revision/energy_forces_v3"
runs = {f"{c} A (40k)": f"{O}/water_pbe_d4_avdz_cutoff_{c}/summary.json" for c in (4, 5, 6)}
runs["6 A prod (100k)"] = "results/revision/energy_forces_v2/water_pbe_d4_avdz_v3_slurm/summary.json"
S = {k: json.load(open(p))["by_split"] for k, p in runs.items() if os.path.exists(p)}
splits = ["id_test", "ood_size", "ood_order", "ood_density"]
for key, label in [("energy_mae_per_water_kcal_mol", "E MAE per water (kcal/mol)"),
                   ("energy_mae_bias_removed_kcal_mol", "E MAE bias-removed (kcal/mol)"),
                   ("force_mae_kcal_mol_A", "F MAE (kcal/mol/A)")]:
    print(f"\n{label}\n{'split':12s}" + "".join(f"{k:>18s}" for k in S))
    for sp in splits:
        print(f"{sp:12s}" + "".join(f"{S[k][sp][key]:18.3f}" for k in S))
loc = {c: json.load(open(f"{O}/water_pbe_d4_avdz_cutoff_{c}/locality.json")) for c in (4, 5, 6) if os.path.exists(f"{O}/water_pbe_d4_avdz_cutoff_{c}/locality.json")}
print("\nLocality (2+6 waters): nonadditivity kcal/mol at 12 A / 30 A, max |dF| on A when a B atom moves (kcal/mol/A)")
for c, d in loc.items():
    p = {q["separation_A"]: q for q in d["pairs"] if q["A"] == "2 waters"}
    print(f"cutoff {c}: " + "  ".join(f"{s} A: {p[s]['nonadditivity_kcal_mol']:+.4f} / dF {p[s]['max_force_change_on_A_when_B_atom_moves']:.4f}" for s in sorted(p)))
