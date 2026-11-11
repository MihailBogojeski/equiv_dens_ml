#!/usr/bin/env python3
"""Energy and force errors of a DenSNet energy head on the water tiers.

The energy counterpart of the ε_ρ tables: the same splits (validation, ID
test, size OOD, ice cutouts, droplets), scored per structure. Reference
energies are the labels minus the composition baseline the head was trained
against (the run's --atomic_energies), never a test-set offset.

Frames are predicted in batches on the GPU. Before per-molecule layer
normalisation (network_blocks.batch_segments), batching changed every
predicted energy; --check-single
recomputes the first frames one at a time and fails if the two disagree.

    .venv/bin/python -B experiments/jacs_revision/analysis/score_energy_forces.py \
        --run results/revision/water_pbe_d4_avdz_energy_ft/<run> --theory pbe_d4_avdz \
        --out results/revision/energy_forces_v2/water_pbe_d4_avdz_sad
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

SPLITS = ["val", "id_test", "ood_size", "ood_order", "ood_density"]


def sha256_head(path, n=1 << 20):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read(n)).hexdigest()


def metrics(err_e, n_water, f_abs, f_sq, n_comp):
    err_e = np.asarray(err_e, float)
    return {
        "n": int(err_e.size),
        "energy_mae_kcal_mol": float(np.abs(err_e).mean()),
        "energy_rmse_kcal_mol": float(np.sqrt(np.mean(err_e ** 2))),
        "energy_bias_kcal_mol": float(err_e.mean()),
        "energy_mae_bias_removed_kcal_mol": float(np.abs(err_e - err_e.mean()).mean()),
        "energy_mae_per_water_kcal_mol": float(np.abs(err_e / np.asarray(n_water)).mean()),
        "force_mae_kcal_mol_A": float(np.sum(f_abs) / np.sum(n_comp)),
        "force_rmse_kcal_mol_A": float(np.sqrt(np.sum(f_sq) / np.sum(n_comp))),
    }


def summarize(rows):
    out = {"by_split": {}, "by_size": []}
    for split in dict.fromkeys(r["split"] for r in rows):
        group = [r for r in rows if r["split"] == split]
        pick = lambda g: metrics([r["energy_error_kcal_mol"] for r in g], [r["n_water"] for r in g],
                                 [r["force_abs_sum"] for r in g], [r["force_sq_sum"] for r in g],
                                 [3 * r["n_atoms"] for r in g])
        out["by_split"][split] = pick(group)
        # Error in energy differences between clusters of equal composition: the mean
        # error of each cluster size is removed, so a size-dependent offset cannot count.
        err = np.array([r["energy_error_kcal_mol"] for r in group], float)
        size = np.array([r["n_water"] for r in group])
        centered = err - np.array([err[size == n].mean() for n in size])
        out["by_split"][split]["energy_mae_size_centered_kcal_mol"] = float(np.abs(centered).mean())
        for n in sorted({r["n_water"] for r in group}):
            out["by_size"].append({"split": split, "n_water": n,
                                   **pick([r for r in group if r["n_water"] == n])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="energy run directory (best_<ID>.pth, args.txt)")
    ap.add_argument("--theory", choices=["pbe_d4_avdz", "wb97mv_def2tzvpd"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--label-file", action="append", default=[], metavar="NAME=PATH",
                    help="score an extra water label file (e.g. ood_modes_water_n2=datasets/revision/ood_geom_labels/...base.npy)")
    ap.add_argument("--batch", type=int, default=16, help="frames per batch at most")
    ap.add_argument("--max-atoms", type=int, default=240, help="atoms per batch at most (GPU memory)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--limit", type=int, help="frames per split (smoke tests only)")
    ap.add_argument("--check-single", type=int, default=4, help="frames recomputed one at a time as a check")
    ap.add_argument("--single-tol", type=float, default=1e-2,
                    help="largest batched-vs-single energy difference tolerated (kcal/mol); raise it only for "
                         "in-training snapshots, whose large head gain amplifies float32 noise")
    ap.add_argument("--single-ulp-factor", type=float, default=4.0,
                    help="per-frame tolerance is max(--single-tol, this x the float32 ulp of the absolute reference "
                         "energy): the model works on absolute energies (~1e5-1e6 kcal/mol), so batched and single "
                         "evaluations differ by accumulation rounding that grows with |E| (0.1 kcal/mol for 16 waters); "
                         "in float64 the difference is 1e-14 (results/revision/energy_training_dashboard/"
                         "batched_vs_single/). 0 restores the fixed gate")
    opt = ap.parse_args()

    import torch
    from equiv_dens.md.dft_network_calculator import geometry_to_model_inputs, load_densnet_calculator
    from equiv_dens.utils import orbitals
    import equiv_dens.utils as utils

    run = opt.run.resolve()
    meta = torch.load(run / "checkpoints/latest_checkpoint.pth", map_location="cpu", weights_only=False)
    args = meta["args"]
    best = run / f"best_{meta['ID']}.pth"
    to_kcal = getattr(utils, args.energy_unit_in + "_to_kcal")
    to_kcal_per_A = lambda f: to_kcal(f) / getattr(utils, args.distance_unit_in + "_to_angstrom")(1.0)
    if not str(args.energy_unit_out).startswith("kcal") or args.distance_unit_out != "angstrom":
        raise ValueError("expects a head with kcal/mol and angstrom outputs")
    atomic = np.load(REPO / args.atomic_energies, allow_pickle=True).item()
    atomic = {int(z): float(e) for z, e in atomic.items()}

    calc = load_densnet_calculator(run, np_dataset=str(REPO / args.np_dataset), repo=REPO,
                                   use_gpu=opt.device == "cuda")
    model = calc.model
    model.eval()
    model.no_force_props = [p for p in model.no_force_props if p == "energy"]
    device = "cuda" if calc.use_gpu else "cpu"

    def predict(Z, P):
        Z = np.asarray(Z, dtype=np.int64)
        inputs = geometry_to_model_inputs(Z, torch.tensor(np.asarray(P), dtype=torch.float32, device=device),
                                          calc.cutoff, calc.use_gpu)
        if calc.remove_atom_density and calc.atom_dens is not None and calc.atom_dens_type in {"mo_coeffs", "df_coeffs"}:
            key = "atom_" + calc.atom_dens_type
            inputs[key], inputs[key + "_occ"], inputs[key + "_basis"] = orbitals.join_atom_coeffs(
                torch.as_tensor(Z, dtype=torch.long), calc.atom_dens, calc.atom_dens_type)
            for k in (key, key + "_occ", key + "_basis"):
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(device)
        out = model(inputs)
        energy = out["energy"].detach().cpu().numpy().reshape(-1)
        forces = out["forces"].detach().cpu().numpy().reshape(-1, 3)
        counts = (Z > 0).sum(1)
        return energy, np.split(forces, np.cumsum(counts)[:-1])

    opt.out.mkdir(parents=True, exist_ok=True)
    rows, checks, t0 = [], [], time.time()
    files = {split: REPO / f"datasets/revision/water_labels/water_{split}_{opt.theory}_base.npy" for split in opt.splits}
    files.update({spec.split("=", 1)[0]: REPO / spec.split("=", 1)[1] for spec in opt.label_file})
    for split, path in files.items():
        data = np.load(path, allow_pickle=True).item()
        count = len(data["energy"]) if opt.limit is None else min(opt.limit, len(data["energy"]))
        sizes = (data["atom_numbers"][:count] > 0).sum(1)
        batches, current = [], []
        for i in range(count):
            if current and (len(current) == opt.batch or sizes[current].sum() + sizes[i] > opt.max_atoms):
                batches.append(current)
                current = []
            current.append(i)
        batches.append(current)
        for idx in batches:
            Z = data["atom_numbers"][idx]
            width = int((Z > 0).sum(1).max())
            e_pred, f_pred = predict(Z[:, :width], data["positions"][idx][:, :width])
            for k, i in enumerate(idx):
                z = data["atom_numbers"][i]
                keep = z > 0
                e_ref = to_kcal(float(np.asarray(data["energy"][i]).squeeze()) - sum(atomic[int(a)] for a in z[keep]))
                f_ref = to_kcal_per_A(np.asarray(data["forces"][i], float)[keep])
                df = f_pred[k] - f_ref
                n_water = int((z == 8).sum())
                if int((z == 1).sum()) != 2 * n_water or keep.sum() != 3 * n_water:
                    raise ValueError(f"{split} frame {i} is not a water cluster")
                rows.append({"split": split, "index": i,
                             "source_index": int(data["source_index"][i]) if "source_index" in data else i,
                             "n_water": n_water, "n_atoms": int(keep.sum()),
                             "energy_pred_kcal_mol": float(e_pred[k]), "energy_ref_kcal_mol": e_ref,
                             "energy_error_kcal_mol": float(e_pred[k]) - e_ref,
                             "force_mae_kcal_mol_A": float(np.abs(df).mean()),
                             "force_abs_sum": float(np.abs(df).sum()), "force_sq_sum": float((df ** 2).sum())})
        for i in range(min(opt.check_single, count)):
            z = data["atom_numbers"][i]
            keep = z > 0
            e1, f1 = predict(z[keep][None], data["positions"][i][keep][None])
            batched = next(r for r in rows if r["split"] == split and r["index"] == i)
            e_abs = abs(to_kcal(float(np.asarray(data["energy"][i]).squeeze())))
            checks.append({"split": split, "index": i, "n_atoms": int(keep.sum()),
                           "energy_single_minus_batched": float(e1[0] - batched["energy_pred_kcal_mol"]),
                           "abs_energy_kcal_mol": e_abs,
                           "tol_kcal_mol": max(opt.single_tol, opt.single_ulp_factor * e_abs * 2.0 ** -23)})
        print(f"{split}: {count} frames, {time.time() - t0:.0f} s", flush=True)

    worst = max((abs(c["energy_single_minus_batched"]) for c in checks), default=0.0)
    (opt.out / "single_structure_check.json").write_text(json.dumps(checks, indent=1) + "\n")
    with open(opt.out / "per_frame.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    violations = [c for c in checks if abs(c["energy_single_minus_batched"]) > c["tol_kcal_mol"]]
    if violations:
        c = max(violations, key=lambda c: abs(c["energy_single_minus_batched"]) / c["tol_kcal_mol"])
        raise RuntimeError(f"batched and single-structure energies differ by {abs(c['energy_single_minus_batched']):.3g} "
                           f"kcal/mol on {c['split']} frame {c['index']} ({c['n_atoms']} atoms), above the tolerance "
                           f"{c['tol_kcal_mol']:.3g} ({len(violations)} of {len(checks)} checks)")
    summary = summarize(rows)
    summary["provenance"] = {
        "run": str(run), "checkpoint": str(best), "checkpoint_sha256_first_MiB": sha256_head(best),
        "theory": opt.theory, "atomic_energies": str(args.atomic_energies),
        "integral_constraint": getattr(args, "integral_constraint", None),
        "positive_coeffs": getattr(args, "positive_coeffs", None),
        "init_from": getattr(args, "init_from", None),
        "batch": opt.batch, "limit": opt.limit, "single_structure_check": checks, "single_tol_kcal_mol": opt.single_tol,
        "single_ulp_factor": opt.single_ulp_factor,
        "worst_single_minus_batched_kcal_mol": worst,
        "worst_single_minus_batched_in_tolerance_units": max((abs(c["energy_single_minus_batched"]) / c["tol_kcal_mol"] for c in checks), default=0.0),
        "units": "energy kcal/mol per cluster (baseline removed); forces kcal/mol/A per Cartesian component",
    }
    (opt.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for split, m in summary["by_split"].items():
        print(f"{split:12s} n={m['n']:4d}  E MAE {m['energy_mae_kcal_mol']:8.3f}  per water "
              f"{m['energy_mae_per_water_kcal_mol']:6.3f}  E MAE bias-removed {m['energy_mae_bias_removed_kcal_mol']:8.3f}  "
              f"size-centered {m['energy_mae_size_centered_kcal_mol']:8.3f}  F MAE {m['force_mae_kcal_mol_A']:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
