#!/usr/bin/env python3
"""Per-frame energy and force errors of a malonaldehyde energy head (in-training snapshot or final).

Counterpart of score_energy_forces.py for the malonaldehyde tiers (train sample, validation,
OOD proton transfer). Writes per_frame.csv with, for every frame: predicted and reference energy,
force MAE over all atoms and per element, the proton-transfer coordinate
q = d(O_a-H*) - d(O_b-H*) (H* = the hydrogen closest to an oxygen) and the O...O distance.

The head is trained without atomic reference energies, so the absolute offset between prediction
and label is a constant of the run. It is measured on the training sample and removed from every
split; the remaining per-split bias is reported, not hidden.

    .venv/bin/python -B experiments/jacs_revision/analysis/score_malon_energy.py \
        --run results/revision/malon_energy_v2/malonaldehyde_pbe_d4_avdz/energy/<run> --theory pbe_d4_avdz --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
SPLITS = {"train": "train", "val": "val", "ood_proton_transfer": "ood_proton_transfer"}
SYM = {1: "H", 6: "C", 8: "O"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--theory", choices=["pbe_d4_avdz", "wb97mv_def2tzvpd"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest"])
    ap.add_argument("--limit", type=int, default=125, help="frames per split")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--label-file", action="append", default=[], metavar="NAME=PATH",
                    help="score an extra malonaldehyde label file after the standard splits (same train-sample offset)")
    opt = ap.parse_args()

    import torch
    from equiv_dens.md.dft_network_calculator import geometry_to_model_inputs, load_densnet_calculator
    from equiv_dens.utils import orbitals
    import equiv_dens.utils as utils

    run = opt.run.resolve()
    meta = torch.load(run / "checkpoints/latest_checkpoint.pth", map_location="cpu", weights_only=False)
    args = meta["args"]
    to_kcal = getattr(utils, args.energy_unit_in + "_to_kcal")
    to_kcal_per_A = lambda f: to_kcal(f) / getattr(utils, args.distance_unit_in + "_to_angstrom")(1.0)
    to_A = getattr(utils, args.distance_unit_in + "_to_angstrom")
    calc = load_densnet_calculator(run, np_dataset=str(REPO / args.np_dataset), repo=REPO, use_gpu=opt.device == "cuda")
    model = calc.model
    if opt.checkpoint == "latest":  # the calculator loads best_<ID>.pth; overwrite with the latest weights
        model.load_state_dict(meta["model_state_dict"], strict=False)
    model.eval()
    model.no_force_props = [p for p in model.no_force_props if p == "energy"]
    device = "cuda" if calc.use_gpu else "cpu"

    def predict(Z, P):
        Z = np.asarray(Z, dtype=np.int64)
        inputs = geometry_to_model_inputs(Z, torch.tensor(np.asarray(P), dtype=torch.float32, device=device), calc.cutoff, calc.use_gpu)
        if calc.remove_atom_density and calc.atom_dens is not None and calc.atom_dens_type in {"mo_coeffs", "df_coeffs"}:
            key = "atom_" + calc.atom_dens_type
            inputs[key], inputs[key + "_occ"], inputs[key + "_basis"] = orbitals.join_atom_coeffs(
                torch.as_tensor(Z, dtype=torch.long), calc.atom_dens, calc.atom_dens_type)
            for k in (key, key + "_occ", key + "_basis"):
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(device)
        out = model(inputs)
        return out["energy"].detach().cpu().numpy().reshape(-1), out["forces"].detach().cpu().numpy().reshape(len(Z), -1, 3)

    opt.out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    files = {split: REPO / f"datasets/revision/malonaldehyde_labels/malonaldehyde_{name}_{opt.theory}_base.npy" for split, name in SPLITS.items()}
    files.update({spec.split("=", 1)[0]: REPO / spec.split("=", 1)[1] for spec in opt.label_file})
    for split, path in files.items():
        data = np.load(path, allow_pickle=True).item()
        n = len(data["energy"]) if split not in SPLITS else min(opt.limit, len(data["energy"]))
        for s in range(0, n, opt.batch):
            idx = list(range(s, min(n, s + opt.batch)))
            Z, P = data["atom_numbers"][idx], data["positions"][idx]
            e_pred, f_pred = predict(Z, P)
            for k, i in enumerate(idx):
                z, pos = Z[k], to_A(np.asarray(P[k], float))
                e_ref = to_kcal(float(np.asarray(data["energy"][i]).squeeze()))
                df = f_pred[k] - to_kcal_per_A(np.asarray(data["forces"][i], float))
                ox, hy = np.flatnonzero(z == 8), np.flatnonzero(z == 1)
                dOH = np.linalg.norm(pos[hy][:, None] - pos[ox][None], axis=2)  # (nH, 2)
                h = int(np.argmin(dOH.min(1)))
                row = {"split": split, "index": i, "source_index": int(data["source_index"][i]) if "source_index" in data else i,
                       "energy_pred_raw_kcal_mol": float(e_pred[k]), "energy_ref_abs_kcal_mol": e_ref,
                       "force_mae_kcal_mol_A": float(np.abs(df).mean()), "force_max_kcal_mol_A": float(np.abs(df).max()),
                       "q_pt_A": float(dOH[h, 0] - dOH[h, 1]), "d_OO_A": float(np.linalg.norm(pos[ox[0]] - pos[ox[1]])),
                       "f_mae_transfer_H": float(np.abs(df[hy[h]]).mean())}
                for zz, sym in SYM.items():
                    row[f"f_mae_{sym}"] = float(np.abs(df[z == zz]).mean())
                rows.append(row)
        print(f"{split}: {n} frames, {time.time() - t0:.0f} s", flush=True)

    # constant offset of the run, from the training sample only
    off = float(np.median([r["energy_pred_raw_kcal_mol"] - r["energy_ref_abs_kcal_mol"] for r in rows if r["split"] == "train"]))
    e0 = float(np.mean([r["energy_ref_abs_kcal_mol"] for r in rows if r["split"] == "train"]))
    for r in rows:
        r["energy_error_kcal_mol"] = r["energy_pred_raw_kcal_mol"] - r["energy_ref_abs_kcal_mol"] - off
        r["energy_ref_rel_kcal_mol"] = r["energy_ref_abs_kcal_mol"] - e0
        r["energy_pred_rel_kcal_mol"] = r["energy_ref_rel_kcal_mol"] + r["energy_error_kcal_mol"]
    with open(opt.out / "per_frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"by_split": {}, "offset_train_median_kcal_mol": off,
               "provenance": {"run": str(run), "checkpoint": opt.checkpoint, "step": int(meta["step"]), "theory": opt.theory, "limit": opt.limit,
                              "units": "energy kcal/mol (train-sample offset removed); forces kcal/mol/A per Cartesian component"}}
    for split in files:
        g = [r for r in rows if r["split"] == split]
        err = np.array([r["energy_error_kcal_mol"] for r in g])
        summary["by_split"][split] = {"n": len(g), "energy_mae_kcal_mol": float(np.abs(err).mean()), "energy_rmse_kcal_mol": float(np.sqrt((err ** 2).mean())),
                                      "energy_bias_kcal_mol": float(err.mean()), "force_mae_kcal_mol_A": float(np.mean([r["force_mae_kcal_mol_A"] for r in g])),
                                      **{f"force_mae_{s}": float(np.mean([r[f"f_mae_{s}"] for r in g])) for s in SYM.values()}}
        m = summary["by_split"][split]
        print(f"{split:20s} n={m['n']:4d}  E MAE {m['energy_mae_kcal_mol']:7.3f}  bias {m['energy_bias_kcal_mol']:+7.3f}  F MAE {m['force_mae_kcal_mol_A']:6.3f}")
    (opt.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
