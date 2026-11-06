#!/usr/bin/env python3
"""Measured energy and force MAE of the paper's DenSNet models (Table S2, Fig. 4e).

Table S2's E/F columns were typed in by hand and cannot be traced to any
evaluation. This scores a checkpoint on its held-out test set, one structure
at a time as in MD, so neither batching nor the pre-2026-09-28 layer-norm
pooling can enter.

Energies are compared the way the model was trained: labels minus the mean
energy of the checkpoint's own training subset (--train-npy with the stored
training indices). The MAE after also removing the test-set mean offset is
reported alongside, labelled as such.

    .venv/bin/python -B experiments/jacs_revision/analysis/score_paper_energy_forces.py \
        --run paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4 \
        --test-npy datasets/ethanethiol_rand-1000.npy \
        --train-npy datasets/ethanethiol_combo_Aidx-1000.npy \
        --out results/revision/paper_energy_forces/ethanethiol_delta
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))


def load_npy(path):
    data = np.load(path, allow_pickle=True)
    data = data.item() if data.dtype == object and data.shape == () else data
    n = len(data["energy"])
    numbers = np.asarray(data["atom_numbers"])
    if numbers.ndim == 1:
        numbers = np.repeat(numbers[None], n, axis=0)
    return numbers, np.asarray(data["positions"], float), np.asarray(data["energy"], float).reshape(n), \
        np.asarray(data["forces"], float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--test-npy", type=Path, required=True)
    ap.add_argument("--train-npy", type=Path,
                    help="the run's training file; its training-subset mean energy is the reference")
    ap.add_argument("--args-file", type=Path,
                    help="MD override config for 2024 checkpoints (config/md/nn/<molecule>.txt)")
    ap.add_argument("--timing-system",
                    help="build the calculator as benchmark_step_timing.densnet_calculator does for this system "
                         "(needed for spline-prior checkpoints such as resorcinol)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--force-scale", type=float,
                    help="multiply predicted energies and forces (checkpoints saved without scaling.std)")
    opt = ap.parse_args()

    import torch
    from ase import Atoms, units
    from equiv_dens.md.dft_network_calculator import load_densnet_calculator
    import equiv_dens.utils as utils

    run = opt.run.resolve()
    meta = torch.load(run / "checkpoints/latest_checkpoint.pth", map_location="cpu", weights_only=False)
    args = meta["args"]
    to_kcal = getattr(utils, args.energy_unit_in + "_to_kcal")
    per_A = 1.0 / getattr(utils, args.distance_unit_in + "_to_angstrom")(1.0)
    if getattr(args, "atomic_energies", None):
        raise NotImplementedError("composition baselines are handled by score_energy_forces.py")

    train_mean = None
    if opt.train_npy is not None:
        _, _, e_train, _ = load_npy(opt.train_npy)
        idx = np.asarray(meta["data_split_indices"]["train"], int)
        train_mean = float(e_train[idx].mean())

    if opt.timing_system:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "benchmark_step_timing", REPO / "experiments/jacs_revision/analysis/benchmark_step_timing.py")
        timing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(timing)
        if timing.REPO / timing.DENSNET[opt.timing_system][0] != run:
            raise ValueError(f"--timing-system {opt.timing_system} does not use {run}")
        calc, _ = timing.densnet_calculator(opt.timing_system, opt.device)
    else:
        calc = load_densnet_calculator(run, args_file=str(opt.args_file) if opt.args_file else None,
                                       np_dataset=str(opt.train_npy or opt.test_npy), repo=REPO,
                                       use_gpu=opt.device == "cuda", pyscf_grid=False)
    model = calc.model
    model.no_force_props = [p for p in model.no_force_props if p == "energy"]
    scale = opt.force_scale or 1.0
    kcal = units.kcal / units.mol

    Z, R, E, F = load_npy(opt.test_npy)
    n = len(E) if opt.limit is None else min(opt.limit, len(E))
    e_pred, e_ref, f_abs, f_sq, n_comp = [], [], 0.0, 0.0, 0
    t0 = time.time()
    for i in range(n):
        keep = Z[i] > 0
        atoms = Atoms(Z[i][keep], positions=R[i][keep])
        atoms.calc = calc
        e_pred.append(atoms.get_potential_energy() / kcal * scale)
        df = atoms.get_forces() / kcal * scale - to_kcal(F[i][keep]) * per_A
        f_abs += float(np.abs(df).sum())
        f_sq += float((df ** 2).sum())
        n_comp += df.size
        e_ref.append(to_kcal(E[i] - train_mean) if train_mean is not None else np.nan)
    e_pred, e_ref = np.array(e_pred), np.array(e_ref)
    err = e_pred - e_ref
    result = {
        "run": str(run), "test_npy": str(opt.test_npy), "train_npy": str(opt.train_npy), "n": n,
        "checkpoint_sha256_first_MiB": hashlib.sha256((run / f"best_{meta['ID']}.pth").read_bytes()[:1 << 20]).hexdigest(),
        "normalize_en": getattr(args, "normalize_en", None), "remove_atom_density": args.remove_atom_density,
        "force_scale_applied": opt.force_scale,
        "force_mae_kcal_mol_A": f_abs / n_comp, "force_rmse_kcal_mol_A": float(np.sqrt(f_sq / n_comp)),
        "energy_mae_kcal_mol": float(np.nanmean(np.abs(err))) if train_mean is not None else None,
        "energy_mean_offset_kcal_mol": float(np.nanmean(err)) if train_mean is not None else None,
        "energy_mae_offset_removed_kcal_mol": float(np.mean(np.abs(
            (e_pred - e_pred.mean()) - (to_kcal(E[:n]) - to_kcal(E[:n]).mean())))),
        "reference": "training-subset mean energy" if train_mean is not None else "none (offset-removed MAE only)",
        "seconds": round(time.time() - t0, 1),
    }
    opt.out.mkdir(parents=True, exist_ok=True)
    np.savez(opt.out / "per_frame.npz", energy_pred_kcal_mol=e_pred, energy_ref_kcal_mol=e_ref,
             energy_label_hartree=E[:n])
    (opt.out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("n", "energy_mae_kcal_mol", "energy_mean_offset_kcal_mol",
                                             "energy_mae_offset_removed_kcal_mol", "force_mae_kcal_mol_A")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
