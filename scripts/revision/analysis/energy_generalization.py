#!/usr/bin/env python3
"""Evaluate a frozen energy head against disjoint JACS reference sets.

Offsets come exclusively from the training configuration and training indices.
No test-set centering, refitting, or frame selection is performed. Each output
contains the exact model, label hashes, per-frame errors, and counts per bin.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE / "src"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def baseline_spec(args, train, train_indices, repo, system):
    """Reconstruct the exact fitted reference, never from evaluation labels."""
    if not args.center_energy:
        return {"kind": "none"}
    if args.atomic_energies:
        path = Path(args.atomic_energies)
        if not path.is_absolute():
            path = repo / path
        values = np.load(path, allow_pickle=True).item()
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Non-finite atomic energy reference")
        return {"kind": "composition", "values_hartree": {int(k): float(v) for k, v in values.items()},
                "path": str(path), "sha256": digest(path)}
    if system == "water":
        raise ValueError("A single energy mean is unsuitable for mixed-size water evaluation")
    indices = np.asarray(train_indices, dtype=int)
    if indices.size == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("Missing or repeated training indices")
    if indices.min() < 0 or indices.max() >= len(train["energy"]):
        raise ValueError("Training indices do not match the recorded dataset")
    mean = float(np.asarray(train["energy"], dtype=float)[indices].mean())
    if not np.isfinite(mean):
        raise ValueError("Non-finite training energy mean")
    return {"kind": "training_mean", "mean_hartree": mean, "train_indices": indices.tolist()}


def reference_offset(spec, numbers):
    if spec["kind"] == "composition":
        return sum(spec["values_hartree"][int(z)] for z in numbers if z > 0)
    if spec["kind"] == "training_mean":
        return spec["mean_hartree"]
    return 0.0


def metrics(rows):
    error = np.array([r["error_kcal_mol"] for r in rows])
    out = {"n": len(rows), "mae_kcal_mol": float(np.abs(error).mean()),
           "rmse_kcal_mol": float(np.sqrt(np.mean(error ** 2))),
           "bias_kcal_mol": float(error.mean()), "max_abs_kcal_mol": float(np.abs(error).max())}
    if "n_water" in rows[0]:
        per = error / np.array([r["n_water"] for r in rows])
        out.update(mae_per_water_kcal_mol=float(np.abs(per).mean()),
                   rmse_per_water_kcal_mol=float(np.sqrt(np.mean(per ** 2))))
    return out


def summarize(rows, system):
    summary = {"by_split": {}, "by_coordinate": []}
    for split in dict.fromkeys(r["split"] for r in rows):
        group = [r for r in rows if r["split"] == split]
        summary["by_split"][split] = metrics(group)
        keys = sorted(set(r["n_water"] if system == "water" else
                          int(np.floor(r["delta_pt_A"] / 0.2)) for r in group))
        for key in keys:
            selected = [r for r in group if (r["n_water"] if system == "water" else
                        int(np.floor(r["delta_pt_A"] / 0.2))) == key]
            center = key if system == "water" else (key + .5) * .2
            summary["by_coordinate"].append({"split": split, "coordinate": center, **metrics(selected)})
    return summary


def plot_results(rows, summary, system, theory, out, train_delta_range=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for split in summary["by_split"]:
        groups = [r for r in summary["by_coordinate"] if r["split"] == split]
        axes[0].plot([r["coordinate"] for r in groups], [r["mae_kcal_mol"] for r in groups],
                     "o-", label=split)
        if system == "water":
            axes[1].plot([r["coordinate"] for r in groups], [r["mae_per_water_kcal_mol"] for r in groups], "o-", label=split)
        else:
            group = [r for r in rows if r["split"] == split]
            axes[1].scatter([r["delta_pt_A"] for r in group], [r["error_kcal_mol"] for r in group], s=12, alpha=.5, label=split)
    for ax in axes:
        ax.set_xlabel("Water molecules n" if system == "water" else r"$\delta=r(O_d-H)-r(H-O_a)$ / Å")
        ax.grid(alpha=.2)
        if train_delta_range:
            ax.axvspan(*train_delta_range, alpha=.1, color="gray", label="training coordinate range")
    axes[0].set_ylabel("Energy MAE / kcal mol⁻¹ (cluster)")
    axes[1].set_ylabel("Energy MAE / kcal mol⁻¹ per water" if system == "water" else "Signed energy error / kcal mol⁻¹")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{system}: {theory} — fixed head, training-defined reference")
    fig.savefig(out / "energy_generalization.png", dpi=200)
    fig.savefig(out / "energy_generalization.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--system", choices=["water", "malonaldehyde"], required=True)
    parser.add_argument("--theory", choices=["pbe_d4_avdz", "wb97mv_def2tzvpd"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-frames", type=int, help="Smoke test only; output explicitly marked partial")
    parser.add_argument("--dataset", action="append", default=[], metavar="LABEL=PATH",
                        help="Override default sets, e.g. an assembled expanded OOD tier")
    opt = parser.parse_args()
    repo, run, out = opt.repo.resolve(), opt.run.resolve(), opt.out.resolve()
    os.chdir(repo)
    import torch
    from ase import Atoms, units
    from equiv_dens.md.dft_network_calculator import load_densnet_calculator

    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(f"Completed evaluation exists: {out}; select a new output directory")
    checkpoint_path = run / "checkpoints/latest_checkpoint.pth"
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = metadata["args"]
    if args.energy_unit_in != "hartree" or not str(args.energy_unit_out).startswith("kcal"):
        raise ValueError("This evaluator requires Hartree labels and kcal/mol model outputs")
    best = run / ("best_" + metadata["ID"] + ".pth")
    if not best.exists():
        raise FileNotFoundError(f"Validation-selected model missing: {best}")
    frozen = out / "frozen_model"
    (frozen / "checkpoints").mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source, dest in [(best, frozen / best.name), (checkpoint_path, frozen / "checkpoints/latest_checkpoint.pth"),
                         (run / "args.txt", frozen / "args.txt")]:
        before = digest(source)
        shutil.copy2(source, dest)
        if digest(dest) != before or digest(source) != before:
            raise RuntimeError("Model changed while being frozen; evaluate only a stopped training run")
        hashes[str(source)] = before
    train_path = repo / args.np_dataset
    train = np.load(train_path, allow_pickle=True).item()
    baseline = baseline_spec(args, train, metadata["data_split_indices"]["train"], repo, opt.system)
    kcal_ev = units.kcal / units.mol
    calculator = load_densnet_calculator(frozen, np_dataset=str(train_path), repo=repo,
                                       use_gpu=opt.device == "cuda")
    # The density/dipole heads have no effect on energy or its derivatives.
    # Skip their integrals during an energy-only study, retaining all weights.
    calculator.model.no_force_props = [p for p in calculator.model.no_force_props if p == "energy"]
    indices = None
    train_delta_range = None
    if opt.system == "malonaldehyde":
        manifest = json.loads((repo / "datasets/revision/malonaldehyde/manifest.json").read_text())
        indices = manifest["indices"]
        train_delta_range = [manifest["splits"]["train"]["delta_min"], manifest["splits"]["train"]["delta_max"]]
    splits = ["val", "id_test", "ood_size", "ood_order", "ood_density"] if opt.system == "water" else ["val", "ood_proton_transfer"]
    sources = {split: repo / f"datasets/revision/{opt.system}_labels/{opt.system}_{split}_{opt.theory}_base.npy" for split in splits}
    if opt.dataset:
        sources = {item.split("=", 1)[0]: Path(item.split("=", 1)[1]).resolve() for item in opt.dataset}
    rows, data_hashes = [], {}
    provenance = {"system": opt.system, "theory": opt.theory, "source_run": str(run), "model_files_sha256": hashes,
                  "training_dataset": str(train_path), "training_sha256": digest(train_path),
                  "baseline": baseline, "model_selection": "validation best", "partial_smoke_test": opt.max_frames is not None,
                  "units": "energy: kcal/mol; positions: A", "coordinate_indices": indices,
                  "uncertainty": "single fitted model; coordinate bins are descriptive, not independent replicates"}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for split, path in sources.items():
        before = digest(path)
        data = np.load(path, allow_pickle=True).item()
        count = len(data["energy"])
        count = min(count, opt.max_frames) if opt.max_frames is not None else count
        if count <= 0:
            raise ValueError(f"Empty evaluation set: {path}")
        for i in range(count):
            numbers = np.asarray(data["atom_numbers"][i], dtype=int)
            keep = numbers > 0
            z = numbers[keep]
            xyz = np.asarray(data["positions"][i], dtype=float)[keep]
            atoms = Atoms(z, positions=xyz)
            atoms.calc = calculator
            prediction = atoms.get_potential_energy() / kcal_ev
            reference = (float(np.asarray(data["energy"][i]).squeeze()) - reference_offset(baseline, z)) * units.Hartree / kcal_ev
            if not np.isfinite([prediction, reference]).all():
                raise ValueError(f"Non-finite energy in {split} frame {i}; no frames silently dropped")
            row = {"split": split, "index": i, "n_atoms": len(z), "prediction_kcal_mol": prediction,
                   "reference_kcal_mol": reference, "error_kcal_mol": prediction - reference}
            if opt.system == "water":
                n = int(np.sum(z == 8))
                if n == 0 or np.sum(z == 1) != 2 * n or len(z) != 3 * n:
                    raise ValueError("Non-water geometry in water evaluation")
                row["n_water"] = n
                row["error_per_water_kcal_mol"] = row["error_kcal_mol"] / n
            else:
                donor, hydrogen, acceptor = (indices[k] for k in ("donor", "hydrogen", "acceptor"))
                if [int(z[k]) for k in (donor, hydrogen, acceptor)] != [8, 1, 8]:
                    raise ValueError("Proton-transfer atom indices do not match labels")
                row["delta_pt_A"] = float(np.linalg.norm(xyz[donor] - xyz[hydrogen]) - np.linalg.norm(xyz[acceptor] - xyz[hydrogen]))
            rows.append(row)
        if digest(path) != before:
            raise RuntimeError(f"Labels changed during evaluation: {path}")
        data_hashes[split] = {"path": str(path), "sha256": before, "evaluated": count, "available": len(data["energy"])}
        print(split, metrics([r for r in rows if r["split"] == split]), flush=True)
    with (out / "per_frame.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, opt.system)
    plot_results(rows, summary, opt.system, opt.theory, out, train_delta_range)
    summary.update(provenance, datasets=data_hashes)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
