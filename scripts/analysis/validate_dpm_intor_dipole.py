#!/usr/bin/env python3
"""
Validate dipole accuracy when using analytic dipole path (dpm_intor).

Compares dipole moments from grid-based (DensityExpansion + DipoleMomentCalc)
vs analytic integral (DipoleMomentIntorCalc) on a few frames from the dataset.
Ensures dpm_intor produces numerically consistent results for IR spectrum quality.

Usage:
    python scripts/analysis/validate_dpm_intor_dipole.py
    python scripts/analysis/validate_dpm_intor_dipole.py --n-frames 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import equiv_dens.compat  # noqa: F401
from analysis.load_density_model import load_density_model
from analysis.benchmark_md_inference import prepare_model_input


def main():
    parser = argparse.ArgumentParser(description="Validate dpm_intor dipole accuracy")
    parser.add_argument("--n-frames", type=int, default=5, help="Number of frames to compare")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tol", type=float, default=0.15,
                        help="Max allowed relative |dpm_intor - grid| / |grid| per component")
    cli = parser.parse_args()

    print("=" * 60)
    print("Dipole validation: grid-based vs analytic (dpm_intor)")
    print("=" * 60)

    device = cli.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    # Load dataset once, then build both models
    from equiv_dens.training.model_loader import load_model

    print("\nLoading grid-based dipole model (density_weight=1, dipole_moment_weight=1, dpm_intor=False)...")
    _, dataset, args_base = load_density_model("ethanethiol", device=device, verbose=False)
    args_grid = argparse.Namespace(**vars(args_base))
    args_grid.dipole_moment_weight = 1.0
    args_grid.density_weight = 1.0
    args_grid.dpm_intor = False
    model_grid = load_model(args_grid, dataset, train=False)
    model_grid.eval()
    if device == "cuda":
        model_grid.cuda()

    print("Loading analytic dipole model (dpm_intor)...")
    model_intor, _, args_intor = load_density_model(
        "ethanethiol", device=device, dpm_intor=True, verbose=False
    )
    if device == "cuda":
        model_intor.cuda()

    n_frames = min(cli.n_frames, len(dataset))
    print(f"\nComparing dipoles on {n_frames} frames...")

    diffs = []
    for i in range(n_frames):
        inp_grid = prepare_model_input(dataset, model_grid, batch_idx=i, device=device, args=args_grid)
        inp_intor = prepare_model_input(dataset, model_intor, batch_idx=i, device=device, args=args_intor)

        for k, v in inp_grid.items():
            if isinstance(v, torch.Tensor) and k not in inp_intor:
                inp_intor[k] = v.to(device) if hasattr(v, "to") else v

        with torch.no_grad():
            model_grid.eval()
            model_intor.eval()
            out_grid = model_grid(inp_grid)
            out_intor = model_intor(inp_intor)

        d_grid = out_grid["dipole_moment"].detach().cpu().numpy().flatten()
        d_intor = out_intor["dipole_moment"].detach().cpu().numpy().flatten()

        diff = np.abs(d_intor - d_grid)
        denom = np.abs(d_grid) + 1e-8
        rel_err = diff / denom
        diffs.append(rel_err)
        max_rel = np.max(rel_err)
        status = "OK" if max_rel <= cli.tol else "LARGE"
        print(f"  Frame {i}: max_rel_err={max_rel:.4f} [{status}]")

    diffs = np.array(diffs)
    mean_max = np.mean(np.max(diffs, axis=1))
    worst = np.max(diffs)
    print(f"\nMean max relative error: {mean_max:.4f}")
    print(f"Worst component: {worst:.4f}")
    print(f"Tolerance: {cli.tol}")

    if worst > cli.tol:
        print(f"\nWARNING: Some frames exceed tolerance {cli.tol}. Review numerical settings.")
        return 1

    print("\nValidation PASSED: dpm_intor dipoles within tolerance of grid-based.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
