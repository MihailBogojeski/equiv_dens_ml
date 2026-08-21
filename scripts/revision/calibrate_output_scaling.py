#!/usr/bin/env python3
"""Recover a checkpoint's output scaling from in-distribution frames.

A DenSNet model's energies and forces are expressed in units of the standard
deviation of its training-set forces. Checkpoints written before that constant
became a buffer do not carry it, and it can normally be recomputed from the
training set -- except that the ethanol training set the published model was
fitted on, ``datasets/ethanol_dft_pyscf_ccpvdz_train.npy``, is not on disk.

Without the constant every prediction is off by the same factor, which is not
visible in any single number: the model still produces smooth, finite forces
that simply do not respond to geometry. It shows up only when predictions are
put beside references across a range of geometries.

The constant is one scalar, so one scalar's worth of in-distribution data
recovers it. This labels the frames the model was trained on at the same level
of theory it was trained against, then takes the least-squares scale

    s = <F_pred . F_ref> / <F_pred . F_pred>

Calibrating on training geometries and applying the result to the
out-of-distribution set keeps the OOD numbers honest: nothing about the OOD
frames enters the fit. The residual in-distribution force error after scaling is
reported too, and it is the thing to check before trusting ``s`` -- a model that
had genuinely lost its energy surface would not be brought into agreement by any
single factor, and a large residual here says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.units import Hartree

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "md"))


def dft_forces(atoms, xc, basis):
    """PBE forces in eV/A for one structure."""
    from pyscf import dft, gto

    mol = gto.Mole()
    mol.atom = [(s, tuple(p)) for s, p in
                zip(atoms.get_chemical_symbols(), atoms.get_positions())]
    mol.basis = basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.kernel()
    grad = mf.nuc_grad_method().kernel()
    # pyscf gradients are dE/dR in Hartree/Bohr; forces are the negative, and
    # the Bohr-to-Angstrom factor has to come with them.
    return -grad * Hartree / 0.529177210903, float(mf.e_tot)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures", type=Path,
                    default=Path("datasets/ethanol_train_10.xyz"),
                    help="in-distribution geometries; training frames, not OOD ones")
    ap.add_argument("--model", default="paper/models/ethanol/2024-03-22_96w7KyGG")
    ap.add_argument("--args-file", default="config/md/nn/ethanol_500ps.txt")
    ap.add_argument("--xc", default="pbe")
    ap.add_argument("--basis", default="augccpvdz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("results/revision/output_scaling_ethanol.json"))
    cli = ap.parse_args()

    from equiv_dens.md.dft_network_calculator import load_densnet_calculator

    frames = read(str(cli.structures), index=":")
    if cli.limit:
        frames = frames[: cli.limit]
    print(f"calibrating on {len(frames)} in-distribution frames from {cli.structures}")

    calc = load_densnet_calculator(cli.model, args_file=cli.args_file, use_gpu=cli.use_gpu)
    scale_before = None
    for module in getattr(calc, "model", calc).modules():
        if type(module).__name__ == "VarianceScaling":
            scale_before = float(module.std)
            break
    print(f"scaling currently stored in the model: {scale_before}")

    pred, ref, rows = [], [], []
    for i, atoms in enumerate(frames):
        a = atoms.copy()
        a.calc = calc
        fp = np.asarray(a.get_forces(), dtype=float)
        fr, e_dft = dft_forces(atoms, cli.xc, cli.basis)
        pred.append(fp)
        ref.append(fr)
        rows.append({
            "frame": i,
            "pred_mean_abs": float(np.abs(fp).mean()),
            "ref_mean_abs": float(np.abs(fr).mean()),
            "dft_energy_hartree": e_dft,
        })
        print(f"  [{i + 1}/{len(frames)}] |F_pred|={rows[-1]['pred_mean_abs']:.5f} "
              f"|F_ref|={rows[-1]['ref_mean_abs']:.5f}", flush=True)

    P = np.concatenate([p.ravel() for p in pred])
    R = np.concatenate([r.ravel() for r in ref])
    denom = float(np.dot(P, P))
    if denom <= 0:
        raise SystemExit("predicted forces are identically zero; nothing to calibrate")
    s = float(np.dot(P, R) / denom)

    resid = s * P - R
    summary = {
        "structures": str(cli.structures),
        "model": cli.model,
        "n_frames": len(frames),
        "xc": cli.xc,
        "basis": cli.basis,
        "scaling_in_checkpoint": scale_before,
        "fitted_output_scaling": s,
        "in_distribution_force_mae_after_scaling_eV_A": float(np.mean(np.abs(resid))),
        "in_distribution_force_rmse_after_scaling_eV_A": float(np.sqrt(np.mean(resid ** 2))),
        "reference_force_mean_abs_eV_A": float(np.mean(np.abs(R))),
        "correlation_pred_ref": float(
            np.corrcoef(P, R)[0, 1]
        ),
        "per_frame": rows,
        "note": (
            "fitted_output_scaling multiplies the model's raw forces and energies to "
            "put them in eV and eV/A. It is fitted only on the training geometries, so "
            "applying it to the OOD set leaves those numbers independent of the fit. "
            "correlation_pred_ref is the check that matters: a scale factor can only "
            "rescale, so if the model had lost its energy surface the correlation "
            "would be near zero and no factor would help."
        ),
    }
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(summary, indent=2) + "\n")
    brief = {k: v for k, v in summary.items() if k != "per_frame"}
    print()
    print(json.dumps(brief, indent=2))
    print(f"\nwrote {cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
