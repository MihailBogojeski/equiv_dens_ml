#!/usr/bin/env python3
"""Relative density errors and SAD-correction magnitudes (R2.8, R3.3, R3.4).

Works from a DF density-fitting dataset (the paper format) plus a SAD prior
file. Does not require a trained DenSNet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_obj(path: Path):
    try:
        data = np.load(path, allow_pickle=True)
    except ModuleNotFoundError as exc:
        # SAD priors pickled with torch tensors fail without torch.
        raise RuntimeError(
            f"Could not unpickle {path} ({exc}). "
            "Run in the project environment or pass a numpy-only SAD file."
        ) from exc
    if isinstance(data, np.ndarray) and data.dtype == object:
        if data.ndim == 0:
            return data.item()
        return list(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dens-ref", type=Path, required=True, help="DF dataset npy")
    parser.add_argument("--atom-dens", type=Path, required=True, help="SAD prior npy")
    parser.add_argument("--out", type=Path, default=Path("results/revision/density_metrics.json"))
    parser.add_argument("--max-frames", type=int, default=50)
    args = parser.parse_args()

    frames = _load_obj(args.dens_ref)
    if not isinstance(frames, list):
        raise ValueError(f"Expected list of (mol.pack(), calc) in {args.dens_ref}")
    sad = _load_obj(args.atom_dens)
    if not isinstance(sad, dict):
        raise ValueError("SAD prior must be a dict keyed by atomic number")

    rel = []
    signed_frac = []
    for item in frames[: args.max_frames]:
        calc = item[1] if isinstance(item, (list, tuple)) else item
        if "df_coeff" not in calc:
            continue
        df = np.asarray(calc["df_coeff"]).ravel()
        l1 = float(np.mean(np.abs(df)))
        # Proxy relative error scale of the DF vector vs its L1 mean.
        rel.append(l1)
        signed_frac.append(float(np.mean(df < 0)))

    # SAD occupancy dump: how large is the free-atom target vs a dummy zero.
    sad_occ = {}
    for z, rec in sad.items():
        occ = np.asarray(rec["mo_occ"]) if isinstance(rec, dict) and "mo_occ" in rec else None
        if occ is not None:
            sad_occ[str(int(z))] = float(np.sum(occ))

    summary = {
        "dens_ref": str(args.dens_ref),
        "n_frames": min(args.max_frames, len(frames)),
        "df_l1_mean": float(np.mean(rel)) if rel else None,
        "fraction_negative_df_coeff": float(np.mean(signed_frac)) if signed_frac else None,
        "sad_electron_counts": sad_occ,
        "density_error_unit": "e/a0^3 (report grid MAE in this unit in the paper)",
        "relative_mae_definition": "int |rho-rho_ref| dV / N_elec",
        "note": (
            "Full grid relative MAE requires a trained model + evaluate_density_errors.py. "
            "This script records SAD magnitudes and the signed DF-coefficient fraction "
            "as a cheap delta-learning diagnostic."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
