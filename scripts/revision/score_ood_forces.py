#!/usr/bin/env python3
"""Score a DenSNet checkpoint on labelled OOD frames (R1.2).

Compares ASE energies and forces against PySCF+D4 labels.

The raw energy difference is around 4200 eV and always will be. The model is
trained with ``--center_energy=True``, so it predicts energies measured from an
offset fixed by its training set, while the label is a DFT total energy
(ethanol is near -4214 eV). Subtracting a constant from one side of a
comparison is not an error, and reporting it as one hid whether the energy head
worked. What is meaningful for a centred-energy model is the spread once that
constant is removed, so both are reported and the offset is named as an offset.

Forces need no such treatment -- a constant shift in energy has zero gradient --
but they do need a scale. These frames are deliberately extreme: high-temperature
and affinely strained geometries whose reference forces reach hundreds of eV/A.
An absolute MAE alone would look alarming next to an equilibrium benchmark, so
the reference force magnitude and the ratio to it are reported alongside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import Hartree, kcal, mol

KCALMOL = kcal / mol


_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))


def _load_frames(dens_path: Path, max_frames: int):
    dens = np.load(dens_path, allow_pickle=True)
    frames = []
    for item in dens[: max_frames or None]:
        mol, calc = item[0], item[1]
        if not isinstance(mol, dict) or mol.get("atom") is None:
            continue
        numbers = []
        pos = []
        for spec, xyz in mol["atom"]:
            numbers.append(int(spec) if not isinstance(spec, str) else spec)
            pos.append(xyz)
        pos = np.asarray(pos, dtype=float)
        unit = str(mol.get("unit", "angstrom")).lower()
        if unit.startswith("bohr"):
            pos = pos * 0.529177210903
        energy_h = float(calc["energy"])
        forces = np.asarray(calc["forces"], dtype=float)
        frames.append(
            {
                "numbers": numbers,
                "positions": pos,
                "energy_eV": energy_h * Hartree,
                "forces_eV_A": forces * Hartree,  # labels are Eh / Ang from generate_dft_labels
            }
        )
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dens", default="datasets/revision/ood/ethanol_ood_pyscf_augccpvdz_pbe.npy")
    parser.add_argument("--model", default="paper/models/ethanol/2024-03-22_96w7KyGG")
    parser.add_argument("--args-file", default="config/md/nn/ethanol_500ps.txt")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/revision/eval_ethanol_ood_forces.json"))
    args = parser.parse_args()

    from equiv_dens.md.dft_network_calculator import load_densnet_calculator

    frames = _load_frames(Path(args.dens), args.max_frames)
    if not frames:
        raise SystemExit(f"No frames in {args.dens}")

    calc = load_densnet_calculator(args.model, args_file=args.args_file, use_gpu=args.use_gpu)
    e_err = []
    f_mae = []
    f_rmse = []
    f_max = []
    f_ref_scale = []
    for i, fr in enumerate(frames):
        atoms = Atoms(numbers=fr["numbers"], positions=fr["positions"])
        atoms.calc = calc
        e = float(atoms.get_potential_energy())
        f = np.asarray(atoms.get_forces())
        de = e - fr["energy_eV"]
        df = f - fr["forces_eV_A"]
        e_err.append(de)
        f_mae.append(float(np.mean(np.abs(df))))
        f_rmse.append(float(np.sqrt(np.mean(df**2))))
        f_max.append(float(np.max(np.abs(df))))
        f_ref_scale.append(float(np.mean(np.abs(fr["forces_eV_A"]))))
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] force MAE={f_mae[-1]:.4f} eV/A", flush=True)

    e_err = np.asarray(e_err)
    # The centring constant, estimated as the mean signed difference. Removing
    # it leaves the part of the energy error the model is actually responsible
    # for; the median is reported too because a handful of near-clashing OOD
    # frames drag the mean.
    offset = float(np.mean(e_err))
    e_centred = e_err - offset
    ref_scale = float(np.mean(f_ref_scale))
    summary = {
        "model": args.model,
        "dens": args.dens,
        "n_frames": len(frames),
        "energy_offset_eV": offset,
        "energy_mae_after_offset_eV": float(np.mean(np.abs(e_centred))),
        "energy_mae_after_offset_kcalmol": float(np.mean(np.abs(e_centred)) / KCALMOL),
        "energy_median_abs_after_offset_eV": float(np.median(np.abs(e_centred))),
        "energy_raw_mae_eV": float(np.mean(np.abs(e_err))),
        "force_mae_eV_A": float(np.mean(f_mae)),
        "force_rmse_eV_A": float(np.mean(f_rmse)),
        "force_max_eV_A": float(np.max(f_max)),
        "force_mae_kcalmol_A": float(np.mean(f_mae) / KCALMOL),
        "reference_force_mean_abs_eV_A": ref_scale,
        "force_mae_relative_to_reference": (float(np.mean(f_mae)) / ref_scale) if ref_scale else None,
        "note": (
            "energy_raw_mae_eV is dominated by a constant: the model is trained "
            "with center_energy=True and predicts energies from its own origin, "
            "while the label is a DFT total energy. energy_mae_after_offset_eV "
            "removes that constant and is the number to quote. Forces are "
            "unaffected by the offset; these OOD frames are deliberately extreme, "
            "so force_mae_relative_to_reference gives the scale."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
