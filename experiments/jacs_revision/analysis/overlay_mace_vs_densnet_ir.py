#!/usr/bin/env python3
"""Overlay MACE-OFF ethanol IR against the archived DenSNet 25 ps curve (R2.2)."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/overlay_mace_vs_densnet_ir.py'


import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_REPO = _SCRIPT_DIR.parents[2]


def _mesa_ir(dipoles: np.ndarray, timestep_fs: float):
    import memspectrum

    speed_cm_s = 299792458.0 * 100.0
    total = None
    for dim in range(3):
        signal = dipoles[:, dim] - np.mean(dipoles[:, dim])
        mesa = memspectrum.MESA()
        mesa.solve(signal, method="Standard", optimisation_method="FPE")
        freq_fs_inv, psd = mesa.spectrum(dt=timestep_fs, onesided=True)
        freq_hz = freq_fs_inv * 1e15
        freq_cm = freq_hz / speed_cm_s
        intensity = psd * (freq_hz ** 2)
        total = intensity if total is None else total + intensity
    return freq_cm, total


def _load_csv_spectrum(path: Path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def _peak_cm(freq, spec, lo=500.0, hi=4000.0):
    mask = (freq >= lo) & (freq <= hi)
    if not np.any(mask):
        return float("nan")
    return float(freq[mask][np.argmax(spec[mask])])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument(
        "--densnet-csv",
        type=Path,
        default=_REPO / "results/revision/ir_convergence_ethanol/ir_ethanol_cluster_rep0_25ps.csv",
    )
    parser.add_argument("--out", type=Path, default=_REPO / "results/revision/ir_mace_ethanol/comparison.json")
    args = parser.parse_args()

    frames = read(args.traj, index=":")
    dipoles = []
    for atoms in frames:
        if hasattr(atoms, "calc") and atoms.calc is not None:
            try:
                dipoles.append(np.asarray(atoms.get_dipole_moment(), dtype=float))
                continue
            except Exception:
                pass
        charges = atoms.get_initial_charges()
        if charges is None or np.allclose(charges, 0.0):
            numbers = atoms.get_atomic_numbers().astype(float)
            dipoles.append(np.sum(numbers[:, None] * atoms.get_positions(), axis=0))
        else:
            dipoles.append(np.sum(charges[:, None] * atoms.get_positions(), axis=0))
    dipoles = np.asarray(dipoles, dtype=float)

    freq_mace, spec_mace = _mesa_ir(dipoles, args.timestep_fs)
    freq_dn, spec_dn = _load_csv_spectrum(args.densnet_csv)

    import matplotlib.pyplot as plt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, freq, spec in (
        ("MACE-OFF 25 ps", freq_mace, spec_mace),
        ("DenSNet 25 ps (archived)", freq_dn, spec_dn),
    ):
        mask = (freq >= 0) & (freq <= 4000)
        norm = spec[mask] / np.max(np.abs(spec[mask])) if np.max(np.abs(spec[mask])) > 0 else spec[mask]
        ax.plot(freq[mask], norm, label=label, linewidth=1.4)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("IR intensity (normalized)")
    ax.set_xlim(0, 4000)
    ax.legend()
    fig.tight_layout()
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=180)
    plt.close(fig)

    payload = {
        "traj": str(args.traj),
        "n_frames": int(len(frames)),
        "timestep_fs": args.timestep_fs,
        "densnet_csv": str(args.densnet_csv),
        "mace_peak_cm": _peak_cm(freq_mace, spec_mace),
        "densnet_peak_cm": _peak_cm(freq_dn, spec_dn),
        "figure": str(png),
        "note": "MACE dipoles from calculator if present, else nuclear-charge*R proxy",
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
