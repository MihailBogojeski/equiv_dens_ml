#!/usr/bin/env python3
"""25 ps MACE-OFF ethanol MD + IR overlay vs archived DenSNet (R2.2)."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/run_mace_ethanol_ir.py'


import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import Trajectory, read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs as ASE_FS

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.analysis.overlay_mace_vs_densnet_ir import _load_csv_spectrum, _mesa_ir, _peak_cm  # noqa: E402


def _dipole(atoms) -> np.ndarray:
    try:
        return np.asarray(atoms.get_dipole_moment(), dtype=float).reshape(3)
    except Exception:
        numbers = atoms.get_atomic_numbers().astype(float)
        return np.sum(numbers[:, None] * atoms.get_positions(), axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--densnet-csv",
        type=Path,
        default=Path("results/revision/ir_convergence_ethanol/ir_ethanol_cluster_rep0_25ps.csv"),
    )
    args = parser.parse_args()

    from mace.calculators import mace_off

    atoms = read(args.structure, index=0)
    atoms.calc = mace_off(model="medium", device=args.device)
    rng = np.random.RandomState(0)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    dyn = Langevin(
        atoms,
        args.timestep_fs * ASE_FS,
        temperature_K=args.temperature,
        friction=0.01,
        rng=rng,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = args.out_dir / "ethanol_25ps.traj"
    dipoles = []

    def _sample():
        if dyn.nsteps % args.sample_every == 0:
            dipoles.append(_dipole(atoms))

    traj = Trajectory(str(traj_path), "w", atoms)
    dyn.attach(traj.write, interval=args.sample_every)
    dyn.attach(_sample, interval=1)
    dyn.run(args.steps)
    dipoles = np.asarray(dipoles, dtype=float)
    np.save(args.out_dir / "dipoles.npy", dipoles)

    dt_sample = args.timestep_fs * args.sample_every
    freq_mace, spec_mace = _mesa_ir(dipoles, dt_sample)
    freq_dn, spec_dn = _load_csv_spectrum(args.densnet_csv)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, freq, spec in (
        ("MACE-OFF 25 ps", freq_mace, spec_mace),
        ("DenSNet 25 ps (archived)", freq_dn, spec_dn),
    ):
        mask = (freq >= 0) & (freq <= 4000)
        peak = np.max(np.abs(spec[mask])) if np.any(mask) else 1.0
        ax.plot(freq[mask], spec[mask] / peak if peak else spec[mask], label=label, linewidth=1.4)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("IR intensity (normalized)")
    ax.set_xlim(0, 4000)
    ax.legend()
    fig.tight_layout()
    png = args.out_dir / "comparison.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)

    payload = {
        "traj": str(traj_path),
        "n_dipole_frames": int(len(dipoles)),
        "sample_dt_fs": dt_sample,
        "mace_peak_cm": _peak_cm(freq_mace, spec_mace),
        "densnet_peak_cm": _peak_cm(freq_dn, spec_dn),
        "figure": str(png),
    }
    (args.out_dir / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
