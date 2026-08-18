#!/usr/bin/env python3
"""NVE total-energy drift from an ML-MD HDF5 log (R2.4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _load_energy(hdf5_path: Path) -> tuple[np.ndarray, float]:
    if hdf5_path.suffix.lower() in {".traj", ".xyz", ".jsonl"}:
        if hdf5_path.suffix.lower() == ".jsonl":
            energy = []
            dt_fs = 0.5
            for line in hdf5_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "dt_fs" in rec:
                    dt_fs = float(rec["dt_fs"])
                if "energy_eV" in rec:
                    energy.append(float(rec["energy_eV"]))
                elif "energy" in rec:
                    energy.append(float(rec["energy"]))
            if not energy:
                raise ValueError(f"No energy fields in {hdf5_path}")
            return np.asarray(energy), dt_fs
        from ase.io import iread

        energy = []
        for atoms in iread(str(hdf5_path)):
            try:
                energy.append(float(atoms.get_potential_energy()))
            except Exception:
                continue
        if not energy:
            raise ValueError(f"No stored energies in {hdf5_path}")
        return np.asarray(energy), 0.5
    with h5py.File(hdf5_path, "r") as f:
        dt_fs = 0.5
        if "molecules" in f and "time_step" in f["molecules"].attrs:
            dt_fs = float(f["molecules"].attrs["time_step"]) * 1000.0
        if "properties" in f:
            props = f["properties"]
            for key in ("energy", "energies", "potential_energy"):
                if key in props:
                    e = np.asarray(props[key]).reshape(-1)
                    return e, dt_fs
        # Fallback: scan datasets named energy.
        def _walk(g):
            for k, v in g.items():
                if k.lower().startswith("energy") and hasattr(v, "shape"):
                    return np.asarray(v).reshape(-1)
                if hasattr(v, "items"):
                    found = _walk(v)
                    if found is not None:
                        return found
            return None

        e = _walk(f)
        if e is None:
            raise ValueError(f"No energy dataset in {hdf5_path}")
        return e, dt_fs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--n-atoms", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/revision/nve_drift.json"))
    args = parser.parse_args()

    energy, dt_fs = _load_energy(args.hdf5)
    t_ps = np.arange(len(energy)) * dt_fs / 1000.0
    # Linear drift of total energy.
    coef = np.polyfit(t_ps, energy, 1)
    drift = float(coef[0])  # energy units / ps
    n_atoms = args.n_atoms or 1
    # Convert eV/ps/atom → meV/atom/ps if energies look like eV (O(1–100)).
    unit = "native_energy_per_ps"
    drift_per_atom = drift / n_atoms
    summary = {
        "hdf5": str(args.hdf5),
        "n_steps": int(len(energy)),
        "dt_fs": dt_fs,
        "duration_ps": float(t_ps[-1]) if len(t_ps) else 0.0,
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
        "drift_per_ps": drift,
        "drift_per_atom_per_ps": drift_per_atom,
        "max_abs_deviation": float(np.max(np.abs(energy - energy[0]))),
        "unit_note": unit,
        "n_atoms": n_atoms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    np.savez(args.out.with_suffix(".npz"), t_ps=t_ps, energy=energy)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
