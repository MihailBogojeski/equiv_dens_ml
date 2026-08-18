#!/usr/bin/env python3
"""ASE Langevin MD with the working DenSNetCalculator (R2.2, R3.6 fallback).

Use when schnetpack_md_run cannot join the revision SAD. Writes a trajectory
and JSONL energies. Dipoles are recorded only if the calculator exposes them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="datasets/ethanol_train_10.xyz")
    parser.add_argument("--model", default="paper/models/ethanol/2024-03-22_96w7KyGG")
    parser.add_argument("--args-file", default="config/md/nn/ethanol_500ps.txt")
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--nve", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/md_ethanol_ase"))
    args = parser.parse_args()

    from equiv_dens.md.dft_network_calculator import load_densnet_calculator

    atoms = read(args.structure, index=0)
    atoms.calc = load_densnet_calculator(args.model, args_file=args.args_file, use_gpu=args.use_gpu)
    rng = np.random.RandomState(17)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    if args.nve:
        dyn = VelocityVerlet(atoms, args.dt_fs * units.fs)
    else:
        dyn = Langevin(
            atoms,
            args.dt_fs * units.fs,
            temperature_K=args.temperature,
            friction=args.friction,
            rng=rng,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    traj = Trajectory(str(args.out_dir / "md.traj"), "w", atoms)
    jsonl = args.out_dir / "md.jsonl"
    if jsonl.exists():
        jsonl.unlink()

    def _log():
        rec = {
            "step": int(dyn.nsteps),
            "energy_eV": float(atoms.get_potential_energy()),
            "temperature_K": float(atoms.get_temperature()),
            "dt_fs": args.dt_fs,
        }
        with jsonl.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if dyn.nsteps % (args.log_interval * 10) == 0:
            print(json.dumps(rec), flush=True)
        traj.write()

    dyn.attach(_log, interval=args.log_interval)
    print(f"starting {'NVE' if args.nve else 'Langevin'} steps={args.steps}", flush=True)
    dyn.run(args.steps)
    print(f"wrote {args.out_dir / 'md.traj'}")


if __name__ == "__main__":
    main()
