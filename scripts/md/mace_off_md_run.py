#!/usr/bin/env python3
"""
Molecular dynamics with MACE-OFF (off-the-shelf MLIP for organic molecules).

Runs MD using MACE-OFF23, a transferable organic force field (C, H, N, O, P, S,
F, Cl, Br, I). Suitable for polythiophene and related conjugated molecules.
Requires: pip install mace-torch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ase
import ase.io
import ase.units
import numpy as np
from ase import Atoms
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent


def _try_import_mace_off():
    """Try to import mace_off. Returns the calculator class or None."""
    try:
        from mace.calculators import mace_off
        return mace_off
    except ImportError as e:
        print(
            "Error: mace-torch is required. Install with: pip install mace-torch",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def load_structure(path: str) -> Atoms:
    """Load initial structure from XYZ or NPY."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")
    if p.suffix.lower() == ".xyz":
        atoms_list = list(ase.io.iread(str(p)))
        return atoms_list[0]
    if p.suffix.lower() == ".npy":
        data = np.load(p, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item() if data.ndim == 0 else data.flat[0] if data.size == 1 else data
        if not isinstance(data, dict):
            raise ValueError(f"NPY file must contain a dict with 'positions' and 'atom_numbers'; got {type(data)}")
        pos = data["positions"]
        anum = data["atom_numbers"]
        if pos.ndim == 3:
            pos = pos[0]
        if anum.ndim > 1:
            anum = anum[0]
        nonzero = anum > 0
        return Atoms(numbers=anum[nonzero], positions=pos[nonzero])
    raise ValueError(f"Cannot load structure from {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run MD with MACE-OFF (off-the-shelf organic MLIP)"
    )
    parser.add_argument(
        "--structure",
        "-s",
        type=str,
        required=True,
        help="Initial structure: XYZ or NPY",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="mace_off_trajectory.traj",
        help="Output trajectory path (default: mace_off_trajectory.traj)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of MD steps (default: 2000 = 1 ps at 0.5 fs)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.5,
        help="Timestep in fs (default: 0.5)",
    )
    parser.add_argument(
        "--temperature",
        "-T",
        type=float,
        default=300.0,
        help="Temperature in K (default: 300)",
    )
    parser.add_argument(
        "--ensemble",
        choices=["nve", "nvt"],
        default="nvt",
        help="Ensemble: nve (microcanonical) or nvt (Langevin) (default: nvt)",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.01,
        help="Langevin friction (NVT only, default: 0.01)",
    )
    parser.add_argument(
        "--model",
        choices=["small", "medium", "large"],
        default="medium",
        help="MACE-OFF model size (default: medium)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log energy/temperature every N steps (default: 100)",
    )
    parser.add_argument(
        "--traj-interval",
        type=int,
        default=1,
        help="Write trajectory every N steps (default: 1)",
    )
    parser.add_argument(
        "--no-traj",
        action="store_true",
        help="Skip writing trajectory file (for scaling benchmarks)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for initial velocities",
    )

    args = parser.parse_args()

    mace_off = _try_import_mace_off()
    atoms = load_structure(args.structure)

    calc = mace_off(model=args.model, device=args.device)
    atoms.calc = calc

    # Initial velocities
    rng = np.random.RandomState(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)

    # MD integrator
    dt = args.timestep * ase.units.fs
    if args.ensemble == "nvt":
        dyn = Langevin(
            atoms,
            dt,
            temperature_K=args.temperature,
            friction=args.friction,
            rng=rng,
        )
    else:
        dyn = VelocityVerlet(atoms, dt)

    # Logging
    def log_callback():
        e = atoms.get_potential_energy()
        T = atoms.get_temperature()
        print(f"  Step {dyn.nsteps}: E = {e:.4f} eV, T = {T:.1f} K")

    dyn.attach(log_callback, interval=args.log_interval)

    # Trajectory (optional)
    if not args.no_traj:
        traj_path = Path(args.output)
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj = Trajectory(traj_path, "w", atoms)
        dyn.attach(traj.write, interval=args.traj_interval)

    # Run
    print(f"Running {args.ensemble.upper()} MD with MACE-OFF: {args.steps} steps, dt = {args.timestep} fs")
    print(f"Model: {args.model}, device: {args.device}")
    print(f"Temperature: {args.temperature} K")
    if not args.no_traj:
        print(f"Output: {Path(args.output)}")
    start_time = time.time()
    dyn.run(args.steps)
    elapsed_s = time.time() - start_time
    ns_total = args.steps * args.timestep / 1e6  # fs -> ns
    ns_per_day = ns_total / (elapsed_s / 86400.0) if elapsed_s > 0 else 0.0
    fmt = f"{ns_per_day:.3f}" if ns_per_day < 0.1 else f"{ns_per_day:.1f}"
    print(f"Final instantaneous speed: {fmt} ns/day")
    print("Done.")


if __name__ == "__main__":
    main()
