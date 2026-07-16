#!/usr/bin/env python3
"""
Run MD with g-xtb as calculator, reusing SchNetPack MD infrastructure.

Accepts same input structures as NN-based MD: .npy (positions, atom_numbers)
or .xyz. Output format matches NN MD (HDF5 trajectory via FileLogger).

Example:
    run.py gxtb-md @config/md/gxtb/gxtb_polythiophene_2mer.txt
"""

import equiv_dens.compat  # noqa: F401 - apply T_co patch before schnetpack import

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import ase.io

import equiv_dens.utils.base as utils
from equiv_dens.md.gxtb_calculator import GxtbMDCalculator
from equiv_dens.md.md_console_logger import MDConsoleLogger

from schnetpack.md import System
from schnetpack.md import MaxwellBoltzmannInit
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md import Simulator
from schnetpack.md.simulation_hooks import thermostats
from schnetpack.md.simulation_hooks import callback_hooks


def _str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("true", "t", "1"):
        return True
    if s.lower() in ("false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {s}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run g-xtb MD (no model)",
        fromfile_prefix_chars="@",
    )
    # Input
    parser.add_argument(
        "--structure",
        type=str,
        required=True,
        help="Path to .npy (positions, atom_numbers) or .xyz",
    )
    parser.add_argument(
        "--structure-frame",
        type=int,
        default=0,
        help="Frame index for multi-frame .npy (default 0)",
    )
    parser.add_argument(
        "--position-unit",
        type=str,
        default="Bohr",
        choices=["Bohr", "Angstrom"],
        help="Unit of positions in .npy (default Bohr)",
    )
    # g-xtb
    parser.add_argument(
        "--gxtb-path",
        type=str,
        required=True,
        help="Path to gxtb executable",
    )
    parser.add_argument(
        "--gxtb-params",
        type=str,
        required=True,
        help="Path to GXTBHOME (parameter directory)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Scratch dir for g-xtb I/O (default: log-dir/gxtb_workdir)",
    )
    # MD
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature (K)",
    )
    parser.add_argument(
        "--md-steps",
        type=int,
        default=1000,
        help="Number of MD steps",
    )
    parser.add_argument(
        "--langevin",
        type=_str2bool,
        default=True,
        help="Use Langevin thermostat",
    )
    parser.add_argument(
        "--warm-up",
        type=_str2bool,
        default=False,
        help="Warm-up run with thermostat before main simulation",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.5,
        help="Time step (fs)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="scratch/gxtb_md",
        help="Output directory for trajectory and checkpoint",
    )
    parser.add_argument(
        "--log-suffix",
        type=str,
        default="",
        help="Suffix for log files (e.g. '2mer')",
    )
    parser.add_argument(
        "--console-log-interval",
        type=int,
        default=10,
        help="Steps between console output",
    )
    return parser.parse_args()


def _load_structure(structure_path: str, frame: int, position_unit: str) -> list:
    """Load structure from .npy or .xyz; return list of ASE Atoms."""
    path = Path(structure_path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        if data.ndim == 0 and isinstance(data.item(), dict):
            data = data.item()
        positions = np.asarray(data["positions"])
        atom_numbers = np.asarray(data["atom_numbers"])
        if positions.ndim == 3:
            positions = positions[frame]
        if atom_numbers.ndim == 2:
            atom_numbers = atom_numbers[frame] if frame < atom_numbers.shape[0] else atom_numbers[0]
        if position_unit == "Bohr":
            positions = utils.bohr_to_angstrom(positions)
        # npy_to_ase expects (n_frames, n_atoms, 3) and (n_frames, n_atoms) or (n_atoms,)
        if positions.ndim == 2:
            positions = positions[np.newaxis, ...]
        if atom_numbers.ndim == 1:
            atom_numbers = atom_numbers[np.newaxis, :]
        mols = utils.npy_to_ase(positions, atom_numbers)
        return mols
    if suffix == ".xyz":
        mols = ase.io.read(path, index=":")
        if isinstance(mols, list):
            mol = mols[frame] if frame < len(mols) else mols[0]
        else:
            mol = mols
        return [mol]
    raise ValueError(f"Unsupported structure format: {suffix}")


def main():
    args = _parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    workdir = Path(args.workdir) if args.workdir else log_dir / "gxtb_workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    mols = _load_structure(args.structure, args.structure_frame, args.position_unit)
    n_replicas = 1

    md_system = System()
    md_system.load_molecules(mols, n_replicas)

    md_initializer = MaxwellBoltzmannInit(
        args.temperature,
        remove_translation=True,
        remove_rotation=True,
    )
    md_initializer.initialize_system(md_system)

    md_integrator = VelocityVerlet(args.time_step)

    md_calculator = GxtbMDCalculator(
        gxtb_path=args.gxtb_path,
        gxtb_params_dir=args.gxtb_params,
        workdir=workdir,
        required_properties=["energy", "forces"],
        force_key="forces",
        energy_key="energy",
        position_unit="Angstrom",
        energy_unit="kcal/mol",
    )

    log_suffix = f"_{args.log_suffix}" if args.log_suffix else ""
    log_file = log_dir / f"simulation{log_suffix}.hdf5"
    chk_file = log_dir / f"simulation{log_suffix}.chk"

    buffer_size = 100
    data_streams = [
        callback_hooks.MoleculeStream(store_velocities=True),
        callback_hooks.PropertyStream(target_properties=["energy", "forces"]),
    ]
    file_logger = callback_hooks.FileLogger(
        str(log_file),
        buffer_size,
        data_streams=data_streams,
    )

    console_logger = MDConsoleLogger(
        every_n_steps=args.console_log_interval,
        time_step_fs=args.time_step,
        energy_unit="kcal/mol",
    )

    checkpoint = callback_hooks.Checkpoint(str(chk_file), every_n_steps=1000)

    simulation_hooks = [file_logger, console_logger, checkpoint]

    if args.langevin or args.warm_up:
        langevin = thermostats.LangevinThermostat(args.temperature, time_constant=100.0)
        if args.langevin:
            simulation_hooks.append(langevin)
        elif args.warm_up:
            warmup_hooks = simulation_hooks + [langevin]
            warmup_simulator = Simulator(
                md_system,
                md_integrator,
                md_calculator,
                simulator_hooks=warmup_hooks,
                progress=False,
            )
            warmup_steps = min(args.md_steps // 20, 500)
            if warmup_steps > 0:
                warmup_simulator.simulate(warmup_steps)
                print("Warm-up finished")

    md_simulator = Simulator(
        md_system,
        md_integrator,
        md_calculator,
        simulator_hooks=simulation_hooks,
        progress=False,
    )

    if chk_file.exists():
        print("Restarting from checkpoint")
        state_dict = torch.load(str(chk_file), weights_only=False)
        md_simulator.restart_simulation(state_dict)

    steps_done = 0
    chunk = 100
    while steps_done < args.md_steps:
        to_do = min(chunk, args.md_steps - steps_done)
        md_simulator.simulate(to_do)
        steps_done += to_do

    print(f"Simulation complete: {steps_done} steps, trajectory: {log_file}")


if __name__ == "__main__":
    main()
