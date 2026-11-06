#!/usr/bin/env python3
"""Run reproducible ethanethiol MD and record physically defined dipoles for IR."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/run_mlip_ir_md.py'


import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.constraints import FixCom
from ase.io import Trajectory, read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

_HERE = _LEGACY_FILE.resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from experiments.jacs_revision.analysis.overlay_mace_vs_densnet_ir import _mesa_ir  # noqa: E402


def _densnet_calculator(*, use_gpu: bool):
    from equiv_dens.md.dft_network_calculator import load_densnet_calculator

    calculator = load_densnet_calculator(
        "paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4",
        args_file="config/md/nn/ethanethiol.txt",
        np_dataset="datasets/ethanethiol_combo_Aidx-1000.npy",
        use_gpu=use_gpu,
    )
    # IR needs the instantaneous electronic moment. Reusing integrals until
    # atoms move 0.5 A introduces artificial plateaus and discontinuities.
    for head in calculator.model.property_models.values():
        if hasattr(head, "cache_integrals"):
            head.cache_integrals = False
            head.dipole_every_n_steps = 1
    return calculator


def _dynamics_calculator(method: str, *, device: str):
    if method == "densnet":
        return _densnet_calculator(use_gpu=device == "cuda")
    if method == "maceoff":
        from mace.calculators import mace_off

        return mace_off(model="medium", device=device, default_dtype="float64")
    if method == "macepolar":
        from mace.calculators import mace_polar

        return mace_polar(
            model="polar-1-m",
            device=device,
            default_dtype="float64",
        )
    if method == "so3lr":
        import os

        if device == "cpu":
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        from so3lr import So3lrCalculator

        return So3lrCalculator(**so3lr_calculator_kwargs())
    raise ValueError(f"unknown method {method!r}")


def so3lr_calculator_kwargs() -> dict[str, Any]:
    """Ask SO3LR for its electrostatic dipole, not energy/forces only."""
    return {
        "calculate_stress": False,
        "lr_cutoff": 1000.0,
        "dtype": np.float64,
        "has_aux": True,
        "observables": ["dipole_vec"],
    }


def _finite_dipole(value) -> np.ndarray | None:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 3 and np.all(np.isfinite(array)):
        return array
    return None


def _dipole_from_results(results) -> np.ndarray | None:
    """Read a physical dipole from ASE results or an SO3LR aux blob."""
    if not isinstance(results, dict):
        return None
    for key in ("dipole", "dipole_vec", "dipole_moment"):
        if key in results:
            found = _finite_dipole(results[key])
            if found is not None:
                return found
    aux = results.get("aux")
    if isinstance(aux, dict):
        return _dipole_from_results(aux)
    return None


def _calculator_dipole(atoms: Atoms) -> np.ndarray:
    """Return a calculator dipole; never substitute a nuclear-charge proxy."""
    calculator = atoms.calc
    if calculator is None:
        raise RuntimeError("atoms have no calculator")
    implemented = set(getattr(calculator, "implemented_properties", []) or [])
    if "dipole" in implemented:
        try:
            value = _finite_dipole(atoms.get_dipole_moment())
        except Exception as exc:
            raise RuntimeError(
                f"{type(calculator).__name__} does not expose a physical dipole"
            ) from exc
        if value is None:
            raise RuntimeError(
                f"invalid dipole shape/value: {atoms.get_dipole_moment()}"
            )
        return value
    atoms.get_potential_energy()
    found = _dipole_from_results(getattr(calculator, "results", {}))
    if found is not None:
        return found
    raise RuntimeError(
        f"{type(calculator).__name__} does not expose a physical dipole"
    )


def load_resume_dipoles(out_dir: Path) -> list[np.ndarray]:
    """Return already-written dipole samples, or an empty list."""
    path = out_dir / "dipoles.npy"
    if not path.is_file():
        return []
    array = np.asarray(np.load(path), dtype=float)
    if array.size == 0:
        return []
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{path} must have shape (n, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path} contains non-finite dipoles")
    return [row.copy() for row in array]


def remaining_production_steps(
    *, n_samples: int, production_steps: int, sample_every: int
) -> int:
    """Production MD steps still needed after ``n_samples`` saved frames."""
    done = int(n_samples) * int(sample_every)
    return max(0, int(production_steps) - done)


def load_resume_atoms(out_dir: Path, structure: Path) -> Atoms:
    """Last written trajectory frame, or the input structure if none exists."""
    traj_path = out_dir / "trajectory.traj"
    if traj_path.is_file():
        last = read(traj_path, index=-1)
        if last is not None:
            return last
    return read(structure, index=0)


def _write_spectrum(dipoles: np.ndarray, sample_dt_fs: float, path: Path) -> None:
    frequency, intensity = _mesa_ir(dipoles, sample_dt_fs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack((frequency, intensity)),
        delimiter=",",
        header="frequency_cm-1,intensity",
        comments="",
    )


def _write_reached_windows(
    dipoles: list[np.ndarray], args: argparse.Namespace
) -> None:
    """Write IR CSVs for production windows that exactly match saved samples."""
    array = np.asarray(dipoles, dtype=float)
    if array.size == 0:
        return
    nsteps = len(dipoles) * args.sample_every
    for window_ps in args.windows_ps:
        target_steps = int(round(window_ps * 1000.0 / args.timestep_fs))
        if nsteps != target_steps:
            continue
        path = args.out_dir / f"ir_{window_ps:g}ps.csv"
        if path.is_file():
            continue
        _write_spectrum(array, args.timestep_fs * args.sample_every, path)


def _run_md(args: argparse.Namespace) -> dict[str, Any]:
    """Run one method and write trajectory, dipoles, spectra, and metadata."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dipoles = load_resume_dipoles(args.out_dir)
    resumed_samples = len(dipoles)
    atoms = load_resume_atoms(args.out_dir, args.structure)
    if resumed_samples and not any(isinstance(c, FixCom) for c in atoms.constraints):
        raise ValueError("Legacy IR trajectory uses a different thermostat; use a fresh output directory")
    if not any(isinstance(c, FixCom) for c in atoms.constraints):
        atoms.set_constraint([*atoms.constraints, FixCom()])
    atoms.info.update(charge=0, spin=1, external_field=[0.0, 0.0, 0.0])
    atoms.calc = _dynamics_calculator(args.method, device=args.device)
    dipole_calculator = (
        _densnet_calculator(use_gpu=args.device == "cuda")
        if args.method == "maceoff"
        else None
    )

    rng = np.random.RandomState(args.seed)
    momenta = np.asarray(atoms.get_momenta(), dtype=float)
    if resumed_samples == 0 or not np.any(np.abs(momenta) > 0):
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_k, rng=rng)
    dynamics = Langevin(
        atoms,
        args.timestep_fs * units.fs,
        temperature_K=args.temperature_k,
        friction=args.friction,
        rng=rng,
        fixcm=False,
    )
    if resumed_samples == 0:
        dynamics.run(args.equilibration_steps)
    # Count production steps from zero, independently of the equilibration
    # length, so the initial observer is consistently skipped on fresh/resume.
    dynamics.nsteps = 0

    traj_mode = "a" if resumed_samples else "w"
    trajectory = Trajectory(str(args.out_dir / "trajectory.traj"), traj_mode, atoms)
    temperatures: list[float] = []
    energies: list[float] = []
    remaining = remaining_production_steps(
        n_samples=resumed_samples,
        production_steps=args.production_steps,
        sample_every=args.sample_every,
    )
    # Langevin.run() calls observers at the current geometry before stepping.
    # That frame is already in dipoles.npy / trajectory.traj on resume.
    skip_current_frame = True

    def sample() -> None:
        nonlocal skip_current_frame
        if skip_current_frame:
            skip_current_frame = False
            return
        if dipole_calculator is None:
            dipoles.append(_calculator_dipole(atoms))
        else:
            probe = atoms.copy()
            probe.info.update(atoms.info)
            probe.calc = dipole_calculator
            dipoles.append(_calculator_dipole(probe))
        trajectory.write()
        temperatures.append(float(atoms.get_temperature()))
        energies.append(float(atoms.get_potential_energy()))
        temporary = args.out_dir / "dipoles.tmp.npy"
        np.save(temporary, np.asarray(dipoles, dtype=float))
        temporary.replace(args.out_dir / "dipoles.npy")
        _write_reached_windows(dipoles, args)

    if remaining > 0:
        dynamics.attach(sample, interval=args.sample_every)
        dynamics.run(remaining)
    else:
        _write_reached_windows(dipoles, args)
    trajectory.close()

    temperature_mean = (
        float(np.mean(temperatures)) if temperatures else float("nan")
    )
    temperature_std = float(np.std(temperatures)) if temperatures else float("nan")
    energy_mean = float(np.mean(energies)) if energies else float("nan")
    metadata = {
        "method": args.method,
        "structure": str(args.structure),
        "seed": args.seed,
        "thermostat": "Langevin(fixcm=False) + FixCom",
        "dipole_integrals": "recomputed at every geometry",
        "temperature_target_k": args.temperature_k,
        "temperature_mean_k": temperature_mean,
        "temperature_std_k": temperature_std,
        "timestep_fs": args.timestep_fs,
        "sample_every": args.sample_every,
        "n_samples": len(dipoles),
        "resumed_samples": resumed_samples,
        "equilibration_steps": (
            0 if resumed_samples else args.equilibration_steps
        ),
        "production_steps": args.production_steps,
        "production_ps": args.production_steps * args.timestep_fs / 1000.0,
        "dipole_source": (
            "DenSNet density evaluated on MACE-OFF frames"
            if args.method == "maceoff"
            else f"{args.method} native calculator dipole"
        ),
        "energy_mean_ev": energy_mean,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("densnet", "maceoff", "macepolar", "so3lr"),
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path("results/revision/ethanethiol_train_frame0.xyz"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--timestep-fs", type=float)
    parser.add_argument("--sample-every", type=int, default=None,
                        help="MD steps between dipoles (default: 1 fs sample spacing)")
    parser.add_argument("--production-ps", type=float, default=50.0)
    parser.add_argument("--equilibration-ps", type=float, default=2.0)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--windows-ps", type=float, nargs="+", default=(25.0, 50.0))
    args = parser.parse_args()

    if args.timestep_fs is None:
        args.timestep_fs = 0.5 if args.method == "densnet" else 1.0
    if args.sample_every is None:
        args.sample_every = max(1, int(round(1.0 / args.timestep_fs)))
    args.production_steps = int(round(args.production_ps * 1000.0 / args.timestep_fs))
    args.equilibration_steps = int(round(args.equilibration_ps * 1000.0 / args.timestep_fs))
    _run_md(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
