#!/usr/bin/env python3
"""CPU-only MLIP / xTB geo-opt, short NVE, and optional NVT MD (R2.2, R2.11, R2.4).

Does not use the node GPUs. Paper DenSNet checkpoints are not required.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read, Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase import units


def _make_calc(
    method: str,
    gxtb_bin: str,
    device: str = "cpu",
    densnet_model: str = "paper/models/ethanol/2024-03-22_96w7KyGG",
    densnet_args: str = "config/md/nn/ethanol_500ps.txt",
):
    if method == "gxtb":
        from equiv_dens.md.gxtb_calculator import GxTBCalculator

        return GxTBCalculator(gxtb_command=gxtb_bin)
    if method in ("gfn2xtb", "tblite"):
        from tblite.ase import TBLite

        return TBLite(method="GFN2-xTB")
    if method == "maceoff":
        from mace.calculators import mace_off

        return mace_off(model="medium", device=device)
    if method == "aimnet2":
        try:
            from aimnet.calculators import AIMNet2ASE

            return AIMNet2ASE("aimnet2")
        except Exception:
            from aimnet.calculators.ase import AIMNet2ASE

            return AIMNet2ASE("aimnet2")
    if method == "densnet":
        from equiv_dens.md.dft_network_calculator import load_densnet_calculator

        return load_densnet_calculator(
            densnet_model,
            args_file=densnet_args,
            use_gpu=device == "cuda",
        )
    raise ValueError(f"unknown method {method}")


def _geoopt(atoms, fmax: float, traj_path: Path | None):
    t0 = time.perf_counter()
    opt = BFGS(atoms, logfile=None, trajectory=str(traj_path) if traj_path else None)
    opt.run(fmax=fmax)
    return {
        "energy_eV": float(atoms.get_potential_energy()),
        "fmax": float(np.max(np.linalg.norm(atoms.get_forces(), axis=1))),
        "nsteps": int(opt.nsteps),
        "wall_s": time.perf_counter() - t0,
        "positions": atoms.get_positions().tolist(),
    }


def _nve(atoms, steps: int, dt_fs: float, out_traj: Path, energy_jsonl: Path, T: float, seed: int):
    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=rng)
    dyn = VelocityVerlet(atoms, dt_fs * units.fs)
    out_traj.parent.mkdir(parents=True, exist_ok=True)
    traj = Trajectory(str(out_traj), "w", atoms)
    energy_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = []

    def _log():
        rec = {
            "step": int(dyn.nsteps),
            "energy_eV": float(atoms.get_potential_energy()),
            "temperature_K": float(atoms.get_temperature()),
            "dt_fs": dt_fs,
        }
        records.append(rec)
        with energy_jsonl.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        traj.write()

    if energy_jsonl.exists():
        energy_jsonl.unlink()
    dyn.attach(_log, interval=1)
    t0 = time.perf_counter()
    dyn.run(steps)
    wall = time.perf_counter() - t0
    return records, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="datasets/ethanol_train_10.xyz")
    parser.add_argument("--methods", default="gxtb")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--nve-steps", type=int, default=400, help="0.5 fs → 0.2 ps default")
    parser.add_argument("--nvt-steps", type=int, default=0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--gxtb-bin", default="g-xtb/binary/gxtb")
    parser.add_argument("--densnet-model", default="paper/models/ethanol/2024-03-22_96w7KyGG")
    parser.add_argument("--densnet-args", default="config/md/nn/ethanol_500ps.txt")
    parser.add_argument("--out-dir", type=Path, default=Path("results/revision/mlip_cpu"))
    parser.add_argument("--skip-geoopt", action="store_true")
    parser.add_argument("--skip-nve", action="store_true")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Calculator device. Use cuda for local L40S tests.",
    )
    args = parser.parse_args()

    atoms0 = read(args.structure, index=0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "structure": args.structure,
        "n_atoms": int(len(atoms0)),
        "methods": {},
    }
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    for method in methods:
        rec = {"method": method}
        try:
            calc = _make_calc(
                method,
                args.gxtb_bin,
                device=args.device,
                densnet_model=args.densnet_model,
                densnet_args=args.densnet_args,
            )
            if not args.skip_geoopt:
                atoms = atoms0.copy()
                atoms.calc = calc
                rec["geoopt"] = _geoopt(
                    atoms, args.fmax, args.out_dir / f"{method}_geoopt.traj"
                )
                opt_atoms = atoms
            else:
                opt_atoms = atoms0.copy()
                opt_atoms.calc = calc
            if not args.skip_nve and args.nve_steps > 0:
                nve_atoms = opt_atoms.copy()
                nve_atoms.calc = _make_calc(
                    method,
                    args.gxtb_bin,
                    device=args.device,
                    densnet_model=args.densnet_model,
                    densnet_args=args.densnet_args,
                )
                records, wall = _nve(
                    nve_atoms,
                    args.nve_steps,
                    args.dt_fs,
                    args.out_dir / f"{method}_nve.traj",
                    args.out_dir / f"{method}_nve.jsonl",
                    args.temperature,
                    seed=17,
                )
                e = np.array([r["energy_eV"] for r in records])
                t_ps = np.arange(len(e)) * args.dt_fs / 1000.0
                rec["nve"] = {
                    "n_steps": int(len(e)),
                    "duration_ps": float(t_ps[-1]) if len(t_ps) else 0.0,
                    "energy_std_eV": float(np.std(e)),
                    "drift_eV_per_ps": float(np.polyfit(t_ps, e, 1)[0]) if len(e) > 2 else None,
                    "max_abs_dev_eV": float(np.max(np.abs(e - e[0]))) if len(e) else None,
                    "wall_s": wall,
                    "ns_per_day": (
                        (args.nve_steps * args.dt_fs / 1e6) / (wall / 86400.0) if wall else None
                    ),
                }
            if args.nvt_steps > 0:
                nvt_atoms = opt_atoms.copy()
                nvt_atoms.calc = _make_calc(
                    method,
                    args.gxtb_bin,
                    device=args.device,
                    densnet_model=args.densnet_model,
                    densnet_args=args.densnet_args,
                )
                rng = np.random.RandomState(17)
                MaxwellBoltzmannDistribution(nvt_atoms, temperature_K=args.temperature, rng=rng)
                dyn = Langevin(
                    nvt_atoms,
                    args.dt_fs * units.fs,
                    temperature_K=args.temperature,
                    friction=0.01,
                    rng=rng,
                )
                traj = Trajectory(str(args.out_dir / f"{method}_nvt.traj"), "w", nvt_atoms)
                dyn.attach(traj.write, interval=10)
                t0 = time.perf_counter()
                dyn.run(args.nvt_steps)
                rec["nvt"] = {
                    "n_steps": args.nvt_steps,
                    "duration_ps": args.nvt_steps * args.dt_fs / 1000.0,
                    "wall_s": time.perf_counter() - t0,
                }
        except Exception as exc:
            rec["error"] = str(exc)
        summary["methods"][method] = rec
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({method: rec}, indent=2), flush=True)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
