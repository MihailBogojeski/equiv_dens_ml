#!/usr/bin/env python3
"""NVE total-energy drift of the paper's archived polythiophene ML-MD runs (R2.4).

The archived trajectories (paper/trajectories/polythiophene/with_velocity/*_vel_*.npy)
store positions and velocities every 0.5 fs step but no energies. This recomputes
the DenSNet potential energy on every --stride-th frame with the paper's
polythiophene checkpoint (the calculator of benchmark_step_timing), adds the
kinetic energy from the stored velocities, and fits a line to E_tot(t).

Two checks guard the units before any drift is reported:
  * velocity units: stored v against the centred finite difference of positions;
  * force/energy scale: m*a from the finite difference of stored velocities
    against the model force on the same frames (slope should be 1). A checkpoint
    loaded without its training force std gives forces and energies that are
    uniformly too small, which would make any drift look small too.

    .venv/bin/python -B experiments/jacs_revision/analysis/nve_drift_archived.py \
        --traj paper/trajectories/polythiophene/with_velocity/simulation_local_all_12mer_md_001_vel_0.npy \
        --stride 1000 --device cpu --out results/revision/nve_drift_archived/12mer_0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DT_FS = 0.5  # paper ML-MD frames are 0.5 fs apart
AMU_A2_FS2_TO_EV = 103.642697  # 1 amu * (A/fs)^2 in eV


def load(path: Path):
    data = np.load(path, allow_pickle=True).item()
    return np.asarray(data["atom_numbers"]), data["positions"], data["velocities"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--loader", choices=["tierh", "timing"], default="tierh",
                    help="tierh: the polythiophene paper checkpoint with mo_coeffs atom densities "
                         "(experiments/internal/polythiophene/harmonic_ir.py; reproduces the archived MD forces, "
                         "r 0.9999); timing: benchmark_step_timing's calculator (does not, r 0.81)")
    ap.add_argument("--system", default="thiophene_12mer", help="benchmark_step_timing system key (--loader timing)")
    ap.add_argument("--stride", type=int, default=1000)
    ap.add_argument("--n-check", type=int, default=6, help="frames used for the force-scale check")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    opt = ap.parse_args()

    from ase import Atoms
    from ase.data import atomic_masses

    spec = importlib.util.spec_from_file_location(
        "benchmark_step_timing", REPO / "experiments/jacs_revision/analysis/benchmark_step_timing.py")
    timing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing)

    Z, R, V = load(opt.traj)
    Z = Z if Z.ndim == 1 else Z[0]
    n_frames, n_atoms = R.shape[0], len(Z)
    m = atomic_masses[Z][:, None]

    # velocity units: compare with centred position differences (A/fs)
    i = np.arange(1, min(2001, n_frames - 1))
    v_fd = (R[i + 1].astype(float) - R[i - 1].astype(float)) / (2 * DT_FS)
    v_scale = float(np.sum(v_fd * V[i]) / np.sum(V[i].astype(float) ** 2))

    if opt.loader == "tierh":
        sys.path[:0] = [str(REPO), str(REPO / "src")]
        from experiments.internal.polythiophene import harmonic_ir as tierh
        from experiments.jacs_revision.analysis import equilibrium_frequencies as EF

        model = EF.DenSNet(tierh.spec("mo_coeffs"), opt.device)
        info = {"loader": "harmonic_ir.spec('mo_coeffs')", "run": tierh.RUN, "scaling_std": model.scaling_std}

        def evaluate(k: int):  # eV and eV/A
            out = model(Atoms(Z, positions=R[k].astype(float)), dipole=False)
            return float(out["energy"]), np.asarray(out["forces"], float)
    else:
        calc, info = timing.densnet_calculator(opt.system, opt.device)

        def evaluate(k: int):  # ASE calculator: eV and eV/A
            atoms = Atoms(Z, positions=R[k].astype(float))
            atoms.calc = calc
            return atoms.get_potential_energy(), atoms.get_forces()

    # force scale: m * d2x/dt2 (A/fs^2 * amu -> eV/A) against the model force
    chk = np.linspace(1000, n_frames - 1000, opt.n_check).astype(int)
    fm, fa = [], []
    for k in chk:
        _, f = evaluate(k)
        a = (R[k + 1].astype(float) - 2 * R[k].astype(float) + R[k - 1].astype(float)) / DT_FS ** 2
        fm.append(f.ravel())
        fa.append((m * a * AMU_A2_FS2_TO_EV).ravel())
    fm, fa = np.concatenate(fm), np.concatenate(fa)
    force_slope = float(np.sum(fm * fa) / np.sum(fm ** 2))
    force_r = float(np.corrcoef(fm, fa)[0, 1])

    frames = np.arange(0, n_frames, opt.stride)
    epot, ekin = [], []
    t0 = time.time()
    for k in frames:
        e, _ = evaluate(k)
        epot.append(e)
        v = V[k].astype(float) * v_scale
        ekin.append(0.5 * float(np.sum(m * v ** 2)) * AMU_A2_FS2_TO_EV)
    epot, ekin = np.array(epot), np.array(ekin)
    etot = epot + ekin
    t_ps = frames * DT_FS / 1000.0
    slope, _ = np.polyfit(t_ps, etot, 1)
    temp = 2 * ekin.mean() / (3 * n_atoms * 8.617333e-5)

    result = {
        "traj": str(opt.traj), "calculator": info, "n_atoms": n_atoms, "n_frames": int(n_frames),
        "duration_ps": float(n_frames * DT_FS / 1000.0), "stride": opt.stride, "n_evaluated": len(frames),
        "velocity_scale_to_A_per_fs": v_scale,
        "force_check": {"slope_md_over_model": force_slope, "pearson_r": force_r, "frames": chk.tolist()},
        "mean_temperature_K": float(temp),
        "drift_meV_per_atom_per_ps": float(slope * 1000 / n_atoms),
        "drift_meV_per_atom_per_ns": float(slope * 1e6 / n_atoms),
        "etot_std_meV_per_atom": float(etot.std() * 1000 / n_atoms),
        "ekin_std_meV_per_atom": float(ekin.std() * 1000 / n_atoms),
        "seconds": round(time.time() - t0, 1),
    }
    opt.out.mkdir(parents=True, exist_ok=True)
    np.savez(opt.out / "energies.npz", t_ps=t_ps, epot_eV=epot, ekin_eV=ekin)
    (opt.out / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
