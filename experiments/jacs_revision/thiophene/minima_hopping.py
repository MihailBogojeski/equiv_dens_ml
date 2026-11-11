"""Goedecker minima hopping (Frank et al., Nat. Commun. 2024 Methods).

Escape MD (Velocity Verlet) is followed by LBFGS. A candidate is the same
minimum if the COM+SO(3) RMSD is ≤ 0.1 Å. Acceptance uses an adaptive
energy window E_diff; the MD temperature T0 is adapted the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import LBFGS
from ase.units import fs as ASE_FS
from ase.units import kB


def so3_rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    """RMSD after removing translation and a proper rotation (no inversion)."""
    p = np.asarray(mobile, dtype=float)
    q = np.asarray(reference, dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return float("inf")
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    h = p_c.T @ q_c
    try:
        u, _, vt = np.linalg.svd(h)
    except np.linalg.LinAlgError:
        return float("inf")
    det = np.linalg.det(vt.T @ u.T)
    d = 1.0 if det >= 0.0 else -1.0
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = p_c @ rot.T
    return float(np.sqrt(np.mean(np.sum((aligned - q_c) ** 2, axis=1))))


def minima_identical(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    threshold: float = 0.1,
) -> bool:
    return so3_rmsd(pos_a, pos_b) <= threshold


def decide_accept(
    energy_new: float,
    energy_current: float,
    is_current: bool,
    e_diff: float,
) -> bool:
    """Accept a non-current minimum if the uphill step is within E_diff."""
    if is_current:
        return False
    return float(energy_new) - float(energy_current) <= float(e_diff)


def adapt_temperature(temperature_k: float, found_new: bool, beta: float = 1.05) -> float:
    if found_new:
        return float(temperature_k) / float(beta)
    return float(temperature_k) * float(beta)


def adapt_ediff(e_diff: float, accepted: bool, alpha: float = 1.05) -> float:
    if accepted:
        return float(e_diff) / float(alpha)
    return float(e_diff) * float(alpha)


def _set_temperature(atoms: Atoms, temperature_k: float, rng: np.random.Generator) -> None:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    Stationary(atoms)
    current = atoms.get_temperature()
    if current > 1e-8:
        atoms.set_momenta(atoms.get_momenta() * np.sqrt(temperature_k / current))


def _finite_atoms(atoms: Atoms) -> bool:
    pos = atoms.get_positions()
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    return bool(np.isfinite(pos).all() and np.isfinite(energy) and np.isfinite(forces).all())


def _relax(atoms: Atoms, fmax: float, relax_steps: int) -> float:
    start_pos = atoms.get_positions().copy()
    opt = LBFGS(atoms, logfile=None, maxstep=0.05)
    opt.run(fmax=fmax, steps=relax_steps)
    if not _finite_atoms(atoms):
        atoms.set_positions(start_pos)
        if not _finite_atoms(atoms):
            raise RuntimeError("DenSNet energy/forces became non-finite during relaxation")
    return float(atoms.get_potential_energy())


def run_minima_hopping(
    atoms: Atoms,
    *,
    n_escapes: int = 1,
    escape_steps: int = 2000,
    timestep_fs: float = 0.5,
    temperature_k: float = 1000.0,
    e_diff_ev: float = 2.0,
    fmax: float = 1e-4,
    relax_steps: int = 1000,
    rmsd_threshold: float = 0.1,
    alpha: float = 1.05,
    beta: float = 1.05,
    seed: int = 0,
    start_step: int = 0,
    history: list[dict[str, Any]] | None = None,
    on_step=None,
) -> dict[str, Any]:
    """Run ``n_escapes`` MH cycles. Energies are whatever the calculator reports."""
    rng = np.random.default_rng(seed)
    if history:
        current_pos = np.asarray(history[-1]["positions"], dtype=float).copy()
        current_energy = float(history[-1]["energy"])
        atoms.set_positions(current_pos)
        log: list[dict[str, Any]] = []
        t_k = float(temperature_k)
        e_diff = float(e_diff_ev)
    else:
        energy = _relax(atoms, fmax=fmax, relax_steps=relax_steps)
        current_pos = atoms.get_positions().copy()
        current_energy = energy
        history = [
            {
                "positions": current_pos.copy(),
                "energy": current_energy,
                "step": 0,
            }
        ]
        log = []
        t_k = float(temperature_k)
        e_diff = float(e_diff_ev)
        if on_step is not None:
            on_step(0, history, log, t_k, e_diff)

    for step in range(start_step + 1, n_escapes + 1):
        atoms.set_positions(current_pos)
        _set_temperature(atoms, t_k, rng)
        dyn = VelocityVerlet(atoms, timestep=timestep_fs * ASE_FS)
        max_pe = float(atoms.get_potential_energy())
        exploded = False
        for _ in range(escape_steps):
            dyn.run(1)
            if not np.isfinite(atoms.get_positions()).all():
                exploded = True
                break
            pe = float(atoms.get_potential_energy())
            if not np.isfinite(pe):
                exploded = True
                break
            max_pe = max(max_pe, pe)
        if exploded:
            atoms.set_positions(current_pos)
            energy_new = current_energy
            pos_new = current_pos.copy()
            is_current = True
            in_history = True
            accepted = False
            found_new = False
            t_k = adapt_temperature(t_k, found_new=False, beta=beta)
            e_diff = adapt_ediff(e_diff, accepted=False, alpha=alpha)
            log.append(
                {
                    "step": step,
                    "accepted": False,
                    "found_new": False,
                    "is_current": True,
                    "energy": energy_new,
                    "temperature_k": t_k,
                    "e_diff_ev": e_diff,
                    "max_pe_escape": max_pe,
                    "exploded": True,
                }
            )
            if on_step is not None:
                on_step(step, history, log, t_k, e_diff)
            continue
        energy_new = _relax(atoms, fmax=fmax, relax_steps=relax_steps)
        pos_new = atoms.get_positions().copy()
        is_current = minima_identical(pos_new, current_pos, rmsd_threshold)
        in_history = any(
            minima_identical(pos_new, row["positions"], rmsd_threshold) for row in history
        )
        accepted = decide_accept(energy_new, current_energy, is_current, e_diff)
        found_new = accepted and not in_history
        if found_new:
            history.append({"positions": pos_new.copy(), "energy": energy_new, "step": step})
        if accepted:
            current_pos = pos_new
            current_energy = energy_new
        t_k = adapt_temperature(t_k, found_new=found_new, beta=beta)
        e_diff = adapt_ediff(e_diff, accepted=accepted, alpha=alpha)
        log.append(
            {
                "step": step,
                "accepted": accepted,
                "found_new": found_new,
                "is_current": is_current,
                "energy": energy_new,
                "temperature_k": t_k,
                "e_diff_ev": e_diff,
                "max_pe_escape": max_pe,
            }
        )
        if on_step is not None:
            on_step(step, history, log, t_k, e_diff)

    return {
        "n_unique": len(history),
        "minima": history,
        "log": log,
        "final_temperature_k": t_k,
        "final_e_diff_ev": e_diff,
        "kB_ev": float(kB),
    }
