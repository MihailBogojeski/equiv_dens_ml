"""Shared helpers for the H-bond / new-system geometry campaign."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/hbonds/common.py'


import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR, _SCRIPT_DIR.parent / "water")

DEFAULT_SEED = 20260826
HEAVY_ELEMENTS = frozenset({"H", "D", "T"})


def gfn2_calculator():
    """GFN2-xTB calculator for geometry placement only."""
    from tblite.ase import TBLite

    return TBLite(method="GFN2-xTB", verbosity=0)


def heavy_atom_mask(symbols: list[str]) -> np.ndarray:
    """Boolean mask: True for non-hydrogen atoms."""
    return np.array([s not in HEAVY_ELEMENTS for s in symbols], dtype=bool)


def distance(positions: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(positions[i] - positions[j]))


def angle_deg(positions: np.ndarray, i: int, j: int, k: int) -> float:
    """Angle at atom j in degrees."""
    v1 = positions[i] - positions[j]
    v2 = positions[k] - positions[j]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")
    cos_theta = np.clip(float(np.dot(v1, v2) / (n1 * n2)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def kabsch_rmsd(
    p: np.ndarray,
    q: np.ndarray,
    *,
    heavy_only: bool = True,
    z_p: np.ndarray | None = None,
    z_q: np.ndarray | None = None,
) -> float:
    """Hungarian-aligned RMSD between two structures."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    if heavy_only:
        if z_p is None or z_q is None:
            raise ValueError("atomic numbers required for heavy-only RMSD")
        p = p[z_p > 1]
        q = q[z_q > 1]
    p = p - p.mean(axis=0)
    q = q - q.mean(axis=0)
    cost = cdist(p, q)
    ri, ci = linear_sum_assignment(cost)
    p = p[ri]
    q = q[ci]
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    p_rot = p @ r.T
    return float(np.sqrt(np.mean(np.sum((p_rot - q) ** 2, axis=1))))


def nearest_train_rmsds(
    train_frames: list[tuple[list[str], np.ndarray]],
    test_frames: list[tuple[list[str], np.ndarray]],
    *,
    heavy_only: bool = True,
) -> np.ndarray:
    """Min RMSD from each test frame to the training set."""
    from ase.data import atomic_numbers

    rmsds = []
    for tsym, tpos in test_frames:
        z_t = np.array([atomic_numbers[s] for s in tsym]) if heavy_only else None
        best = float("inf")
        for psym, ppos in train_frames:
            if psym != tsym:
                continue
            z_p = np.array([atomic_numbers[s] for s in psym]) if heavy_only else None
            best = min(
                best,
                kabsch_rmsd(
                    tpos,
                    ppos,
                    heavy_only=heavy_only,
                    z_p=z_p,
                    z_q=z_t,
                ),
            )
        rmsds.append(best)
    return np.asarray(rmsds, dtype=float)


def farthest_point_indices(positions_list: list[np.ndarray], n_pick: int, min_dist: float) -> list[int]:
    """Greedy farthest-point subsample using pairwise distance vectors."""
    from scipy.spatial.distance import cdist

    if n_pick >= len(positions_list):
        return list(range(len(positions_list)))
    descs = np.stack([np.sort(cdist(p, p)[np.triu_indices(len(p), k=1)]) for p in positions_list])
    chosen = [0]
    while len(chosen) < n_pick:
        dists = cdist(descs[chosen], descs).min(axis=0)
        for idx in np.argsort(-dists):
            idx = int(idx)
            if idx in chosen:
                continue
            if all(
                float(np.linalg.norm(positions_list[idx] - positions_list[j])) >= min_dist
                for j in chosen
            ):
                chosen.append(idx)
                break
        else:
            break
    return chosen


def thermal_noise_frames(
    atoms,
    n_frames: int,
    *,
    amplitude: float,
    seed: int,
    accept=None,
    max_attempts: int | None = None,
):
    """Draw accepted frames as Gaussian displacements around a reference geometry."""
    rng = np.random.default_rng(seed)
    base = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    frames: list[tuple[list[str], np.ndarray]] = []
    n_rejected = 0
    limit = max_attempts if max_attempts is not None else max(200, 40 * n_frames)
    for _ in range(limit):
        if len(frames) >= n_frames:
            break
        pos = base + rng.normal(0.0, amplitude, size=base.shape)
        if accept is None or accept(pos):
            frames.append((symbols, pos))
        else:
            n_rejected += 1
    if len(frames) < n_frames:
        raise RuntimeError(f"only collected {len(frames)}/{n_frames} accepted thermal frames")
    return frames, n_rejected


def langevin_frames(
    atoms,
    n_frames: int,
    *,
    temperature_k: float,
    steps_between: int,
    seed: int,
    equilibration_steps: int = 2000,
    accept=None,
    max_sweeps: int | None = None,
):
    """Collect Langevin frames that pass an optional acceptance predicate."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    work = atoms.copy()
    work.calc = gfn2_calculator()
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(work, temperature_K=temperature_k, rng=rng)
    dyn = Langevin(work, 0.5 * units.fs, temperature_K=temperature_k, friction=0.02, rng=rng)
    dyn.run(equilibration_steps)

    frames: list[tuple[list[str], np.ndarray]] = []
    n_rejected = 0
    sweep_limit = max_sweeps * n_frames if max_sweeps is not None else max(40 * n_frames, n_frames * 5)
    for _ in range(sweep_limit):
        if len(frames) >= n_frames:
            break
        dyn.run(steps_between)
        positions = work.get_positions().copy()
        if accept is None or accept(positions):
            frames.append((work.get_chemical_symbols(), positions))
        else:
            n_rejected += 1
    if len(frames) < n_frames:
        raise RuntimeError(f"only collected {len(frames)}/{n_frames} accepted frames")
    return frames, n_rejected
