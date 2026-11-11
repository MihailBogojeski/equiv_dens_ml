"""Mass-weighted normal-mode displacements and projections.

xTB (or any calculator) only supplies the Hessian eigenvectors. This
module scales a mode to a requested Cartesian RMSD, projects a
displacement onto the retained subspace, and Kabsch-aligns a frame to
the equilibrium used to define the modes. Atom order is assumed fixed;
the alignment does not permute atoms.
"""

from __future__ import annotations

import numpy as np

from experiments.jacs_revision.lib.rigid_scale import COVALENT_OVERLAP_FACTOR, covalent_overlap_pairs

BOND_STRETCH_FACTOR = 1.5


def drop_rigid_modes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_drop: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the `n_drop` lowest-frequency columns (translations/rotations)."""
    order = np.argsort(np.asarray(eigenvalues, dtype=float))
    keep = order[n_drop:]
    return np.asarray(eigenvalues, dtype=float)[keep], np.asarray(eigenvectors, dtype=float)[:, keep]


def cartesian_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def displace_along_mode(
    positions: np.ndarray,
    mode: np.ndarray,
    rmsd: float,
) -> np.ndarray:
    """Add `mode` scaled so the Cartesian RMSD equals `|rmsd|`."""
    positions = np.asarray(positions, dtype=float)
    mode = np.asarray(mode, dtype=float).reshape(positions.shape)
    current = float(np.sqrt(np.mean(np.sum(mode * mode, axis=1))))
    if current < 1e-18:
        raise ValueError("mode has zero Cartesian RMS amplitude")
    return positions + mode * (rmsd / current)


def mass_weight_sqrt(masses: np.ndarray) -> np.ndarray:
    masses = np.asarray(masses, dtype=float)
    return np.repeat(np.sqrt(masses), 3)


def project_modes(
    displacement: np.ndarray,
    modes: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """q_k = e_k^T M^{1/2} (R - R_eq). `modes` are 3N x n_modes columns."""
    delta = np.asarray(displacement, dtype=float).reshape(-1)
    mw = mass_weight_sqrt(masses)
    weighted = mw * delta
    return np.asarray(modes, dtype=float).T @ weighted


def align_kabsch(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate/translate `mobile` onto `reference` without inversion or permutation."""
    p = np.asarray(mobile, dtype=float)
    q = np.asarray(reference, dtype=float)
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    h = p_c.T @ q_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    if d == 0.0:
        d = 1.0
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return p_c @ rot.T + q.mean(axis=0)


def nearest_mode_distance(q: np.ndarray, cloud: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    cloud = np.asarray(cloud, dtype=float)
    if cloud.size == 0:
        return float("nan")
    return float(np.min(np.linalg.norm(cloud - q, axis=1)))


def equilibrium_bond_pairs(positions: np.ndarray, cutoff: float = 1.8) -> list[tuple[int, int, float]]:
    """Pairs closer than `cutoff` A at equilibrium, treated as covalent bonds."""
    pos = np.asarray(positions, dtype=float)
    pairs = []
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if dist < cutoff:
                pairs.append((i, j, dist))
    return pairs


def validity_tags(
    symbols: list[str],
    positions: np.ndarray,
    equilibrium: np.ndarray,
    bond_pairs: list[tuple[int, int, float]] | None = None,
) -> dict:
    """Tag stretched bonds and covalent overlaps; frames are not dropped."""
    pos = np.asarray(positions, dtype=float)
    eq = np.asarray(equilibrium, dtype=float)
    if bond_pairs is None:
        bond_pairs = equilibrium_bond_pairs(eq)
    stretched = []
    for i, j, r0 in bond_pairs:
        r = float(np.linalg.norm(pos[i] - pos[j]))
        if r > BOND_STRETCH_FACTOR * r0:
            stretched.append({"i": i, "j": j, "r": r, "r0": r0})
    clash, pairs = covalent_overlap_pairs(symbols, pos, factor=COVALENT_OVERLAP_FACTOR)
    return {
        "bond_stretched": bool(stretched),
        "stretched_bonds": stretched,
        "clash": clash,
        "clash_pairs": [{"i": i, "j": j, "dist": d} for i, j, d in pairs],
    }


def displacement_metrics(positions: np.ndarray, equilibrium: np.ndarray) -> dict:
    pos = np.asarray(positions, dtype=float)
    eq = np.asarray(equilibrium, dtype=float)
    delta = pos - eq
    per_atom = np.linalg.norm(delta, axis=1)
    return {
        "rmsd": cartesian_rmsd(pos, eq),
        "disp_norm": float(np.linalg.norm(delta)),
        "max_atom_travel": float(per_atom.max()) if len(per_atom) else 0.0,
    }
