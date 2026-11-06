"""Kabsch RMSD and bonded internals for comparing two molecular minima.

Used by ``optimize_densnet.py`` for Reviewer 2, point 11. Distances are in
angstrom; angles are in degrees.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from ase.data import atomic_numbers, covalent_radii

HEAVY_SYMBOLS = frozenset({"C", "N", "O", "S", "F", "P", "Cl", "Br", "I"})


def align_onto(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate and translate ``mobile`` onto ``reference`` (proper rotation)."""
    p = np.asarray(mobile, dtype=float)
    q = np.asarray(reference, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f"align needs matching shapes, got {p.shape} vs {q.shape}")
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    u, _, vt = np.linalg.svd(p_c.T @ q_c)
    det = np.linalg.det(vt.T @ u.T)
    reflection = 1.0 if det >= 0.0 else -1.0
    rotation = vt.T @ np.diag([1.0, 1.0, reflection]) @ u.T
    return p_c @ rotation.T + q.mean(axis=0)


def kabsch_rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    """All-atom RMSD after removing translation and a proper rotation."""
    aligned = align_onto(mobile, reference)
    q = np.asarray(reference, dtype=float)
    return float(np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=1))))


def heavy_atom_rmsd(
    coords_a: np.ndarray, coords_b: np.ndarray, symbols: list[str]
) -> float:
    """Kabsch RMSD over non-hydrogen atoms. Falls back to all atoms if needed."""
    a_all = np.asarray(coords_a, dtype=float)
    b_all = np.asarray(coords_b, dtype=float)
    if a_all.shape != b_all.shape:
        raise ValueError(f"RMSD needs matching shapes, got {a_all.shape} vs {b_all.shape}")
    mask = np.array([symbol in HEAVY_SYMBOLS for symbol in symbols], dtype=bool)
    if int(mask.sum()) < 2:
        mask = np.ones(a_all.shape[0], dtype=bool)
    return _kabsch(a_all[mask], b_all[mask])


def _kabsch(mobile: np.ndarray, reference: np.ndarray) -> float:
    return kabsch_rmsd(mobile, reference)


def pair_distance(coords: np.ndarray, pair: tuple[int, int]) -> float:
    i, j = pair
    return float(np.linalg.norm(np.asarray(coords[i], dtype=float) - np.asarray(coords[j], dtype=float)))


def angle_deg(coords: np.ndarray, triple: tuple[int, int, int]) -> float:
    i, j, k = triple
    u = np.asarray(coords[i], dtype=float) - np.asarray(coords[j], dtype=float)
    v = np.asarray(coords[k], dtype=float) - np.asarray(coords[j], dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        raise ValueError("zero-length angle side")
    cosine = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def covalent_bonds(
    symbols: list[str], coords: np.ndarray, scale: float = 1.20
) -> list[tuple[int, int]]:
    """Pairs closer than ``scale`` times the sum of covalent radii."""
    positions = np.asarray(coords, dtype=float)
    radii = np.array([covalent_radii[atomic_numbers[symbol]] for symbol in symbols], dtype=float)
    bonds: list[tuple[int, int]] = []
    n_atoms = len(symbols)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            cutoff = scale * (radii[i] + radii[j])
            if np.linalg.norm(positions[i] - positions[j]) <= cutoff:
                bonds.append((i, j))
    return bonds


_BOND_NAME_ORDER = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H")


def _bond_name(left: str, right: str) -> str:
    rank = {symbol: idx for idx, symbol in enumerate(_BOND_NAME_ORDER)}
    a, b = left, right
    if rank.get(a, 99) > rank.get(b, 99):
        a, b = b, a
    return f"{a}-{b}"


def named_bond_groups(
    symbols: list[str], bonds: list[tuple[int, int]]
) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, j in bonds:
        groups[_bond_name(symbols[i], symbols[j])].append((i, j))
    return dict(groups)


def named_angle_groups(
    symbols: list[str], bonds: list[tuple[int, int]]
) -> dict[str, list[tuple[int, int, int]]]:
    """Heavy-atom valence angles (A-B-C with A, B, C all heavy)."""
    neighbors: dict[int, list[int]] = defaultdict(list)
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
    groups: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for center, bonded in neighbors.items():
        if symbols[center] not in HEAVY_SYMBOLS:
            continue
        heavies = [idx for idx in bonded if symbols[idx] in HEAVY_SYMBOLS]
        for a in range(len(heavies)):
            for b in range(a + 1, len(heavies)):
                i, k = heavies[a], heavies[b]
                ends = sorted((symbols[i], symbols[k]))
                name = f"{ends[0]}-{symbols[center]}-{ends[1]}"
                groups[name].append((i, center, k))
    return dict(groups)


def summarize_internal_errors(
    symbols: list[str], coords_a: np.ndarray, coords_b: np.ndarray
) -> dict[str, dict[str, float]]:
    """Mean absolute bond and heavy-angle errors between two geometries."""
    bonds = covalent_bonds(symbols, coords_b)
    bond_mae: dict[str, float] = {}
    for name, pairs in named_bond_groups(symbols, bonds).items():
        deltas = [
            abs(pair_distance(coords_a, pair) - pair_distance(coords_b, pair))
            for pair in pairs
        ]
        bond_mae[name] = float(np.mean(deltas))
    angle_mae: dict[str, float] = {}
    for name, triples in named_angle_groups(symbols, bonds).items():
        deltas = [
            abs(angle_deg(coords_a, triple) - angle_deg(coords_b, triple))
            for triple in triples
        ]
        angle_mae[name] = float(np.mean(deltas))
    return {"bond_mae_ang": bond_mae, "angle_mae_deg": angle_mae}
