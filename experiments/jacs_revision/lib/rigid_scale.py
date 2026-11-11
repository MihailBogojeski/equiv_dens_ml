"""Rigid-monomer intermolecular scaling (OMol25 Appendix H.6).

Scale the vector from the cluster centre of mass to each monomer COM.
Intramolecular geometry is left untouched. The short-range / long-range
split is at 6 A, matching OMol25 and the default water-model cutoff.
"""

from __future__ import annotations

import math

import numpy as np

COVALENT_OVERLAP_FACTOR = 0.55
SR_LR_CUTOFF_A = 6.0

# Covalent radii (A), same values ASE uses for H and O.
_COVALENT_RADII = {
    "H": 0.31,
    "O": 0.66,
    "C": 0.76,
    "N": 0.71,
    "S": 1.05,
}


def split_water_molecules(symbols: list[str], coords: np.ndarray) -> list[np.ndarray]:
    """Group an O,H,H-ordered frame into per-molecule coordinate blocks."""
    mols = []
    i = 0
    while i < len(symbols):
        if symbols[i] != "O":
            raise ValueError(f"expected O at position {i}, found {symbols[i]}")
        mols.append(np.asarray(coords[i : i + 3], dtype=float).copy())
        i += 3
    return mols


def molecules_to_water_frame(mols: list[np.ndarray]) -> tuple[list[str], np.ndarray]:
    symbols: list[str] = []
    rows: list[np.ndarray] = []
    for mol in mols:
        symbols.extend(["O", "H", "H"])
        rows.append(np.asarray(mol, dtype=float))
    return symbols, np.vstack(rows)


def scale_rigid_coms(
    mols: list[np.ndarray],
    factor: float,
    centre: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Move rigid monomers so each COM-to-cluster-COM vector is scaled by `factor`."""
    stacked = np.vstack(mols)
    if centre is None:
        centre = stacked.mean(axis=0)
    out = []
    for mol in mols:
        com = mol.mean(axis=0)
        out.append(mol - com + (centre + (com - centre) * factor))
    return out


def monomer_internal_geometry(mol: np.ndarray) -> tuple[np.ndarray, float]:
    """OH bond lengths (A) and H-O-H angle (degrees) for one water."""
    oxygen = mol[0]
    oh = np.array([np.linalg.norm(mol[1] - oxygen), np.linalg.norm(mol[2] - oxygen)])
    v1 = mol[1] - oxygen
    v2 = mol[2] - oxygen
    cos_theta = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    hoh = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))
    return oh, hoh


def min_intermolecular_oo(mols: list[np.ndarray]) -> float:
    """Shortest oxygen-oxygen distance between distinct monomers (A)."""
    oxygens = np.array([m[0] for m in mols])
    if len(oxygens) < 2:
        return float("inf")
    dist = np.linalg.norm(oxygens[:, None] - oxygens[None], axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def covalent_overlap_pairs(
    symbols: list[str],
    coords: np.ndarray,
    factor: float = COVALENT_OVERLAP_FACTOR,
) -> tuple[bool, list[tuple[int, int, float]]]:
    """Pairs closer than `factor` times the sum of covalent radii.

    Bonded OH (~0.96 A) is above 0.55*(r_H + r_O) and is not flagged.
    """
    coords = np.asarray(coords, dtype=float)
    pairs: list[tuple[int, int, float]] = []
    n = len(symbols)
    for i in range(n):
        ri = _COVALENT_RADII.get(symbols[i])
        if ri is None:
            continue
        for j in range(i + 1, n):
            rj = _COVALENT_RADII.get(symbols[j])
            if rj is None:
                continue
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < factor * (ri + rj):
                pairs.append((i, j, dist))
    return bool(pairs), pairs


def sr_lr_label(min_oo: float, cutoff: float = SR_LR_CUTOFF_A) -> str:
    """Short-range if min intermolecular O-O is at or inside `cutoff` A."""
    return "SR" if min_oo <= cutoff else "LR"


def omol25_scale_factors() -> list[float]:
    """2 compressive, 10 short-range expansive, 10 long-range expansive.

    OMol25 samples randomly inside each bin. A fixed grid with the same
    counts and endpoints is used here so the scan is reproducible.
    The SR/LR boundary 1.8 A-scale appears once, as the last SR point.
    """
    compressive = [0.70, 0.95]
    short_range = np.linspace(1.05, 1.80, 10).tolist()
    long_range = np.linspace(1.92, 3.00, 10).tolist()
    return [float(x) for x in compressive + short_range + long_range]
