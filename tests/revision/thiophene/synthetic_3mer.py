"""Minimal α-terthiophene topology for unit tests (not a DFT geometry)."""

from __future__ import annotations

import numpy as np


def build_synthetic_3mer() -> tuple[list[str], np.ndarray]:
    """Planar C12H8S3 with two α–α links. Inter-ring S–C–C–S ≈ 0°."""
    rings = []
    origins = (0.0, 3.8, 7.6)
    for origin in origins:
        rings.append(
            [
                ("S", origin + 0.0, 1.20, 0.0),
                ("C", origin - 1.20, 0.20, 0.0),
                ("C", origin - 0.80, -1.00, 0.0),
                ("C", origin + 0.80, -1.00, 0.0),
                ("C", origin + 1.20, 0.20, 0.0),
            ]
        )
    hydrogens = [
        (-1.20 - 0.95, 0.55, 0.0),
        (-0.80, -2.00, 0.0),
        (0.80, -2.00, 0.0),
        (3.00, -2.00, 0.0),
        (4.60, -2.00, 0.0),
        (6.80, -2.00, 0.0),
        (8.40, -2.00, 0.0),
        (8.80 + 0.95, 0.55, 0.0),
    ]
    symbols: list[str] = []
    pos = []
    for ring in rings:
        for sym, x, y, z in ring:
            symbols.append(sym)
            pos.append([x, y, z])
    for x, y, z in hydrogens:
        symbols.append("H")
        pos.append([x, y, z])
    return symbols, np.asarray(pos, dtype=float)


def _rotate_about_axis(
    points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(angle_deg)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rot = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
    shifted = points - origin
    return shifted @ rot.T + origin


def twist_ring(
    positions: np.ndarray,
    symbols: list[str],
    ring_index: int,
    angle_deg: float,
) -> np.ndarray:
    """Rotate one ring about its inbound inter-ring C–C axis.

    ``ring_index`` is 0, 1, or 2 in chain order. Ring 0 has no inbound link;
    twisting ring 2 changes θ23.
    """
    if ring_index <= 0:
        raise ValueError("cannot twist the first ring about an inbound link")
    pos = np.asarray(positions, dtype=float).copy()
    # Heavy atoms: rings occupy indices 0:5, 5:10, 10:15
    start = 5 * ring_index
    prev_alpha = 5 * (ring_index - 1) + 4
    this_alpha = start + 1
    origin = pos[this_alpha]
    axis = pos[this_alpha] - pos[prev_alpha]
    members = list(range(start, start + 5))
    # terminal hydrogens sit after the 15 heavy atoms
    if ring_index == 2:
        members.extend([20, 21, 22])
    elif ring_index == 1:
        members.extend([18, 19])
    pos[members] = _rotate_about_axis(pos[members], origin, axis, angle_deg)
    return pos
