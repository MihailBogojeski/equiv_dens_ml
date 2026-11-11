#!/usr/bin/env python3
"""Inter-ring S–Cα–Cα–S dihedrals for α-oligothiophene.

This is the SO3krates Ramachandran analogue: two backbone torsions
(θ12, θ23) on terthiophene, compared to the training cloud rather than
to a single RMSD cutoff.

Usage:
  python scripts/revision/thiophene/thiophene_collective_variables.py \\
    --npy datasets/thiophene_all_train_d4.npy --n-atoms 23 \\
    --out results/revision/thiophene_mh/train_cv.json
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/thiophene/thiophene_collective_variables.py'


import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols, covalent_radii

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)
_REPO_ROOT = repo_root()

BOND_TOLERANCE = 0.45


def wrap_deg(angle: float) -> float:
    """Map an angle in degrees to (-180, 180]."""
    return float((np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0)


def wrap_deg_array(angles: np.ndarray) -> np.ndarray:
    return (np.asarray(angles, dtype=float) + 180.0) % 360.0 - 180.0


def fold_abs_deg(angles: np.ndarray) -> np.ndarray:
    """Fold signed dihedrals to [0, 180].

    +170° and −170° are the same anti well; 10° and −10° are the same syn well.
    """
    return np.abs(wrap_deg_array(angles))


def fold_cv_pair(cv: np.ndarray) -> np.ndarray:
    """Fold a (N, 2) or (2,) CV array onto [0, 180]²."""
    arr = np.asarray(cv, dtype=float)
    if arr.size == 0:
        return arr.reshape(arr.shape)
    return fold_abs_deg(arr)


def dihedral_angle(p0, p1, p2, p3) -> float:
    """Signed dihedral of the quad p0–p1–p2–p3, in degrees."""
    b0 = -1.0 * (np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))
    b1 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    b2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    norm = np.linalg.norm(b1)
    if norm < 1e-12:
        return 0.0
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(-np.degrees(np.arctan2(y, x)))


def _atomic_numbers(symbols: list[str]) -> np.ndarray:
    from ase.data import atomic_numbers

    return np.asarray([atomic_numbers[s] for s in symbols], dtype=int)


def covalent_bonds(positions: np.ndarray, symbols: list[str]) -> list[tuple[int, int]]:
    z = _atomic_numbers(symbols)
    pos = np.asarray(positions, dtype=float)
    n = len(symbols)
    bonds: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = covalent_radii[int(z[i])] + covalent_radii[int(z[j])] + BOND_TOLERANCE
            if np.linalg.norm(pos[i] - pos[j]) < cutoff:
                bonds.append((i, j))
    return bonds


def _adjacency(n: int, bonds: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _heavy_components(
    adj: list[list[int]],
    symbols: list[str],
    forbidden: set[tuple[int, int]],
) -> list[set[int]]:
    heavy = [i for i, sym in enumerate(symbols) if sym != "H"]
    seen: set[int] = set()
    comps: list[set[int]] = []
    for start in heavy:
        if start in seen:
            continue
        queue = deque([start])
        comp = {start}
        seen.add(start)
        while queue:
            node = queue.popleft()
            for neigh in adj[node]:
                if symbols[neigh] == "H":
                    continue
                edge = (min(node, neigh), max(node, neigh))
                if edge in forbidden:
                    continue
                if neigh not in seen:
                    seen.add(neigh)
                    comp.add(neigh)
                    queue.append(neigh)
        comps.append(comp)
    return comps


def inter_ring_cc_bonds(
    positions: np.ndarray,
    symbols: list[str],
) -> list[tuple[int, int]]:
    """C–C bonds whose removal splits the heavy graph into ≥2 S-containing pieces."""
    bonds = covalent_bonds(positions, symbols)
    adj = _adjacency(len(symbols), bonds)
    inter: list[tuple[int, int]] = []
    for i, j in bonds:
        if symbols[i] != "C" or symbols[j] != "C":
            continue
        comps = _heavy_components(adj, symbols, forbidden={(min(i, j), max(i, j))})
        n_s = sum(1 for comp in comps if any(symbols[k] == "S" for k in comp))
        if n_s >= 2:
            inter.append((i, j))
    return inter


def inter_ring_dihedral_quads(
    positions: np.ndarray,
    symbols: list[str],
) -> list[tuple[int, int, int, int]]:
    """Return (S, Cα, Cα, S) quads, ordered along the chain (increasing C midpoint)."""
    bonds = covalent_bonds(positions, symbols)
    adj = _adjacency(len(symbols), bonds)
    pos = np.asarray(positions, dtype=float)
    links = inter_ring_cc_bonds(pos, symbols)
    quads: list[tuple[int, int, int, int, float]] = []
    for ci, cj in links:
        comps = _heavy_components(adj, symbols, forbidden={(min(ci, cj), max(ci, cj))})
        comp_i = next(comp for comp in comps if ci in comp)
        comp_j = next(comp for comp in comps if cj in comp)
        si = next(k for k in adj[ci] if symbols[k] == "S" and k in comp_i)
        sj = next(k for k in adj[cj] if symbols[k] == "S" and k in comp_j)
        midpoint = 0.5 * (pos[ci, 0] + pos[cj, 0])
        quads.append((si, ci, cj, sj, midpoint))
    quads.sort(key=lambda row: row[4])
    return [(si, ci, cj, sj) for si, ci, cj, sj, _ in quads]


def inter_ring_dihedrals(positions: np.ndarray, symbols: list[str]) -> np.ndarray:
    pos = np.asarray(positions, dtype=float)
    quads = inter_ring_dihedral_quads(pos, symbols)
    return np.asarray(
        [dihedral_angle(pos[a], pos[b], pos[c], pos[d]) for a, b, c, d in quads],
        dtype=float,
    )


def backbone_cv_pair(positions: np.ndarray, symbols: list[str]) -> tuple[float, float]:
    """Two-torsion map: (θ12, θ23) on a 3-mer; central pair on longer oligomers."""
    angles = inter_ring_dihedrals(positions, symbols)
    if len(angles) < 2:
        raise ValueError(f"need at least two inter-ring dihedrals, got {len(angles)}")
    if len(angles) == 2:
        return float(angles[0]), float(angles[1])
    mid = len(angles) // 2
    return float(angles[mid - 1]), float(angles[mid])


def terthiophene_theta12_theta23(
    positions: np.ndarray,
    symbols: list[str],
) -> tuple[float, float]:
    return backbone_cv_pair(positions, symbols)


def periodic_cv_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return wrap_deg_array(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def nearest_train_cv_distance(query: np.ndarray, train: np.ndarray) -> float:
    """Minimum circular Euclidean distance in (θ12, θ23) degrees."""
    query = np.asarray(query, dtype=float).reshape(2)
    train = np.asarray(train, dtype=float).reshape(-1, 2)
    delta = periodic_cv_delta(train, query)
    return float(np.min(np.linalg.norm(delta, axis=1)))


def cv_histogram_2d(
    theta12: np.ndarray,
    theta23: np.ndarray,
    bins: int = 18,
    range_deg: tuple[float, float] = (-180.0, 180.0),
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    hist, xedges, yedges = np.histogram2d(
        np.asarray(theta12, dtype=float),
        np.asarray(theta23, dtype=float),
        bins=bins,
        range=(range_deg, range_deg),
    )
    return hist, (xedges, yedges)


def low_density_bins(
    hist: np.ndarray,
    edges: tuple[np.ndarray, np.ndarray],
    max_count: int = 0,
) -> list[tuple[float, float, float, float]]:
    xedges, yedges = edges
    boxes: list[tuple[float, float, float, float]] = []
    nx, ny = hist.shape
    for ix in range(nx):
        for iy in range(ny):
            if hist[ix, iy] <= max_count:
                boxes.append(
                    (
                        float(xedges[ix]),
                        float(xedges[ix + 1]),
                        float(yedges[iy]),
                        float(yedges[iy + 1]),
                    )
                )
    return boxes


def symbols_from_z(atom_numbers: np.ndarray) -> list[str]:
    return [chemical_symbols[int(z)] for z in atom_numbers if int(z) > 0]


def load_npy_frames(
    path: Path,
    n_atoms: int | None = None,
) -> list[tuple[list[str], np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item() if data.ndim == 0 else data
    if not isinstance(data, dict):
        raise ValueError(f"unsupported npy in {path}")
    pos = np.asarray(data["positions"], dtype=float)
    z = np.asarray(data["atom_numbers"])
    if pos.ndim == 2:
        pos = pos[None, ...]
    if z.ndim == 1:
        z = np.broadcast_to(z, (pos.shape[0], z.shape[0]))
    frames: list[tuple[list[str], np.ndarray]] = []
    for i in range(len(pos)):
        mask = np.asarray(z[i]) > 0
        if n_atoms is not None and int(mask.sum()) != n_atoms:
            continue
        frames.append((symbols_from_z(z[i]), pos[i, mask]))
    return frames


def frames_to_cv(frames: list[tuple[list[str], np.ndarray]]) -> np.ndarray:
    rows = []
    for symbols, pos in frames:
        theta12, theta23 = terthiophene_theta12_theta23(pos, symbols)
        rows.append((theta12, theta23))
    return np.asarray(rows, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", type=Path, required=True)
    parser.add_argument("--n-atoms", type=int, default=23)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "results/revision/thiophene_mh/train_cv.json",
    )
    args = parser.parse_args()
    frames = load_npy_frames(args.npy, n_atoms=args.n_atoms)
    cv = frames_to_cv(frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "npy": str(args.npy),
        "n_atoms": args.n_atoms,
        "n_frames": int(len(cv)),
        "theta12": cv[:, 0].tolist() if len(cv) else [],
        "theta23": cv[:, 1].tolist() if len(cv) else [],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out} ({len(cv)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
