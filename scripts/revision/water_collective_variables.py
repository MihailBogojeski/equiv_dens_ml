#!/usr/bin/env python3
"""Collective variables describing a water configuration.

Asserting that a test set is out of distribution is not evidence, and the
reviewers said as much. These are the order parameters the water literature
actually uses to tell one phase from another, so a claim of non-overlap can be
made in variables a reader already knows how to interpret rather than in the
distance metric of whatever descriptor the model happens to use.

Per frame:
  q             tetrahedral order (Errington-Debenedetti); 0 for an ideal gas,
                1 for a perfect tetrahedral network such as ice
  n_hbond       hydrogen bonds per molecule, geometric criterion
  ring_6_frac   fraction of hydrogen-bond rings that are six-membered; ice Ih is
                dominated by them, small gas-phase motifs are not
  local_density mass density within the cluster's own radius of gyration
  r_gyration    size of the oxygen framework
  surface_frac  fraction of molecules with fewer than four O neighbours
  mean_nn_oo    mean nearest-neighbour O-O distance

Usage:
  python scripts/revision/water_collective_variables.py \\
    --xyz datasets/revision/water_clusters/train.xyz --label train \\
    --out results/revision/cv/train.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

M_WATER_G = 18.01528 / 6.02214076e23

# Standard geometric hydrogen-bond definition for water.
HB_OO_MAX = 3.5
HB_OH_MIN = 1.2
HB_OH_MAX = 2.5
HB_ANGLE_MAX_DEG = 30.0

CV_NAMES = (
    "q_tetrahedral",
    "n_hbond",
    "ring_6_frac",
    "local_density",
    "r_gyration",
    "surface_frac",
    "mean_nn_oo",
    "n_water",
)


def read_xyz_frames(path: Path) -> list[tuple[np.ndarray, np.ndarray, str]]:
    from ase.data import atomic_numbers

    lines = Path(path).read_text().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        nat = int(lines[i].strip())
        comment = lines[i + 1].strip() if i + 1 < len(lines) else ""
        z, xyz = [], []
        for row in lines[i + 2 : i + 2 + nat]:
            parts = row.split()
            sym = parts[0]
            z.append(int(sym) if sym.isdigit() else atomic_numbers[sym[0].upper() + sym[1:].lower()])
            xyz.append([float(v) for v in parts[1:4]])
        frames.append((np.asarray(z, dtype=int), np.asarray(xyz, dtype=float), comment))
        i += 2 + nat
    return frames


def tetrahedral_order(oxygens: np.ndarray) -> float:
    """Errington-Debenedetti q averaged over molecules with four neighbours.

    q = 1 - (3/8) sum_{j<k} (cos(psi_jkl) + 1/3)^2 over the four nearest
    neighbours. Molecules with fewer than four neighbours are skipped rather
    than padded, because on a small cluster most molecules are surface and
    padding them would wash out the very contrast this is measuring.
    """
    n = len(oxygens)
    if n < 5:
        return float("nan")
    dist = np.linalg.norm(oxygens[:, None] - oxygens[None], axis=-1)
    np.fill_diagonal(dist, np.inf)
    values = []
    for i in range(n):
        order = np.argsort(dist[i])
        neighbours = [j for j in order[:4] if dist[i, j] <= HB_OO_MAX]
        if len(neighbours) < 4:
            continue
        total = 0.0
        for a in range(3):
            for b in range(a + 1, 4):
                v1 = oxygens[neighbours[a]] - oxygens[i]
                v2 = oxygens[neighbours[b]] - oxygens[i]
                cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                total += (cos + 1.0 / 3.0) ** 2
        values.append(1.0 - 3.0 / 8.0 * total)
    return float(np.mean(values)) if values else float("nan")


def hydrogen_bonds(oxygens: np.ndarray, hydrogens: np.ndarray, owner: np.ndarray) -> list[tuple[int, int]]:
    """Donor-acceptor pairs under the standard geometric criterion."""
    bonds = []
    for h_idx, h in enumerate(hydrogens):
        donor = int(owner[h_idx])
        for acceptor in range(len(oxygens)):
            if acceptor == donor:
                continue
            r_oo = float(np.linalg.norm(oxygens[acceptor] - oxygens[donor]))
            if r_oo > HB_OO_MAX:
                continue
            r_ha = float(np.linalg.norm(oxygens[acceptor] - h))
            if not (HB_OH_MIN <= r_ha <= HB_OH_MAX):
                continue
            v1 = h - oxygens[donor]
            v2 = oxygens[acceptor] - oxygens[donor]
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) <= HB_ANGLE_MAX_DEG:
                bonds.append((donor, acceptor))
    return bonds


def ring_statistics(n_waters: int, bonds: list[tuple[int, int]], max_ring: int = 8) -> dict[str, float]:
    """Fraction of small hydrogen-bond rings that are six-membered."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n_waters))
    graph.add_edges_from(bonds)
    try:
        cycles = nx.minimum_cycle_basis(graph)
    except Exception:
        cycles = []
    sizes = [len(c) for c in cycles if len(c) <= max_ring]
    if not sizes:
        return {"ring_6_frac": 0.0, "n_rings": 0, "mean_ring_size": float("nan")}
    return {
        "ring_6_frac": float(sum(1 for s in sizes if s == 6) / len(sizes)),
        "n_rings": int(len(sizes)),
        "mean_ring_size": float(np.mean(sizes)),
    }


def assign_owner(oxygens: np.ndarray, hydrogens: np.ndarray) -> np.ndarray:
    """Nearest oxygen for each hydrogen (covalent partner)."""
    dist = np.linalg.norm(hydrogens[:, None] - oxygens[None], axis=-1)
    return np.argmin(dist, axis=1)


def frame_cvs(z: np.ndarray, xyz: np.ndarray) -> dict[str, float]:
    oxygens = xyz[z == 8]
    hydrogens = xyz[z == 1]
    n = len(oxygens)
    if n == 0:
        return {name: float("nan") for name in CV_NAMES}

    centre = oxygens.mean(axis=0)
    r_gyr = float(np.sqrt(((oxygens - centre) ** 2).sum(axis=1).mean()))

    dist = np.linalg.norm(oxygens[:, None] - oxygens[None], axis=-1)
    np.fill_diagonal(dist, np.inf)
    mean_nn = float(np.min(dist, axis=1).mean()) if n > 1 else float("nan")
    coordination = (dist <= HB_OO_MAX).sum(axis=1)
    surface_frac = float((coordination < 4).mean())

    # Density inside the sphere of the radius of gyration, which is a size
    # invariant way to say "how tightly packed is this cluster".
    volume_a3 = 4.0 / 3.0 * math.pi * max(r_gyr, 1e-6) ** 3
    local_density = float(n * M_WATER_G / (volume_a3 * 1e-24)) if volume_a3 > 0 else float("nan")

    owner = assign_owner(oxygens, hydrogens) if len(hydrogens) else np.zeros(0, dtype=int)
    bonds = hydrogen_bonds(oxygens, hydrogens, owner) if len(hydrogens) else []
    rings = ring_statistics(n, bonds)

    return {
        "q_tetrahedral": tetrahedral_order(oxygens),
        "n_hbond": float(len(bonds) / n),
        "ring_6_frac": rings["ring_6_frac"],
        "n_rings": rings["n_rings"],
        "mean_ring_size": rings["mean_ring_size"],
        "local_density": local_density,
        "r_gyration": r_gyr,
        "surface_frac": surface_frac,
        "mean_nn_oo": mean_nn,
        "n_water": float(n),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    frames = read_xyz_frames(args.xyz)
    if args.max_frames:
        frames = frames[: args.max_frames]

    rows = []
    for index, (z, xyz, comment) in enumerate(frames):
        cvs = frame_cvs(z, xyz)
        cvs["index"] = index
        cvs["comment"] = comment
        rows.append(cvs)
        if (index + 1) % 250 == 0:
            print(f"  {index + 1}/{len(frames)}", flush=True)

    summary = {}
    for name in CV_NAMES:
        values = np.array([r[name] for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            summary[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "p01": float(np.percentile(values, 1)),
                "p99": float(np.percentile(values, 99)),
                "n_finite": int(len(values)),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"label": args.label, "xyz": str(args.xyz), "n_frames": len(rows), "summary": summary, "frames": rows},
            indent=2,
        )
    )
    print(f"wrote {args.out} ({len(rows)} frames)")
    for name in ("q_tetrahedral", "n_hbond", "ring_6_frac", "local_density", "r_gyration"):
        if name in summary:
            s = summary[name]
            print(f"  {name:16s} mean={s['mean']:8.3f} std={s['std']:7.3f} [{s['min']:.3f}, {s['max']:.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
