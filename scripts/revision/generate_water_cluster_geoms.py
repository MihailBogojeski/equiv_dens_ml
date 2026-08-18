#!/usr/bin/env python3
"""Build (H2O)_n cluster geometries for the JACS revision.

Stdlib only. Places oxygens on known motifs (cycle, prism, cube, prisms),
attaches hydrogens with r(OH)=0.9572 Angstrom and aHOH=104.52 deg, then
adds Gaussian thermal noise with a simple steric filter.

Output: multi-frame XYZ files plus a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROH = 0.9572
HOH = math.radians(104.52)
OO = 2.76


def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _add(a, b, s=1.0):
    return (a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return _scale(a, 1.0 / n)


def _rotate_about(axis, angle, vec):
    axis = _unit(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    return _add(
        _add(_scale(vec, c), _cross(axis, vec), s),
        axis,
        _dot(axis, vec) * (1.0 - c),
    )


def _water_hydrogens(oxygen, hbond_dir, up):
    """One H along hbond_dir, the second at the HOH angle in the plane of `up`."""
    e1 = _unit(hbond_dir)
    up_u = _unit(up)
    # If up is parallel to e1, pick another axis.
    if abs(_dot(e1, up_u)) > 0.95:
        up_u = (1.0, 0.0, 0.0) if abs(e1[0]) < 0.9 else (0.0, 1.0, 0.0)
    e2 = _unit(_add(up_u, e1, -_dot(up_u, e1)))
    h1 = _add(oxygen, e1, ROH)
    e_h2 = _add(_scale(e1, math.cos(HOH)), e2, math.sin(HOH))
    h2 = _add(oxygen, _unit(e_h2), ROH)
    return h1, h2


def _cycle_oxygens(n, oo=OO):
    radius = oo / (2.0 * math.sin(math.pi / n))
    return [
        (radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n), 0.0)
        for i in range(n)
    ]


def _cycle_cluster(n, pucker=0.15):
    oxygens = _cycle_oxygens(n)
    atoms = []
    for i, o in enumerate(oxygens):
        nxt = oxygens[(i + 1) % n]
        z = 1.0 if i % 2 == 0 else -1.0
        o = (o[0], o[1], z * pucker)
        h1, h2 = _water_hydrogens(o, _vec(o, nxt), (0.0, 0.0, z))
        atoms.extend([("O", o), ("H", h1), ("H", h2)])
    return atoms


def _prism_cluster(n_ring, height=2.70):
    """Two stacked n_ring cycles (hexamer prism n_ring=3; decamer n_ring=5; dodecamer n_ring=6)."""
    bottom = _cycle_oxygens(n_ring)
    top = [(x, y, height) for x, y, _ in bottom]
    atoms = []
    rings = (bottom, top)
    for layer, oxygens in enumerate(rings):
        for i, o in enumerate(oxygens):
            # Alternate H-bonds around the ring vs the vertical edge.
            if (i + layer) % 2 == 0:
                target = oxygens[(i + 1) % n_ring]
            else:
                other = rings[1 - layer][i]
                target = other
            up = (0.0, 0.0, 1.0 if layer == 0 else -1.0)
            h1, h2 = _water_hydrogens(o, _vec(o, target), up)
            atoms.extend([("O", o), ("H", h1), ("H", h2)])
    return atoms


def _cube_octamer(edge=2.75):
    h = edge
    corners = [
        (0, 0, 0),
        (h, 0, 0),
        (h, h, 0),
        (0, h, 0),
        (0, 0, h),
        (h, 0, h),
        (h, h, h),
        (0, h, h),
    ]
    # Each O donates along one cube edge (D2d-like pattern).
    donate_to = [1, 2, 3, 0, 7, 4, 5, 6]
    atoms = []
    center = (h / 2, h / 2, h / 2)
    for i, o in enumerate(corners):
        target = corners[donate_to[i]]
        up = _vec(center, o)
        h1, h2 = _water_hydrogens(o, _vec(o, target), up)
        atoms.extend([("O", o), ("H", h1), ("H", h2)])
    return atoms


def _book_hexamer():
    """Two fused tetramer-like rhombi (approximate book isomer)."""
    a = OO
    oxygens = [
        (0.0, 0.0, 0.0),
        (a, 0.0, 0.0),
        (1.5 * a, 0.85 * a, 0.15 * a),
        (0.5 * a, 0.85 * a, 0.15 * a),
        (a, 1.70 * a, 0.0),
        (0.0, 1.70 * a, 0.0),
    ]
    donate = [1, 2, 4, 0, 5, 3]
    atoms = []
    for i, o in enumerate(oxygens):
        h1, h2 = _water_hydrogens(o, _vec(o, oxygens[donate[i]]), (0.0, 0.0, 1.0))
        atoms.extend([("O", o), ("H", h1), ("H", h2)])
    return atoms


def _cage_hexamer():
    """Approximate cage: tetrahedral O scaffold plus two capping waters."""
    a = OO * 0.85
    oxygens = [
        (a, a, a),
        (a, -a, -a),
        (-a, a, -a),
        (-a, -a, a),
        (0.0, 0.0, 1.6 * a),
        (0.0, 0.0, -1.6 * a),
    ]
    donate = [4, 5, 4, 5, 0, 1]
    atoms = []
    for i, o in enumerate(oxygens):
        h1, h2 = _water_hydrogens(o, _vec(o, oxygens[donate[i]]), (1.0, 0.0, 0.0))
        atoms.extend([("O", o), ("H", h1), ("H", h2)])
    return atoms


MOTIFS = {
    2: {"dimer": lambda: _cycle_cluster(2, pucker=0.0)},
    3: {"cyclic": lambda: _cycle_cluster(3)},
    4: {"cyclic": lambda: _cycle_cluster(4)},
    5: {"cyclic": lambda: _cycle_cluster(5)},
    6: {
        "cyclic": lambda: _cycle_cluster(6),
        "prism": lambda: _prism_cluster(3),
        "book": _book_hexamer,
        "cage": _cage_hexamer,
    },
    8: {"cube": _cube_octamer},
    10: {"pent_prism": lambda: _prism_cluster(5, height=2.72)},
    12: {"hex_prism": lambda: _prism_cluster(6, height=2.72)},
}


def _min_pair_distance(atoms, z_i, z_j):
    best = 1e9
    for si, pi in atoms:
        if si != z_i:
            continue
        for sj, pj in atoms:
            if sj != z_j:
                continue
            if si == sj and pi == pj:
                continue
            best = min(best, _norm(_vec(pi, pj)))
    return best


def _valid(atoms):
    if _min_pair_distance(atoms, "O", "O") < 2.20:
        return False
    # Each O should have two nearby H.
    for i, (s, p) in enumerate(atoms):
        if s != "O":
            continue
        n_h = 0
        for j, (t, q) in enumerate(atoms):
            if t != "H" or i == j:
                continue
            d = _norm(_vec(p, q))
            if 0.75 < d < 1.25:
                n_h += 1
        if n_h < 2:
            return False
    return True


def _jitter(atoms, sigma, rng):
    out = []
    for s, p in atoms:
        out.append(
            (
                s,
                (
                    p[0] + rng.gauss(0.0, sigma),
                    p[1] + rng.gauss(0.0, sigma),
                    p[2] + rng.gauss(0.0, sigma),
                ),
            )
        )
    return out


def _write_xyz(path, frames, comment):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for fi, atoms in enumerate(frames):
            fh.write(f"{len(atoms)}\n")
            fh.write(f"{comment} frame={fi}\n")
            for s, p in atoms:
                fh.write(f"{s:2s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}\n")


def _sample_motifs(n, n_frames, sigmas, rng):
    motifs = MOTIFS[n]
    names = list(motifs)
    frames = []
    meta = []
    attempts = 0
    while len(frames) < n_frames and attempts < n_frames * 40:
        attempts += 1
        name = names[len(frames) % len(names)]
        base = motifs[name]()
        sigma = sigmas[len(frames) % len(sigmas)]
        trial = _jitter(base, sigma, rng)
        if _valid(trial):
            frames.append(trial)
            meta.append({"n": n, "isomer": name, "sigma": sigma})
    if len(frames) < n_frames:
        raise RuntimeError(f"Only accepted {len(frames)}/{n_frames} frames for n={n}")
    return frames, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/revision/water_clusters"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-per-size", type=int, default=250)
    parser.add_argument("--val-per-size", type=int, default=50)
    parser.add_argument("--id-test-per-size", type=int, default=50)
    parser.add_argument("--ood-per-size", type=int, default=100)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    sigmas_id = (0.04, 0.06, 0.08)
    sigmas_ood = (0.05, 0.07, 0.10)

    split_frames = {"train": [], "val": [], "id_test": [], "ood_size": []}
    split_meta = {k: [] for k in split_frames}

    minima = {}
    for n, motifs in MOTIFS.items():
        minima[str(n)] = {}
        for name, fn in motifs.items():
            atoms = fn()
            minima[str(n)][name] = [(s, p) for s, p in atoms]
            _write_xyz(out / "minima" / f"n{n}_{name}.xyz", [atoms], f"water n={n} {name} minimum")

    for n in (2, 3, 4, 5, 6):
        for split, count in (
            ("train", args.train_per_size),
            ("val", args.val_per_size),
            ("id_test", args.id_test_per_size),
        ):
            frames, meta = _sample_motifs(n, count, sigmas_id, rng)
            split_frames[split].extend(frames)
            split_meta[split].extend(meta)

    for n in (8, 10, 12):
        frames, meta = _sample_motifs(n, args.ood_per_size, sigmas_ood, rng)
        split_frames["ood_size"].extend(frames)
        split_meta["ood_size"].extend(meta)

    manifest = {"seed": args.seed, "splits": {}}
    for split, frames in split_frames.items():
        xyz = out / f"{split}.xyz"
        _write_xyz(xyz, frames, f"water clusters {split}")
        manifest["splits"][split] = {
            "xyz": str(xyz),
            "n_frames": len(frames),
            "by_n": {},
        }
        for m in split_meta[split]:
            key = str(m["n"])
            manifest["splits"][split]["by_n"].setdefault(key, 0)
            manifest["splits"][split]["by_n"][key] += 1

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
