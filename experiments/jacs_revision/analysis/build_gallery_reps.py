#!/usr/bin/env python3
"""Write a few chemically intact frames for the reviewer-structure gallery.

The August --quick XYZ trees fused the NMA dimer, left AcAc as an open enol
(O···O ~4.2 Å), and jittered Phe-Gly-Gly apart. Those files stay as the
campaign trees; this script only writes display frames.
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/build_gallery_reps.py'


import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

_HERE = _LEGACY_FILE.resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from experiments.jacs_revision.analysis.render_dataset_structures import Frame, heavy_min_distance, write_xyz

_REPO = _LEGACY_FILE.resolve().parents[3]
OUT = _REPO / "results" / "revision" / "gallery_reps"


def _embed(smiles: str, seed: int) -> Atoms:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"embed failed for {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return Atoms(symbols=symbols, positions=pos)


def _write(name: str, atoms: Atoms, comment: str) -> Path:
    dest = OUT / f"{name}.xyz"
    frame = Frame(list(atoms.get_chemical_symbols()), atoms.get_positions().tolist(), comment)
    if heavy_min_distance(frame) < 1.15:
        raise RuntimeError(f"{name} still has a heavy-atom clash")
    write_xyz(dest, frame)
    return dest


def _rotation_onto(src: np.ndarray, dest: np.ndarray) -> np.ndarray:
    a = src / np.linalg.norm(src)
    b = dest / np.linalg.norm(dest)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999:
        return -np.eye(3)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def nma_dimer(seed: int = 7) -> Path:
    """Linear amide H-bond: O···H–N along +x, monomers otherwise untouched."""
    mono = _embed("CC(=O)NC", seed)
    partner = _embed("CC(=O)NC", seed + 3)
    p1 = mono.get_positions().copy()
    p2 = partner.get_positions().copy()
    s1 = list(mono.get_chemical_symbols())
    s2 = list(partner.get_chemical_symbols())
    o1 = next(i for i, s in enumerate(s1) if s == "O")
    c1 = min((i for i, s in enumerate(s1) if s == "C"), key=lambda i: np.linalg.norm(p1[i] - p1[o1]))
    n2 = next(i for i, s in enumerate(s2) if s == "N")
    h2 = next(
        i for i, s in enumerate(s2)
        if s == "H" and float(np.linalg.norm(p2[i] - p2[n2])) < 1.2
    )
    p1 = p1 - p1[o1]
    p1 = p1 @ _rotation_onto(p1[c1], np.array([-1.0, 0.0, 0.0])).T
    p2 = p2 - p2[h2]
    p2 = p2 @ _rotation_onto(p2[n2] - p2[h2], np.array([1.0, 0.0, 0.0])).T
    p2 = p2 + np.array([1.90, 0.0, 0.0])
    heavy1 = [i for i, s in enumerate(s1) if s != "H"]
    heavy2 = [i for i, s in enumerate(s2) if s != "H"]
    sep = min(float(np.linalg.norm(p1[i] - p2[j])) for i in heavy1 for j in heavy2)
    if sep < 1.70:
        raise RuntimeError(f"NMA dimer still clashing (min heavy inter={sep:.2f} Å)")
    dimer = Atoms(symbols=s1 + s2, positions=np.vstack([p1, p2]))
    return _write("nma_dimer", dimer, "nma_dimer gallery H-bond")


def acac_enol(seed: int = 11) -> Path:
    best = None
    best_oo = 99.0
    for offset in range(40):
        atoms = _embed("CC(=O)C=C(C)O", seed + offset)
        pos = atoms.get_positions()
        o_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"]
        if len(o_idx) != 2:
            continue
        oo = float(np.linalg.norm(pos[o_idx[0]] - pos[o_idx[1]]))
        if oo < best_oo:
            best_oo = oo
            best = atoms
        if oo <= 2.80:
            break
    if best is None or best_oo > 2.80:
        raise RuntimeError(f"no chelate AcAc enol (best O···O={best_oo:.2f} Å)")
    dest = _write("acac", best, f"acac gallery enol r_oo={best_oo:.3f}")
    pos = best.get_positions().copy()
    symbols = list(best.get_chemical_symbols())
    o_idx = [i for i, s in enumerate(symbols) if s == "O"]
    h_idx = min(
        (i for i, s in enumerate(symbols) if s == "H"),
        key=lambda i: min(np.linalg.norm(pos[i] - pos[o]) for o in o_idx),
    )
    mid = 0.5 * (pos[o_idx[0]] + pos[o_idx[1]])
    pos[h_idx] = mid
    scan = Atoms(symbols=symbols, positions=pos)
    _write("acac_scan", scan, "acac gallery scan delta=0.000")
    return dest


def pgg_tripeptide(seed: int = 3) -> Path:
    atoms = _embed("N[C@@H](Cc1ccccc1)C(=O)NCC(=O)O", seed)
    return _write("pgg", atoms, "pgg gallery Phe-Gly-Gly")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (nma_dimer(), acac_enol(), pgg_tripeptide()):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
