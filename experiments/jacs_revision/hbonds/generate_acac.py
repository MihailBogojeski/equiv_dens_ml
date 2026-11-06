#!/usr/bin/env python3
"""Build acetylacetone enol proton-transfer splits (Reviewer 1, R1.1)."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/hbonds/generate_acac.py'


import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
_WATER = _SCRIPT_DIR.parent / "water"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_WATER) not in sys.path:
    sys.path.insert(0, str(_WATER))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR, _WATER)

from ase import Atoms  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from experiments.jacs_revision.hbonds.common import DEFAULT_SEED, gfn2_calculator  # noqa: E402
from experiments.jacs_revision.water.generate_malonaldehyde import (  # noqa: E402
    delta_coordinate,
    proton_transfer_indices,
    relaxed_scan,
    sample_enol_basin,
)
from experiments.jacs_revision.hbonds.generator_defaults import add_quick_flags, apply_quick  # noqa: E402
from experiments.jacs_revision.hbonds.xyz_io import write_xyz  # noqa: E402

_REPO_ROOT = repo_root()
SMILES_ENOL = "CC(=O)C=C(C)O"
CHELATE_OO_MAX_A = 3.0


def _oo_distance(positions: np.ndarray, idx: dict[str, int]) -> float:
    return float(np.linalg.norm(positions[idx["donor"]] - positions[idx["acceptor"]]))


def build_acac_enol(seed: int, *, n_embed: int = 40) -> tuple[Atoms, dict[str, int]]:
    """Optimised enol minimum with the tightest intramolecular O···O contact."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    best_atoms: Atoms | None = None
    best_idx: dict[str, int] | None = None
    best_oo = float("inf")

    for trial in range(n_embed):
        mol = Chem.AddHs(Chem.MolFromSmiles(SMILES_ENOL))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed + trial
        if AllChem.EmbedMolecule(mol, params) != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = gfn2_calculator()
        BFGS(atoms, logfile=None).run(fmax=0.02, steps=200 if n_embed <= 10 else 500)
        idx = proton_transfer_indices(atoms)
        oo = _oo_distance(atoms.get_positions(), idx)
        if oo < best_oo:
            best_oo = oo
            best_atoms = atoms
            best_idx = idx

    if best_atoms is None or best_idx is None:
        raise RuntimeError("failed to embed acetylacetone enol")
    return best_atoms, best_idx


def raw_delta_path(
    atoms: Atoms,
    idx: dict[str, int],
    delta_targets: np.ndarray,
) -> tuple[list[tuple[list[str], np.ndarray]], list[dict[str, float]]]:
    """Place the transferring H on the O_donor→O_acceptor axis without reoptimisation."""
    frames: list[tuple[list[str], np.ndarray]] = []
    records: list[dict[str, float]] = []
    positions = atoms.get_positions().copy()
    donor = positions[idx["donor"]]
    acceptor = positions[idx["acceptor"]]
    oo = float(np.linalg.norm(acceptor - donor))
    axis = (acceptor - donor) / oo
    symbols = atoms.get_chemical_symbols()

    for target_delta in delta_targets:
        s = 0.5 * (oo + float(target_delta))
        s = float(np.clip(s, 0.85, oo - 0.85))
        trial = positions.copy()
        trial[idx["hydrogen"]] = donor + s * axis
        frames.append((symbols, trial))
        records.append(
            {
                "target_delta": float(target_delta),
                "delta": delta_coordinate(trial, idx),
                "r_oo": oo,
            }
        )
    return frames, records


def relaxed_delta_path(
    atoms: Atoms,
    idx: dict[str, int],
    delta_targets: np.ndarray,
) -> tuple[list[tuple[list[str], np.ndarray]], list[dict[str, float]]]:
    """Relax geometries with H placed to target delta along the O_donor→O_acceptor axis."""
    from ase.optimize import BFGS

    frames: list[tuple[list[str], np.ndarray]] = []
    records: list[dict[str, float]] = []
    work = atoms.copy()
    positions = work.get_positions()
    donor = positions[idx["donor"]]
    acceptor = positions[idx["acceptor"]]
    oo = float(np.linalg.norm(acceptor - donor))
    axis = (acceptor - donor) / oo

    for target_delta in delta_targets:
        s = 0.5 * (oo + float(target_delta))
        s = float(np.clip(s, 0.85, oo - 0.85))
        positions = work.get_positions().copy()
        positions[idx["hydrogen"]] = donor + s * axis
        work.set_positions(positions)
        work.calc = gfn2_calculator()
        BFGS(work, logfile=None).run(fmax=0.05, steps=150)
        snapshot = work.get_positions().copy()
        frames.append((work.get_chemical_symbols(), snapshot))
        records.append(
            {
                "target_delta": float(target_delta),
                "delta": delta_coordinate(snapshot, idx),
                "r_oo": oo,
                "energy_gfn2_ev": float(work.get_potential_energy()),
            }
        )
    return frames, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "datasets/revision/acac")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=80)
    parser.add_argument("--n-id", type=int, default=80)
    parser.add_argument("--train-delta-max", type=float, default=-0.40)
    parser.add_argument("--steps-between", type=int, default=50)
    parser.add_argument("--equilibration-steps", type=int, default=2000)
    parser.add_argument("--max-sweeps", type=int, default=40)
    add_quick_flags(parser)
    args = parser.parse_args()
    apply_quick(args)
    scan_points = 10 if args.quick else 25
    jitter = 2 if args.quick else 4
    args.output_dir.mkdir(parents=True, exist_ok=True)

    atoms, idx = build_acac_enol(args.seed, n_embed=5 if args.quick else 40)
    pool, n_rejected = sample_enol_basin(
        atoms,
        idx,
        args.n_train + args.n_val + args.n_id,
        300.0,
        args.steps_between,
        args.seed,
        args.train_delta_max,
        max_sweeps=args.max_sweeps,
        equilibration_steps=args.equilibration_steps,
    )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))
    train = [pool[i] for i in order[: args.n_train]]
    val = [pool[i] for i in order[args.n_train : args.n_train + args.n_val]]
    id_test = [pool[i] for i in order[args.n_train + args.n_val :]]

    manifest = {
        "seed": args.seed,
        "ood_split": "ood_proton_transfer",
        "indices": idx,
        "train_delta_max": args.train_delta_max,
        "n_rejected_crossed_barrier": n_rejected,
        "oo_distance_A": _oo_distance(atoms.get_positions(), idx),
        "splits": {},
    }
    for name, frames in (("train", train), ("val", val), ("id_test", id_test)):
        path = args.output_dir / f"{name}.xyz"
        deltas = [delta_coordinate(np.asarray(p), idx) for _, p in frames]
        write_xyz(path, frames, [f"acac {name} delta={d:+.3f}" for d in deltas])
        manifest["splits"][name] = {"xyz": str(path), "n_frames": len(frames)}

    oo = _oo_distance(atoms.get_positions(), idx)
    if oo <= CHELATE_OO_MAX_A:
        targets = np.linspace(0.98, 1.60, scan_points)
        scan_frames, scan_records = relaxed_scan(atoms, idx, targets)
    else:
        delta_targets = np.linspace(-0.95, 0.95, scan_points)
        scan_frames, scan_records = raw_delta_path(atoms, idx, delta_targets)
    frames, comments = [], []
    for (symbols, positions), record in zip(scan_frames, scan_records):
        frames.append((symbols, positions))
        comments.append(f"acac scan delta={record['delta']:+.3f}")
        for _ in range(jitter):
            jittered = positions + rng.normal(0.0, 0.03, size=positions.shape)
            frames.append((symbols, jittered))
            comments.append("acac scan-jitter")
    ood_path = args.output_dir / "ood_proton_transfer.xyz"
    write_xyz(ood_path, frames, comments)
    manifest["splits"]["ood_proton_transfer"] = {"xyz": str(ood_path), "n_frames": len(frames)}

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
