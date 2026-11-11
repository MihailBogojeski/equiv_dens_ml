#!/usr/bin/env python3
"""Build paracetamol train/val/id/OOD splits for IR comparison (Reviewer 3)."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/hbonds/generate_paracetamol.py'


import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)

from ase import Atoms  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from experiments.jacs_revision.hbonds.common import DEFAULT_SEED, farthest_point_indices, gfn2_calculator, langevin_frames, thermal_noise_frames  # noqa: E402
from experiments.jacs_revision.hbonds.generator_defaults import add_quick_flags, apply_quick  # noqa: E402
from experiments.jacs_revision.hbonds.xyz_io import write_xyz  # noqa: E402

_REPO_ROOT = repo_root()
SMILES = "CC(=O)NC1=CC=C(O)C=C1"


def build_minimum(seed: int, *, bfgs_steps: int = 400) -> Atoms:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    atoms = Atoms(symbols=symbols, positions=pos)
    atoms.calc = gfn2_calculator()
    BFGS(atoms, logfile=None).run(fmax=0.02, steps=bfgs_steps)
    return atoms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "datasets/revision/paracetamol")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-id", type=int, default=100)
    parser.add_argument("--n-ood", type=int, default=100)
    parser.add_argument("--pool-size", type=int, default=3000)
    parser.add_argument("--steps-between", type=int, default=20)
    parser.add_argument("--equilibration-steps", type=int, default=2000)
    parser.add_argument("--max-sweeps", type=int, default=40)
    add_quick_flags(parser)
    args = parser.parse_args()
    apply_quick(args)
    gap_a = 0.35 if args.quick else 0.25
    bfgs_steps = 80 if args.quick else 400
    if args.quick:
        args.pool_size = max(args.pool_size, 150)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = build_minimum(args.seed, bfgs_steps=bfgs_steps)
    if args.quick:
        traj, _ = thermal_noise_frames(base, args.pool_size, amplitude=0.30, seed=args.seed)
    else:
        traj, _ = langevin_frames(
            base,
            args.pool_size,
            temperature_k=300.0,
            steps_between=args.steps_between,
            seed=args.seed,
            equilibration_steps=args.equilibration_steps,
            max_sweeps=args.max_sweeps,
        )
    positions = [p for _, p in traj]
    pick = farthest_point_indices(positions, args.n_train, min_dist=gap_a)
    train = [traj[i] for i in pick]

    remaining = [traj[i] for i in range(len(traj)) if i not in pick]
    rem_pos = [p for _, p in remaining]
    val_pick = farthest_point_indices(rem_pos, args.n_val, min_dist=gap_a)
    val = [remaining[i] for i in val_pick]
    rem2 = [remaining[i] for i in range(len(remaining)) if i not in val_pick]
    id_pick = farthest_point_indices([p for _, p in rem2], args.n_id, min_dist=gap_a)
    id_test = [rem2[i] for i in id_pick]

    if args.quick:
        ood, _ = thermal_noise_frames(base, args.n_ood, amplitude=0.55, seed=args.seed + 7)
    else:
        hot, _ = langevin_frames(
            base,
            args.n_ood * 3,
            temperature_k=600.0,
            steps_between=args.steps_between,
            seed=args.seed + 7,
            equilibration_steps=args.equilibration_steps,
            max_sweeps=args.max_sweeps,
        )
        rng = np.random.default_rng(args.seed + 7)
        ood = hot[: args.n_ood]
        for i in range(len(ood)):
            syms, pos = ood[i]
            pos = pos.copy()
            pos += rng.normal(0.0, 0.15, size=pos.shape)
            ood[i] = (syms, pos)

    manifest = {"seed": args.seed, "ood_split": "ood_highT", "splits": {}}
    for name, frames in (
        ("train", train),
        ("val", val),
        ("id_test", id_test),
    ):
        path = args.output_dir / f"{name}.xyz"
        write_xyz(path, frames, [f"paracetamol {name}"] * len(frames))
        manifest["splits"][name] = {"xyz": str(path), "n_frames": len(frames)}
    ood_path = args.output_dir / "ood_highT.xyz"
    write_xyz(ood_path, ood, [f"paracetamol ood_highT"] * len(ood))
    manifest["splits"]["ood_highT"] = {"xyz": str(ood_path), "n_frames": len(ood)}

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
