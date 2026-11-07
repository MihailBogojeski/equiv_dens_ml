#!/usr/bin/env python3
"""Build PGG (Phe-Gly-Gly) conformational-family splits (Reviewer 3)."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/hbonds/generate_pgg.py'


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
from experiments.jacs_revision.hbonds.common import DEFAULT_SEED, gfn2_calculator, langevin_frames, thermal_noise_frames  # noqa: E402
from experiments.jacs_revision.hbonds.generator_defaults import add_quick_flags, apply_quick  # noqa: E402
from experiments.jacs_revision.hbonds.xyz_io import write_xyz  # noqa: E402

_REPO_ROOT = repo_root()
SMILES = "N[C@@H](Cc1ccccc1)C(=O)NCC(=O)O"


def embed_minima(n_embed: int, seed: int) -> list[Atoms]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
    minima: list[Atoms] = []
    for i in range(n_embed):
        params = AllChem.ETKDGv3()
        params.randomSeed = seed + i
        code = AllChem.EmbedMolecule(mol, params)
        if code != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        atoms = Atoms(symbols=symbols, positions=pos)
        atoms.calc = gfn2_calculator()
        BFGS(atoms, logfile=None).run(fmax=0.05, steps=80)
        minima.append(atoms)
    return minima


def cluster_minima(minima: list[Atoms], rmsd_cut: float = 0.6) -> list[list[Atoms]]:
    from experiments.jacs_revision.hbonds.common import kabsch_rmsd
    from ase.data import atomic_numbers

    families: list[list[Atoms]] = []
    for atoms in minima:
        placed = False
        for fam in families:
            ref = fam[0]
            z = np.array([atomic_numbers[s] for s in ref.get_chemical_symbols()])
            rmsd = kabsch_rmsd(
                atoms.get_positions(), ref.get_positions(), heavy_only=True, z_p=z, z_q=z,
            )
            if rmsd < rmsd_cut:
                fam.append(atoms)
                placed = True
                break
        if not placed:
            families.append([atoms])
    families.sort(key=lambda f: len(f), reverse=True)
    return families


def sample_family(ref: Atoms, n_frames: int, seed: int, *, quick: bool) -> list[tuple[list[str], np.ndarray]]:
    if quick:
        frames, _ = thermal_noise_frames(ref, n_frames, amplitude=0.35, seed=seed)
        return frames
    frames, _ = langevin_frames(
        ref,
        n_frames,
        temperature_k=200.0,
        steps_between=40,
        seed=seed,
    )
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "datasets/revision/pgg")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-embed", type=int, default=80)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=80)
    parser.add_argument("--n-id", type=int, default=80)
    parser.add_argument("--n-ood", type=int, default=80)
    parser.add_argument("--steps-between", type=int, default=40)
    parser.add_argument("--equilibration-steps", type=int, default=2000)
    parser.add_argument("--max-sweeps", type=int, default=40)
    add_quick_flags(parser)
    args = parser.parse_args()
    apply_quick(args)
    if args.quick:
        args.n_embed = min(args.n_embed, 12)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    minima = embed_minima(args.n_embed, args.seed)
    if len(minima) < 4:
        raise RuntimeError(f"only {len(minima)} PGG minima embedded")
    families = cluster_minima(minima)[:4]
    while len(families) < 4:
        families.append([families[-1][0]])

    train_frames, val_frames, id_frames, ood_frames = [], [], [], []
    train_ids, ood_ids = [1, 2, 3], [4]
    for fam_id, fam in enumerate(families, start=1):
        ref = fam[0]
        if fam_id in train_ids:
            frames = sample_family(ref, (args.n_train + args.n_val + args.n_id) // 3, args.seed + fam_id, quick=args.quick)
            n = len(frames)
            train_frames.extend(frames[: n * 2 // 3])
            val_frames.extend(frames[n * 2 // 3 : n * 5 // 6])
            id_frames.extend(frames[n * 5 // 6 :])
        elif fam_id in ood_ids:
            frames = sample_family(ref, args.n_ood, args.seed + fam_id + 100, quick=args.quick)
            ood_frames.extend(frames)

    manifest = {
        "seed": args.seed,
        "ood_split": "ood_conformer",
        "train_family_ids": train_ids,
        "ood_family_ids": ood_ids,
        "n_families_found": len(families),
        "splits": {},
    }
    for name, frames in (
        ("train", train_frames[: args.n_train]),
        ("val", val_frames[: args.n_val]),
        ("id_test", id_frames[: args.n_id]),
    ):
        path = args.output_dir / f"{name}.xyz"
        write_xyz(path, frames, [f"pgg {name}"] * len(frames))
        manifest["splits"][name] = {"xyz": str(path), "n_frames": len(frames)}
    ood_path = args.output_dir / "ood_conformer.xyz"
    write_xyz(ood_path, ood_frames[: args.n_ood], [f"pgg ood_conformer"] * min(len(ood_frames), args.n_ood))
    manifest["splits"]["ood_conformer"] = {"xyz": str(ood_path), "n_frames": min(len(ood_frames), args.n_ood)}

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
