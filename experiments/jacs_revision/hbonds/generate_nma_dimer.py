#!/usr/bin/env python3
"""Build N-methylacetamide dimer train/val/test splits (Reviewer 1, R1.1)."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/hbonds/generate_nma_dimer.py'


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

ensure_revision_imports(_SCRIPT_DIR, _SCRIPT_DIR.parent / "water")

from ase import Atoms  # noqa: E402
from ase.constraints import FixBondLength  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from experiments.jacs_revision.hbonds.collective_variables import compute_frame_cvs, nma_hbond_contacts  # noqa: E402
from experiments.jacs_revision.hbonds.common import DEFAULT_SEED, distance, gfn2_calculator, langevin_frames, thermal_noise_frames  # noqa: E402
from experiments.jacs_revision.hbonds.generator_defaults import add_quick_flags, apply_quick  # noqa: E402
from experiments.jacs_revision.hbonds.xyz_io import write_xyz  # noqa: E402

_REPO_ROOT = repo_root()
SMILES = "CC(=O)NC"


def build_monomer(seed: int, *, bfgs_steps: int = 300) -> Atoms:
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
    monomer = Atoms(symbols=symbols, positions=pos)
    if bfgs_steps > 0:
        monomer.calc = gfn2_calculator()
        BFGS(monomer, logfile=None).run(fmax=0.02, steps=bfgs_steps)
    return monomer


def _align_hbond_dimer(monomer: Atoms, partner: Atoms) -> Atoms:
    """Translate ``partner`` so an N–H group sits near the monomer carbonyl O."""
    pos1 = monomer.get_positions()
    pos2 = partner.get_positions()
    syms1 = monomer.get_chemical_symbols()
    syms2 = partner.get_chemical_symbols()
    o1 = next(i for i, s in enumerate(syms1) if s == "O")
    n2 = next(i for i, s in enumerate(syms2) if s == "N")
    h2 = next(
        i for i, s in enumerate(syms2)
        if s == "H" and np.linalg.norm(pos2[i] - pos2[n2]) < 1.2
    )
    target_h = pos1[o1] + np.array([1.78, 0.30, 0.0])
    partner.translate(target_h - pos2[h2])
    return partner


def build_cyclic_dimer(seed: int = DEFAULT_SEED, *, bfgs_steps: int = 300) -> Atoms:
    monomer = build_monomer(seed, bfgs_steps=bfgs_steps)
    partner = build_monomer(seed + 1, bfgs_steps=0 if bfgs_steps <= 80 else bfgs_steps)
    partner = _align_hbond_dimer(monomer, partner)
    dimer = Atoms(
        symbols=monomer.get_chemical_symbols() + partner.get_chemical_symbols(),
        positions=np.vstack([monomer.get_positions(), partner.get_positions()]),
    )
    if bfgs_steps > 0:
        dimer.calc = gfn2_calculator()
        BFGS(dimer, logfile=None).run(fmax=0.05, steps=bfgs_steps)
    return dimer


def accept_bound(positions: np.ndarray, symbols: list[str], r_max: float = 2.4) -> bool:
    nitrogens = [i for i, s in enumerate(symbols) if s == "N"]
    oxygens = [i for i, s in enumerate(symbols) if s == "O"]
    hydrogens = [i for i, s in enumerate(symbols) if s == "H"]
    if len(nitrogens) != 2 or len(oxygens) != 2:
        return False
    heavy = [i for i, s in enumerate(symbols) if s not in {"H", "D", "T"}]
    for a, i in enumerate(heavy):
        for j in heavy[a + 1 :]:
            if distance(positions, i, j) < 1.15:
                return False
    for n in nitrogens:
        candidates = [h for h in hydrogens if distance(positions, n, h) < 1.2]
        if not candidates:
            return False
        best_r = min(
            distance(positions, h, o)
            for h in candidates
            for o in oxygens
        )
        if best_r > r_max:
            return False
    return True


def relaxed_n_o_stretch(dimer: Atoms, targets: np.ndarray, n_idx: int, o_idx: int, *, bfgs_steps: int = 300):
    frames, records = [], []
    work = dimer.copy()
    for target in targets:
        positions = work.get_positions().copy()
        vec = positions[o_idx] - positions[n_idx]
        current = np.linalg.norm(vec)
        positions[o_idx] = positions[n_idx] + vec / current * target
        work.set_positions(positions)
        if bfgs_steps > 0:
            work.calc = gfn2_calculator()
            work.set_constraint(FixBondLength(n_idx, o_idx))
            BFGS(work, logfile=None).run(fmax=0.05, steps=bfgs_steps)
            work.set_constraint()
        snap = work.get_positions().copy()
        syms = work.get_chemical_symbols()
        frames.append((syms, snap))
        cvs = compute_frame_cvs("nma_dimer", syms, snap)
        records.append({"r_no": float(target), **cvs})
    return frames, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "datasets/revision/nma_dimer")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=80)
    parser.add_argument("--n-id", type=int, default=80)
    parser.add_argument("--steps-between", type=int, default=50)
    parser.add_argument("--equilibration-steps", type=int, default=2000)
    parser.add_argument("--max-sweeps", type=int, default=40)
    parser.add_argument("--thermal-amplitude", type=float, default=0.05)
    parser.add_argument("--langevin", action="store_true")
    add_quick_flags(parser)
    args = parser.parse_args()
    apply_quick(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bfgs_steps = 0 if args.quick else 300

    dimer = build_cyclic_dimer(args.seed, bfgs_steps=bfgs_steps)
    symbols = dimer.get_chemical_symbols()
    n_idx = [i for i, s in enumerate(symbols) if s == "N"][0]
    o_idx = [i for i, s in enumerate(symbols) if s == "O"][0]

    accept = lambda p: accept_bound(p, symbols)
    n_needed = args.n_train + args.n_val + args.n_id
    if args.langevin:
        pool, n_rejected = langevin_frames(
            dimer,
            n_needed,
            temperature_k=300.0,
            steps_between=args.steps_between,
            seed=args.seed,
            accept=accept,
            equilibration_steps=args.equilibration_steps,
            max_sweeps=max(args.max_sweeps, 80 if args.quick else args.max_sweeps),
        )
    else:
        pool, n_rejected = thermal_noise_frames(
            dimer,
            n_needed,
            amplitude=args.thermal_amplitude,
            seed=args.seed,
            accept=accept,
        )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))
    splits = {
        "train": [pool[i] for i in order[: args.n_train]],
        "val": [pool[i] for i in order[args.n_train : args.n_train + args.n_val]],
        "id_test": [pool[i] for i in order[args.n_train + args.n_val :]],
    }

    manifest = {"seed": args.seed, "ood_split": "ood_stretch", "n_rejected": n_rejected, "splits": {}}
    for name, frames in splits.items():
        path = args.output_dir / f"{name}.xyz"
        comments = [f"nma_dimer {name}" for _ in frames]
        write_xyz(path, frames, comments)
        manifest["splits"][name] = {"xyz": str(path), "n_frames": len(frames)}

    scan_points = 10 if args.quick else 25
    jitter = 2 if args.quick else 4
    targets = np.linspace(4.0, 6.5, scan_points)
    scan_frames, scan_records = relaxed_n_o_stretch(
        dimer, targets, n_idx, o_idx, bfgs_steps=0 if args.quick else 300,
    )
    ood_frames, comments = [], []
    for (syms, pos), rec in zip(scan_frames, scan_records):
        ood_frames.append((syms, pos))
        comments.append(f"nma_dimer stretch r_ho={rec['r_ho']:.3f}")
        for _ in range(jitter):
            jittered_pos = pos + rng.normal(0.0, 0.02, size=pos.shape)
            ood_frames.append((syms, jittered_pos))
            comments.append("nma_dimer stretch-jitter")
    ood_path = args.output_dir / "ood_stretch.xyz"
    write_xyz(ood_path, ood_frames, comments)
    manifest["splits"]["ood_stretch"] = {"xyz": str(ood_path), "n_frames": len(ood_frames), "scan": scan_records}

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
