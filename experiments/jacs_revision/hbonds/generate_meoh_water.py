#!/usr/bin/env python3
"""Build methanol–water dimer splits with isomer-A train and isomer-B OOD."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/hbonds/generate_meoh_water.py'


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
from experiments.jacs_revision.hbonds.collective_variables import meoh_water_cv, meoh_water_donor_is_methanol  # noqa: E402
from experiments.jacs_revision.hbonds.common import DEFAULT_SEED, gfn2_calculator, thermal_noise_frames  # noqa: E402
from experiments.jacs_revision.hbonds.generator_defaults import add_quick_flags, apply_quick  # noqa: E402
from experiments.jacs_revision.hbonds.xyz_io import write_xyz  # noqa: E402

_REPO_ROOT = repo_root()


def build_dimer(isomer_a: bool, seed: int, *, bfgs_steps: int = 400) -> Atoms:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    meoh = Chem.AddHs(Chem.MolFromSmiles("CO"))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(meoh, params)
    AllChem.MMFFOptimizeMolecule(meoh)
    conf = meoh.GetConformer()
    syms = [a.GetSymbol() for a in meoh.GetAtoms()]
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(meoh.GetNumAtoms())])
    m = Atoms(symbols=syms, positions=pos)
    if bfgs_steps >= 200:
        m.calc = gfn2_calculator()
        BFGS(m, logfile=None).run(fmax=0.02, steps=min(200, bfgs_steps))

    w = Atoms("OHH", positions=[[4.0, 0.0, 0.0], [4.9, 0.0, 0.0], [3.5, 0.8, 0.0]])
    if bfgs_steps >= 200:
        w.calc = gfn2_calculator()
        BFGS(w, logfile=None).run(fmax=0.02, steps=min(200, bfgs_steps))

    dimer = Atoms(
        symbols=m.get_chemical_symbols() + w.get_chemical_symbols(),
        positions=np.vstack([m.get_positions(), w.get_positions()]),
    )
    dimer.calc = gfn2_calculator()
    BFGS(dimer, logfile=None).run(fmax=0.05, steps=bfgs_steps)

    if isomer_a:
        return dimer

    positions = dimer.get_positions().copy()
    symbols = dimer.get_chemical_symbols()
    carbons = [i for i, s in enumerate(symbols) if s == "C"][0]
    meoh_o = next(
        i for i, s in enumerate(symbols)
        if s == "O" and np.linalg.norm(positions[i] - positions[carbons]) < 1.5
    )
    water_o = next(i for i, s in enumerate(symbols) if s == "O" and i != meoh_o)
    meoh_h = next(
        i for i, s in enumerate(symbols)
        if s == "H" and np.linalg.norm(positions[i] - positions[meoh_o]) < 1.2
    )
    water_h = next(
        i for i, s in enumerate(symbols)
        if s == "H" and i != meoh_h and np.linalg.norm(positions[i] - positions[water_o]) < 1.2
    )
    midpoint = 0.5 * (positions[meoh_o] + positions[water_o])
    positions[meoh_h] = midpoint + (positions[meoh_o] - midpoint) * 0.3
    positions[water_h] = midpoint + (positions[water_o] - midpoint) * 0.3
    dimer.set_positions(positions)
    dimer.calc = gfn2_calculator()
    BFGS(dimer, logfile=None).run(fmax=0.05, steps=bfgs_steps)
    return dimer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "datasets/revision/meoh_water")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=80)
    parser.add_argument("--n-id", type=int, default=80)
    parser.add_argument("--n-ood", type=int, default=125)
    parser.add_argument("--thermal-amplitude", type=float, default=0.10)
    parser.add_argument("--langevin", action="store_true", help="use Langevin instead of thermal noise")
    parser.add_argument("--steps-between", type=int, default=50)
    parser.add_argument("--equilibration-steps", type=int, default=2000)
    parser.add_argument("--max-sweeps", type=int, default=40)
    add_quick_flags(parser)
    args = parser.parse_args()
    apply_quick(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bfgs_steps = 80 if args.quick else 400

    dimer_a = build_dimer(True, args.seed, bfgs_steps=bfgs_steps)
    symbols_a = dimer_a.get_chemical_symbols()

    def accept_a(positions):
        return meoh_water_donor_is_methanol(positions, symbols_a)

    train_temp_k = 200.0 if args.quick else 300.0

    def collect_frames(dimer, n_frames, accept_fn, seed, amplitude=None):
        if args.langevin:
            from experiments.jacs_revision.hbonds.common import langevin_frames

            return langevin_frames(
                dimer,
                n_frames,
                temperature_k=train_temp_k,
                steps_between=args.steps_between,
                seed=seed,
                accept=accept_fn,
                equilibration_steps=args.equilibration_steps,
                max_sweeps=max(args.max_sweeps, 80 if args.quick else args.max_sweeps),
            )
        return thermal_noise_frames(
            dimer,
            n_frames,
            amplitude=amplitude or args.thermal_amplitude,
            seed=seed,
            accept=accept_fn,
        )

    pool, n_rejected = collect_frames(dimer_a, args.n_train + args.n_val + args.n_id, accept_a, args.seed)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))
    splits = {
        "train": [pool[i] for i in order[: args.n_train]],
        "val": [pool[i] for i in order[args.n_train : args.n_train + args.n_val]],
        "id_test": [pool[i] for i in order[args.n_train + args.n_val :]],
    }

    manifest = {
        "seed": args.seed,
        "ood_split": "ood_isomer_b",
        "train_temperature_k": train_temp_k,
        "placement": "langevin" if args.langevin else "thermal_noise",
        "thermal_amplitude_A": args.thermal_amplitude,
        "n_rejected_a": n_rejected,
        "splits": {},
    }
    for name, frames in splits.items():
        path = args.output_dir / f"{name}.xyz"
        write_xyz(path, frames, [f"meoh_water {name}"] * len(frames))
        manifest["splits"][name] = {"xyz": str(path), "n_frames": len(frames)}

    dimer_b = build_dimer(False, args.seed + 1, bfgs_steps=bfgs_steps)
    symbols_b = dimer_b.get_chemical_symbols()

    def accept_b(positions):
        return not meoh_water_donor_is_methanol(positions, symbols_b)

    ood_pool, n_rejected_b = collect_frames(
        dimer_b,
        args.n_ood,
        accept_b,
        args.seed + 99,
        amplitude=max(args.thermal_amplitude, 0.12),
    )
    ood_path = args.output_dir / "ood_isomer_b.xyz"
    write_xyz(ood_path, ood_pool, [f"meoh_water ood_isomer_b"] * len(ood_pool))
    manifest["splits"]["ood_isomer_b"] = {"xyz": str(ood_path), "n_frames": len(ood_pool), "n_rejected_b": n_rejected_b}

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
