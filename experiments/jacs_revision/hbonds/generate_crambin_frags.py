#!/usr/bin/env python3
"""Build crambin fragment-proxy splits (~40-70 atoms, not the 25k-atom JACS test)."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/hbonds/generate_crambin_frags.py'


import argparse
import json
import sys
import urllib.request
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
from experiments.jacs_revision.hbonds.common import DEFAULT_SEED, langevin_frames, thermal_noise_frames  # noqa: E402
from experiments.jacs_revision.hbonds.generator_defaults import add_quick_flags, apply_quick  # noqa: E402
from experiments.jacs_revision.hbonds.xyz_io import write_xyz  # noqa: E402

_REPO_ROOT = repo_root()
PDB_ID = "1CRN"
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"


def fetch_pdb(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{PDB_ID}.pdb"
    if not path.exists():
        urllib.request.urlretrieve(PDB_URL, path)
    return path


def load_protein(pdb_path: Path) -> Atoms:
    from ase.io import read

    return read(str(pdb_path))


def carve_fragment(protein: Atoms, center_idx: int, radius: float, n_waters: int, seed: int) -> Atoms:
    """Select protein atoms near center_idx and add explicit waters."""
    pos = protein.get_positions()
    dists = np.linalg.norm(pos - pos[center_idx], axis=1)
    keep = dists <= radius
    symbols = [s for s, k in zip(protein.get_chemical_symbols(), keep) if k]
    frag_pos = pos[keep]
    rng = np.random.default_rng(seed)
    waters = []
    for _ in range(n_waters):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        waters.append(("O", frag_pos.mean(0) + direction * (radius + 2.5)))
        waters.append(("H", frag_pos.mean(0) + direction * (radius + 2.5) + np.array([0.8, 0.0, 0.0])))
        waters.append(("H", frag_pos.mean(0) + direction * (radius + 2.5) + np.array([-0.4, 0.7, 0.0])))
    w_syms = [w[0] for w in waters]
    w_pos = np.array([w[1] for w in waters])
    return Atoms(
        symbols=symbols + w_syms,
        positions=np.vstack([frag_pos, w_pos]),
    )


def refuse_full_protein_train() -> None:
    raise RuntimeError(
        "refuse to write a full-protein train split: crambin proxy uses QM fragments only"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "datasets/revision/crambin_frags")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--n-id", type=int, default=40)
    parser.add_argument("--n-ood", type=int, default=60)
    parser.add_argument("--steps-between", type=int, default=30)
    parser.add_argument("--equilibration-steps", type=int, default=2000)
    parser.add_argument("--max-sweeps", type=int, default=40)
    add_quick_flags(parser)
    args = parser.parse_args()
    apply_quick(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdb = fetch_pdb(_REPO_ROOT / "datasets/revision/cache")
    protein = load_protein(pdb)
    n = len(protein)
    region_specs = {
        "helix_nt": {"center_idx": min(10, n - 1), "radius": 5.5, "n_waters": 8},
        "loop_ct": {"center_idx": min(n // 2, n - 1), "radius": 5.5, "n_waters": 6},
        "disulfide": {"center_idx": min(n - 15, n - 1), "radius": 6.0, "n_waters": 10},
    }
    regions = {
        name: carve_fragment(protein, seed=args.seed + i, **spec)
        for i, (name, spec) in enumerate(region_specs.items())
    }
    fragments = {name: {"natoms": len(atoms)} for name, atoms in regions.items()}

    train_regions = ["helix_nt", "loop_ct"]
    ood_regions = ["disulfide"]
    train_frames, val_frames, id_frames = [], [], []
    for rid in train_regions:
        if args.quick:
            frames, _ = thermal_noise_frames(
                regions[rid], (args.n_train + args.n_val + args.n_id) // 2, amplitude=0.20, seed=args.seed + hash(rid) % 1000,
            )
        else:
            frames, _ = langevin_frames(
                regions[rid],
                (args.n_train + args.n_val + args.n_id) // 2,
                temperature_k=300.0,
                steps_between=args.steps_between,
                seed=args.seed + hash(rid) % 1000,
                equilibration_steps=args.equilibration_steps,
                max_sweeps=args.max_sweeps,
            )
        n = len(frames)
        train_frames.extend(frames[: n // 2])
        val_frames.extend(frames[n // 2 : n * 3 // 4])
        id_frames.extend(frames[n * 3 // 4 :])
    if args.quick:
        ood_frames, _ = thermal_noise_frames(regions["disulfide"], args.n_ood, amplitude=0.25, seed=args.seed + 999)
    else:
        ood_frames, _ = langevin_frames(
            regions["disulfide"],
            args.n_ood,
            temperature_k=300.0,
            steps_between=args.steps_between,
            seed=args.seed + 999,
            equilibration_steps=args.equilibration_steps,
            max_sweeps=args.max_sweeps,
        )

    full_protein_train_written = False
    if full_protein_train_written:
        refuse_full_protein_train()

    manifest = {
        "seed": args.seed,
        "pdb_id": PDB_ID,
        "ood_split": "ood_region",
        "fragments": fragments,
        "train_region_ids": train_regions,
        "ood_region_ids": ood_regions,
        "full_protein_train_written": full_protein_train_written,
        "note": "fragment proxy; not the 25k-atom SO3LR crambin-in-water benchmark",
        "splits": {},
    }
    for name, frames in (
        ("train", train_frames[: args.n_train]),
        ("val", val_frames[: args.n_val]),
        ("id_test", id_frames[: args.n_id]),
    ):
        path = args.output_dir / f"{name}.xyz"
        write_xyz(path, frames, [f"crambin_frags {name}"] * len(frames))
        manifest["splits"][name] = {"xyz": str(path), "n_frames": len(frames)}
    ood_path = args.output_dir / "ood_region.xyz"
    write_xyz(ood_path, ood_frames, [f"crambin_frags ood_region"] * len(ood_frames))
    manifest["splits"]["ood_region"] = {"xyz": str(ood_path), "n_frames": len(ood_frames)}

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
