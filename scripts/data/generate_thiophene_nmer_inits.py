#!/usr/bin/env python3
"""
Generate polythiophene n-mer initial structures for ML-MD scaling benchmarks.

Produces .npy files in the format expected by AtomsDensityData (positions,
atom_numbers) for n = 1..20. Uses polythiophene_generator for 3D embedding;
optionally runs g-xtb optimization for better geometries.
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

import importlib.util
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Load polythiophene_generator from same directory (no package dependency)
_pg_spec = importlib.util.spec_from_file_location(
    "polythiophene_generator",
    _SCRIPT_DIR / "polythiophene_generator.py",
)
_pg = importlib.util.module_from_spec(_pg_spec)
_pg_spec.loader.exec_module(_pg)
generate_nmer_from_smiles = _pg.generate_nmer_from_smiles
optimize_with_gxtb = _pg.optimize_with_gxtb


def atoms_to_npy_dict(atoms) -> dict:
    """
    Convert ASE Atoms to dict format expected by AtomsDensityData.

    Parameters
    ----------
    atoms : ase.Atoms
        Single structure.

    Returns
    -------
    dict
        Keys: positions (1, natoms, 3), atom_numbers (natoms,).
    """
    positions = atoms.get_positions()
    atom_numbers = atoms.get_atomic_numbers()
    return {
        "positions": np.array([positions], dtype=np.float64),
        "atom_numbers": atom_numbers.astype(np.int64),
    }


def generate_nmer_init(
    n: int,
    output_path: Path,
    optimize: bool = False,
    gxtb_path: Path | None = None,
    gxtb_params: Path | None = None,
    opt_loose: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Generate n-mer polythiophene and save as .npy for MD.

    Parameters
    ----------
    n : int
        Number of thiophene rings (1..20).
    output_path : Path
        Output .npy path (e.g. datasets/thiophene5mer_init.npy).
    optimize : bool
        If True, run g-xtb geometry optimization.
    gxtb_path : Path, optional
        Path to gxtb binary.
    gxtb_params : Path, optional
        Path to g-xtb parameters directory (GXTBHOME).
    opt_loose : bool
        Use loose convergence for g-xtb optimization.
    verbose : bool
        Print progress.

    Returns
    -------
    Path
        Path to generated file.
    """
    if verbose:
        print(f"[Generate] {n}-mer polythiophene -> {output_path}")

    atoms = generate_nmer_from_smiles(n)

    if optimize:
        if gxtb_path is None:
            gxtb_path = _REPO_ROOT / "g-xtb" / "binary" / "gxtb"
        if gxtb_params is None:
            gxtb_params = _REPO_ROOT / "g-xtb" / "parameters"
        if not gxtb_params.exists():
            raise FileNotFoundError(
                f"g-xtb parameters not found at {gxtb_params}. "
                "Ensure g-xtb submodule is initialized."
            )
        workdir = output_path.parent / ".nmer_opt_workdir"
        atoms = optimize_with_gxtb(
            atoms,
            xtb_cmd="xtb",
            gxtb_path=str(gxtb_path),
            gxtb_params_dir=str(gxtb_params),
            workdir=workdir,
            opt_level="loose" if opt_loose else "normal",
            verbose=verbose,
        )

    data = atoms_to_npy_dict(atoms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data, allow_pickle=True)

    if verbose:
        natoms = len(atoms)
        print(f"[Save] {natoms} atoms -> {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate polythiophene n-mer initial structures for MD scaling."
    )
    parser.add_argument(
        "--n-mers",
        type=str,
        default="1-20",
        help="Comma-separated list or range (e.g. 1-20, 1,2,4,8,10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "datasets",
        help="Output directory for .npy files",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run g-xtb geometry optimization",
    )
    parser.add_argument(
        "--gxtb-path",
        type=Path,
        default=_REPO_ROOT / "g-xtb" / "binary" / "gxtb",
        help="Path to gxtb binary",
    )
    parser.add_argument(
        "--opt-loose",
        action="store_true",
        help="Use loose convergence for g-xtb optimization",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Parse n-mers
    n_list: list[int] = []
    for part in args.n_mers.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            n_list.extend(range(int(lo), int(hi) + 1))
        else:
            n_list.append(int(part))
    n_list = sorted(set(n_list))

    if verbose:
        print(f"[Start] Generating polythiophene n-mer inits for n = {n_list}")

    failed: list[int] = []
    for n in n_list:
        if n < 1:
            continue
        output_path = args.output_dir / f"thiophene{n}mer_init.npy"
        try:
            generate_nmer_init(
                n=n,
                output_path=output_path,
                optimize=args.optimize,
                gxtb_path=args.gxtb_path,
                gxtb_params=_REPO_ROOT / "g-xtb" / "parameters",
                opt_loose=args.opt_loose,
                verbose=verbose,
            )
        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"[Skip] {n}-mer failed: {e}")
            failed.append(n)

    if verbose and failed:
        print(f"[Done] Generated successfully; failed for n = {failed}")
    elif verbose:
        print("[Done] Finished successfully")


if __name__ == "__main__":
    main()
