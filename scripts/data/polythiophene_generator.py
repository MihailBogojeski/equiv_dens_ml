"""
N-mer polythiophene generator using SMILES + RDKit.

Builds polythiophene oligomers from SMILES strings (2-5 alpha linkage),
embeds 3D coordinates with RDKit, and optionally optimizes with g-xtb.
Guarantees closed-shell structures.
"""

from __future__ import annotations

import argparse

import numpy as np
import os
import re
import subprocess
from pathlib import Path

import ase.io
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS

# Use project root for imports when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def _ring_label(i: int) -> str:
    """SMILES ring closure label; use % for i >= 10."""
    return f"%{i}" if i >= 10 else str(i)


def polythiophene_smiles(n: int) -> str:
    """
    SMILES string for n-mer polythiophene (2-5 alpha linkage).

    Parameters
    ----------
    n : int
        Number of thiophene rings.

    Returns
    -------
    str
        SMILES string. 1-mer: c1ccc[s]1. n-mer: c1ccc(s1)c2ccc(s2)...cnccc[s]n.
        Ring numbers >= 10 use %10, %11, etc. per SMILES spec.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return "c1ccc[s]1"
    parts = []
    for i in range(1, n):
        r = _ring_label(i)
        parts.append(f"c{r}ccc(s{r})")
    rn = _ring_label(n)
    parts.append(f"c{rn}ccc[s]{rn}")
    return "".join(parts)


def smiles_to_ase_atoms(smiles: str) -> Atoms:
    """
    Convert SMILES to ASE Atoms with 3D coordinates via RDKit embedding.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    Atoms
        ASE Atoms with embedded 3D coordinates (Angstrom).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol)
    if result != 0:
        raise RuntimeError(f"RDKit 3D embedding failed for SMILES: {smiles}")

    conf = mol.GetConformer()
    symbols = []
    positions = []
    for atom in mol.GetAtoms():
        symbols.append(atom.GetSymbol())
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])

    return Atoms(symbols=symbols, positions=positions)


def _check_closed_shell(atoms: Atoms) -> None:
    """Raise ValueError if structure has odd electron count."""
    from pyscf import gto

    nelec = sum(gto.charge(int(z)) for z in atoms.get_atomic_numbers() if z > 0)
    if nelec % 2 != 0:
        raise ValueError(
            f"Structure has odd electron count ({nelec}); only closed-shell "
            "systems are supported."
        )


def generate_nmer_from_smiles(n: int) -> Atoms:
    """
    Generate n-mer polythiophene as ASE Atoms.

    Parameters
    ----------
    n : int
        Number of thiophene rings.

    Returns
    -------
    Atoms
        ASE Atoms with 3D coordinates.
    """
    smiles = polythiophene_smiles(n)
    atoms = smiles_to_ase_atoms(smiles)
    _check_closed_shell(atoms)
    return atoms


class _GxtbCalculator(Calculator):
    """
    ASE calculator that calls g-xtb for energy and gradient.

    Bypasses xtb entirely; xtb's driver expects TURBOMOLE format but g-xtb
    produces a different gradient format, causing parse errors.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        gxtb_path: str,
        gxtb_params_dir: str,
        workdir: Path,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gxtb_path = str(Path(gxtb_path).resolve())
        self.gxtb_params_dir = str(Path(gxtb_params_dir).resolve())
        self.workdir = Path(workdir)
        self.verbose = verbose
        self._coord_file = "coord.xyz"

    def _run_gxtb(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        """Run g-xtb -grad and return (energy, gradient). Gradient in Eh/bohr."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        coord_path = self.workdir / self._coord_file
        ase.io.write(coord_path, atoms, format="xyz")

        env = os.environ.copy()
        env["GXTBHOME"] = self.gxtb_params_dir
        gxtb_dir = str(Path(self.gxtb_path).resolve().parent)
        env["PATH"] = gxtb_dir + os.pathsep + env.get("PATH", "")

        result = subprocess.run(
            [self.gxtb_path, "-grad", "-c", str(coord_path)],
            cwd=str(self.workdir),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            tail = (result.stdout or "")[-2000:] if result.stdout else ""
            raise RuntimeError(
                f"g-xtb failed (exit {result.returncode}). Output:\n{tail}"
            )

        energy = self._parse_energy()
        gradient = self._parse_gradient(len(atoms))
        return energy, gradient

    def _parse_energy(self) -> float:
        """Parse energy from $energy or gradient file."""
        energy_path = self.workdir / "energy"
        if energy_path.exists():
            with open(energy_path) as f:
                for line in f:
                    if line.strip().startswith("$"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1])
        grad_path = self.workdir / "gradient"
        if grad_path.exists():
            with open(grad_path) as f:
                for line in f:
                    if "SCF energy" in line:
                        match = re.search(r"SCF energy\s*=\s*([-\d.]+)", line)
                        if match:
                            return float(match.group(1))
        raise FileNotFoundError("Could not parse energy from g-xtb output")

    def _parse_gradient(self, n: int) -> np.ndarray:
        """Parse gradient from gradient file. g-xtb format: 3 numbers + element per line."""
        grad_path = self.workdir / "gradient"
        if not grad_path.exists():
            raise FileNotFoundError(f"g-xtb did not produce {grad_path}")

        gradient = np.zeros((n, 3))
        idx = 0
        with open(grad_path) as f:
            for line in f:
                if "cycle" in line.lower() or line.strip().startswith("$"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        gx, gy, gz = float(parts[0]), float(parts[1]), float(parts[2])
                        if idx < n:
                            gradient[idx] = [gx, gy, gz]
                            idx += 1
                    except ValueError:
                        continue
                if idx >= n:
                    break
        if idx != n:
            raise ValueError(f"Expected {n} gradient rows, got {idx}")
        return gradient

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if atoms is None:
            atoms = self.atoms
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)
        energy, gradient = self._run_gxtb(atoms)
        self.results = {
            "energy": energy,
            "forces": -gradient,  # ASE uses forces = -dE/dR
        }


def optimize_with_gxtb(
    atoms: Atoms,
    xtb_cmd: str,
    gxtb_path: str,
    gxtb_params_dir: str,
    workdir: Path,
    opt_level: str = "normal",
    verbose: bool = True,
) -> Atoms:
    """
    Run geometry optimization using g-xtb directly (no xtb driver).

    Parameters
    ----------
    atoms : Atoms
        Input structure.
    xtb_cmd : str
        Unused (kept for API compatibility).
    gxtb_path : str
        Full path to gxtb binary.
    gxtb_params_dir : str
        Path to directory containing .gxtb, .eeq, .basisq (GXTBHOME).
    workdir : Path
        Working directory for input/output files.
    opt_level : str
        "loose" or "normal" - affects fmax convergence (loose: 0.05, normal: 0.01).
    verbose : bool
        Print optimization progress.

    Returns
    -------
    Atoms
        Optimized structure.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    for f in ("energy", "gradient", "coord.xyz", "gxtbrestart"):
        p = workdir / f
        if p.exists():
            p.unlink()

    calc = _GxtbCalculator(
        gxtb_path=gxtb_path,
        gxtb_params_dir=gxtb_params_dir,
        workdir=workdir,
        verbose=verbose,
    )
    atoms = atoms.copy()
    atoms.calc = calc

    fmax = 0.05 if opt_level == "loose" else 0.01
    dyn = BFGS(atoms, maxstep=0.2, logfile=None)
    if verbose:
        print(f"[Optimize] Running g-xtb optimization ({len(atoms)} atoms, fmax={fmax})...")

    dyn.run(fmax=fmax, steps=200)

    opt_xyz = workdir / "opt.xyz"
    ase.io.write(opt_xyz, atoms, format="xyz")
    if verbose:
        print(f"[Optimize] Geometry optimized successfully, wrote {opt_xyz}")
    return atoms


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate N-mer polythiophene oligomers from SMILES (RDKit) with optional g-xtb optimization."
    )
    parser.add_argument("--n-mer", type=int, required=True, help="Target oligomer size (e.g. 8)")
    parser.add_argument("--output", type=Path, required=True, help="Output .xyz path")
    default_root = _PROJECT_ROOT
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run g-xtb geometry optimization",
    )
    parser.add_argument(
        "--xtb-cmd",
        type=str,
        default="xtb",
        help="xtb executable (PATH or full path)",
    )
    parser.add_argument(
        "--gxtb-path",
        type=Path,
        default=default_root / "g-xtb" / "binary" / "gxtb",
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

    if verbose:
        print(f"[Start] Polythiophene {args.n_mer}-mer generator (SMILES + RDKit)")
        print(f"[Start] Output: {args.output}")

    mol = generate_nmer_from_smiles(args.n_mer)

    if args.optimize:
        workdir = args.output.parent / ".nmer_opt_workdir"
        gxtb_params = default_root / "g-xtb" / "parameters"
        if not gxtb_params.exists():
            raise FileNotFoundError(
                f"g-xtb parameters not found at {gxtb_params}. "
                "Ensure g-xtb submodule is initialized."
            )
        mol = optimize_with_gxtb(
            mol,
            xtb_cmd=args.xtb_cmd,
            gxtb_path=str(args.gxtb_path),
            gxtb_params_dir=str(gxtb_params),
            workdir=workdir,
            opt_level="loose" if args.opt_loose else "normal",
            verbose=verbose,
        )
    else:
        if verbose:
            print("[Skip] Geometry optimization disabled (use --optimize to enable)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(args.output, [mol], format="xyz")

    if verbose:
        print(f"[Save] Wrote {len(mol)} atoms to {args.output}")
        print("[Done] Finished successfully")


if __name__ == "__main__":
    main()
