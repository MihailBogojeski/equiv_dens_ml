"""DFTB/3ob ASE calculator factory for revision Figure 2 timings."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.dftb import Dftb
from ase.units import Bohr, Hartree

from experiments.jacs_revision.lib.paths import repo_root

# 3ob-3-1 Hubbard derivatives (atomic units); see dftbparams/3ob README.
# DFTB+ 24.1 expects shell-resolved Real arrays here; ASE only writes scalars,
# so third-order (DFTB3) input is opt-in and may fail until ASE catches up.
HUBBARD_DERIVS_3OB: dict[str, float] = {
    "H": -0.1857,
    "C": -0.1492,
    "O": -0.1575,
    "N": -0.1535,
    "S": -0.11,
    "P": -0.14,
}


def default_dftb_paths(start: Path | None = None) -> tuple[Path, Path]:
    """Return default DFTB+ binary and 3ob-3-1 Slater–Koster directory."""
    root = repo_root(start)
    binary = root / "tools" / "dftbplus" / "bin" / "dftb+"
    sk_dir = root / "tools" / "dftbplus" / "skfiles" / "3ob-3-1"
    return binary, sk_dir


def _angular_momentum_for_symbol(symbol: str) -> str:
    if symbol == "H":
        return "s"
    return "p"


class DftbPlus24Calculator(Dftb):
    """ASE Dftb with DFTB+ 24 ``results.tag`` force parsing."""

    def read_forces(self):
        for iline, line in enumerate(self.lines):
            stripped = line.strip()
            if not stripped.startswith("forces"):
                continue
            if ":real:" in stripped:
                match = re.search(r":real:\d+:(\d+),(\d+)", stripped)
                if match is None:
                    break
                ncols = int(match.group(1))
                nrows = int(match.group(2))
                gradients = []
                for j in range(iline + 1, iline + 1 + nrows):
                    gradients.append([float(x) for x in self.lines[j].split()])
                arr = np.array(gradients)
                if arr.shape != (nrows, ncols):
                    arr = arr.reshape(nrows, ncols)
                return arr * Hartree / Bohr
            break
        return super().read_forces()


def make_dftb3_3ob_calculator(
    atoms: Atoms,
    command: str,
    sk_dir: str | Path,
    label: str = "dftb",
    *,
    third_order: bool = False,
) -> DftbPlus24Calculator:
    """Return an ASE Dftb calculator using the 3ob-3-1 Slater–Koster set.

    By default this runs SCC-DFTB (DFTB2) with the 3ob parameters, which is
    compatible with ASE's HSD writer and DFTB+ 24.1. Pass ``third_order=True``
    to request DFTB3 keywords (Hubbard derivatives + H–X damping); this may
    fail on DFTB+ 24.1 because ASE writes scalar HubbardDerivs entries.

    Parameters
    ----------
    atoms
        Structure whose element symbols set ``MaxAngularMomentum``.
    command
        Shell command for DFTB+ (must redirect stdout to ``PREFIX.out``).
    sk_dir
        Directory containing ``X-Y.skf`` Slater–Koster files (3ob-3-1).
    label
        ASE label / working directory prefix.
    third_order
        If True, enable DFTB3/3ob keywords (experimental with ASE 3.26).
    """
    sk_text = str(sk_dir).rstrip("/") + "/"
    symbols = sorted(set(atoms.get_chemical_symbols()))
    params: dict = {
        "label": label,
        "command": command,
        "slako_dir": sk_text,
        "Hamiltonian_SCC": "Yes",
        "Hamiltonian_SCCTolerance": 1e-8,
        "Hamiltonian_MaxAngularMomentum_": "",
        "Analysis_": "",
        "Analysis_CalculateForces": "Yes",
        "ParserOptions_": "",
        "ParserOptions_ParserVersion": 1,
        "ParserOptions_IgnoreUnprocessedNodes": "Yes",
    }
    for sym in symbols:
        params[f"Hamiltonian_MaxAngularMomentum_{sym}"] = _angular_momentum_for_symbol(sym)

    if third_order:
        params.update(
            {
                "Hamiltonian_ThirdOrderFull": "Yes",
                "Hamiltonian_DampXH": "Yes",
                "Hamiltonian_DampXHExponent": 4.0,
                "Hamiltonian_HubbardDerivs_": "",
            }
        )
        for sym in symbols:
            if sym in HUBBARD_DERIVS_3OB:
                params[f"Hamiltonian_HubbardDerivs_{sym}"] = HUBBARD_DERIVS_3OB[sym]

    return DftbPlus24Calculator(atoms=atoms, **params)
