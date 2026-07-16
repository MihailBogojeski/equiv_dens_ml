"""
g-xtb MD calculator for SchNetPack molecular dynamics.

Uses g-xtb for energy and gradient evaluation, reusing SchNetPack MD
infrastructure (integrators, thermostats, logging). Energy and gradient
are parsed from g-xtb output files; forces = -gradient.
"""

import equiv_dens.compat  # noqa: F401 - apply T_co patch before schnetpack import

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import ase.io

import equiv_dens.utils.base as utils
from schnetpack.md.calculators import MDCalculator, MDCalculatorError
from schnetpack import properties


class GxtbMDCalculator(MDCalculator):
    """
    SchNetPack MD calculator that uses g-xtb for energy and gradient.

    Extends schnetpack.md.calculators.MDCalculator. Writes coord.xyz,
    runs gxtb -grad -c coord.xyz, parses energy (Eh) and gradient (Eh/bohr),
    converts to kcal/mol and kcal/mol/Å, and updates the system.

    Args:
        gxtb_path: Path to the gxtb executable.
        gxtb_params_dir: Path to GXTBHOME (parameter directory).
        workdir: Scratch directory for g-xtb I/O (created per step).
        required_properties: Properties to compute (default: ['energy', 'forces']).
        force_key: Key for forces in results (default: 'forces').
        energy_key: Key for energy in results (default: 'energy').
        position_unit: Unit of positions passed to calculator (default: 'Angstrom').
        energy_unit: Unit of energy in results (default: 'kcal/mol').
    """

    def __init__(
        self,
        gxtb_path: str,
        gxtb_params_dir: str,
        workdir: str | Path,
        required_properties: list[str] | None = None,
        force_key: str = "forces",
        energy_key: str = "energy",
        position_unit: str = "Angstrom",
        energy_unit: str = "kcal/mol",
    ):
        if required_properties is None:
            required_properties = ["energy", "forces"]
        super().__init__(
            required_properties=required_properties,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            energy_key=energy_key,
        )
        self.gxtb_path = str(Path(gxtb_path).resolve())
        self.gxtb_params_dir = str(Path(gxtb_params_dir).resolve())
        self.workdir = Path(workdir)
        self._coord_file = "coord.xyz"

    def calculate(self, system):
        """
        Compute energy and forces with g-xtb and update the system.

        Args:
            system: schnetpack.md.System with current positions and atom types.
        """
        sys_mols = self._get_system_molecules(system)
        positions = sys_mols[properties.R]  # (n_atoms_total, 3)
        atom_types = sys_mols[properties.Z]  # (n_atoms_total,)
        n_atoms = int(sys_mols[properties.n_atoms][0])

        positions_np = positions.detach().cpu().numpy()
        atom_types_np = atom_types.detach().cpu().numpy()

        # Reshape for single molecule: (1, n_atoms, 3) -> (n_atoms, 3)
        positions_flat = positions_np.reshape(-1, 3)[:n_atoms]
        atom_types_flat = atom_types_np.reshape(-1)[:n_atoms]

        # Positions from _get_system_molecules are in position_unit (Angstrom)
        atoms = _atoms_from_arrays(atom_types_flat, positions_flat)

        energy_hartree, gradient_eh_bohr = self._run_gxtb(atoms)

        # Convert: energy Eh -> kcal/mol; gradient Eh/bohr -> forces kcal/mol/Å
        energy_kcal = utils.hartree_to_kcal(energy_hartree)
        # forces = -gradient; gradient in Eh/bohr -> force in Eh/bohr
        # 1 Eh/bohr = (627.5 kcal/mol) / (0.529 Å) = 1186 kcal/mol/Å
        gradient_au = -gradient_eh_bohr  # forces in Eh/bohr (positive = repulsion)
        force_factor = utils.hartree_to_kcal(1.0) / utils.bohr_to_angstrom(
            np.array([1.0])
        )
        forces_kcal_ang = gradient_au * float(force_factor)

        # Store as torch tensors; SchNetPack expects (n_replicas, n_atoms, 3)
        self.results = {
            "energy": torch.tensor(
                [[energy_kcal]], dtype=positions.dtype, device=positions.device
            ),
            "forces": torch.tensor(
                forces_kcal_ang, dtype=positions.dtype, device=positions.device
            ).unsqueeze(0),
        }
        self._update_system(system)

    def _run_gxtb(self, atoms: "ase.Atoms") -> tuple[float, np.ndarray]:
        """Run g-xtb -grad and return (energy_Eh, gradient_Eh_bohr)."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        coord_path = self.workdir / self._coord_file
        ase.io.write(coord_path, atoms, format="xyz")

        env = os.environ.copy()
        env["GXTBHOME"] = self.gxtb_params_dir
        gxtb_dir = str(Path(self.gxtb_path).resolve().parent)
        env["PATH"] = gxtb_dir + os.pathsep + env.get("PATH", "")

        result = subprocess.run(
            [self.gxtb_path, "-grad", "-c", self._coord_file],
            cwd=str(self.workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
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
        """Parse energy (Eh) from $energy or gradient file."""
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
        """Parse gradient (Eh/bohr) from gradient file."""
        grad_path = self.workdir / "gradient"
        if not grad_path.exists():
            raise FileNotFoundError(f"g-xtb did not produce {grad_path}")

        gradient = np.zeros((n, 3), dtype=np.float64)
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


def _atoms_from_arrays(atom_numbers: np.ndarray, positions: np.ndarray) -> "ase.Atoms":
    """Build ASE Atoms from atom numbers and positions (Angstrom)."""
    nonzero = atom_numbers != 0
    return ase.Atoms(
        atom_numbers[nonzero].astype(int),
        positions=positions[nonzero],
    )
