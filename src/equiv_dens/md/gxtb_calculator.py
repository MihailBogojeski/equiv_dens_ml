"""SchNetPack MD calculator backed by g-xtb (subprocess, parse energy/gradient)."""

import equiv_dens.compat  # noqa: F401 - apply T_co patch before schnetpack import

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import ase.io
from ase.data import atomic_numbers

import equiv_dens.utils.base as utils
from schnetpack.md.calculators import MDCalculator, MDCalculatorError
from schnetpack import properties


_CYCLE_HEADER = re.compile(
    r"cycle\s*=\s*(\d+)\s+SCF energy\s*=\s*(\S+)\s+"
    r"\|dE/dxyz\|\s*=\s*(\S+)",
    re.IGNORECASE,
)


def _finite_float(value: str, context: str) -> float:
    """Read the Fortran exponent notation used by g-xTB, rejecting bad output."""
    try:
        number = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as exc:
        raise ValueError(f"Invalid number {value!r} in {context}") from exc
    if not np.isfinite(number):
        raise ValueError(f"Nonfinite number {value!r} in {context}")
    return number


def _output_groups(path: Path, marker: str) -> list[list[str]]:
    """Read complete Turbomole data groups; never reuse a truncated old result."""
    groups = []
    current = None
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line.lower() == marker and current is None:
            current = []
        elif line.lower() == "$end" and current is not None:
            if not current:
                raise ValueError(f"Empty {marker} group in {path}")
            groups.append(current)
            current = None
        elif line.startswith("$") or current is None:
            raise ValueError(
                f"Unexpected output in {path} at line {line_number}: {line}"
            )
        else:
            current.append(line)
    if current is not None or not groups:
        raise ValueError(f"Missing or incomplete {marker}/$end group in {path}")
    return groups


class GxtbMDCalculator(MDCalculator):
    """Run gxtb -grad on coord.xyz each step; forces = -gradient.

    Writes scratch files under workdir. Energy in Eh, gradient in Eh/bohr;
    converted to kcal/mol and kcal/mol/Å for SchNetPack.
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
        """Read the final energy row, or the final complete gradient-cycle header."""
        energy_path = self.workdir / "energy"
        if energy_path.exists():
            for group in _output_groups(energy_path, "$energy"):
                for line in group:
                    parts = line.split()
                    if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) < 1:
                        raise ValueError(
                            f"Malformed energy row in {energy_path}: {line}"
                        )
                    values = [
                        _finite_float(part, str(energy_path)) for part in parts[1:]
                    ]
                    energy = values[0]
            return energy
        grad_path = self.workdir / "gradient"
        if grad_path.exists():
            return self._read_gradient_cycles()[0]
        raise FileNotFoundError("Could not parse energy from g-xtb output")

    def _parse_gradient(self, n: int) -> np.ndarray:
        """Read derivatives from the final complete cycle, excluding coordinates."""
        return self._read_gradient_cycles(n)[1]

    def _read_gradient_cycles(self, n: int | None = None) -> tuple[float, np.ndarray]:
        """Validate every cycle before accepting its energy and Eh/bohr derivatives.

        Each cycle contains a header, labeled coordinate rows, then the same
        number of unlabeled derivative rows. A malformed trailing cycle fails
        rather than returning stale derivatives from an earlier cycle.
        """
        grad_path = self.workdir / "gradient"
        if not grad_path.exists():
            raise FileNotFoundError(f"g-xtb did not produce {grad_path}")
        if n is not None and n <= 0:
            raise ValueError(f"Expected a positive atom count, got {n}")

        for group in _output_groups(grad_path, "$grad"):
            cursor = 0
            while cursor < len(group):
                header = _CYCLE_HEADER.fullmatch(group[cursor])
                if header is None or int(header[1]) < 1:
                    raise ValueError(
                        f"Malformed gradient cycle header in {grad_path}: {group[cursor]}"
                    )
                energy = _finite_float(header[2], str(grad_path))
                _finite_float(header[3], str(grad_path))
                cursor += 1

                coordinate_count = 0
                while cursor < len(group) and len(group[cursor].split()) == 4:
                    parts = group[cursor].split()
                    if not atomic_numbers.get(parts[3].capitalize(), 0):
                        raise ValueError(
                            f"Invalid coordinate atom label in {grad_path}: {parts[3]}"
                        )
                    for part in parts[:3]:
                        _finite_float(part, str(grad_path))
                    coordinate_count += 1
                    cursor += 1
                if not coordinate_count or (n is not None and coordinate_count != n):
                    raise ValueError(
                        f"Expected {n or 'nonzero'} coordinate rows in {grad_path}, "
                        f"got {coordinate_count}"
                    )

                derivatives = []
                for _ in range(coordinate_count):
                    if cursor >= len(group):
                        raise ValueError(f"Incomplete derivative rows in {grad_path}")
                    parts = group[cursor].split()
                    if len(parts) != 3:
                        raise ValueError(
                            f"Malformed derivative row in {grad_path}: {group[cursor]}"
                        )
                    derivatives.append(
                        [_finite_float(part, str(grad_path)) for part in parts]
                    )
                    cursor += 1
                gradient = np.asarray(derivatives, dtype=np.float64)
        return energy, gradient


def _atoms_from_arrays(atom_numbers: np.ndarray, positions: np.ndarray) -> "ase.Atoms":
    """Build ASE Atoms from atom numbers and positions (Angstrom)."""
    nonzero = atom_numbers != 0
    return ase.Atoms(
        atom_numbers[nonzero].astype(int),
        positions=positions[nonzero],
    )
