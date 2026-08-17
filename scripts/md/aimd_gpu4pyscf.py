#!/usr/bin/env python3
"""
Ab initio molecular dynamics with PySCF/gpu4pyscf.

Runs AIMD using PBE/aug-cc-pVDZ + D4 dispersion. Uses GPU acceleration
when gpu4pyscf is available.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ase
import ase.io
import ase.units
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pyscf import gto, dft, scf
import dftd4.pyscf as d4disp

# Add src to path for equiv_dens imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

BASIS = "augccpvdz"
AUXBASIS = "augccpvdzjkfit"  # JK-fit for aug-cc-pVDZ; enables GPU4PySCF density-fitting speedup


def _try_import_gpu4pyscf():
    """Try to import gpu4pyscf. Returns True if available."""
    try:
        import gpu4pyscf  # noqa: F401
        return True
    except ImportError:
        return False


class PySCFCalculator(Calculator):
    """
    ASE Calculator wrapping PySCF/gpu4pyscf DFT with PBE/aug-cc-pVDZ + D4.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        basis: str = BASIS,
        xc: str = "pbe",
        use_gpu: bool = True,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.basis = basis
        self.xc = xc
        self.use_gpu = use_gpu and _try_import_gpu4pyscf()
        self.verbose = verbose
        self._dm_prev: np.ndarray | None = None
        self._mol_prev = None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if atoms is None:
            atoms = self.atoms
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
        atom = [(int(z), pos) for z, pos in zip(numbers, positions) if z > 0]

        mol = gto.M(atom=atom, basis=self.basis)
        mol.build()

        mf = dft.RKS(mol).density_fit(auxbasis=AUXBASIS)
        mf.chkfile = None
        mf.xc = self.xc
        mf.verbose = self.verbose
        # Looser convergence for MD: fewer SCF cycles per step
        mf.conv_tol = 1e-5
        mf.conv_tol_grad = 1e-3

        if self.use_gpu:
            mf = mf.to_gpu()

        # Warm restart: project previous step's density matrix as initial guess
        dm0 = None
        if self._dm_prev is not None and self._mol_prev is not None:
            try:
                dm0 = scf.addons.project_dm_nr2nr(
                    self._mol_prev, self._dm_prev, mol
                )
            except (ValueError, TypeError):
                self._dm_prev = None
                self._mol_prev = None

        mf.kernel(dm0=dm0)

        # dftd4.pyscf requires standard PySCF SCF; convert GPU object to CPU
        if self.use_gpu:
            mf = mf.to_cpu()

        d4mf = d4disp.energy(mf).run()
        grad = d4mf.nuc_grad_method()
        gradients = grad.kernel()

        # Convert to numpy if CuPy
        if hasattr(gradients, "get"):
            gradients = gradients.get()
        gradients = np.asarray(gradients)

        # Energy: Hartree -> eV
        energy_ev = float(d4mf.e_tot) * ase.units.Hartree
        # Forces: -dE/dR in Hartree/Bohr -> eV/Å (ASE convention: F = -dE/dR)
        forces_ev_ang = -gradients * ase.units.Hartree / ase.units.Bohr

        self.results = {
            "energy": energy_ev,
            "forces": forces_ev_ang,
        }

        # Store for warm restart at next step
        dm1 = d4mf.make_rdm1(d4mf.mo_coeff, d4mf.mo_occ)
        if hasattr(dm1, "get"):
            dm1 = dm1.get()
        self._dm_prev = np.asarray(dm1)
        self._mol_prev = mol


def _check_closed_shell(atoms: Atoms) -> None:
    """Raise ValueError if structure has odd electron count (open shell)."""
    nelec = sum(gto.charge(int(z)) for z in atoms.get_atomic_numbers() if z > 0)
    if nelec % 2 != 0:
        raise ValueError(
            f"Structure has odd electron count ({nelec}); only closed-shell systems "
            "are supported. Check input structure."
        )


def load_structure(path: str) -> Atoms:
    """Load initial structure from XYZ or NPY."""
    p = Path(path)
    if p.suffix.lower() == ".xyz":
        atoms_list = list(ase.io.iread(str(p)))
        return atoms_list[0]
    if p.suffix.lower() == ".npy":
        data = np.load(p, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item() if data.ndim == 0 else data
        if isinstance(data, dict):
            pos = data["positions"]
            anum = data["atom_numbers"]
            if pos.ndim == 3:
                pos = pos[0]
            if anum.ndim > 1:
                anum = anum[0]
            nonzero = anum > 0
            return Atoms(numbers=anum[nonzero], positions=pos[nonzero])
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            item = data[0]
            if isinstance(item, (list, tuple)) and len(item) == 2:
                mol = gto.Mole.unpack(item[0])
                mol.build()
                pos = mol.atom_coords()
                anum = np.array([a[0] for a in item[0]["atom"]])
                return Atoms(numbers=anum, positions=pos)
    raise ValueError(f"Cannot load structure from {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ab initio MD with PySCF/gpu4pyscf (PBE/aug-cc-pVDZ + D4)"
    )
    parser.add_argument(
        "--structure",
        "-s",
        type=str,
        required=True,
        help="Initial structure: XYZ or NPY",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="aimd_trajectory.traj",
        help="Output trajectory path (default: aimd_trajectory.traj)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of MD steps (default: 1000)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.5,
        help="Timestep in fs (default: 0.5)",
    )
    parser.add_argument(
        "--temperature",
        "-T",
        type=float,
        default=300.0,
        help="Temperature in K (default: 300)",
    )
    parser.add_argument(
        "--ensemble",
        choices=["nve", "nvt"],
        default="nvt",
        help="Ensemble: nve (microcanonical) or nvt (Langevin) (default: nvt)",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.01,
        help="Langevin friction (NVT only, default: 0.01)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU (use CPU PySCF only)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log energy/temperature every N steps (default: 10)",
    )
    parser.add_argument(
        "--traj-interval",
        type=int,
        default=1,
        help="Write trajectory every N steps (default: 1)",
    )
    parser.add_argument(
        "--no-traj",
        action="store_true",
        help="Skip writing trajectory file (for scaling benchmarks)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for initial velocities",
    )

    args = parser.parse_args()

    use_gpu = not args.no_gpu and _try_import_gpu4pyscf()
    print(f"GPU acceleration: {'enabled' if use_gpu else 'disabled'}")

    atoms = load_structure(args.structure)
    _check_closed_shell(atoms)
    calc = PySCFCalculator(use_gpu=use_gpu)
    atoms.calc = calc

    # Initial velocities
    rng = np.random.RandomState(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)

    # MD integrator
    dt = args.timestep * ase.units.fs
    if args.ensemble == "nvt":
        dyn = Langevin(
            atoms,
            dt,
            temperature_K=args.temperature,
            friction=args.friction,
            rng=rng,
        )
    else:
        dyn = VelocityVerlet(atoms, dt)

    # Logging
    def log_callback():
        e = atoms.get_potential_energy()
        T = atoms.get_temperature()
        print(f"  Step {dyn.nsteps}: E = {e:.4f} eV, T = {T:.1f} K")

    dyn.attach(log_callback, interval=args.log_interval)

    # Trajectory (optional)
    if not args.no_traj:
        traj_path = Path(args.output)
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj = Trajectory(traj_path, "w", atoms)
        dyn.attach(traj.write, interval=args.traj_interval)

    # Run
    print(f"Running {args.ensemble.upper()} MD: {args.steps} steps, dt = {args.timestep} fs")
    print(f"Temperature: {args.temperature} K")
    if not args.no_traj:
        print(f"Output: {Path(args.output)}")
    start_time = time.time()
    dyn.run(args.steps)
    elapsed_s = time.time() - start_time
    ns_total = args.steps * args.timestep / 1e6  # fs -> ns
    ns_per_day = ns_total / (elapsed_s / 86400.0) if elapsed_s > 0 else 0.0
    fmt = f"{ns_per_day:.3f}" if ns_per_day < 0.1 else f"{ns_per_day:.1f}"
    print(f"Final instantaneous speed: {fmt} ns/day")
    print("Done.")


if __name__ == "__main__":
    main()
