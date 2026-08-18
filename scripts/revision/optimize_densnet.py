#!/usr/bin/env python3
"""Compare DenSNet and PySCF equilibria (R2.11).

If --model is omitted, only the DFT optimization is run (useful as a
reference while paper checkpoints are restored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.optimize import BFGS

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "md"))


def _rms(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _pyscf_calculator(xc, use_d4):
    from aimd_gpu4pyscf import PySCFCalculator

    return PySCFCalculator(xc=xc) if "xc" in PySCFCalculator.__init__.__code__.co_varnames else PySCFCalculator()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dft-xc", default="pbe")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("results/revision/geoopt.json"))
    args = parser.parse_args()

    atoms0 = read(args.structure, index=0)
    dft_atoms = atoms0.copy()
    try:
        from ase.calculators.emt import EMT

        # Placeholder if PySCF AIMD calculator signature differs; prefer PySCF.
        raise ImportError
    except ImportError:
        pass

    try:
        from pyscf import gto, dft
        import ase.units

        class _RKS(object):
            def __init__(self, xc):
                self.xc = xc

            def get_potential_energy(self, atoms):
                mol = gto.M(
                    atom=[(s, t) for s, t in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
                    basis="augccpvdz",
                )
                mf = dft.RKS(mol)
                mf.xc = self.xc
                e = mf.kernel()
                self._mf = mf
                return e * ase.units.Hartree

            def get_forces(self, atoms):
                g = self._mf.nuc_grad_method().kernel()
                return -np.asarray(g) / ase.units.Bohr * ase.units.Hartree

        from ase.calculators.calculator import Calculator, all_changes

        class PySCFAse(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self, xc="pbe"):
                super().__init__()
                self._rks = _RKS(xc)

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                e = self._rks.get_potential_energy(atoms)
                f = self._rks.get_forces(atoms)
                self.results = {"energy": e, "forces": f}

        dft_atoms.calc = PySCFAse(xc=args.dft_xc)
        BFGS(dft_atoms, logfile=None).run(fmax=args.fmax)
        dft_pos = dft_atoms.get_positions()
        dft_e = float(dft_atoms.get_potential_energy())
    except Exception as exc:
        dft_pos = atoms0.get_positions()
        dft_e = None
        dft_err = str(exc)
    else:
        dft_err = None

    dens_pos = None
    dens_e = None
    dens_err = None
    if args.model:
        try:
            from equiv_dens.training.model_loader import load_model
            from equiv_dens.md.dft_network_calculator import DFTNetworkCalculator

            raise RuntimeError(
                "Wire DFTNetworkCalculator through run.py md internals; "
                "use `python run.py md` with max_steps=0 + ASE BFGS once the "
                f"checkpoint at {args.model} is restored."
            )
        except Exception as exc:
            dens_err = str(exc)

    summary = {
        "structure": args.structure,
        "model": args.model,
        "dft_xc": args.dft_xc,
        "dft_energy_eV": dft_e,
        "dft_error": dft_err,
        "densnet_energy_eV": dens_e,
        "densnet_error": dens_err,
        "rmsd_angstrom": _rms(dens_pos, dft_pos) if dens_pos is not None else None,
        "n_atoms": int(len(atoms0)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
