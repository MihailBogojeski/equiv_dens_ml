#!/usr/bin/env python3
"""Compare DenSNet and PySCF equilibria (R2.11).

If --model is omitted, only the DFT optimization is run (useful as a
reference while paper checkpoints are restored).
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/optimize_densnet.py'


import argparse
import json
import sys
from pathlib import Path


_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402
ensure_revision_imports(_SCRIPT_DIR)
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from experiments.jacs_revision.lib.geoopt_metrics import heavy_atom_rmsd, kabsch_rmsd, summarize_internal_errors

_REPO_ROOT = repo_root()
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "md"))


def _geometry_report(symbols, dens_pos, dft_pos):
    internals = summarize_internal_errors(symbols, dens_pos, dft_pos)
    return {
        "rmsd_angstrom": kabsch_rmsd(dens_pos, dft_pos),
        "heavy_atom_rmsd_angstrom": heavy_atom_rmsd(dens_pos, dft_pos, symbols),
        "bond_mae_ang": internals["bond_mae_ang"],
        "angle_mae_deg": internals["angle_mae_deg"],
    }


def _pyscf_calculator(xc, use_d4):
    from aimd_gpu4pyscf import PySCFCalculator

    return PySCFCalculator(xc=xc) if "xc" in PySCFCalculator.__init__.__code__.co_varnames else PySCFCalculator()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--args-file", default=None)
    parser.add_argument(
        "--np-dataset",
        default=None,
        help="training NPY with forces, for output-scaling recovery",
    )
    parser.add_argument("--dft-xc", default="pbe")
    parser.add_argument(
        "--basis",
        default="cc-pVDZ",
        help="PySCF basis for the DFT reference minimum (ethanol training is cc-pVDZ)",
    )
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--skip-dft", action="store_true")
    parser.add_argument(
        "--reference-xyz",
        default=None,
        help="use this DFT-relaxed xyz instead of running PySCF",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/revision/geoopt.json"))
    args = parser.parse_args()

    atoms0 = read(args.structure, index=0)
    dft_atoms = atoms0.copy()
    if args.reference_xyz:
        dft_atoms = read(args.reference_xyz, index=0)
        dft_pos = dft_atoms.get_positions()
        dft_e = dft_atoms.info.get("energy")
        if dft_e is not None:
            dft_e = float(dft_e)
        dft_err = None
    elif args.skip_dft:
        dft_pos = atoms0.get_positions()
        dft_e = None
        dft_err = "skipped"
    else:
        try:
            from pyscf import gto, dft
            import ase.units
            from ase.calculators.calculator import Calculator, all_changes

            class _RKS(object):
                def __init__(self, xc, basis):
                    self.xc = xc
                    self.basis = basis

                def get_potential_energy(self, atoms):
                    mol = gto.M(
                        atom=[(s, t) for s, t in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
                        basis=self.basis,
                    )
                    mf = dft.RKS(mol)
                    mf.xc = self.xc
                    e = mf.kernel()
                    self._mf = mf
                    return e * ase.units.Hartree

                def get_forces(self, atoms):
                    g = self._mf.nuc_grad_method().kernel()
                    return -np.asarray(g) / ase.units.Bohr * ase.units.Hartree

            class PySCFAse(Calculator):
                implemented_properties = ["energy", "forces"]

                def __init__(self, xc="pbe", basis="cc-pVDZ"):
                    super().__init__()
                    self._rks = _RKS(xc, basis)

                def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
                    super().calculate(atoms, properties, system_changes)
                    e = self._rks.get_potential_energy(atoms)
                    f = self._rks.get_forces(atoms)
                    self.results = {"energy": e, "forces": f}

            dft_atoms.calc = PySCFAse(xc=args.dft_xc, basis=args.basis)
            BFGS(dft_atoms, logfile=None).run(fmax=args.fmax)
            dft_pos = dft_atoms.get_positions()
            dft_e = float(dft_atoms.get_potential_energy())
            dft_err = None
        except Exception as exc:
            dft_pos = atoms0.get_positions()
            dft_e = None
            dft_err = str(exc)

    dens_pos = None
    dens_e = None
    dens_err = None
    dens_atoms = None
    if args.model:
        try:
            from equiv_dens.md.dft_network_calculator import load_densnet_calculator

            dens_atoms = (dft_atoms if args.reference_xyz else atoms0).copy()
            dens_atoms.calc = load_densnet_calculator(
                args.model,
                args_file=args.args_file,
                np_dataset=args.np_dataset,
                use_gpu=args.use_gpu,
                pyscf_grid=False,
            )
            log_path = args.out.with_suffix(".bfgs.log")
            LBFGS(dens_atoms, logfile=str(log_path), maxstep=0.05).run(
                fmax=args.fmax, steps=args.max_steps
            )
            dens_pos = dens_atoms.get_positions()
            dens_e = float(dens_atoms.get_potential_energy())
        except Exception as exc:
            dens_err = str(exc)

    symbols = list(atoms0.get_chemical_symbols())
    geometry = (
        _geometry_report(symbols, dens_pos, dft_pos) if dens_pos is not None else {}
    )
    summary = {
        "structure": args.structure,
        "model": args.model,
        "dft_xc": args.dft_xc,
        "basis": args.basis,
        "dft_energy_eV": dft_e,
        "dft_error": dft_err,
        "densnet_energy_eV": dens_e,
        "densnet_error": dens_err,
        "n_atoms": int(len(atoms0)),
        **geometry,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    dft_xyz = Path(f"{stem}_dft.xyz")
    dens_xyz = Path(f"{stem}_densnet.xyz")
    write(dft_xyz, dft_atoms)
    summary["dft_xyz"] = str(dft_xyz)
    if dens_atoms is not None:
        write(dens_xyz, dens_atoms)
        summary["densnet_xyz"] = str(dens_xyz)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
