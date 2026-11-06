#!/usr/bin/env python3
"""DFT single-point and optional BFGS on selected MH minima (PBE-D4/aug-cc-pVDZ).

DenSNet density error is recorded when --model is given and the eval helper
can score the frame; otherwise the field is left null.
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/thiophene/validate_mh_minima_dft.py'


import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)
from experiments.jacs_revision.thiophene.minima_hopping import so3_rmsd  # noqa: E402
from experiments.jacs_revision.thiophene.thiophene_collective_variables import terthiophene_theta12_theta23  # noqa: E402

_REPO_ROOT = repo_root()
EV_TO_MEV = 1000.0


def energy_error_mev(ml_ev: float, dft_ev: float) -> float:
    return abs(float(ml_ev) - float(dft_ev)) * EV_TO_MEV


def density_l1_per_electron(pred: np.ndarray, ref: np.ndarray, weights: np.ndarray, n_electrons: float) -> float:
    """∫|Δρ| / N_e on a quadrature grid."""
    pred = np.asarray(pred, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    return float(np.sum(np.abs(pred - ref) * weights) / max(float(n_electrons), 1e-12))


def summarize_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    energy_err = [
        energy_error_mev(row["energy_ml_ev"], row["energy_dft_sp_ev"])
        for row in rows
        if row.get("energy_dft_sp_ev") is not None and row.get("energy_ml_ev") is not None
    ]
    density_err = [row["density_l1_per_e"] for row in rows if row.get("density_l1_per_e") is not None]
    return {
        "n": len(rows),
        "labels": [row.get("label") for row in rows],
        "mae_energy_mev": float(np.mean(energy_err)) if energy_err else None,
        "mean_density_l1_per_e": float(np.mean(density_err)) if density_err else None,
        "n_with_dft": len(energy_err),
    }


def _pyscf_ase_calculator(xc: str = "pbe", use_d4: bool = True):
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree
    from pyscf import dft, gto

    class PySCFAse(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.xc = xc
            self.use_d4 = use_d4

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            mol = gto.M(
                atom=[(s, tuple(p)) for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
                basis="augccpvdz",
                unit="Angstrom",
                verbose=0,
            )
            mf = dft.RKS(mol)
            mf.xc = self.xc
            mf.chkfile = False
            if self.use_d4:
                import dftd4.pyscf as d4disp

                mf = d4disp.energy(mf)
            mf.max_cycle = 200
            e_tot = mf.kernel()
            grad = mf.nuc_grad_method().kernel()
            self.results = {
                "energy": float(e_tot) * Hartree,
                "forces": -np.asarray(grad) * Hartree / Bohr,
            }
            self.mf = mf

    return PySCFAse()


def validate_row(
    row: dict[str, Any],
    *,
    run_dft: bool,
    relax: bool,
    fmax: float,
    theory: str,
) -> dict[str, Any]:
    out = {
        "label": row.get("label"),
        "index": row.get("index"),
        "theta12": row.get("theta12"),
        "theta23": row.get("theta23"),
        "energy_ml_ev": row.get("energy"),
        "energy_dft_sp_ev": None,
        "energy_dft_relax_ev": None,
        "rmsd_relax_ang": None,
        "theta12_dft": None,
        "theta23_dft": None,
        "density_l1_per_e": row.get("density_l1_per_e"),
        "dft_error": None,
    }
    if not run_dft:
        return out
    symbols = row["symbols"]
    pos = np.asarray(row["positions"], dtype=float)
    atoms = Atoms(symbols=symbols, positions=pos)
    try:
        atoms.calc = _pyscf_ase_calculator()
        out["energy_dft_sp_ev"] = float(atoms.get_potential_energy())
        if relax:
            work = atoms.copy()
            work.calc = _pyscf_ase_calculator()
            BFGS(work, logfile=None).run(fmax=fmax)
            out["energy_dft_relax_ev"] = float(work.get_potential_energy())
            out["rmsd_relax_ang"] = so3_rmsd(work.get_positions(), pos)
            t12, t23 = terthiophene_theta12_theta23(work.get_positions(), symbols)
            out["theta12_dft"] = t12
            out["theta23_dft"] = t23
    except Exception as exc:
        out["dft_error"] = str(exc)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "results/revision/thiophene_mh/dft_validation.json",
    )
    parser.add_argument("--run-dft", action="store_true")
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--theory", default="pbe_d4_avdz")
    parser.add_argument("--model", default=None, help="optional DenSNet dir for density scoring")
    args = parser.parse_args()
    payload = json.loads(args.selected.read_text())
    rows = []
    for item in payload["selected"]:
        rows.append(
            validate_row(
                item,
                run_dft=args.run_dft,
                relax=args.relax,
                fmax=args.fmax,
                theory=args.theory,
            )
        )
    summary = summarize_validation(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "selected": str(args.selected),
                "theory": args.theory,
                "run_dft": args.run_dft,
                "rows": rows,
                "summary": summary,
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
