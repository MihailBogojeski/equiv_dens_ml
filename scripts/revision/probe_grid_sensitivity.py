#!/usr/bin/env python3
"""How much of the ORCA-vs-PySCF density difference is just the XC grid?

calibrate_theory.py found PBE-D4 densities differing by 2.4e-4 relative L2
between the two codes even with ORCA's RI switched off, while wB97M-V with the
same settings agreed to 8e-6. An integration-grid mismatch is the obvious
suspect -- ORCA's DEFGRID3 is far denser than PySCF's default level 3 -- and it
is worth settling, because if the grid is the whole story then PBE labels are
engine-independent once the grid is pinned, and the two codes can share a split.

Converging PySCF against itself is the cheap half of the test: if the density
still moves by ~1e-4 between level 3 and level 7, the default grid alone can
account for the disagreement.

Usage:
  python scripts/revision/probe_grid_sensitivity.py --theory pbe_d4_avdz --n-water 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import dft

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from pyscf_labeler import build_mol, density_fit_coeffs  # noqa: E402
from theory_levels import get_level  # noqa: E402

# A relaxed water dimer, the same geometry the calibration used.
DIMER = (
    [8, 1, 1, 8, 1, 1],
    [
        [0.0000, 0.0000, 0.0000],
        [0.7580, 0.5860, 0.0000],
        [-0.7580, 0.5860, 0.0000],
        [0.0000, 0.0000, 2.8500],
        [0.0000, 0.9000, 3.1600],
        [0.0000, -0.2400, 3.7800],
    ],
)


def run(mol, xc: str, level: int, auxbasis: str, prune: bool) -> tuple[float, np.ndarray]:
    mf = dft.RKS(mol)
    mf.chkfile = None
    mf.xc = xc
    mf.grids.level = level
    if not prune:
        mf.grids.prune = None
    mf.max_cycle = 200
    mf.kernel()
    dm1 = mf.make_rdm1()
    return float(mf.e_tot), density_fit_coeffs(mol, np.asarray(dm1), auxbasis)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", default="pbe_d4_avdz")
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "results/revision/calibration/grid_sensitivity.json"
    )
    args = parser.parse_args()

    level = get_level(args.theory)
    mol = build_mol(*DIMER, level)

    runs = {}
    for grid_level in args.levels:
        energy, coeffs = run(mol, level.pyscf_xc, grid_level, level.auxbasis, prune=True)
        runs[grid_level] = (energy, coeffs)
        print(f"grid level {grid_level}: E = {energy:.10f}")

    # The densest grid run is the converged answer everything else is measured
    # against; the question is how far the production default sits from it.
    ref_level = max(runs)
    e_ref, c_ref = runs[ref_level]
    report = {"theory": args.theory, "reference_grid_level": ref_level, "levels": {}}
    for grid_level, (energy, coeffs) in sorted(runs.items()):
        entry = {
            "energy": energy,
            "d_energy_vs_ref": energy - e_ref,
            "df_rel_l2_vs_ref": rel_l2(coeffs, c_ref),
        }
        report["levels"][str(grid_level)] = entry
        print(
            f"level {grid_level}: dE = {entry['d_energy_vs_ref']:+.3e} Ha  "
            f"df rel L2 = {entry['df_rel_l2_vs_ref']:.3e}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
