#!/usr/bin/env python3
"""Write an ORCA job that prints the AO ordering and overlap for a CSH entry.

Reverse-engineering the stored AO order from Fock matrices runs into a noise
floor: reproducing ORCA's wB97M-V/RIJCOSX numbers in PySCF leaves ~0.01-0.1 Ha
of method error per element, which is the same size as the spacing between
near-degenerate polarization components.

The overlap matrix has no such problem. It depends only on the basis and the
geometry - no functional, no grid, no RI - so ORCA's S and PySCF's S are the
same numbers to machine precision, and any disagreement is purely ordering.
Printing the MO block additionally gives ORCA's AO labels in order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import parse_charge_mult  # noqa: E402

SYMBOLS = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 35: "Br", 53: "I",
}

TEMPLATE = """! HF def2-TZVPD NoIter NoPop
%output
  Print[P_Overlap] 1
  Print[P_MOs] 1
end
%maxcore 2000
* xyz {charge} {mult}
{geometry}
*
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--subset", default="ani1xbb")
    parser.add_argument("--entry", default=None)
    parser.add_argument("--outdir", default="results/revision/orca_overlap")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as fh:
        group = fh[args.subset]
        name = args.entry or sorted(group.keys())[0]
        node = group[name]
        elements = np.asarray(node["elements"][()])
        coords = np.asarray(node["coords"][()])

    charge, mult = parse_charge_mult(name)
    lines = []
    for z, xyz in zip(elements, coords):
        symbol = SYMBOLS.get(int(z))
        if symbol is None:
            raise SystemExit(f"add element Z={int(z)} to SYMBOLS")
        lines.append(f"  {symbol:<2} {xyz[0]:>14.8f} {xyz[1]:>14.8f} {xyz[2]:>14.8f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    inp = outdir / "csh_ao.inp"
    inp.write_text(TEMPLATE.format(charge=charge, mult=mult, geometry="\n".join(lines)))

    (outdir / "entry.txt").write_text(f"{args.subset}/{name}\n")
    print(f"entry   : {args.subset}/{name}")
    print(f"natoms  : {len(elements)}  charge={charge} mult={mult}")
    print(f"wrote   : {inp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
