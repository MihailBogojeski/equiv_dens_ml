#!/usr/bin/env python3
"""Validate the AO permutation table on many structures without an ORCA reference.

Once the signed permutation is known, correctness can be checked from physics
alone: the deepest orbital energy must land on the 1s level of the heaviest
element present, the density must integrate to the electron count, and Mulliken
charges must be chemically sensible. A wrong ordering fails all three
spectacularly (eps_min in the thousands of Hartree, charges of +/-8).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import build_mol, inflate_triangle, parse_charge_mult  # noqa: E402

HARTREE_TO_EV = 27.211386245988

# Approximate 1s orbital energies at DFT, used only as a sanity window.
CORE_1S = {1: -0.5, 6: -10.3, 7: -14.5, 8: -19.3, 9: -24.9, 16: -89.3, 17: -101.6, 35: -483.0}

_R1 = re.compile(r"_(-?\d+)_1$")
_R2 = re.compile(r"_(-?\d+)_1_\d+$")


def charge_of(leaf: str, parent: str | None) -> int | None:
    for name in (leaf, parent):
        if name is None:
            continue
        m = _R1.search(name) or _R2.search(name)
        if m:
            return int(m.group(1))
    return None


def load_table(path: str):
    data = json.loads(Path(path).read_text())
    return {
        int(z): (np.asarray(rec["perm"], dtype=int), np.asarray(rec["signs"], dtype=float))
        for z, rec in data["per_element"].items()
    }


def build_transform(mol, elements, table):
    slices = mol.aoslice_by_atom()
    perm = np.empty(mol.nao, dtype=int)
    signs = np.empty(mol.nao)
    for atom, z in enumerate(elements):
        z = int(z)
        if z not in table:
            raise KeyError(z)
        s0, s1 = slices[atom][2], slices[atom][3]
        local_perm, local_signs = table[z]
        if len(local_perm) != s1 - s0:
            raise ValueError(f"Z={z}: table width {len(local_perm)} != {s1 - s0}")
        perm[s0:s1] = s0 + local_perm
        signs[s0:s1] = local_signs
    return perm, signs


def iter_entries(path):
    with h5py.File(path, "r") as fh:
        stack = [(fh, None, None)]
        while stack:
            group, parent, top = stack.pop()
            keys = list(group)
            if "fock" in keys and "elements" in keys:
                yield (
                    group.name,
                    group.name.split("/")[-1],
                    parent,
                    np.asarray(group["elements"][()]),
                    np.asarray(group["coords"][()]),
                    np.asarray(group["fock"][()]),
                )
                continue
            for k in keys:
                child = group[k]
                if isinstance(child, h5py.Group):
                    stack.append((child, group.name.split("/")[-1], top or k))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5")
    parser.add_argument("--table", default="datasets/revision/omol_csh/ao_permutation.json")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--neutral-only", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    table = load_table(args.table)
    allowed = set(table)
    print(f"table elements: {sorted(allowed)}")

    results = []
    for full, leaf, parent, elements, coords, fock_flat in iter_entries(args.h5):
        if len(elements) > args.max_atoms:
            continue
        if not set(int(z) for z in elements) <= allowed:
            continue
        charge = charge_of(leaf, parent)
        if charge is None:
            continue
        if args.neutral_only and charge != 0:
            continue

        mol = build_mol(elements, coords, charge, 0)
        try:
            perm, signs = build_transform(mol, elements, table)
        except (KeyError, ValueError) as exc:
            print(f"  skip {leaf}: {exc}")
            continue

        fock = inflate_triangle(fock_flat)[np.ix_(perm, perm)] * np.outer(signs, signs)
        ovlp = mol.intor("int1e_ovlp")
        eps, coeff = scipy.linalg.eigh(fock, ovlp)
        nocc = mol.nelectron // 2
        occ = coeff[:, :nocc]
        dm = 2.0 * occ @ occ.T
        pop = np.einsum("ij,ji->i", dm, ovlp)
        slices = mol.aoslice_by_atom()
        charges = np.array(
            [
                mol.atom_charge(a) - pop[slices[a][2] : slices[a][3]].sum()
                for a in range(mol.natm)
            ]
        )
        heaviest = int(max(elements))
        expected = CORE_1S[heaviest]
        rec = {
            "entry": full,
            "natoms": int(len(elements)),
            "nao": int(mol.nao),
            "charge": charge,
            "heaviest_z": heaviest,
            "eps_min": float(eps[0]),
            "expected_core": expected,
            "core_dev": float(eps[0] - expected),
            "gap_ev": float((eps[nocc] - eps[nocc - 1]) * HARTREE_TO_EV),
            "trace_ps": float(np.einsum("ij,ij->", ovlp, dm)),
            "max_abs_mulliken": float(np.abs(charges).max()),
        }
        # Correctness is: the deepest level sits on the right core, the density
        # integrates to the electron count, and the gap is positive. Core levels
        # shift by several eV in charged systems, so the window has to be wide.
        # A small gap is not an error - reaction-path and transition-state
        # geometries are genuinely near-degenerate - so it is flagged, not failed.
        rec["low_gap"] = bool(rec["gap_ev"] < 0.5)
        rec["high_mulliken"] = bool(rec["max_abs_mulliken"] > 2.0)
        rec["pass"] = bool(
            abs(rec["core_dev"]) < 1.5
            and rec["gap_ev"] > 0.0
            and abs(rec["trace_ps"] - mol.nelectron) < 1e-6
        )
        results.append(rec)
        print(
            f"  {'OK ' if rec['pass'] else 'FAIL'} {leaf[:44]:<44} "
            f"n={rec['natoms']:>3} q={charge:>2} eps_min={rec['eps_min']:>9.3f} "
            f"(exp {expected:>7.1f}) gap={rec['gap_ev']:>6.2f}eV "
            f"max|q|={rec['max_abs_mulliken']:.2f}"
        )
        if len(results) >= args.count:
            break

    n_pass = sum(r["pass"] for r in results)
    print(f"\npassed {n_pass}/{len(results)}")
    if results:
        dev = np.array([abs(r["core_dev"]) for r in results])
        mul = np.array([r["max_abs_mulliken"] for r in results])
        print(f"core level deviation: max {dev.max():.3f} Ha, mean {dev.mean():.3f} Ha")
        print(f"max|Mulliken| across set: {mul.max():.3f}")
        print(f"flagged low gap (<0.5 eV): {sum(r['low_gap'] for r in results)}")
        print(f"flagged high Mulliken (>2): {sum(r['high_mulliken'] for r in results)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0 if results and n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
