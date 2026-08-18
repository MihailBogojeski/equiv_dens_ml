#!/usr/bin/env python3
"""Brute-force the AO layout of the stored CSH Fock matrices.

Four independent axes are scanned, since the stored order turns out not to be a
plain ORCA dump:

  m_order   within-shell component order: PySCF (m ascending, but (x,y,z) for
            l=1) or ORCA (m = 0, +1, -1, +2, -2, ...)
  signs     ORCA inverts the phase of the |m| >= 3 real solid harmonics
  layout    for a given l with several shells, whether AOs run shell-major
            (all components of shell 1, then shell 2, ...) or component-major
            (component 1 of every shell, then component 2, ...)
  shell_rev whether shells of a given l are listed tight-to-diffuse or reversed

Scoring uses physics rather than a fitted reference: the reconstructed density
must give sane Mulliken charges, and the deepest orbital energy must land on
the 1s level of the heaviest element present.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import build_mol, inflate_triangle, parse_charge_mult  # noqa: E402

# Approximate 1s orbital energies (Hartree) at DFT, for scoring only.
CORE_1S = {1: -0.5, 6: -10.2, 7: -14.4, 8: -19.2, 9: -24.7, 16: -88.9, 17: -101.5, 35: -481.0}


def m_list(l: int, convention: str) -> list[int]:
    if convention == "pyscf":
        return [1, -1, 0] if l == 1 else list(range(-l, l + 1))
    if convention == "ascending":
        return list(range(-l, l + 1))
    if convention == "orca":
        out = [0]
        for m in range(1, l + 1):
            out.extend([m, -m])
        return out
    raise ValueError(convention)


def build_perm(mol, m_order: str, layout: str, shell_rev: bool) -> np.ndarray:
    """perm[i] = stored index holding the AO that PySCF puts at position i."""
    perm = np.empty(mol.nao, dtype=int)
    ao_loc = mol.ao_loc_nr()

    shells_by_atom: dict[int, list[tuple[int, int]]] = {}
    for shell in range(mol.nbas):
        shells_by_atom.setdefault(mol.bas_atom(shell), []).append((shell, mol.bas_angular(shell)))

    for atom, shells in shells_by_atom.items():
        base = min(ao_loc[s] for s, _ in shells)
        by_l: dict[int, list[int]] = {}
        for shell, l in shells:
            by_l.setdefault(l, []).append(shell)

        # Stored offset of each l-group, in ascending l (both codes sort by l).
        offset = base
        group_offset = {}
        for l in sorted(by_l):
            group_offset[l] = offset
            offset += len(by_l[l]) * (2 * l + 1)

        for l, shell_list in by_l.items():
            n_shell = len(shell_list)
            ncomp = 2 * l + 1
            pyscf_ms = m_list(l, "pyscf")
            stored_ms = m_list(l, m_order)
            for shell_idx, shell in enumerate(shell_list):
                shell_pos = n_shell - 1 - shell_idx if shell_rev else shell_idx
                for j in range(ncomp):
                    m_pos = stored_ms.index(pyscf_ms[j])
                    if layout == "shell":
                        local = shell_pos * ncomp + m_pos
                    else:
                        local = m_pos * n_shell + shell_pos
                    perm[ao_loc[shell] + j] = group_offset[l] + local
    return perm


def build_signs(mol, flip: bool) -> np.ndarray:
    signs = np.ones(mol.nao)
    if not flip:
        return signs
    ao_loc = mol.ao_loc_nr()
    for shell in range(mol.nbas):
        l = mol.bas_angular(shell)
        if l < 3:
            continue
        for j, m in enumerate(m_list(l, "pyscf")):
            if abs(m) >= 3:
                signs[ao_loc[shell] + j] = -1.0
    return signs


def evaluate(fock_raw, mol, ovlp, perm, signs, s_thresh=1e-6):
    fock = fock_raw[np.ix_(perm, perm)] * signs[:, None] * signs[None, :]
    val, vec = np.linalg.eigh(ovlp)
    keep = val > s_thresh
    x = vec[:, keep] / np.sqrt(val[keep])
    eps, c = np.linalg.eigh(x.T @ fock @ x)
    coeff = x @ c
    nocc = mol.nelectron // 2
    dm = 2.0 * coeff[:, :nocc] @ coeff[:, :nocc].T
    pop = np.einsum("ij,ji->i", dm, ovlp)
    slices = mol.aoslice_by_atom()
    charges = np.array(
        [mol.atom_charge(a) - pop[slices[a][2] : slices[a][3]].sum() for a in range(mol.natm)]
    )
    return eps, dm, charges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--subset", default="ani1xbb")
    parser.add_argument("--entry", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with h5py.File(args.path, "r") as fh:
        group = fh[args.subset]
        name = args.entry or sorted(group.keys())[0]
        node = group[name]
        elements = np.asarray(node["elements"][()])
        coords = np.asarray(node["coords"][()])
        fock_flat = np.asarray(node["fock"][()])

    charge, mult = parse_charge_mult(name)
    mol = build_mol(elements, coords, charge, mult - 1)
    fock_raw = inflate_triangle(fock_flat)
    ovlp = mol.intor("int1e_ovlp")
    target_core = CORE_1S[int(max(elements))]
    print(f"entry={name}  nao={mol.nao}  nelec={mol.nelectron}")
    print(f"heaviest element Z={int(max(elements))}, expected eps_min ~ {target_core:.1f} Ha\n")

    rows = []
    for m_order, layout, shell_rev, flip in itertools.product(
        ("pyscf", "orca", "ascending"), ("shell", "component"), (False, True), (False, True)
    ):
        perm = build_perm(mol, m_order, layout, shell_rev)
        signs = build_signs(mol, flip)
        eps, _, charges = evaluate(fock_raw, mol, ovlp, perm, signs)
        rows.append(
            {
                "m_order": m_order,
                "layout": layout,
                "shell_rev": shell_rev,
                "flip_f": flip,
                "eps_min": float(eps[0]),
                "max_abs_charge": float(np.abs(charges).max()),
                "core_err": float(abs(eps[0] - target_core)),
            }
        )

    rows.sort(key=lambda r: (r["max_abs_charge"], r["core_err"]))
    print(f"{'m_order':<10}{'layout':<11}{'rev':<6}{'flip_f':<8}{'eps_min':>12}{'max|q|':>10}")
    for r in rows:
        print(
            f"{r['m_order']:<10}{r['layout']:<11}{str(r['shell_rev']):<6}{str(r['flip_f']):<8}"
            f"{r['eps_min']:>12.3f}{r['max_abs_charge']:>10.2f}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
