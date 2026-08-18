#!/usr/bin/env python3
"""Discover the within-atom AO permutation of the stored CSH Fock matrices.

Guessing which order ORCA wrote def2-TZVPD in is error prone, so instead the
permutation is recovered from the data. A reference Fock matrix is built in
PySCF from a superposition-of-atomic-densities guess (one Fock build, no SCF).
Its diagonal is a fingerprint of the AO layout: core functions sit at large
negative values, diffuse ones near zero. Matching the stored diagonal onto the
reference diagonal by optimal assignment - restricted to AOs of the same atom
and the same angular momentum, which any sane convention preserves - recovers
the permutation without assuming anything about ORCA's shell order.

The recovered map is then validated on the off-diagonal structure, which the
assignment never saw.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import scipy.linalg
from pyscf import dft
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import build_mol, inflate_triangle, parse_charge_mult, triangle_dim  # noqa: E402

HARTREE_TO_EV = 27.211386245988


def load_entry(path: str, subset: str | None, entry: str | None, max_atoms: int) -> dict:
    with h5py.File(path, "r") as fh:
        tops = [subset] if subset else list(fh.keys())
        for top in tops:
            group = fh[top]
            names = [entry] if entry else sorted(group.keys())
            for name in names:
                node = group[name]
                if node["elements"].shape[0] > max_atoms:
                    continue
                return {
                    "top": top,
                    "name": name,
                    "elements": np.asarray(node["elements"][()]),
                    "coords": np.asarray(node["coords"][()]),
                    "fock": np.asarray(node["fock"][()]),
                }
    raise SystemExit("no entry matched the selection")


def reference_fock(mol, xc: str = "pbe") -> np.ndarray:
    """Single Fock build from the atomic-density guess (no SCF iterations)."""
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.chkfile = None
    mf.verbose = 0
    dm0 = mf.get_init_guess(key="atom")
    return mf.get_fock(dm=dm0)


def ao_groups(mol) -> tuple[np.ndarray, np.ndarray]:
    """Per-AO atom index and angular momentum."""
    atom_of = np.empty(mol.nao, dtype=int)
    l_of = np.empty(mol.nao, dtype=int)
    ao_loc = mol.ao_loc_nr()
    for shell in range(mol.nbas):
        l = mol.bas_angular(shell)
        atom = mol.bas_atom(shell)
        atom_of[ao_loc[shell] : ao_loc[shell + 1]] = atom
        l_of[ao_loc[shell] : ao_loc[shell + 1]] = l
    return atom_of, l_of


def discover_permutation(diag_stored, diag_ref, atom_of, l_of) -> np.ndarray:
    """perm[i] = index in the stored ordering that belongs at PySCF position i."""
    perm = np.empty(diag_ref.shape[0], dtype=int)
    for atom in np.unique(atom_of):
        for l in np.unique(l_of[atom_of == atom]):
            idx = np.flatnonzero((atom_of == atom) & (l_of == l))
            cost = np.abs(diag_ref[idx][:, None] - diag_stored[idx][None, :])
            rows, cols = linear_sum_assignment(cost)
            perm[idx[rows]] = idx[cols]
    return perm


def solve_density(fock, ovlp, nelec, s_thresh):
    """Generalised eigenproblem with canonical orthogonalisation.

    def2-TZVPD is diffuse enough that the overlap matrix is numerically singular
    for these system sizes, so near-null eigenvectors have to be projected out
    before diagonalising or the spectrum blows up.
    """
    s_val, s_vec = np.linalg.eigh(ovlp)
    keep = s_val > s_thresh
    x = s_vec[:, keep] / np.sqrt(s_val[keep])
    eps, c_orth = np.linalg.eigh(x.T @ fock @ x)
    coeff = x @ c_orth
    occ = coeff[:, : nelec // 2]
    return 2.0 * occ @ occ.T, eps, int(keep.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--subset", default="ani1xbb")
    parser.add_argument("--entry", default=None)
    parser.add_argument("--max-atoms", type=int, default=10)
    parser.add_argument("--s-thresh", type=float, default=1e-5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    entry = load_entry(args.path, args.subset, args.entry, args.max_atoms)
    charge, mult = parse_charge_mult(entry["name"])
    mol = build_mol(entry["elements"], entry["coords"], charge, mult - 1)
    n = triangle_dim(entry["fock"].shape[0])
    print(f"entry   : {entry['top']}/{entry['name']}")
    print(f"natoms  : {len(entry['elements'])}  charge={charge} mult={mult}")
    print(f"nao     : stored={n} pyscf={mol.nao}  nelec={mol.nelectron}")
    assert n == mol.nao

    fock_stored = inflate_triangle(entry["fock"])
    ovlp = mol.intor("int1e_ovlp")

    print("building reference Fock from atomic-density guess ...", flush=True)
    fock_ref = reference_fock(mol)

    atom_of, l_of = ao_groups(mol)
    diag_ref = np.diag(fock_ref)
    diag_stored = np.diag(fock_stored)

    corr_before = float(np.corrcoef(diag_stored, diag_ref)[0, 1])
    perm = discover_permutation(diag_stored, diag_ref, atom_of, l_of)
    fock_perm = fock_stored[np.ix_(perm, perm)]
    corr_after = float(np.corrcoef(np.diag(fock_perm), diag_ref)[0, 1])

    off = ~np.eye(n, dtype=bool)
    corr_off_before = float(np.corrcoef(fock_stored[off], fock_ref[off])[0, 1])
    corr_off_after = float(np.corrcoef(fock_perm[off], fock_ref[off])[0, 1])

    print(f"\ndiagonal corr : before={corr_before:+.6f}  after={corr_after:+.6f}")
    print(f"offdiag corr  : before={corr_off_before:+.6f}  after={corr_off_after:+.6f}")
    print(f"permutation is identity: {bool(np.all(perm == np.arange(n)))}")

    # Report the recovered map as a per-element shell reordering.
    print("\nrecovered within-atom map (first atom of each element):")
    seen = set()
    labels = mol.ao_labels()
    for atom in np.unique(atom_of):
        z = int(mol.atom_charge(atom)) if not mol.has_ecp() else int(entry["elements"][atom])
        if z in seen:
            continue
        seen.add(z)
        idx = np.flatnonzero(atom_of == atom)
        local = perm[idx] - idx[0]
        print(f"  Z={z:>2}  perm={list(local)}")
        if len(idx) <= 40:
            for i in idx[:8]:
                print(f"      {labels[i]:<20} stored_pos={perm[i] - idx[0]:>3}")

    nelec = int(mol.nelectron)
    summary = {
        "entry": f"{entry['top']}/{entry['name']}",
        "nao": int(n),
        "nelectron": nelec,
        "corr_diag_before": corr_before,
        "corr_diag_after": corr_after,
        "corr_offdiag_before": corr_off_before,
        "corr_offdiag_after": corr_off_after,
        "identity": bool(np.all(perm == np.arange(n))),
    }

    for tag, mat in (("raw", fock_stored), ("permuted", fock_perm)):
        dm, eps, nkeep = solve_density(mat, ovlp, nelec, args.s_thresh)
        n_elec_check = float(np.einsum("ij,ij->", ovlp, dm))
        homo = eps[nelec // 2 - 1]
        lumo = eps[nelec // 2]
        print(
            f"\n[{tag}] kept {nkeep}/{n} orbitals  eps_min={eps[0]:+.4f} "
            f"HOMO={homo:+.4f} LUMO={lumo:+.4f} gap={(lumo - homo) * HARTREE_TO_EV:.3f} eV "
            f"Tr(PS)={n_elec_check:.6f}"
        )
        print(f"   lowest 8 eps: {np.array2string(eps[:8], precision=3)}")
        summary[tag] = {
            "eps_min": float(eps[0]),
            "homo": float(homo),
            "lumo": float(lumo),
            "gap_ev": float((lumo - homo) * HARTREE_TO_EV),
            "trace_ps": n_elec_check,
            "n_kept": nkeep,
        }

    counts = Counter(int(z) for z in entry["elements"])
    print(f"\ncomposition: {dict(counts)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
