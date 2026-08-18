#!/usr/bin/env python3
"""Solve the stored CSH AO ordering against a full-precision ORCA reference.

Earlier attempts matched the stored Fock matrix against a PySCF SCF or against
ORCA's printed MO coefficients. Both hit a precision floor (0.01-0.1 Ha from
RI/grid differences, 3e-3 from six-decimal printing) that is the same size as
the spacing between near-degenerate polarization components, so the assignment
fit noise instead of the permutation.

`orca_2json` dumps the GBW at full double precision. Transforming those MOs
into PySCF's AO order (using the ORCA convention validated exactly against the
overlap matrix) and rebuilding

    F = S C eps C^T S

gives a reference limited only by SCF convergence, which makes the per-atom
assignment unambiguous.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import build_mol, inflate_triangle, parse_charge_mult  # noqa: E402
from omol_csh_orca_labels import pyscf_ao_index  # noqa: E402

_L_OF_LETTER = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
_COMPONENT_M = {
    0: {"s": 0},
    1: {"pz": 0, "px": 1, "py": -1},
    2: {"dz2": 0, "dxz": 1, "dyz": -1, "dx2y2": 2, "dxy": -2},
}
# ORCA inverts the phase of the |m| >= 3 real solid harmonics relative to PySCF.
FLIP_ABS_M = 3


def parse_json_labels(labels: list[str]) -> list[tuple[int, int, int, int]]:
    """'0C   3dxz' -> (atom, l, shell_index, m), in ORCA order."""
    out = []
    for lab in labels:
        m = re.match(r"^\s*(\d+)([A-Za-z]{1,2})\s+(\S+)\s*$", lab)
        if m is None:
            raise ValueError(f"cannot parse AO label {lab!r}")
        atom = int(m.group(1))
        orbital = m.group(3)
        shell = int(re.match(r"\d+", orbital).group(0))
        rest = orbital[len(str(shell)) :]
        l = _L_OF_LETTER[rest[0]]
        mq = _COMPONENT_M[l][rest] if l <= 2 else int(rest[1:])
        out.append((atom, l, shell, mq))
    return out


def orca_to_pyscf(mol, labels) -> tuple[np.ndarray, np.ndarray]:
    """perm[i] = ORCA index of the AO PySCF places at i, plus PySCF-order signs."""
    parsed = parse_json_labels(labels)
    if len(parsed) != mol.nao:
        raise ValueError(f"ORCA has {len(parsed)} AOs, PySCF has {mol.nao}")
    lookup = pyscf_ao_index(mol)
    perm = np.empty(mol.nao, dtype=int)
    signs = np.ones(mol.nao)
    for orca_index, key in enumerate(parsed):
        i = lookup[key]
        perm[i] = orca_index
        if abs(key[3]) >= FLIP_ABS_M:
            signs[i] = -1.0
    if sorted(perm.tolist()) != list(range(mol.nao)):
        raise ValueError("ORCA label map is not a permutation")
    return perm, signs


def reference_fock(mol, data) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact Fock matrix in PySCF AO order from full-precision ORCA MOs."""
    mo_block = data["Molecule"]["MolecularOrbitals"]
    labels = mo_block["OrbitalLabels"]
    perm, signs = orca_to_pyscf(mol, labels)

    nao = mol.nao
    coeff_orca = np.empty((nao, nao))
    energies = np.empty(nao)
    for j, mo in enumerate(mo_block["MOs"]):
        coeff_orca[:, j] = mo["MOCoefficients"]
        energies[j] = mo["OrbitalEnergy"]

    # Reindex rows (AOs) into PySCF order and apply the phase convention.
    coeff = coeff_orca[perm, :] * signs[:, None]
    ovlp = mol.intor("int1e_ovlp")
    orthonormality = np.abs(coeff.T @ ovlp @ coeff - np.eye(nao)).max()
    fock = ovlp @ coeff @ np.diag(energies) @ coeff.T @ ovlp
    return fock, ovlp, float(orthonormality)


def atom_block_fingerprint(fock, slices, natm) -> np.ndarray:
    """Per-AO signature that does not depend on the unknown column ordering.

    The diagonal alone cannot separate components of the same shell: the level
    spacing inside a heavy atom drops to ~1e-4 while the difference between
    OMol25's ORCA run and a local one is ~3e-3. The norm of each AO's coupling
    into each *atom block* is far more discriminating (components of one d shell
    point at different neighbours) and, because it aggregates over a whole
    block, it is invariant to how the columns are permuted within that block.
    """
    feats = [np.diag(fock)[:, None]]
    for atom in range(natm):
        s0, s1 = slices[atom][2], slices[atom][3]
        block = fock[:, s0:s1]
        feats.append(np.linalg.norm(block, axis=1)[:, None])
        feats.append(np.abs(block).max(axis=1)[:, None])
    return np.concatenate(feats, axis=1)


def _assemble(per_element, elements, slices, nao) -> np.ndarray:
    perm = np.empty(nao, dtype=int)
    for atom, z in enumerate(elements):
        s0 = slices[atom][2]
        perm[s0 : s0 + len(per_element[int(z)])] = s0 + np.asarray(per_element[int(z)])
    return perm


def solve_permutation(fock_stored, fock_ref, mol, elements) -> tuple[np.ndarray, float]:
    """Fit one shared permutation per element, not one per atom.

    Within a single atom the objective is badly conditioned: level spacings drop
    below the ORCA-to-ORCA noise, so a per-atom search settles into whichever
    local minimum it started near. But the stored layout is a property of the
    element, so the same permutation must apply to every atom of that element.
    Sharing it across atoms multiplies the evidence per parameter and shrinks
    the search space, which is what makes the fit identifiable.
    """
    slices = mol.aoslice_by_atom()
    fp_stored = atom_block_fingerprint(fock_stored, slices, mol.natm)
    fp_ref = atom_block_fingerprint(fock_ref, slices, mol.natm)

    # Per-element initialisation: average the assignment cost over all atoms of
    # the element so noise on any single atom cannot drive the choice.
    atoms_of: dict[int, list[int]] = defaultdict(list)
    for atom, z in enumerate(elements):
        atoms_of[int(z)].append(atom)

    per_element: dict[int, list[int]] = {}
    for z, atoms in atoms_of.items():
        width = slices[atoms[0]][3] - slices[atoms[0]][2]
        cost = np.zeros((width, width))
        for atom in atoms:
            s0, s1 = slices[atom][2], slices[atom][3]
            cost += np.linalg.norm(
                fp_ref[s0:s1, None, :] - fp_stored[None, s0:s1, :], axis=2
            )
        rows, cols = linear_sum_assignment(cost)
        local = np.empty(width, dtype=int)
        local[rows] = cols
        per_element[z] = local.tolist()

    # The diagonal is a hard constraint, not a soft cost. Per-atom the stored and
    # reference diagonals are the same multiset up to the ORCA-to-ORCA noise, so
    # any pairing whose diagonals differ by more than that is impossible. Ruling
    # those out keeps the refinement from wandering into the wrong basin; the
    # off-diagonal structure then only has to break ties inside near-degenerate
    # groups.
    ds, dr = np.diag(fock_stored), np.diag(fock_ref)
    tol = {}
    for z, atoms in atoms_of.items():
        worst = 0.0
        for atom in atoms:
            s0, s1 = slices[atom][2], slices[atom][3]
            worst = max(
                worst, np.abs(np.sort(ds[s0:s1]) - np.sort(dr[s0:s1])).max()
            )
        tol[z] = max(10.0 * worst, 1e-3)

    # Rows are compared on magnitude. The stored layout may be a *signed*
    # permutation - ORCA already flips the phase of the |m| >= 3 harmonics
    # relative to PySCF - and a sign error leaves a large residual even when the
    # permutation is right, which is exactly what derails a signed comparison.
    # The diagonal is unaffected because the two phases cancel there, so it can
    # still be used as the hard constraint, and signs are fitted afterwards.
    def refine(perm_in: np.ndarray) -> np.ndarray:
        perm_out = np.empty(mol.nao, dtype=int)
        abs_ref = np.abs(fock_ref)
        abs_stored = np.abs(fock_stored)
        for atom, z in enumerate(elements):
            s0, s1 = slices[atom][2], slices[atom][3]
            ref_rows = abs_ref[s0:s1, :]
            stored_rows = abs_stored[s0:s1, :][:, perm_in]
            cost = np.linalg.norm(
                ref_rows[:, None, :] - stored_rows[None, :, :], axis=2
            )
            forbidden = np.abs(dr[s0:s1][:, None] - ds[s0:s1][None, :]) > tol[int(z)]
            cost = np.where(forbidden, 1e6, cost)
            rows, cols = linear_sum_assignment(cost)
            perm_out[s0 + rows] = s0 + cols
        return perm_out

    perm = _assemble(per_element, elements, slices, mol.nao)
    for _ in range(30):
        new = refine(perm)
        if np.array_equal(new, perm):
            break
        perm = new

    # Enforce the per-element rule by majority vote, then re-verify.
    votes: dict[int, list] = defaultdict(list)
    for atom, z in enumerate(elements):
        s0, s1 = slices[atom][2], slices[atom][3]
        votes[int(z)].append(tuple((perm[s0:s1] - s0).tolist()))
    for z, cand in votes.items():
        best = max(set(cand), key=cand.count)
        per_element[z] = list(best)
    perm = _assemble(per_element, elements, slices, mol.nao)
    signs = fit_signs(fock_stored[np.ix_(perm, perm)], fock_ref)
    residual = np.abs(
        fock_stored[np.ix_(perm, perm)] * np.outer(signs, signs) - fock_ref
    ).max()
    return perm, signs, float(residual)


def fit_signs(mat, ref, thresh=1e-2) -> np.ndarray:
    """Recover per-AO phases from mat[i,j] = s_i s_j ref[i,j].

    Propagated over the graph of matrix elements large enough to determine a
    sign reliably; components never reached keep +1, which is harmless because
    their couplings are negligible either way.
    """
    n = ref.shape[0]
    signs = np.zeros(n)
    big = np.abs(ref) > thresh
    for start in range(n):
        if signs[start] != 0:
            continue
        signs[start] = 1.0
        queue = [start]
        while queue:
            i = queue.pop()
            for j in np.flatnonzero(big[i]):
                if signs[j] != 0 or mat[i, j] == 0:
                    continue
                signs[j] = np.sign(mat[i, j] / ref[i, j]) * signs[i]
                queue.append(j)
    signs[signs == 0] = 1.0
    return signs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="orca_2json output for the SCF")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--subset", default="ani1xbb")
    parser.add_argument("--entry", default=None)
    parser.add_argument("--h5-path", default=None, help="full HDF5 path to a leaf group")
    parser.add_argument("--charge", type=int, default=None)
    parser.add_argument("--mult", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text())
    with h5py.File(args.h5, "r") as fh:
        if args.h5_path:
            node = fh[args.h5_path]
            name = args.h5_path
        else:
            group = fh[args.subset]
            name = args.entry or sorted(group.keys())[0]
            node = group[name]
        elements = np.asarray(node["elements"][()])
        coords = np.asarray(node["coords"][()])
        fock_flat = np.asarray(node["fock"][()])

    if args.charge is not None:
        charge, mult = args.charge, args.mult
    else:
        charge, mult = parse_charge_mult(name)
    mol = build_mol(elements, coords, charge, mult - 1)
    fock_stored = inflate_triangle(fock_flat)

    fock_ref, _, orthonormality = reference_fock(mol, data)
    print(f"entry={name} nao={mol.nao}")
    print(f"MO orthonormality max|C^T S C - I| = {orthonormality:.3e}")

    perm, signs, residual = solve_permutation(fock_stored, fock_ref, mol, elements)
    print(f"residual max|F_perm - F_ref| = {residual:.4e}")
    print(f"valid permutation: {sorted(perm.tolist()) == list(range(mol.nao))}")
    print(f"sign flips: {int((signs < 0).sum())} of {mol.nao}")

    slices = mol.aoslice_by_atom()
    by_element = defaultdict(list)
    sign_by_element = defaultdict(list)
    for atom in range(mol.natm):
        s0, s1 = slices[atom][2], slices[atom][3]
        by_element[int(elements[atom])].append((perm[s0:s1] - s0).tolist())
        sign_by_element[int(elements[atom])].append(signs[s0:s1].astype(int).tolist())

    consistent = True
    per_element = {}
    for z, maps in sorted(by_element.items()):
        sgn = sign_by_element[z]
        same = all(m == maps[0] for m in maps) and all(s == sgn[0] for s in sgn)
        consistent &= same
        per_element[str(z)] = {"perm": maps[0], "signs": sgn[0]}
        print(
            f"  Z={z:>2} n_atoms={len(maps)} consistent={same} "
            f"flips={sum(1 for s in sgn[0] if s < 0)}"
        )

    print(f"consistent across like atoms: {consistent}")

    # Physical check: the reconstructed density must have a sane spectrum and
    # chemically reasonable Mulliken charges.
    ovlp = mol.intor("int1e_ovlp")
    fock = fock_stored[np.ix_(perm, perm)] * np.outer(signs, signs)
    import scipy.linalg

    eps, coeff = scipy.linalg.eigh(fock, ovlp)
    nocc = mol.nelectron // 2
    dm = 2.0 * coeff[:, :nocc] @ coeff[:, :nocc].T
    pop = np.einsum("ij,ji->i", dm, ovlp)
    charges = np.array(
        [mol.atom_charge(a) - pop[slices[a][2] : slices[a][3]].sum() for a in range(mol.natm)]
    )
    gap_ev = (eps[nocc] - eps[nocc - 1]) * 27.211386245988
    print(f"eps_min={eps[0]:.4f} HOMO={eps[nocc-1]:.4f} gap={gap_ev:.3f} eV")
    print(f"Mulliken max|q|={np.abs(charges).max():.3f}  Tr(PS)={np.einsum('ij,ij->', ovlp, dm):.6f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "entry": name,
                    "residual_max_abs": residual,
                    "consistent": bool(consistent),
                    "eps_min": float(eps[0]),
                    "gap_ev": float(gap_ev),
                    "mulliken_max_abs": float(np.abs(charges).max()),
                    "per_element": per_element,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
