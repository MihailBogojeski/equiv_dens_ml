#!/usr/bin/env python3
"""Shared helpers for turning OMol_CSH Fock matrices into PySCF densities.

The CSH HDF5 files store, per structure, only ``coords`` (Angstrom), ``elements``
(Z) and ``fock`` (the flat upper triangle of the converged Kohn-Sham matrix at
wB97M-V/def2-TZVPD). Charge and spin multiplicity are encoded in the trailing
two fields of the group name, e.g. ``aniBB_010_194018_-1_1`` is charge -1,
multiplicity 1.

Reconstructing the density follows the standard closed-shell route:

    F C = S C eps,    P = 2 C_occ C_occ^T

with S rebuilt in PySCF. The only non-trivial part is that the stored matrix
uses the AO ordering of the code that produced it (ORCA), which differs from
PySCF's, so a per-shell permutation/sign transform has to be applied first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from pyscf import gto

BASIS = "def2-tzvpd"
# Heavier elements in the full 58-element CSH set need the matching def2 ECP;
# it is a no-op for the H-Br subset in the "common" split.
ECP = "def2-tzvpd"

_NAME_RE = re.compile(r"_(-?\d+)_(\d+)$")


def parse_charge_mult(name: str) -> tuple[int, int]:
    """Extract (charge, multiplicity) from a CSH group name."""
    match = _NAME_RE.search(name)
    if match is None:
        raise ValueError(f"cannot parse charge/multiplicity from {name!r}")
    return int(match.group(1)), int(match.group(2))


def triangle_dim(length: int) -> int:
    """Side length n of a symmetric matrix stored as n(n+1)/2 upper-triangle values."""
    n = int(round((np.sqrt(8.0 * length + 1.0) - 1.0) / 2.0))
    if n * (n + 1) // 2 != length:
        raise ValueError(f"length {length} is not a triangular number")
    return n


def inflate_triangle(vec: np.ndarray) -> np.ndarray:
    """Re-inflate an upper-triangle vector into the full symmetric matrix."""
    n = triangle_dim(vec.shape[0])
    mat = np.zeros((n, n), dtype=vec.dtype)
    mat[np.triu_indices(n)] = vec
    return mat + mat.T - np.diag(np.diag(mat))


def build_mol(elements: np.ndarray, coords: np.ndarray, charge: int, spin: int = 0) -> gto.Mole:
    """PySCF molecule at the CSH level of theory. Coordinates are in Angstrom."""
    atom = [(int(z), tuple(float(x) for x in xyz)) for z, xyz in zip(elements, coords)]
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = BASIS
    mol.ecp = ECP
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.unit = "Angstrom"
    mol.cart = False
    mol.verbose = 0
    mol.build()
    return mol


def _orca_to_m_ascending(l: int) -> np.ndarray:
    """Index map taking an ORCA-ordered shell (m = 0, +1, -1, +2, -2, ...) to m ascending."""
    idx = np.zeros(2 * l + 1, dtype=int)
    for i in range(2 * l + 1):
        m = i - l
        if m == 0:
            pos = 0
        elif m > 0:
            pos = 2 * m - 1
        else:
            pos = 2 * (-m)
        idx[i] = pos
    return idx


def _pyscf_from_m_ascending(l: int) -> np.ndarray:
    """PySCF's within-shell order expressed as indices into an m-ascending shell.

    PySCF is m-ascending for every l except l=1, where it uses (x, y, z),
    i.e. (m=+1, m=-1, m=0).
    """
    if l == 1:
        return np.array([2, 0, 1])
    return np.arange(2 * l + 1)


def shell_permutation(l: int, source: str) -> np.ndarray:
    """Within-shell index map so that ``pyscf_vec[i] = source_vec[perm[i]]``."""
    if source == "pyscf":
        return np.arange(2 * l + 1)
    if source == "orca":
        return _orca_to_m_ascending(l)[_pyscf_from_m_ascending(l)]
    if source == "m_ascending":
        return _pyscf_from_m_ascending(l)
    raise ValueError(f"unknown source convention {source!r}")


def shell_signs(l: int, flip_high_m: bool) -> np.ndarray:
    """Optional phase flips, in PySCF within-shell order.

    ORCA carries an opposite Condon-Shortley phase relative to PySCF for the
    |m| >= 3 real solid harmonics, which shows up as soon as f functions are in
    the basis (they are, for every heavy atom in def2-TZVP).
    """
    signs = np.ones(2 * l + 1)
    if flip_high_m and l >= 3:
        for i in range(2 * l + 1):
            m = i - l
            if abs(m) >= 3:
                signs[i] = -1.0
    return signs


@dataclass
class AOTransform:
    """Full-molecule AO permutation and phases mapping stored order -> PySCF order."""

    perm: np.ndarray
    signs: np.ndarray

    def apply(self, mat: np.ndarray) -> np.ndarray:
        out = mat[np.ix_(self.perm, self.perm)]
        return out * self.signs[:, None] * self.signs[None, :]


def build_ao_transform(mol: gto.Mole, source: str, flip_high_m: bool = False) -> AOTransform:
    """Assemble the AO-level transform by walking PySCF's shell layout."""
    ao_loc = mol.ao_loc_nr()
    perm = np.empty(mol.nao, dtype=int)
    signs = np.empty(mol.nao)
    for shell in range(mol.nbas):
        l = mol.bas_angular(shell)
        nctr = mol.bas_nctr(shell)
        ncomp = 2 * l + 1
        sub_perm = shell_permutation(l, source)
        sub_signs = shell_signs(l, flip_high_m)
        start = ao_loc[shell]
        for c in range(nctr):
            block = start + c * ncomp
            perm[block : block + ncomp] = block + sub_perm
            signs[block : block + ncomp] = sub_signs
    return AOTransform(perm=perm, signs=signs)


def basis_file_order_transform(mol: gto.Mole, basis: str = BASIS) -> AOTransform:
    """Map a matrix written in basis-file shell order into PySCF's shell order.

    PySCF groups an atom's shells by angular momentum, whereas the def2-TZVPD
    basis file lists the standard def2-TZVP shells first and appends the diffuse
    augmentation at the end of each element's block. A code that keeps file
    order therefore interleaves those diffuse shells differently from PySCF, so
    the two layouts differ by a stable-sort-by-l permutation.
    """
    perm = np.empty(mol.nao, dtype=int)
    ao_loc = mol.ao_loc_nr()
    atom_start = {}
    for shell in range(mol.nbas):
        atom_start.setdefault(mol.bas_atom(shell), ao_loc[shell])

    for atom in range(mol.natm):
        symbol = mol.atom_pure_symbol(atom)
        shells = gto.basis.load(basis, symbol)

        sizes, angular = [], []
        for shell_def in shells:
            l = shell_def[0]
            nctr = max(1, len(shell_def[1]) - 1)
            sizes.append(nctr * (2 * l + 1))
            angular.append(l)

        offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]
        # Stable sort by l reproduces the order PySCF builds shells in.
        order = sorted(range(len(shells)), key=lambda i: angular[i])

        base = atom_start[atom]
        cursor = base
        for idx in order:
            size = sizes[idx]
            perm[cursor : cursor + size] = base + offsets[idx] + np.arange(size)
            cursor += size

    return AOTransform(perm=perm, signs=np.ones(mol.nao))


def density_from_fock(fock: np.ndarray, ovlp: np.ndarray, nelec: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-shell density matrix from a converged Fock matrix.

    Returns (P, eps) with P normalised to ``nelec`` electrons.
    """
    if nelec % 2 != 0:
        raise ValueError(f"open-shell electron count {nelec}; CSH is closed shell")
    eps, coeff = scipy.linalg.eigh(fock, ovlp)
    occ = coeff[:, : nelec // 2]
    return 2.0 * occ @ occ.T, eps
