"""Tests for GPU dipole integrals (int1e_r_gpu) vs PySCF."""

import pytest
import numpy as np
from pyscf import gto

from equiv_dens.integral.dipole_gpu import int1e_r_gpu
from equiv_dens.utils import orbitals


def _max_abs_diff(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max()


class TestInt1eRGpu:
    """Test int1e_r_gpu against PySCF intor / intor_cross."""

    def test_water_self(self):
        """Water with sto3g: self integrals vs mol.intor('int1e_r')."""
        mol = gto.M(
            atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
            basis="sto3g",
        )
        mol.build()
        ref = mol.intor("int1e_r")
        out = int1e_r_gpu(mol, mol, xp=np, r_origin=(0.0, 0.0, 0.0))
        assert _max_abs_diff(ref, out) < 1e-10

    def test_h2_self(self):
        """H2 with sto3g: self integrals."""
        mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto3g")
        mol.build()
        ref = mol.intor("int1e_r")
        out = int1e_r_gpu(mol, mol, xp=np, r_origin=(0.0, 0.0, 0.0))
        assert _max_abs_diff(ref, out) < 1e-10

    def test_cross_helper_auxmol(self):
        """Cross integrals helper_mol vs auxmol (equiv_dens dipole use case)."""
        mol = gto.M(
            atom="C 0 0 0; H 0.63 0.63 0.63; H -0.63 0.63 -0.63; "
            "H -0.63 -0.63 0.63; H 0.63 -0.63 -0.63",
            basis="sto3g",
        )
        mol.build()
        auxmol = gto.M(atom=mol.atom, basis="sto3g")
        auxmol.build()
        helper = orbitals.build_1c1e_helper_mol(auxmol)
        ref = gto.mole.intor_cross("int1e_r", helper, auxmol)
        out = int1e_r_gpu(helper, auxmol, xp=np, r_origin=(0.0, 0.0, 0.0))
        assert _max_abs_diff(ref, out) < 1e-10

    def test_methane_sto3g(self):
        """Methane with sto3g: multiple s and p shells."""
        mol = gto.M(
            atom="C 0 0 0; H 0.63 0.63 0.63; H -0.63 0.63 -0.63; "
            "H -0.63 -0.63 0.63; H 0.63 -0.63 -0.63",
            basis="sto3g",
        )
        mol.build()
        ref = mol.intor("int1e_r")
        out = int1e_r_gpu(mol, mol, xp=np, r_origin=(0.0, 0.0, 0.0))
        assert _max_abs_diff(ref, out) < 1e-10
