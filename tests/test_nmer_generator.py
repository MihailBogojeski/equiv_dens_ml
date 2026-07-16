"""
Unit tests for N-mer polythiophene generator (SMILES + RDKit).
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import after potential sys.path setup
import sys

# Ensure equiv_dens_ml is on path when running from project root
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.data.polythiophene_generator import (
    generate_nmer_from_smiles,
    optimize_with_gxtb,
    polythiophene_smiles,
    smiles_to_ase_atoms,
)


def test_polythiophene_smiles():
    """polythiophene_smiles returns correct SMILES for 1-mer, 2-mer, 8-mer."""
    assert polythiophene_smiles(1) == "c1ccc[s]1"
    assert polythiophene_smiles(2) == "c1ccc(s1)c2ccc[s]2"
    assert polythiophene_smiles(8) == "c1ccc(s1)c2ccc(s2)c3ccc(s3)c4ccc(s4)c5ccc(s5)c6ccc(s6)c7ccc(s7)c8ccc[s]8"
    # 12-mer uses %10, %11, %12 for ring numbers >= 10
    assert "c%10ccc(s%10)" in polythiophene_smiles(12)
    assert "c%12ccc[s]%12" in polythiophene_smiles(12)


def test_polythiophene_smiles_stoichiometry():
    """SMILES-derived structures have correct stoichiometry."""
    from pyscf import gto

    # 1-mer: C4H4S = 9 atoms, 44 electrons
    mol1 = smiles_to_ase_atoms(polythiophene_smiles(1))
    assert len(mol1) == 9
    nelec1 = sum(gto.charge(int(z)) for z in mol1.get_atomic_numbers() if z > 0)
    assert nelec1 == 44  # 4*6 + 4*1 + 16 = 24+4+16

    # 2-mer: C8H6S2 = 16 atoms, 86 electrons
    mol2 = smiles_to_ase_atoms(polythiophene_smiles(2))
    assert len(mol2) == 16
    nelec2 = sum(gto.charge(int(z)) for z in mol2.get_atomic_numbers() if z > 0)
    assert nelec2 == 86  # 8*6 + 6*1 + 2*16 = 48+6+32

    # 8-mer: C32H18S8 = 58 atoms (terminal H), 338 electrons, closed shell
    mol8 = smiles_to_ase_atoms(polythiophene_smiles(8))
    assert len(mol8) == 58
    nelec8 = sum(gto.charge(int(z)) for z in mol8.get_atomic_numbers() if z > 0)
    assert nelec8 == 338  # 32*6+18*1+8*16 = 192+18+128
    assert nelec8 % 2 == 0


def test_generate_nmer_closed_shell():
    """generate_nmer_from_smiles produces closed-shell structures."""
    from pyscf import gto

    for n in [1, 2, 8, 12]:
        atoms = generate_nmer_from_smiles(n)
        nelec = sum(gto.charge(int(z)) for z in atoms.get_atomic_numbers() if z > 0)
        assert nelec % 2 == 0, f"n={n} has odd electron count {nelec}"


def test_generate_nmer_returns_atoms():
    """generate_nmer_from_smiles returns ASE Atoms with 3D coordinates."""
    from ase import Atoms

    atoms = generate_nmer_from_smiles(8)
    assert isinstance(atoms, Atoms)
    assert atoms.positions.shape == (len(atoms), 3)
    assert atoms.positions.shape[0] == 58  # C32H18S8 (terminal H included)


def test_polythiophene_smiles_invalid():
    """polythiophene_smiles raises for n < 1."""
    with pytest.raises(ValueError, match="n must be >= 1"):
        polythiophene_smiles(0)


def test_optimize_with_gxtb_mock():
    """optimize_with_gxtb runs g-xtb and returns optimized structure."""
    from ase import Atoms

    mol = Atoms(
        "C2H4",
        positions=[
            [0, 0, 0],
            [1.3, 0, 0],
            [0.5, 0.9, 0],
            [0.5, -0.9, 0],
            [-0.5, 0.9, 0],
            [-0.5, -0.9, 0],
        ],
    )
    n = len(mol)

    def _create_gxtb_output_and_return(*args, **kwargs):
        cwd = kwargs.get("cwd", ".")
        workdir = Path(cwd)
        (workdir / "energy").write_text(
            "$energy\n  1     -10.5   -10.5  99.9 99.9 99.9\n$end\n"
        )
        grad_lines = [
            "$grad\n",
            "  cycle =      1    SCF energy = -10.5   |dE/dxyz| =  0.001\n",
        ]
        for _ in range(n):
            grad_lines.append("  0.0  0.0  0.0  H\n")
        grad_lines.append("$end\n")
        (workdir / "gradient").write_text("".join(grad_lines))
        return MagicMock(returncode=0, stdout="", stderr="")

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        with patch("subprocess.run", side_effect=_create_gxtb_output_and_return) as mock_run:
            result = optimize_with_gxtb(
                mol,
                xtb_cmd="xtb",
                gxtb_path="/fake/gxtb",
                gxtb_params_dir="/fake/params",
                workdir=workdir,
                verbose=False,
            )

        assert mock_run.call_count >= 1
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "gxtb" in cmd[0] or "gxtb" in str(cmd)
        assert "-grad" in cmd
        env = call_args[1].get("env", {})
        assert "GXTBHOME" in env
        assert env["GXTBHOME"] == "/fake/params"

        assert result is not None
        assert len(result) == len(mol)
