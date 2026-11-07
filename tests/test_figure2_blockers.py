"""Figure 2 blockers: Triton headers, AIMNet2 device, DFTB3/3ob factory."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LIB = _REPO_ROOT / "scripts" / "revision" / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from dftb_calculator import (  # noqa: E402
    HUBBARD_DERIVS_3OB,
    default_dftb_paths,
    make_dftb3_3ob_calculator,
)
from triton_headers import ensure_triton_python_headers, mlip_device, make_aimnet2_calculator  # noqa: E402


def test_mlip_device_respects_use_gpu_flag() -> None:
    assert mlip_device(False) == "cpu"
    assert mlip_device(True) == "cuda"


def test_ensure_triton_python_headers_noop_when_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 't'\n")
    tools = tmp_path / "tools" / "python312-headers"
    tools.mkdir(parents=True)
    for key in ("CPATH", "C_INCLUDE_PATH"):
        monkeypatch.delenv(key, raising=False)
    status = ensure_triton_python_headers(start=tmp_path / "anchor.py")
    assert status["applied"] is False
    assert "missing" in status["reason"]


def test_ensure_triton_python_headers_prepends_cpath(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 't'\n")
    inc = tmp_path / "tools" / "python312-headers"
    inc.mkdir(parents=True)
    (inc / "Python.h").write_text("/* stub */\n")
    monkeypatch.setenv("CPATH", "/existing")
    status = ensure_triton_python_headers(start=tmp_path / "anchor.py")
    assert status["applied"] is True
    assert str(inc) in os.environ["CPATH"].split(":")
    assert os.environ["CPATH"].startswith(str(inc))


def test_dftb3_3ob_factory_sets_hubbard_and_angular_momentum() -> None:
    atoms = Atoms("CH3CH2OH", positions=[[0, 0, 0]] * 9)
    calc = make_dftb3_3ob_calculator(
        atoms,
        command="/nonexistent/dftb+ > PREFIX.out",
        sk_dir="/tmp/skfiles",
    )
    params = calc.parameters
    assert params["Hamiltonian_SCC"] == "Yes"
    assert params["Hamiltonian_SCCTolerance"] == 1e-8
    assert "Hamiltonian_ThirdOrderFull" not in params
    assert params["Hamiltonian_MaxAngularMomentum_H"] == "s"
    assert params["Hamiltonian_MaxAngularMomentum_C"] == "p"
    assert params["Hamiltonian_MaxAngularMomentum_O"] == "p"
    assert calc.slako_dir.endswith("/")


def test_dftb3_3ob_third_order_keywords() -> None:
    atoms = Atoms("CH3CH2OH", positions=[[0, 0, 0]] * 9)
    calc = make_dftb3_3ob_calculator(
        atoms,
        command="/nonexistent/dftb+ > PREFIX.out",
        sk_dir="/tmp/skfiles/",
        third_order=True,
    )
    params = calc.parameters
    assert params["Hamiltonian_ThirdOrderFull"] == "Yes"
    assert params["Hamiltonian_DampXH"] == "Yes"
    assert params["Hamiltonian_DampXHExponent"] == 4.0
    assert params["Hamiltonian_HubbardDerivs_H"] == HUBBARD_DERIVS_3OB["H"]
    assert params["Hamiltonian_HubbardDerivs_C"] == HUBBARD_DERIVS_3OB["C"]
    assert params["Hamiltonian_HubbardDerivs_O"] == HUBBARD_DERIVS_3OB["O"]


def test_default_dftb_paths_under_tools() -> None:
    binary, sk_dir = default_dftb_paths(start=_LIB)
    assert "dftbplus" in str(binary)
    assert "3ob" in str(sk_dir)


def test_make_aimnet2_calculator_uses_explicit_device() -> None:
    pytest.importorskip("aimnet")
    calc = make_aimnet2_calculator("cpu")
    assert calc.base_calc.device == "cpu"


def test_fake_dftb_stub_energy_forces(tmp_path: Path) -> None:
    pytest.importorskip("ase")
    stub = tmp_path / "fake_dftbplus"
    stub.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
cwd = Path(".")
(cwd / "detailed.out").write_text(
    "Total energy:                        -0.5000000000 H\\n"
)
(cwd / "results.tag").write_text(
    "forces   : 5\\n"
    + "\\n".join("0.0 0.0 0.0" for _ in range(5))
    + "\\n"
)
"""
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    atoms = Atoms("CH4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]])
    write(tmp_path / "mol.xyz", atoms)
    sk_dir = tmp_path / "sk"
    sk_dir.mkdir()
    calc = make_dftb3_3ob_calculator(
        atoms,
        command=f"{stub} > PREFIX.out",
        sk_dir=str(sk_dir),
        label=str(tmp_path / "run"),
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert energy == pytest.approx(-0.5 * 27.211386246, rel=1e-6)
    assert forces.shape == (5, 3)
