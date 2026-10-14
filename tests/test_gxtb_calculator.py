"""Regression cases for the g-xTB v1.1.0 Turbomole output adapter."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from equiv_dens.md.gxtb_calculator import GxtbMDCalculator


# Independently captured from g-xTB v1.1.0 for H2 at 0.74 Angstrom.
# The labeled rows are coordinates in Bohr; only the following rows are dE/dR.
# Capture: docs/analysis/runtime-2026-10-11/gxtb-captured-output.txt.
H2_GRADIENT = """$grad
  cycle =      1    SCF energy =    -1.16504144152   |dE/dxyz| =  0.033714
    0.00000000000000      0.00000000000000      0.00000000000000      H
    0.00000000000000      0.00000000000000      1.39839723292800      H
   0.0000000000000D+00   0.0000000000000D+00  -0.2383910121102D-01
   0.0000000000000D+00   0.0000000000000D+00   0.2383910121102D-01
$end
"""
H2_ENERGY = """$energy
  1        -1.16504144152114       80.37730371662956  99.9 99.9 99.9
$end
"""
H2_DERIVATIVES = [[0.0, 0.0, -0.02383910121102], [0.0, 0.0, 0.02383910121102]]

# A second cycle follows the recorded schema, with deliberately different data.
SECOND_CYCLE = """  cycle = 2    SCF energy = -1.2D+00   |dE/dxyz| = 0.056569
  0.1 0.2 0.3 H
  0.1 0.2 1.8 H
  0.0D+00 0.0D+00 -4.0D-02
  0.0D+00 0.0D+00  4.0D-02
"""


@pytest.fixture
def calculator(tmp_path):
    return GxtbMDCalculator("/unused/gxtb", "/unused/params", tmp_path)


def test_recorded_gradient_excludes_coordinates(calculator):
    (calculator.workdir / "gradient").write_text(H2_GRADIENT)
    gradient = calculator._parse_gradient(2)
    np.testing.assert_allclose(gradient, H2_DERIVATIVES, rtol=0, atol=1e-15)
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, rtol=0, atol=1e-15)


def test_recorded_energy_precision(calculator):
    (calculator.workdir / "energy").write_text(H2_ENERGY)
    assert calculator._parse_energy() == -1.16504144152114


def test_energy_fallback_to_complete_gradient(calculator):
    (calculator.workdir / "gradient").write_text(H2_GRADIENT)
    assert calculator._parse_energy() == -1.16504144152


@pytest.mark.parametrize("exponent", ["D", "d", "E", "e"])
def test_gradient_exponent_notation(calculator, exponent):
    (calculator.workdir / "gradient").write_text(H2_GRADIENT.replace("D", exponent))
    np.testing.assert_allclose(
        calculator._parse_gradient(2), H2_DERIVATIVES, rtol=0, atol=1e-15
    )


def test_latest_cycle_in_gradient_group(calculator):
    (calculator.workdir / "gradient").write_text(
        H2_GRADIENT.replace("$end\n", SECOND_CYCLE + "$end\n")
    )
    np.testing.assert_allclose(
        calculator._parse_gradient(2),
        [[0, 0, -0.04], [0, 0, 0.04]],
        rtol=0, atol=1e-15,
    )
    assert calculator._parse_energy() == -1.2


def test_latest_complete_gradient_group(calculator):
    (calculator.workdir / "gradient").write_text(
        H2_GRADIENT + "$grad\n" + SECOND_CYCLE + "$end\n"
    )
    np.testing.assert_allclose(
        calculator._parse_gradient(2),
        [[0, 0, -0.04], [0, 0, 0.04]],
        rtol=0, atol=1e-15,
    )
    assert calculator._parse_energy() == -1.2


@pytest.mark.parametrize("second_group", [False, True])
def test_latest_energy_row(calculator, second_group):
    next_row = "  2 -1.200000000001D+00 80.0 99.9 99.9 99.9\n"
    if second_group:
        text = H2_ENERGY + "$energy\n" + next_row + "$end\n"
    else:
        text = H2_ENERGY.replace("$end\n", next_row + "$end\n")
    (calculator.workdir / "energy").write_text(text)
    assert calculator._parse_energy() == -1.200000000001


@pytest.mark.parametrize("content", [
    "",
    "$grad\n$end\n",
    H2_GRADIENT.replace("$grad\n", ""),
    H2_GRADIENT.replace("$end\n", ""),
    H2_GRADIENT.replace("cycle =      1", "not a cycle"),
    H2_GRADIENT.replace("1.39839723292800      H", "1.39839723292800"),
    H2_GRADIENT.replace("1.39839723292800      H", "1.39839723292800      Unknown"),
    H2_GRADIENT.replace("1.39839723292800", "not-a-number"),
    H2_GRADIENT.replace("-0.2383910121102D-01", "bad"),
    H2_GRADIENT.replace("-0.2383910121102D-01", "nan"),
    H2_GRADIENT.replace("-0.2383910121102D-01", "inf"),
    H2_GRADIENT.replace("-0.2383910121102D-01", "-0.2383910121102D-01 extra"),
    H2_GRADIENT.replace(
        "   0.0000000000000D+00   0.0000000000000D+00   0.2383910121102D-01\n", ""
    ),
    H2_GRADIENT.replace("$end\n", " 0 0 0\n$end\n"),
    H2_GRADIENT.replace("-1.16504144152", "nan"),
    H2_GRADIENT.replace("0.033714", "inf"),
    H2_GRADIENT.replace("$end", "$unknown"),
], ids=["empty", "empty-group", "missing-start", "missing-end", "bad-header",
        "missing-label", "unknown-label", "bad-coordinate", "bad-derivative",
        "nan-derivative", "inf-derivative", "extra-derivative-token",
        "missing-derivative", "extra-derivative", "nan-energy", "inf-norm", "unknown-marker"])
def test_rejects_malformed_gradient(calculator, content):
    (calculator.workdir / "gradient").write_text(content)
    with pytest.raises(ValueError):
        calculator._parse_gradient(2)


@pytest.mark.parametrize("atom_count", [0, 1, 3])
def test_rejects_wrong_atom_count(calculator, atom_count):
    (calculator.workdir / "gradient").write_text(H2_GRADIENT)
    with pytest.raises(ValueError):
        calculator._parse_gradient(atom_count)


def test_rejects_truncated_latest_cycle(calculator):
    trailing = "  cycle = 2 SCF energy = -1.2 |dE/dxyz| = 0.01\n  0 0 0 H\n"
    (calculator.workdir / "gradient").write_text(
        H2_GRADIENT.replace("$end\n", trailing + "$end\n")
    )
    with pytest.raises(ValueError):
        calculator._parse_gradient(2)
    with pytest.raises(ValueError):
        calculator._parse_energy()


@pytest.mark.parametrize("content", [
    "", "$energy\n$end\n", H2_ENERGY.replace("$end\n", ""),
    H2_ENERGY.replace("$energy\n", ""),
    H2_ENERGY.replace("-1.16504144152114", "nan"),
    H2_ENERGY.replace("$end\n", "  2 bad 80.0\n$end\n"),
    H2_ENERGY.replace("$end\n", "  2\n$end\n"),
], ids=["empty", "empty-group", "missing-end", "missing-start", "nan-energy",
        "bad-latest-row", "truncated-latest-row"])
def test_rejects_bad_energy_even_with_valid_gradient(calculator, content):
    (calculator.workdir / "energy").write_text(content)
    (calculator.workdir / "gradient").write_text(H2_GRADIENT)
    with pytest.raises(ValueError):
        calculator._parse_energy()


def test_missing_output(calculator):
    with pytest.raises(FileNotFoundError):
        calculator._parse_gradient(2)
    with pytest.raises(FileNotFoundError):
        calculator._parse_energy()


def test_run_returns_recorded_derivatives(calculator, monkeypatch):
    def completed_run(command, **kwargs):
        workdir = Path(kwargs["cwd"])
        (workdir / "gradient").write_text(H2_GRADIENT)
        (workdir / "energy").write_text(H2_ENERGY)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("equiv_dens.md.gxtb_calculator.subprocess.run", completed_run)
    energy, gradient = calculator._run_gxtb(Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]))
    assert energy == -1.16504144152114
    np.testing.assert_allclose(gradient, H2_DERIVATIVES, rtol=0, atol=1e-15)
