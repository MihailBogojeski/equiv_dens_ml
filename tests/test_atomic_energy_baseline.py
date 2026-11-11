"""Mixed-N energy labels need an extensive atomic baseline, not a global mean.

The paper's variable-N model (polythiophene) passed --atomic_energies. Revision
water and QM7x used the fixed-N recipe (center_energy only) on mixed cluster
sizes, so the targets were E - mean(E) and no local energy head could fit them.
These tests lock the padded-Z=0 path and the least-squares residual scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "revision" / "analysis"))

from equiv_dens.data.density_dataset import AtomsDensityData
from fit_atomic_energies import fit_atomic_energies

HARTREE_TO_KCAL = 627.5094740631
WATER_TRAIN_PBE = (
    _REPO_ROOT / "datasets/revision/water_labels/water_train_pbe_d4_avdz_base.npy"
)
ENERGY_OVERLAYS = (
    _REPO_ROOT / "config/training/water_energy_pbe.txt",
    _REPO_ROOT / "config/training/water_energy_wb97mv.txt",
    _REPO_ROOT / "experiments/jacs_revision/config/training/qm7x_energy_001.txt",
)


def _stub(atom_numbers: np.ndarray, energy: np.ndarray) -> AtomsDensityData:
    dataset = AtomsDensityData.__new__(AtomsDensityData)
    dataset.verbose = 0
    dataset.energy_centered = False
    dataset.atoms = {
        "atom_numbers": np.asarray(atom_numbers),
        "energy": np.asarray(energy, dtype=float).copy(),
    }
    return dataset


def _padded_dimer_hexamer():
    z = np.zeros((2, 18), dtype=int)
    z[0, :6] = [8, 1, 1, 8, 1, 1]
    z[1, :] = [8, 1, 1] * 6
    energy = np.array([-2.0 * 76.4, -6.0 * 76.4])
    return z, energy


def test_normalize_energy_skips_padded_zeros():
    z, energy = _padded_dimer_hexamer()
    e_water = 76.4
    table = {1: 0.0, 8: -e_water}
    dataset = _stub(z, energy)
    dataset.normalize_energy(table)
    leftover = dataset.atoms["energy"]
    assert leftover.shape == (2,)
    assert leftover == pytest.approx(0.0, abs=1e-12)


def test_normalize_energy_does_not_apply_a_global_mean():
    z, energy = _padded_dimer_hexamer()
    table = {1: 0.0, 8: -76.4}
    dataset = _stub(z, energy)
    dataset.normalize_energy(table)
    global_centered = energy - energy.mean()
    assert not np.allclose(dataset.atoms["energy"], global_centered)


def test_normalize_energy_errors_on_missing_element():
    z, energy = _padded_dimer_hexamer()
    dataset = _stub(z, energy)
    with pytest.raises(KeyError, match=r"Z=8"):
        dataset.normalize_energy({1: 0.0})


def test_least_squares_water_train_residual_is_conformational():
    if not WATER_TRAIN_PBE.is_file():
        pytest.skip(f"missing {WATER_TRAIN_PBE}")
    data = np.load(WATER_TRAIN_PBE, allow_pickle=True).item()
    table = fit_atomic_energies(data["atom_numbers"], data["energy"])
    assert table[0] == 0.0
    dataset = _stub(data["atom_numbers"], data["energy"])
    dataset.normalize_energy(table)
    mae_kcal = float(np.mean(np.abs(dataset.atoms["energy"])) * HARTREE_TO_KCAL)
    global_mae_kcal = float(
        np.mean(np.abs(data["energy"] - np.mean(data["energy"]))) * HARTREE_TO_KCAL
    )
    assert mae_kcal < 50.0
    assert global_mae_kcal > 1.0e4


@pytest.mark.local_data
@pytest.mark.parametrize("overlay", ENERGY_OVERLAYS)
def test_energy_overlay_names_an_existing_atomic_energy_file(overlay):
    assert overlay.is_file(), overlay
    paths = []
    for line in overlay.read_text().splitlines():
        if line.startswith("--atomic_energies="):
            paths.append(line.split("=", 1)[1].strip())
    assert paths, f"{overlay.name} has no --atomic_energies"
    for rel in paths:
        assert rel not in {"None", ""}, f"{overlay.name} leaves atomic_energies unset"
        path = (_REPO_ROOT / rel).resolve()
        assert path.is_file(), f"{overlay.name}: missing {path}"
