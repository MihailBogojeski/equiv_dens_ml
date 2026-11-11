"""Energy-per-water histograms for the response letter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

import plot_response_letter_figures as figs  # noqa: E402


def test_energy_offset_is_round_kcal_near_water():
    assert figs.ENERGY_OFFSET_KCAL == 47900.0
    assert "47" in figs.ENERGY_XLABEL
    assert r"kcal" in figs.ENERGY_XLABEL


def test_shift_energy_kcal_removes_the_absolute_offset():
    values = np.array([-47900.0, -47850.0, -47960.0])
    assert figs.shift_energy_kcal(values).tolist() == [0.0, 50.0, -60.0]


def test_water_energy_includes_id_hybrid():
    assert "ID $\\omega$B97M-V" in figs.WATER_ENERGY
    assert figs.WATER_ENERGY["ID $\\omega$B97M-V"].endswith(
        "water_id_test_wb97mv_def2tzvpd_base.npy"
    )


def test_faded_histogram_style_matches_other_response_figures():
    assert figs.HIST_FILL_ALPHA == 0.28


def _write_label_npy(directory: Path, name: str, energy_ha: list[float], n_atoms: int) -> str:
    dest = directory / name
    atom_numbers = np.ones((len(energy_ha), n_atoms), dtype=int)
    np.save(
        dest,
        {
            "energy": np.asarray(energy_ha, dtype=float),
            "atom_numbers": atom_numbers,
        },
    )
    return name


def test_plot_dataset_energy_histograms_write_png(tmp_path, monkeypatch):
    monkeypatch.setattr(figs, "_REPO", tmp_path)
    monkeypatch.setattr(figs, "OUT", tmp_path)
    monkeypatch.setattr(
        figs,
        "WATER_ENERGY",
        {
            "Train PBE": _write_label_npy(tmp_path, "tr_pbe.npy", [-152.5, -152.6], 6),
            "ID PBE": _write_label_npy(tmp_path, "id_pbe.npy", [-152.55], 6),
            "Size OOD PBE": _write_label_npy(tmp_path, "ood_pbe.npy", [-610.6, -610.7], 24),
            "Train $\\omega$B97M-V": _write_label_npy(tmp_path, "tr_hyb.npy", [-152.7, -152.8], 6),
            "ID $\\omega$B97M-V": _write_label_npy(tmp_path, "id_hyb.npy", [-152.75], 6),
            "Size OOD $\\omega$B97M-V": _write_label_npy(tmp_path, "ood_hyb.npy", [-611.2, -611.3], 24),
        },
    )
    dest = tmp_path / "dataset_water_energy_histograms.pdf"
    written = figs.plot_dataset_energy_histograms()
    assert written == dest
    assert dest.is_file()
    assert dest.stat().st_size > 0
