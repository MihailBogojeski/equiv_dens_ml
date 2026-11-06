"""Synthetic regression test for the ethanethiol IR comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


_REPO = Path(__file__).resolve().parents[2]
_ANALYSIS = _REPO / "scripts" / "revision" / "analysis"
sys.path.insert(0, str(_ANALYSIS))


def test_synthetic_spectra_write_overlay_and_metrics(tmp_path: Path) -> None:
    from plot_ethanethiol_ir_compare import compare_spectra

    frequency = np.linspace(0.0, 4000.0, 801)

    def peak(center: float) -> np.ndarray:
        return np.exp(-0.5 * ((frequency - center) / 35.0) ** 2)

    spectra = {
        "DenSNet": (frequency, peak(2580.0) + 0.4 * peak(700.0)),
        "MACE-POLAR-1-M": (frequency, peak(2600.0) + 0.4 * peak(710.0)),
        "SO3LR": (frequency, peak(2550.0) + 0.4 * peak(690.0)),
        "NIST": (frequency, peak(2570.0) + 0.4 * peak(695.0)),
    }
    metrics_path = tmp_path / "metrics.json"
    figure_path = tmp_path / "overlay.png"

    payload = compare_spectra(spectra, figure_path=figure_path, metrics_path=metrics_path)

    assert figure_path.is_file()
    assert metrics_path.is_file()
    assert set(payload["methods"]) == set(spectra)
    for method in spectra:
        assert "correlation_vs_nist" in payload["methods"][method]
        assert set(payload["methods"][method]["peak_cm"]) == {"c_s", "s_h", "c_h"}
    assert json.loads(metrics_path.read_text())["figure"] == str(figure_path)


def test_ir_sbatch_is_cpu_only() -> None:
    text = (_REPO / "scripts" / "revision" / "submit_ethanethiol_ir_compare.sbatch").read_text()
    assert "--gres=gpu" not in text
    assert "--device cpu" in text
    assert "#SBATCH --partition=cs" in text
    assert "#SBATCH --qos=cpu48" in text


def test_nist_jcamp_parser_applies_axis_and_intensity_factors() -> None:
    from fetch_ethanethiol_nist_ir import parse_jcamp

    text = """\
##XFACTOR=2
##YFACTOR=0.1
##DELTAX=4
##NPOINTS=3
##XYDATA=(X++(Y..Y))
100 1 2 3
##END=
"""
    frequency, intensity = parse_jcamp(text)
    np.testing.assert_allclose(frequency, [100.0, 108.0, 116.0])
    np.testing.assert_allclose(intensity, [0.1, 0.2, 0.3])
