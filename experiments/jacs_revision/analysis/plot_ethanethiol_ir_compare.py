#!/usr/bin/env python3
"""Plot and quantify ethanethiol IR spectra from comparable MD trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from experiments.jacs_revision.analysis.mpl_classic_tex import save_response_figure


Spectrum = tuple[np.ndarray, np.ndarray]
PEAK_WINDOWS_CM = {
    "c_s": (600.0, 800.0),
    "s_h": (2400.0, 2700.0),
    "c_h": (2800.0, 3200.0),
}


def _normalized_on_grid(spectrum: Spectrum, grid: np.ndarray) -> np.ndarray:
    frequency, intensity = (np.asarray(value, dtype=float) for value in spectrum)
    order = np.argsort(frequency)
    interpolated = np.interp(grid, frequency[order], intensity[order], left=0.0, right=0.0)
    scale = float(np.max(np.abs(interpolated)))
    return interpolated / scale if scale > 0.0 else interpolated


def _peak_positions(frequency: np.ndarray, intensity: np.ndarray) -> dict[str, float | None]:
    peaks: dict[str, float | None] = {}
    for name, (lower, upper) in PEAK_WINDOWS_CM.items():
        mask = (frequency >= lower) & (frequency <= upper)
        peaks[name] = (
            float(frequency[mask][np.argmax(intensity[mask])]) if np.any(mask) else None
        )
    return peaks


def _similarity(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    if np.std(reference) == 0.0 or np.std(prediction) == 0.0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(reference, prediction)[0, 1])
    rmse = float(np.sqrt(np.mean((reference - prediction) ** 2)))
    return correlation, rmse


def compare_spectra(
    spectra: Mapping[str, Spectrum],
    *,
    figure_path: Path,
    metrics_path: Path,
) -> dict:
    """Write an IR overlay and quantitative comparison.

    Parameters
    ----------
    spectra
        Mapping from display label to ``(frequency_cm-1, intensity)`` arrays.
        It must include ``NIST`` and should include ``DenSNet``.
    figure_path
        Destination PNG or PDF.
    metrics_path
        Destination JSON containing peak positions and spectral similarities.

    Returns
    -------
    dict
        The JSON-serializable metrics payload.
    """
    if "NIST" not in spectra:
        raise ValueError("spectra must contain a NIST reference")

    grid = np.linspace(0.0, 4000.0, 2001)
    normalized = {name: _normalized_on_grid(spec, grid) for name, spec in spectra.items()}
    nist = normalized["NIST"]
    densnet = normalized.get("DenSNet")

    methods = {}
    for name, values in normalized.items():
        corr_nist, rmse_nist = _similarity(nist, values)
        corr_densnet, rmse_densnet = (
            _similarity(densnet, values) if densnet is not None else (float("nan"), float("nan"))
        )
        methods[name] = {
            "peak_cm": _peak_positions(grid, values),
            "correlation_vs_nist": corr_nist,
            "rmse_vs_nist": rmse_nist,
            "correlation_vs_densnet": corr_densnet,
            "rmse_vs_densnet": rmse_densnet,
        }

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for name, values in normalized.items():
        linestyle = "--" if name == "NIST" else "-"
        axis.plot(grid, values, label=name, linewidth=1.4, linestyle=linestyle)
    axis.set_xlim(4000.0, 400.0)
    axis.set_xlabel(r"Wavenumber / cm$^{-1}$")
    axis.set_ylabel("Normalized IR intensity")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_response_figure(fig, figure_path)

    payload = {
        "figure": str(figure_path),
        "frequency_range_cm": [0.0, 4000.0],
        "peak_windows_cm": PEAK_WINDOWS_CM,
        "methods": methods,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    return payload


def _read_csv(path: Path) -> Spectrum:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain frequency,intensity columns")
    return data[:, 0], data[:, 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectrum",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Repeat for DenSNet, MACE-POLAR-1-M, SO3LR, and NIST.",
    )
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    spectra = {}
    for item in args.spectrum:
        label, separator, raw_path = item.partition("=")
        if not separator:
            raise ValueError(f"invalid --spectrum {item!r}; expected LABEL=CSV")
        spectra[label] = _read_csv(Path(raw_path))
    compare_spectra(spectra, figure_path=args.figure, metrics_path=args.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
