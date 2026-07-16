#!/usr/bin/env python3
"""
Compute IR spectrum comparison: MACE-OFF + ML density vs ML-DFT trajectory.

Direct comparison addressing Kieron's question (Section 12): "Do IR spectra
change much?" when dynamics come from another MLIP (MACE-OFF) vs our ML-DFT model.
Both use our ML density network for dipole moments.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import memspectrum
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.ethanethiol_experimental_ir import get_ethanethiol_experimental_spectrum

# Physical constants
SPEED_OF_LIGHT_CM_S = 299792458.0 * 100  # cm/s
MLDFT_TIMESTEP_FS = 0.5  # SchNetPack default
MACEOFF_TIMESTEP_FS = 1.0  # MACE-OFF trajectory


def load_mldft_dipoles(traj_path: Path) -> np.ndarray:
    """Load dipole moments from ML-DFT trajectory (.npy dict)."""
    data = np.load(traj_path, allow_pickle=True).item()
    return data['dipole_moment']


def load_maceoff_dipoles(npz_path: Path) -> np.ndarray:
    """Load dipole moments from MACE-OFF dipoles NPZ file."""
    data = np.load(npz_path)
    return data['dipole_moments']


def compute_ir_spectrum_mesa(dipoles: np.ndarray, timestep_fs: float) -> tuple:
    """
    Compute IR spectrum using Maximum Entropy Spectral Analysis (MESA).

    Returns:
        freq_cm: Frequencies in cm^-1
        spectrum: IR intensity (arbitrary units)
    """
    total_spectrum = None

    for dim in range(3):
        signal = dipoles[:, dim] - np.mean(dipoles[:, dim])

        m = memspectrum.MESA()
        m.solve(signal, method='Standard', optimisation_method='FPE')

        freq_fs_inv, psd = m.spectrum(dt=timestep_fs, onesided=True)

        freq_hz = freq_fs_inv * 1e15
        freq_cm = freq_hz / SPEED_OF_LIGHT_CM_S
        omega_sq = freq_hz ** 2
        ir_intensity = psd * omega_sq

        if total_spectrum is None:
            total_spectrum = ir_intensity
        else:
            total_spectrum += ir_intensity

    return freq_cm, total_spectrum


def compute_spectrum_similarity(freq1, spec1, freq2, spec2, freq_range=(0, 4000)):
    """Compute correlation and RMSE between two spectra on common grid."""
    freq_common = np.linspace(freq_range[0], freq_range[1], 1000)

    mask1 = (freq1 >= freq_range[0]) & (freq1 <= freq_range[1])
    mask2 = (freq2 >= freq_range[0]) & (freq2 <= freq_range[1])

    spec1_interp = np.interp(freq_common, freq1[mask1], spec1[mask1])
    spec2_interp = np.interp(freq_common, freq2[mask2], spec2[mask2])

    spec1_norm = spec1_interp / np.max(spec1_interp) if np.max(spec1_interp) > 0 else spec1_interp
    spec2_norm = spec2_interp / np.max(spec2_interp) if np.max(spec2_interp) > 0 else spec2_interp

    correlation = np.corrcoef(spec1_norm, spec2_norm)[0, 1]
    rmse = np.sqrt(np.mean((spec1_norm - spec2_norm) ** 2))

    return {'correlation': float(correlation), 'rmse': float(rmse)}


def create_comparison_plot(spectra_dict: dict, output_path: Path):
    """Create IR spectrum comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        'MACE-OFF + ML density': '#2E86AB',
        'ML-DFT': '#1A535C',
        'Experimental (NIST)': '#E94F37',
    }
    for label, (freq, spec) in spectra_dict.items():
        mask = (freq >= 0) & (freq <= 4000)
        spec_norm = spec[mask] / np.max(np.abs(spec[mask])) if np.max(np.abs(spec[mask])) > 0 else spec[mask]
        linestyle = '--' if label == 'Experimental (NIST)' else '-'

        ax.plot(freq[mask], spec_norm, label=label, color=colors.get(label, 'gray'),
                linewidth=1.5, alpha=0.85, linestyle=linestyle)

    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
    ax.set_ylabel('IR Intensity (normalized)', fontsize=12)
    ax.set_title('Ethanethiol: IR Spectrum — MACE-OFF + ML Density vs ML-DFT vs Experiment', fontsize=14)
    ax.set_xlim(0, 4000)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    region_labels = [(3000, 'C-H'), (2600, 'S-H'), (1000, 'C-C'), (700, 'C-S')]
    for freq, label in region_labels:
        ax.axvline(x=freq, color='gray', linestyle=':', alpha=0.4)
        ax.text(freq + 30, 0.95, label, fontsize=8, alpha=0.6, transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare IR spectra: MACE-OFF + ML density vs ML-DFT trajectory'
    )
    parser.add_argument('--mldft-traj', type=str,
                        default=str(_REPO_ROOT / 'paper' / 'trajectories' / 'ethanethiol' /
                                   'simulation_-ethanethiol_cluster_all_001_compressed_0.npy'),
                        help='Path to ML-DFT trajectory (.npy)')
    parser.add_argument('--maceoff-dipoles', type=str,
                        default=str(_REPO_ROOT / 'results' / 'ethanethiol_subsampling' /
                                   'dipoles_subsample_1.npz'),
                        help='Path to MACE-OFF dipoles NPZ')
    parser.add_argument('--output-dir', type=str,
                        default=str(_REPO_ROOT / 'results' / 'ethanethiol_subsampling'),
                        help='Output directory for plot and metrics')
    args = parser.parse_args()

    mldft_path = Path(args.mldft_traj)
    maceoff_path = Path(args.maceoff_dipoles)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IR Spectrum Comparison: MACE-OFF + ML Density vs ML-DFT")
    print("=" * 70)
    print(f"ML-DFT trajectory: {mldft_path}")
    print(f"MACE-OFF dipoles: {maceoff_path}")
    print(f"Output directory: {output_dir}")
    print()

    if not mldft_path.exists():
        print(f"ERROR: ML-DFT trajectory not found: {mldft_path}")
        return 1
    if not maceoff_path.exists():
        print(f"ERROR: MACE-OFF dipoles not found: {maceoff_path}")
        return 1

    # Load dipoles
    print("Loading dipoles...")
    dipoles_mldft = load_mldft_dipoles(mldft_path)
    dipoles_maceoff = load_maceoff_dipoles(maceoff_path)
    print(f"  ML-DFT: {len(dipoles_mldft)} frames, timestep {MLDFT_TIMESTEP_FS} fs")
    print(f"  MACE-OFF: {len(dipoles_maceoff)} frames, timestep {MACEOFF_TIMESTEP_FS} fs")
    print()

    # Compute IR spectra
    print("Computing IR spectra (MESA/FPE)...")
    freq_mldft, spec_mldft = compute_ir_spectrum_mesa(dipoles_mldft, MLDFT_TIMESTEP_FS)
    freq_maceoff, spec_maceoff = compute_ir_spectrum_mesa(dipoles_maceoff, MACEOFF_TIMESTEP_FS)
    print("  Done.")
    print()

    # Experimental spectrum (NIST gas phase)
    freq_exp, spec_exp = get_ethanethiol_experimental_spectrum()

    spectra = {
        'MACE-OFF + ML density': (freq_maceoff, spec_maceoff),
        'ML-DFT': (freq_mldft, spec_mldft),
        'Experimental (NIST)': (freq_exp, spec_exp),
    }

    # Create comparison plot
    output_plot = output_dir / 'ir_spectrum_maceoff_vs_mldft.png'
    print(f"Creating comparison plot: {output_plot}")
    create_comparison_plot(spectra, output_plot)
    print(f"  Saved: {output_plot}")
    print(f"  Saved: {output_plot.with_suffix('.pdf')}")
    print()

    # Compute similarity metrics
    print("Computing similarity metrics...")
    sim_maceoff_mldft = compute_spectrum_similarity(
        freq_maceoff, spec_maceoff, freq_mldft, spec_mldft
    )
    sim_mldft_exp = compute_spectrum_similarity(freq_mldft, spec_mldft, freq_exp, spec_exp)
    sim_maceoff_exp = compute_spectrum_similarity(
        freq_maceoff, spec_maceoff, freq_exp, spec_exp
    )
    print(f"  Correlation (MACE-OFF vs ML-DFT): {sim_maceoff_mldft['correlation']:.4f}")
    print(f"  Correlation (ML-DFT vs Experiment): {sim_mldft_exp['correlation']:.4f}")
    print(f"  Correlation (MACE-OFF vs Experiment): {sim_maceoff_exp['correlation']:.4f}")
    print()

    # Save metrics JSON
    metrics = {
        'mldft': {
            'n_frames': int(len(dipoles_mldft)),
            'timestep_fs': MLDFT_TIMESTEP_FS,
        },
        'maceoff': {
            'n_frames': int(len(dipoles_maceoff)),
            'timestep_fs': MACEOFF_TIMESTEP_FS,
        },
        'similarity': sim_maceoff_mldft,
        'similarity_maceoff_vs_mldft': sim_maceoff_mldft,
        'similarity_mldft_vs_experiment': sim_mldft_exp,
        'similarity_maceoff_vs_experiment': sim_maceoff_exp,
        'computed_at': datetime.now().isoformat(),
    }
    metrics_file = output_dir / 'ir_maceoff_vs_mldft_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_file}")

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
