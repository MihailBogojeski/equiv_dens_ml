#!/usr/bin/env python3
"""
Generate combined IR spectrum comparison plots.

This script creates publication-ready plots comparing:
1. IR spectra across all oligomers (2-6 mers)
2. Subsampling effects on IR spectra quality

Usage:
    python scripts/analysis/generate_ir_comparison_plots.py \
        --ir_results_dir results/ir_spectra_lower_oligomers \
        --subsampling_dir results/mlip_subsampling_experiment \
        --output_dir results/paper_figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_ir_spectra(ir_dir: Path) -> dict:
    """Load all IR spectra from the results directory."""
    spectra = {}
    
    for oligomer_dir in sorted(ir_dir.iterdir()):
        if not oligomer_dir.is_dir():
            continue
        
        oligomer = oligomer_dir.name
        spectra[oligomer] = {}
        
        for replica_dir in sorted(oligomer_dir.iterdir()):
            if not replica_dir.is_dir() or not replica_dir.name.startswith('replica'):
                continue
            
            npz_file = replica_dir / 'ir_spectrum.npz'
            if npz_file.exists():
                data = np.load(npz_file)
                rep = int(replica_dir.name.split('_')[1])
                spectra[oligomer][rep] = {
                    'freq': data['frequency_cm'],
                    'spectrum': data['spectrum'],
                    'n_frames': int(data['n_frames'])
                }
    
    return spectra


def load_subsampling_results(subsample_dir: Path) -> dict:
    """Load subsampling experiment results."""
    results = {}
    
    for oligomer_dir in subsample_dir.iterdir():
        if not oligomer_dir.is_dir():
            continue
        
        oligomer = oligomer_dir.name
        results[oligomer] = {}
        
        for npz_file in oligomer_dir.glob('subsampling_results_rep*.npz'):
            data = np.load(npz_file, allow_pickle=True)
            rep = int(npz_file.stem.split('rep')[1])
            results[oligomer][rep] = dict(data)
    
    return results


def create_combined_oligomer_plot(spectra: dict, output_dir: Path) -> None:
    """Create combined IR spectrum plot for all oligomers."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        '2mer': '#1f77b4',
        '3mer': '#ff7f0e', 
        '4mer': '#2ca02c',
        '5mer': '#d62728',
        '6mer': '#9467bd'
    }
    
    offsets = {
        '2mer': 0,
        '3mer': 1.2,
        '4mer': 2.4,
        '5mer': 3.6,
        '6mer': 4.8
    }
    
    for oligomer in sorted(spectra.keys(), key=lambda x: int(x.replace('mer', ''))):
        if oligomer not in colors:
            continue
            
        all_spectra = []
        freq_ref = None
        
        for rep, data in spectra[oligomer].items():
            freq = data['freq']
            spec = data['spectrum']
            mask = (freq >= 0) & (freq <= 4000)
            
            if freq_ref is None:
                freq_ref = freq[mask]
            
            spec_norm = spec[mask]
            if np.max(np.abs(spec_norm)) > 0:
                spec_norm = spec_norm / np.max(np.abs(spec_norm))
            all_spectra.append(spec_norm)
        
        if all_spectra and freq_ref is not None:
            avg = np.mean(all_spectra, axis=0)
            std = np.std(all_spectra, axis=0)
            
            offset = offsets.get(oligomer, 0)
            color = colors.get(oligomer, 'gray')
            
            ax.plot(freq_ref, avg + offset, label=oligomer.upper(),
                   color=color, linewidth=1.5)
            ax.fill_between(freq_ref, avg + offset - std, avg + offset + std,
                          alpha=0.2, color=color)
    
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
    ax.set_ylabel('IR Intensity (a.u., offset for clarity)', fontsize=14)
    ax.set_title('IR Spectra of Polythiophene Oligomers (ML-MD, MESA)', fontsize=16)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(0, 4000)
    ax.set_ylim(-0.3, 6.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ir_spectra_oligomers_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ir_spectra_oligomers_combined.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved oligomer comparison plot to {output_dir}")


def create_subsampling_summary_plot(results: dict, output_dir: Path) -> None:
    """Create summary plot for subsampling experiment."""
    if not results:
        print("No subsampling results to plot")
        return
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    
    oligomer = list(results.keys())[0]
    rep = list(results[oligomer].keys())[0]
    data = results[oligomer][rep]
    
    factors = [1, 2, 5, 10]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(factors)))
    
    for factor, color in zip(factors, colors):
        key = f'factor_{factor}'
        if key in data:
            factor_data = data[key].item()
            freq = factor_data['freq_cm']
            spec = factor_data['spectrum']
            n_frames = factor_data['n_frames']
            
            label = f'{factor}x ({n_frames} frames)'
            ax1.plot(freq, spec, label=label, color=color, linewidth=1.5, alpha=0.8)
    
    ax1.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12)
    ax1.set_ylabel('IR Intensity (normalized)', fontsize=12)
    ax1.set_title(f'IR Spectra at Different Subsampling Levels ({oligomer})', fontsize=14)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_xlim(0, 4000)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    
    all_correlations = {f: [] for f in factors}
    
    for oligomer, reps in results.items():
        for rep, data in reps.items():
            ref_key = 'factor_1'
            if ref_key not in data:
                continue
            ref_data = data[ref_key].item()
            ref_freq = ref_data['freq_cm']
            ref_spec = ref_data['spectrum']
            
            for factor in factors:
                key = f'factor_{factor}'
                if key in data:
                    factor_data = data[key].item()
                    test_freq = factor_data['freq_cm']
                    test_spec = factor_data['spectrum']
                    
                    common_freq = np.linspace(500, 3500, 500)
                    ref_interp = np.interp(common_freq, ref_freq, ref_spec)
                    test_interp = np.interp(common_freq, test_freq, test_spec)
                    
                    if np.max(ref_interp) > 0:
                        ref_interp /= np.max(ref_interp)
                    if np.max(test_interp) > 0:
                        test_interp /= np.max(test_interp)
                    
                    corr = np.corrcoef(ref_interp, test_interp)[0, 1]
                    all_correlations[factor].append(corr)
    
    means = [np.mean(all_correlations[f]) for f in factors]
    stds = [np.std(all_correlations[f]) for f in factors]
    
    x_labels = [f'{f}x' for f in factors]
    bars = ax2.bar(x_labels, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Subsampling Factor', fontsize=12)
    ax2.set_ylabel('Correlation with Full Trajectory', fontsize=12)
    ax2.set_title('Spectrum Correlation vs Subsampling', fontsize=14)
    ax2.set_ylim(0, 1.15)
    ax2.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{mean:.2f}', ha='center', fontsize=10)
    
    ax3 = fig.add_subplot(gs[1, 1])
    
    cost_reductions = [1/f * 100 for f in factors]
    colors_cost = plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(factors)))
    bars = ax3.bar(x_labels, cost_reductions, color=colors_cost, alpha=0.8)
    ax3.set_xlabel('Subsampling Factor', fontsize=12)
    ax3.set_ylabel('Computational Cost (%)', fontsize=12)
    ax3.set_title('Cost Reduction from Subsampling', fontsize=14)
    ax3.set_ylim(0, 120)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, cost in zip(bars, cost_reductions):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{cost:.0f}%', ha='center', fontsize=10)
    
    plt.savefig(output_dir / 'subsampling_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'subsampling_summary.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved subsampling summary plot to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate combined IR comparison plots'
    )
    parser.add_argument('--ir_results_dir', type=Path, 
                       default=Path('results/ir_spectra_lower_oligomers'),
                       help='Directory containing IR spectra results')
    parser.add_argument('--subsampling_dir', type=Path,
                       default=Path('results/mlip_subsampling_experiment'),
                       help='Directory containing subsampling results')
    parser.add_argument('--output_dir', type=Path,
                       default=Path('results/paper_figures'),
                       help='Output directory for figures')
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING IR COMPARISON PLOTS")
    print("=" * 70)
    
    if args.ir_results_dir.exists():
        print(f"\nLoading IR spectra from: {args.ir_results_dir}")
        spectra = load_ir_spectra(args.ir_results_dir)
        print(f"  Loaded spectra for: {list(spectra.keys())}")
        
        create_combined_oligomer_plot(spectra, args.output_dir)
    else:
        print(f"IR results not found: {args.ir_results_dir}")
    
    if args.subsampling_dir.exists():
        print(f"\nLoading subsampling results from: {args.subsampling_dir}")
        subsampling = load_subsampling_results(args.subsampling_dir)
        print(f"  Loaded results for: {list(subsampling.keys())}")
        
        create_subsampling_summary_plot(subsampling, args.output_dir)
    else:
        print(f"Subsampling results not found: {args.subsampling_dir}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutput saved to: {args.output_dir}")
    print(f"Files created:")
    for f in args.output_dir.iterdir():
        print(f"  - {f.name}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
