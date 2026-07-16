#!/usr/bin/env python3
"""
Plot dipole moment time series for all replicas and all oligomers.

Generates plots for:
- 8mer, 10mer, 12mer
- Replicas 0, 1, 2, 3
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
import sys

# Add parent directory to path to import from plot_dipole_continuity
sys.path.insert(0, str(Path(__file__).parent))
from plot_dipole_continuity import load_and_combine_dipoles


def plot_single_timeseries(
    frames: np.ndarray,
    dipoles: np.ndarray,
    gaps: list,
    ax,
    oligomer: str,
    replica: int,
    dt_fs: float = 0.5
):
    """
    Plot a single dipole time series on the given axes.
    """
    # Calculate dipole magnitude
    dipole_mag = np.linalg.norm(dipoles, axis=1)
    
    # Convert frames to time in picoseconds
    time_ps = frames * dt_fs / 1000.0
    
    # Plot time series
    ax.plot(time_ps, dipole_mag, 'b-', linewidth=0.5, alpha=0.8)
    
    # Highlight gaps
    for gap_start, gap_end in gaps:
        gap_time_start = gap_start * dt_fs / 1000.0
        gap_time_end = gap_end * dt_fs / 1000.0
        ax.axvspan(gap_time_start, gap_time_end, color='red', alpha=0.3)
    
    # Set labels and title
    ax.set_ylabel('|μ| (Debye)', fontsize=10)
    ax.set_xlabel('Time (ps)', fontsize=10)
    ax.set_title(f'{oligomer} Replica {replica}\n({len(frames)} frames, {len(gaps)} gaps)', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {dipole_mag.mean():.2f} D\nStd: {dipole_mag.std():.2f} D'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))


def plot_all_timeseries(
    results_base_dir: Path,
    output_dir: Path,
    dt_fs: float = 0.5
):
    """
    Plot dipole time series for all oligomers and replicas.
    
    Creates:
    1. Individual plots for each oligomer/replica combination
    2. A combined summary plot with all replicas for each oligomer
    """
    oligomers = ['8mer', '10mer', '12mer']
    replicas = [0, 1, 2, 3]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Individual plots for each oligomer/replica
    print("="*70)
    print("Generating individual plots for each oligomer/replica...")
    print("="*70)
    
    for oligomer in oligomers:
        for replica in replicas:
            results_dir = results_base_dir / oligomer / f"replica_{replica}"
            
            if not results_dir.exists():
                print(f"WARNING: Skipping {oligomer} replica {replica} - directory not found")
                continue
            
            try:
                print(f"\nProcessing {oligomer} replica {replica}...")
                frames, dipoles, gaps, coverage = load_and_combine_dipoles(results_dir)
                
                # Create individual plot
                fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                plot_single_timeseries(frames, dipoles, gaps, ax, oligomer, replica, dt_fs)
                
                plt.tight_layout()
                output_file = output_dir / f"{oligomer}_replica_{replica}_dipole_timeseries.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Saved: {output_file}")
                
            except Exception as e:
                print(f"  ✗ Error processing {oligomer} replica {replica}: {e}")
                continue
    
    # Plot 2: Combined plots - all replicas for each oligomer
    print("\n" + "="*70)
    print("Generating combined plots (all replicas per oligomer)...")
    print("="*70)
    
    for oligomer in oligomers:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        all_data_loaded = True
        for idx, replica in enumerate(replicas):
            results_dir = results_base_dir / oligomer / f"replica_{replica}"
            ax = axes[idx]
            
            if not results_dir.exists():
                ax.text(0.5, 0.5, f'No data for {oligomer} replica {replica}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(f'{oligomer} Replica {replica} - No data', fontsize=11)
                all_data_loaded = False
                continue
            
            try:
                frames, dipoles, gaps, coverage = load_and_combine_dipoles(results_dir)
                plot_single_timeseries(frames, dipoles, gaps, ax, oligomer, replica, dt_fs)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading {oligomer} replica {replica}: {e}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                ax.set_title(f'{oligomer} Replica {replica} - Error', fontsize=11)
                all_data_loaded = False
        
        if all_data_loaded:
            axes[-1].set_xlabel('Time (ps)', fontsize=12)
            fig.suptitle(f'Dipole Moment Time Series: {oligomer} (All Replicas)', 
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            output_file = output_dir / f"{oligomer}_all_replicas_dipole_timeseries.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {output_file}")
        else:
            plt.close()
            print(f"  WARNING: Skipped {oligomer} combined plot due to missing data")
    
    # Plot 3: Summary comparison - one replica from each oligomer
    print("\n" + "="*70)
    print("Generating summary comparison plot...")
    print("="*70)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    
    for idx, oligomer in enumerate(oligomers):
        ax = axes[idx]
        results_dir = results_base_dir / oligomer / "replica_0"
        
        if not results_dir.exists():
            ax.text(0.5, 0.5, f'No data for {oligomer}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            continue
        
        try:
            frames, dipoles, gaps, coverage = load_and_combine_dipoles(results_dir)
            plot_single_timeseries(frames, dipoles, gaps, ax, oligomer, 0, dt_fs)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading {oligomer}: {e}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=10)
    
    axes[-1].set_xlabel('Time (ps)', fontsize=12)
    fig.suptitle('Dipole Moment Time Series Comparison (Replica 0)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = output_dir / "all_oligomers_comparison_dipole_timeseries.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_file}")
    
    print("\n" + "="*70)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Plot dipole moment time series for all replicas and oligomers')
    parser.add_argument('--results_dir', type=str, 
                       default='results/dipole_parallel',
                       help='Base directory containing results (default: results/dipole_parallel)')
    parser.add_argument('--output_dir', type=str, 
                       default='results/dipole_timeseries_plots',
                       help='Output directory for plots (default: results/dipole_timeseries_plots)')
    parser.add_argument('--dt_fs', type=float, default=0.5,
                       help='MD timestep in femtoseconds (default: 0.5)')
    
    args = parser.parse_args()
    
    results_base_dir = Path(args.results_dir)
    if not results_base_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_base_dir}")
    
    output_dir = Path(args.output_dir)
    
    plot_all_timeseries(results_base_dir, output_dir, args.dt_fs)


if __name__ == '__main__':
    main()
