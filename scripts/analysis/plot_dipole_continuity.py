#!/usr/bin/env python3
"""
Plot dipole moment time series and verify continuity across parallel job outputs.

This script loads all .npz files from a results directory, combines them into
a continuous time series, and checks for gaps or discontinuities.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re


def load_and_combine_dipoles(results_dir: Path) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Load all .npz files and combine into a single time series.
    
    Returns:
        frames: Array of frame indices
        dipoles: Array of dipole vectors (N, 3)
        gaps: List of (start, end) tuples indicating gaps in coverage
        coverage: List of (start, end) tuples for each job's actual coverage
    """
    npz_files = sorted(results_dir.glob("frames_*.npz"))
    
    if not npz_files:
        raise ValueError(f"No .npz files found in {results_dir}")
    
    print(f"Found {len(npz_files)} result files")
    
    # Parse frame ranges from filenames
    file_info = []
    for f in npz_files:
        match = re.match(r"frames_(\d+)_(\d+)\.npz", f.name)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            file_info.append((start, end, f))
    
    # Sort by start frame
    file_info.sort(key=lambda x: x[0])
    
    # Load data and track coverage
    all_frames = []
    all_dipoles = []
    coverage = []  # List of (start, end) tuples
    
    for start, end, filepath in file_info:
        try:
            data = np.load(filepath)
            if 'dipole_moments' in data:
                dipoles = data['dipole_moments']
            elif 'dipoles' in data:
                dipoles = data['dipoles']
            else:
                print(f"  Warning: No dipole data in {filepath.name}")
                continue
            
            n_frames = len(dipoles)
            frames = np.arange(start, start + n_frames)
            
            all_frames.append(frames)
            all_dipoles.append(dipoles)
            coverage.append((start, start + n_frames))
            
            print(f"  Loaded {filepath.name}: frames {start}-{start + n_frames - 1} ({n_frames} frames)")
            
        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")
    
    if not all_frames:
        raise ValueError("No valid data loaded")
    
    # Combine all data
    frames = np.concatenate(all_frames)
    dipoles = np.concatenate(all_dipoles)
    
    # Sort by frame index
    sort_idx = np.argsort(frames)
    frames = frames[sort_idx]
    dipoles = dipoles[sort_idx]
    
    # Find gaps in coverage
    gaps = []
    coverage.sort()
    for i in range(len(coverage) - 1):
        end_current = coverage[i][1]
        start_next = coverage[i + 1][0]
        if start_next > end_current:
            gaps.append((end_current, start_next))
    
    return frames, dipoles, gaps, coverage


def plot_dipole_timeseries(
    frames: np.ndarray, 
    dipoles: np.ndarray, 
    gaps: list,
    output_path: Path,
    dt_fs: float = 0.5,  # MD timestep in femtoseconds
    coverage: list = None  # List of (start, end) tuples for each job
):
    """
    Plot dipole magnitude over time and highlight any gaps.
    """
    # Calculate dipole magnitude
    dipole_mag = np.linalg.norm(dipoles, axis=1)
    
    # Convert frames to time in picoseconds
    time_ps = frames * dt_fs / 1000.0
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Full time series
    ax1 = axes[0]
    ax1.plot(time_ps, dipole_mag, 'b-', linewidth=0.3, alpha=0.7)
    ax1.set_ylabel('|μ| (Debye)', fontsize=12)
    ax1.set_title(f'Dipole Moment Time Series ({len(frames)} frames)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Highlight gaps
    for gap_start, gap_end in gaps:
        gap_time_start = gap_start * dt_fs / 1000.0
        gap_time_end = gap_end * dt_fs / 1000.0
        ax1.axvspan(gap_time_start, gap_time_end, color='red', alpha=0.3, label='Gap')
    
    # Add statistics
    stats_text = f'Mean: {dipole_mag.mean():.2f} D\nStd: {dipole_mag.std():.2f} D\nMin: {dipole_mag.min():.2f} D\nMax: {dipole_mag.max():.2f} D'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Check for discontinuities (frame-to-frame changes)
    ax2 = axes[1]
    if len(dipole_mag) > 1:
        # Calculate frame-to-frame change
        delta_dipole = np.abs(np.diff(dipole_mag))
        delta_time = time_ps[1:]
        
        ax2.plot(delta_time, delta_dipole, 'g-', linewidth=0.3, alpha=0.7)
        ax2.set_ylabel('Δ|μ| (Debye)', fontsize=12)
        ax2.set_xlabel('Time (ps)', fontsize=12)
        ax2.set_title('Frame-to-Frame Dipole Change (discontinuity check)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Highlight large jumps (potential discontinuities)
        threshold = delta_dipole.mean() + 5 * delta_dipole.std()
        large_jumps = delta_dipole > threshold
        if np.any(large_jumps):
            ax2.scatter(delta_time[large_jumps], delta_dipole[large_jumps], 
                       c='red', s=20, zorder=5, label=f'Large jumps (>{threshold:.1f} D)')
            ax2.legend()
        
        jump_stats = f'Mean Δ: {delta_dipole.mean():.3f} D\nMax Δ: {delta_dipole.max():.3f} D\nLarge jumps: {large_jumps.sum()}'
        ax2.text(0.02, 0.98, jump_stats, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Plot 3: Zoom into a random job boundary (+/- 10 fs)
    ax3 = axes[2]
    
    # Find continuous boundaries (where one job ends and next begins without gap)
    continuous_boundaries = []
    if coverage is not None and len(coverage) > 1:
        gap_set = set((g[0], g[1]) for g in gaps)
        for i in range(len(coverage) - 1):
            end_current = coverage[i][1]
            start_next = coverage[i + 1][0]
            # Check if this is a continuous boundary (no gap)
            if start_next == end_current:
                continuous_boundaries.append(end_current)
    
    if continuous_boundaries:
        # Pick a random continuous boundary
        np.random.seed(42)  # For reproducibility
        boundary_frame = np.random.choice(continuous_boundaries)
        
        # Calculate window: +/- 10 fs = +/- 10/dt_fs frames
        window_frames = int(10.0 / dt_fs)  # 10 fs / 0.5 fs = 20 frames
        
        # Find frames within window
        mask = (frames >= boundary_frame - window_frames) & (frames <= boundary_frame + window_frames)
        
        if np.sum(mask) > 0:
            zoom_frames = frames[mask]
            zoom_dipoles = dipole_mag[mask]
            zoom_time_fs = zoom_frames * dt_fs  # Time in femtoseconds
            
            ax3.plot(zoom_time_fs, zoom_dipoles, 'b-', linewidth=1.5, marker='o', markersize=3)
            ax3.axvline(boundary_frame * dt_fs, color='orange', linestyle='--', linewidth=2, 
                       label=f'Job boundary (frame {boundary_frame})')
            ax3.set_ylabel('|μ| (Debye)', fontsize=12)
            ax3.set_xlabel('Time (fs)', fontsize=12)
            ax3.set_title(f'Zoom: Job Boundary at frame {boundary_frame} (±10 fs)', fontsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Add continuity check
            boundary_idx = np.where(zoom_frames == boundary_frame)[0]
            if len(boundary_idx) > 0 and boundary_idx[0] > 0:
                idx = boundary_idx[0]
                delta_at_boundary = abs(zoom_dipoles[idx] - zoom_dipoles[idx - 1])
                ax3.text(0.02, 0.98, f'Δ at boundary: {delta_at_boundary:.4f} D', 
                        transform=ax3.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        else:
            ax3.text(0.5, 0.5, 'No data in zoom window', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=14)
    else:
        ax3.text(0.5, 0.5, 'No continuous job boundaries found\n(all boundaries have gaps)', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot dipole moment continuity')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to results directory containing .npz files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output plot path (default: results_dir/dipole_timeseries.png)')
    parser.add_argument('--dt_fs', type=float, default=0.5,
                        help='MD timestep in femtoseconds (default: 0.5)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    output_path = Path(args.output) if args.output else results_dir / 'dipole_timeseries.png'
    
    print(f"Loading dipoles from: {results_dir}")
    frames, dipoles, gaps, coverage = load_and_combine_dipoles(results_dir)
    
    print(f"\n{'='*60}")
    print(f"CONTINUITY REPORT")
    print(f"{'='*60}")
    print(f"Total frames loaded: {len(frames)}")
    print(f"Frame range: {frames.min()} - {frames.max()}")
    print(f"Expected frames: {frames.max() - frames.min() + 1}")
    print(f"Actual unique frames: {len(np.unique(frames))}")
    
    # Check for duplicates
    unique, counts = np.unique(frames, return_counts=True)
    duplicates = unique[counts > 1]
    if len(duplicates) > 0:
        print(f"WARNING: {len(duplicates)} duplicate frames found!")
        print(f"   First few duplicates: {duplicates[:10]}")
    else:
        print(f"✓ No duplicate frames")
    
    # Check for gaps
    if gaps:
        print(f"WARNING: {len(gaps)} gaps found in coverage!")
        for gap_start, gap_end in gaps:
            print(f"   Gap: frames {gap_start} - {gap_end} ({gap_end - gap_start} frames missing)")
    else:
        # Check if we have all frames in sequence
        expected_frames = np.arange(frames.min(), frames.max() + 1)
        missing = np.setdiff1d(expected_frames, frames)
        if len(missing) > 0:
            print(f"WARNING: {len(missing)} frames missing within range!")
            print(f"   First few missing: {missing[:10]}")
        else:
            print(f"✓ No gaps - continuous coverage from frame {frames.min()} to {frames.max()}")
    
    print(f"{'='*60}\n")
    
    # Plot
    plot_dipole_timeseries(frames, dipoles, gaps, output_path, args.dt_fs, coverage)
    
    return frames, dipoles


if __name__ == '__main__':
    main()
