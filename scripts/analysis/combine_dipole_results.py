#!/usr/bin/env python3
"""
Combine results from parallel dipole moment computation.

This script combines the partial results from SLURM array jobs into
complete trajectory dipole moment arrays.

"""

import argparse
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import sys


def find_partial_results(base_dir: Path):
    """
    Find all partial result files organized by oligomer and replica.
    
    Returns:
        dict: {(oligomer, replica): [list of result files]}
    """
    results_map = defaultdict(list)
    
    for oligomer_dir in base_dir.iterdir():
        if not oligomer_dir.is_dir():
            continue
        
        oligomer = oligomer_dir.name
        
        for replica_dir in oligomer_dir.iterdir():
            if not replica_dir.is_dir():
                continue
            
            # Extract replica number from "replica_X"
            replica = int(replica_dir.name.split('_')[1])
            
            # Find all .npz files
            for result_file in replica_dir.glob('frames_*.npz'):
                results_map[(oligomer, replica)].append(result_file)
    
    return results_map


def combine_trajectory_results(result_files: list, output_file: Path):
    """
    Combine partial results into a single trajectory.
    
    Args:
        result_files: List of .npz files containing partial results
        output_file: Path to save combined results
    
    Returns:
        dict: Combined results metadata
    """
    # Sort files by start frame to ensure correct order
    def get_start_frame(filepath):
        # Extract start frame from filename like "frames_0_50.npz"
        parts = filepath.stem.split('_')
        return int(parts[1])
    
    result_files = sorted(result_files, key=get_start_frame)
    
    print(f"  Combining {len(result_files)} partial results...")
    
    # Load all partial results
    all_frame_indices = []
    all_dipole_moments = []
    all_energies = []
    all_metadata = []
    
    for result_file in result_files:
        try:
            data = np.load(result_file, allow_pickle=True)
            
            all_frame_indices.append(data['frame_indices'])
            all_dipole_moments.append(data['dipole_moments'])
            all_energies.append(data['energies'])
            
            # Load metadata
            metadata_str = str(data['metadata'])
            metadata = json.loads(metadata_str)
            all_metadata.append(metadata)
            
            print(f"    ✓ {result_file.name}: frames {metadata['start_frame']}-{metadata['end_frame']}")
            
        except Exception as e:
            print(f"    Error loading {result_file.name}: {e}")
            continue
    
    if not all_frame_indices:
        print(f"    Error: No valid results found!")
        return None
    
    # Concatenate arrays
    frame_indices = np.concatenate(all_frame_indices)
    dipole_moments = np.concatenate(all_dipole_moments, axis=0)
    energies = np.concatenate(all_energies)
    
    # Check for missing frames
    expected_frames = np.arange(frame_indices[0], frame_indices[-1] + 1)
    if len(frame_indices) != len(expected_frames):
        print(f"    WARNING: Missing frames detected!")
        print(f"    Expected: {len(expected_frames)}, Got: {len(frame_indices)}")
        missing = set(expected_frames) - set(frame_indices)
        print(f"    Missing frames: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    
    # Sort by frame index (in case files were processed out of order)
    sort_idx = np.argsort(frame_indices)
    frame_indices = frame_indices[sort_idx]
    dipole_moments = dipole_moments[sort_idx]
    energies = energies[sort_idx]
    
    # Compute statistics
    dipole_magnitudes = np.linalg.norm(dipole_moments, axis=1)
    
    combined_metadata = {
        'trajectory_file': all_metadata[0]['trajectory_file'],
        'oligomer': all_metadata[0]['oligomer'],
        'replica': all_metadata[0]['replica'],
        'n_frames': len(frame_indices),
        'n_partial_files': len(result_files),
        'frame_range': [int(frame_indices[0]), int(frame_indices[-1]) + 1],
        'statistics': {
            'dipole_magnitude_mean': float(np.mean(dipole_magnitudes)),
            'dipole_magnitude_std': float(np.std(dipole_magnitudes)),
            'dipole_magnitude_min': float(np.min(dipole_magnitudes)),
            'dipole_magnitude_max': float(np.max(dipole_magnitudes)),
            'energy_mean': float(np.mean(energies[~np.isnan(energies)])),
            'energy_std': float(np.std(energies[~np.isnan(energies)])),
        },
        'partial_results': [str(f) for f in result_files]
    }
    
    # Save combined results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as .npy (standard format)
    dipole_file = output_file.parent / f"{output_file.stem}_dipoles.npy"
    energy_file = output_file.parent / f"{output_file.stem}_energies.npy"
    metadata_file = output_file.parent / f"{output_file.stem}_metadata.json"
    
    np.save(dipole_file, dipole_moments)
    np.save(energy_file, energies)
    
    with open(metadata_file, 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    mean_dipole = combined_metadata['statistics']['dipole_magnitude_mean']
    std_dipole = combined_metadata['statistics']['dipole_magnitude_std']
    print(f"    ✓ Combined: {len(frame_indices)} frames, |μ| = {mean_dipole:.2f}±{std_dipole:.2f}D")
    print(f"    ✓ Saved to: {dipole_file.name}, {energy_file.name}")
    
    return combined_metadata


def main():
    parser = argparse.ArgumentParser(
        description='Combine parallel dipole moment computation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent / 'paper' / 'results' / 'dipole_parallel'),
                        help='Input directory containing partial results')
    parser.add_argument('--output_dir', type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent / 'paper' / 'results' / 'dipole_combined'),
                        help='Output directory for combined results')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print(f"{'='*70}")
    print("Combining Parallel Dipole Moment Results")
    print(f"{'='*70}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    # Find all partial results
    print("Discovering partial results...")
    results_map = find_partial_results(input_dir)
    
    if not results_map:
        print("Error: No partial results found!")
        return 1
    
    print(f"✓ Found results for {len(results_map)} trajectories")
    print()
    
    # Combine results for each trajectory
    all_combined_metadata = []
    
    for (oligomer, replica), result_files in sorted(results_map.items()):
        print(f"{oligomer} Replica {replica}:")
        print(f"{'-'*70}")
        
        output_file = output_dir / oligomer / f"replica_{replica}"
        combined_metadata = combine_trajectory_results(result_files, output_file)
        
        if combined_metadata is not None:
            all_combined_metadata.append(combined_metadata)
        
        print()
    
    # Create summary
    print(f"{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    summary = {
        'n_trajectories': len(all_combined_metadata),
        'total_frames': sum(m['n_frames'] for m in all_combined_metadata),
        'trajectories': {}
    }
    
    for metadata in all_combined_metadata:
        oligomer = metadata['oligomer']
        replica = metadata['replica']
        
        if oligomer not in summary['trajectories']:
            summary['trajectories'][oligomer] = {}
        
        summary['trajectories'][oligomer][f'replica_{replica}'] = {
            'n_frames': metadata['n_frames'],
            'statistics': metadata['statistics']
        }
    
    # Save global summary
    summary_file = output_dir / 'summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Total trajectories combined: {summary['n_trajectories']}")
    print(f"Total frames processed: {summary['total_frames']}")
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"✓ Summary saved to: {summary_file}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

