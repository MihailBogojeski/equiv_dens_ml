#!/usr/bin/env python3
"""
Compute dipole moments from MACE-OFF trajectories using the ML density network.

This script implements the MLIP + density subsampling experiment (Kieron Item 12):
- Load MACE-OFF trajectories (positions only)
- Compute dipole moments using trained density network
- Support frame subsampling to test cost vs accuracy tradeoff

Usage:
    python scripts/analysis/compute_dipoles_from_traj.py \
        --traj_file scratch/md_logs/mace_off/thiophene3mer_rep0.traj \
        --model_dir paper/models/polythiophene/2024-03-23_1XDL67zp_ext \
        --output_dir results/mlip_subsampling_experiment/3mer \
        --subsample 1  # every frame (no subsampling)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from ase.io import read
from tqdm import tqdm

# Set seeds for reproducibility
GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)
torch.manual_seed(GLOBAL_RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_RANDOM_SEED)

# Add the equiv_dens package to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))

E_BOHR_TO_DEBYE = 4.80320425


def load_model_and_calculator(model_dir: Path, device: str = 'cuda'):
    """
    Load the polythiophene model and create a calculator.
    
    Uses the load_polythiophene_model function from the parallel script.
    """
    sys.path.insert(0, str(_REPO_ROOT / 'scripts'))
    
    from recompute_polythiophene_dipoles_parallel import load_polythiophene_model
    
    calculator, args = load_polythiophene_model(
        model_path=model_dir,
        device=device,
        cpu_grid_generation=True,
        enable_grid_caching=True,
        gpu_grid_acceleration=False
    )
    
    return calculator


def compute_dipoles_from_trajectory(traj_file: Path, calculator, 
                                    subsample: int = 1,
                                    max_frames: int = None) -> dict:
    """
    Compute dipole moments from ASE trajectory file.
    
    Args:
        traj_file: Path to ASE .traj file
        calculator: Model calculator with dipole capability
        subsample: Take every Nth frame (1 = all frames)
        max_frames: Maximum frames to process (None = all)
        
    Returns:
        dict with frame_indices, dipole_moments, energies, timing info
    """
    print(f"Loading trajectory: {traj_file}")
    atoms_list = read(str(traj_file), index=':')
    n_total = len(atoms_list)
    print(f"Total frames in trajectory: {n_total}")
    
    frame_indices = list(range(0, n_total, subsample))
    if max_frames and len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]
    
    n_frames = len(frame_indices)
    print(f"Processing {n_frames} frames (subsample={subsample})")
    
    dipole_moments = []
    energies = []
    frame_times = []
    
    pbar = tqdm(frame_indices, desc=f"Computing dipoles (1/{subsample})", unit="frame")
    
    for frame_idx in pbar:
        atoms = atoms_list[frame_idx]
        
        np.random.seed(GLOBAL_RANDOM_SEED)
        torch.manual_seed(GLOBAL_RANDOM_SEED)
        
        t0 = time.time()
        
        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy()
        
        dipole = calculator.results.get('dipole_moment', np.zeros(3))
        if isinstance(dipole, torch.Tensor):
            dipole = dipole.detach().cpu().numpy()
        dipole = np.array(dipole).flatten()[:3]
        
        t1 = time.time()
        
        dipole_moments.append(dipole)
        energies.append(energy)
        frame_times.append(t1 - t0)
        
        dipole_mag = np.linalg.norm(dipole) * E_BOHR_TO_DEBYE
        pbar.set_postfix({'|μ|': f'{dipole_mag:.2f}D', 't': f'{(t1-t0)*1000:.0f}ms'})
    
    dipole_moments = np.array(dipole_moments)
    energies = np.array(energies)
    
    dipole_moments_debye = dipole_moments * E_BOHR_TO_DEBYE
    
    return {
        'frame_indices': np.array(frame_indices),
        'dipole_moments': dipole_moments_debye,
        'energies': energies,
        'n_total_frames': n_total,
        'n_computed_frames': n_frames,
        'subsample': subsample,
        'frame_times': np.array(frame_times),
        'total_time': sum(frame_times),
        'mean_time_per_frame': np.mean(frame_times),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute dipole moments from MACE-OFF trajectories'
    )
    parser.add_argument('--traj_file', type=Path, required=True,
                       help='Path to ASE .traj file')
    parser.add_argument('--model_dir', type=Path, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--output_dir', type=Path, required=True,
                       help='Output directory')
    parser.add_argument('--subsample', type=int, default=1,
                       help='Take every Nth frame (default: 1 = all frames)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    if not args.traj_file.exists():
        print(f"Error: Trajectory file not found: {args.traj_file}")
        return 1
    
    if not args.model_dir.exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        return 1
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DIPOLE COMPUTATION FROM MACE-OFF TRAJECTORIES")
    print("=" * 70)
    print(f"Trajectory:  {args.traj_file}")
    print(f"Model:       {args.model_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"Subsample:   every {args.subsample} frame(s)")
    print(f"Device:      {args.device}")
    print("=" * 70 + "\n")
    
    print("Loading model...")
    calculator = load_model_and_calculator(args.model_dir, args.device)
    print("Model loaded.\n")
    
    results = compute_dipoles_from_trajectory(
        args.traj_file,
        calculator,
        subsample=args.subsample,
        max_frames=args.max_frames
    )
    
    dipole_mags = np.linalg.norm(results['dipole_moments'], axis=1)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Frames computed:     {results['n_computed_frames']} / {results['n_total_frames']}")
    print(f"Subsample factor:    {results['subsample']}x")
    print(f"Total compute time:  {results['total_time']:.2f} s")
    print(f"Time per frame:      {results['mean_time_per_frame']*1000:.1f} ms")
    print(f"|μ| mean:            {np.mean(dipole_mags):.3f} D")
    print(f"|μ| std:             {np.std(dipole_mags):.3f} D")
    
    subsample_str = f"subsample_{args.subsample}x" if args.subsample > 1 else "full"
    output_file = args.output_dir / f"dipoles_{subsample_str}.npz"
    
    np.savez_compressed(
        output_file,
        frame_indices=results['frame_indices'],
        dipole_moments=results['dipole_moments'],
        energies=results['energies'],
        n_total_frames=results['n_total_frames'],
        n_computed_frames=results['n_computed_frames'],
        subsample=results['subsample'],
        total_time=results['total_time'],
        mean_time_per_frame=results['mean_time_per_frame'],
        traj_file=str(args.traj_file),
        model_dir=str(args.model_dir),
        computed_at=datetime.now().isoformat()
    )
    
    print(f"\nSaved to: {output_file}")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
