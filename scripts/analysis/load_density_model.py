#!/usr/bin/env python3
"""
Generic density model loader for different molecules.

Handles path remapping from training environment to local repo paths.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np
import torch

# Add the equiv_dens package to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))

from equiv_dens.training.model_loader import load_model
from equiv_dens.data.density_dataset import AtomsDensityData


# Model configurations for different molecules
MODEL_CONFIGS = {
    'ethanethiol': {
        'model_dir': 'paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4',
        'checkpoint': 'best_checkpoint_density.pth',
        'np_dataset': 'paper/models/ethanethiol/experimental_variants/unstable/datasets/ethanethiol_combo_Aidx-1000.npy',
        'dens_dataset': 'paper/models/ethanethiol/experimental_variants/unstable/datasets/ethanethiol_combo_Aidx-1000_pyscf_augccpvdz.npy',
        'radial_coeffs': 'datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy',
        'orbitals': 'datasets/augccpvqzjkfit_orbital_basis_df.npy',
        'atom_dens': 'datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy',
    },
    'ethanol': {
        'model_dir': 'paper/models/ethanol/2024-04-18_5K9Ne7it_ext',
        'checkpoint': 'best_checkpoint_density.pth',
        'np_dataset': 'datasets/ethanol_train_10.npy',
        'dens_dataset': None,  # May need to find this
        'radial_coeffs': 'datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy',
        'orbitals': 'datasets/augccpvqzjkfit_orbital_basis_df.npy',
        'atom_dens': 'datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy',
    },
    'resorcinol': {
        'model_dir': 'paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1',
        'checkpoint': 'best_checkpoint_density.pth',
        'np_dataset': 'datasets/resorcinol_combo_kmeansidx-1000_train.npy',
        'dens_dataset': None,  # May need to find this
        'radial_coeffs': 'datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy',
        'orbitals': 'datasets/augccpvqzjkfit_orbital_basis_df.npy',
        'atom_dens': 'datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy',
    },
}


def load_density_model(
    molecule: str,
    device: str = 'cuda',
    verbose: bool = True,
    dpm_intor: bool = False,
    energy_forces_only: bool = False,
) -> Tuple[Any, Any, Any]:
    """
    Load a density model for a specific molecule.
    
    Args:
        molecule: Molecule name ('ethanethiol', 'ethanol', 'resorcinol')
        device: 'cuda' or 'cpu'
        verbose: Print loading progress
        dpm_intor: If True, use analytic dipole path (dpm_intor=True, density_weight=0,
            dipole_moment_weight=1) for faster MD inference; overrides checkpoint args.
        energy_forces_only: If True, build energy+forces only (density_weight=0,
            dipole_moment_weight=0). Use for throughput benchmark without dipole computation.
        
    Returns:
        model: Loaded DFTNetwork model
        dataset: AtomsDensityData dataset object
        args: Training arguments from checkpoint (with overrides if dpm_intor=True)
    """
    if molecule not in MODEL_CONFIGS:
        raise ValueError(f"Unknown molecule: {molecule}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[molecule]
    base_dir = _REPO_ROOT
    
    # Build paths
    model_dir = base_dir / config['model_dir']
    checkpoint_path = model_dir / 'checkpoints' / config['checkpoint']
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if verbose:
        print(f"Loading {molecule} model from: {model_dir.name}")
        print(f"Device: {device}")
    
    # Load checkpoint to get training args
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    
    # Remap paths to local repo
    args.restart = str(model_dir)
    args.use_gpu = (device == 'cuda' and torch.cuda.is_available())
    
    # Set local paths
    args.orbitals_file = str(base_dir / config['orbitals'])
    args.radial_coeffs_file = str(base_dir / config['radial_coeffs'])
    args.atom_dens_path = str(base_dir / config['atom_dens'])
    args.np_dataset = str(base_dir / config['np_dataset'])
    
    # For density computation we need the density dataset
    if config['dens_dataset']:
        args.dens_dataset = str(base_dir / config['dens_dataset'])
    else:
        args.dens_dataset = None
    
    if verbose:
        print(f"  np_dataset: {args.np_dataset}")
        print(f"  dens_dataset: {args.dens_dataset}")
        print(f"  radial_coeffs: {args.radial_coeffs_file}")
    
    # Verify files exist
    for path_name, path_val in [
        ('np_dataset', args.np_dataset),
        ('radial_coeffs', args.radial_coeffs_file),
        ('orbitals', args.orbitals_file),
    ]:
        if path_val and not Path(path_val).exists():
            raise FileNotFoundError(f"{path_name} not found: {path_val}")
    
    if args.dens_dataset and not Path(args.dens_dataset).exists():
        raise FileNotFoundError(f"dens_dataset not found: {args.dens_dataset}")

    # MD config overrides (apply before dataset so atom_dens_type is correct)
    if energy_forces_only:
        args.density_weight = 0.0
        args.dipole_moment_weight = 0.0
        args.dpm_intor = False
        args.ignore_missing_keywords = True
        args.ignore_unexpected_keywords = True
        if verbose:
            print("  Using energy+forces only (no density, no dipole)")
    elif dpm_intor:
        args.dpm_intor = True
        args.density_weight = 0.0
        args.dipole_moment_weight = 1.0
        # remove_atom_density path needs mo_coeffs for intor_dipole_moment_free_atom
        if getattr(args, "atom_dens_type", "spline") == "spline":
            args.atom_dens_type = "mo_coeffs"
        if verbose:
            print("  Using analytic dipole path (dpm_intor=True, density_weight=0)")
    
    if verbose:
        print("  Loading dataset for model initialization...")
    
    dataset = AtomsDensityData(
        np_path=args.np_dataset,
        density_path=args.dens_dataset,
        orbitals_path=args.orbitals_file,
        density_n_samp=getattr(args, 'density_subsamples', 10000),
        radial_coeffs_file=args.radial_coeffs_file,
        L0_coeffs_file=getattr(args, 'L0_coeffs_file', None),
        pyscf_grid=getattr(args, 'pyscf_grid', False),
        atom_dens_path=args.atom_dens_path,
        atom_dens_type=getattr(args, 'atom_dens_type', 'spline'),
        cutoff=getattr(args, 'cutoff', 5.0),
        verbose=getattr(args, 'verbose', False),
        timing=getattr(args, 'timing', False),
        use_gpu=args.use_gpu,
        projected_density=getattr(args, 'projected_density', True),
        df_loss_weights=getattr(args, 'df_loss_weights', False),
        density_grad=getattr(args, 'density_grad', False),
        calc_basis_path=getattr(args, 'calc_basis_file', None),
        dtype=getattr(args, 'dtype', 'float32')
    )
    
    if verbose:
        print("  Loading model weights...")
    
    model = load_model(args, dataset, train=False)
    model.eval()
    
    if verbose:
        print(f"Model loaded successfully")
    
    return model, dataset, args


def create_calculator(
    model: Any,
    dataset: Any,
    args: Any,
    cpu_grid_generation: bool = True
) -> Any:
    """
    Create an ASE-compatible calculator from a loaded model.
    
    Args:
        model: Loaded DFTNetwork model
        dataset: AtomsDensityData dataset object
        args: Training arguments
        cpu_grid_generation: Use CPU for grid generation (safer)
        
    Returns:
        ASECalculatorWrapper instance
    """
    # Import the calculator wrapper
    sys.path.insert(0, str(_REPO_ROOT / 'scripts'))
    from recompute_polythiophene_dipoles_parallel import ASECalculatorWrapper
    
    data_atoms = dataset.get_properties(0)
    
    calculator = ASECalculatorWrapper(
        model=model,
        data_atoms=data_atoms,
        grid_spec=dataset.grid_spec,
        grid_sampling_fn=dataset.sampling_fn,
        density_expansion=(getattr(args, 'density_weight', 1.0) > 0),
        use_gpu=args.use_gpu,
        cutoff=getattr(args, 'cutoff', 5.0),
        atom_dens=dataset.atom_dens,
        atom_dens_type=getattr(args, 'atom_dens_type', 'spline'),
        cpu_grid_generation=cpu_grid_generation,
        gpu_grid_acceleration=False,
    )
    
    return calculator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test density model loading')
    parser.add_argument('--molecule', type=str, default='ethanethiol',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print(f"Testing model loading for {args.molecule}...")
    model, dataset, model_args = load_density_model(args.molecule, args.device)
    print(f"\nModel loaded successfully!")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
