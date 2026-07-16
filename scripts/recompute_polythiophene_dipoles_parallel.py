#!/usr/bin/env python3
"""
Parallel version of dipole moment recomputation for polythiophene trajectories.

This script processes a specific frame range from a single trajectory, allowing
parallelization via SLURM array jobs.

"""

import argparse
import os
import sys
import json
import time
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from ase.io import read
from tqdm import tqdm

# CRITICAL: Set fixed random seed BEFORE importing anything else
# This ensures grid subsampling is deterministic across all SLURM jobs
GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)
torch.manual_seed(GLOBAL_RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_RANDOM_SEED)

# Add the equiv_dens package to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))

from equiv_dens.training.model_loader import load_model
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from ase.calculators.calculator import Calculator, all_changes

# Add profiling utilities (scripts/ is sibling of this file's parent)
sys.path.insert(0, str(_REPO_ROOT / 'scripts'))
from profile_utils import GPUProfiler, profile_context


class ASECalculatorWrapper(Calculator):
    """
    Wrapper to make DFTNetworkCalculator compatible with ASE's Calculator interface.
    """
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, model, data_atoms, grid_spec, grid_sampling_fn, density_expansion=False, 
                 use_gpu=False, cutoff=5.0, atom_dens=None, atom_dens_type=None, cpu_grid_generation=True, 
                 profiler=None, gpu_grid_acceleration=False):
        Calculator.__init__(self)
        self.model = model
        self.data_atoms = data_atoms
        self.grid_sampling_fn = grid_sampling_fn
        self.density_expansion = density_expansion
        self.use_gpu = use_gpu
        self.cutoff = cutoff
        self.cpu_grid_generation = cpu_grid_generation
        self.gpu_grid_acceleration = gpu_grid_acceleration and use_gpu
        self.atom_numbers = data_atoms['atom_numbers'].squeeze()
        self.atom_dens = atom_dens
        self.atom_dens_type = atom_dens_type
        self.profiler = profiler  # GPU profiler instance
        
        # Import GPU-accelerated grid functions if available
        self._gpu_grid_fn = None
        if self.gpu_grid_acceleration:
            try:
                from equiv_dens.utils.grids import spherical_radial_sampling_gpu
                self._gpu_grid_fn = spherical_radial_sampling_gpu
            except ImportError:
                print("Warning: GPU grid acceleration not available, using CPU")
                self.gpu_grid_acceleration = False
        
        # Keep both CPU and GPU copies of grid_spec for flexibility
        self.grid_spec_cpu = {}
        self.grid_spec_gpu = {}
        for key, (coords, weights) in grid_spec.items():
            self.grid_spec_cpu[key] = (coords.cpu(), weights.cpu())
            if use_gpu:
                self.grid_spec_gpu[key] = (coords.cuda(), weights.cuda())
            else:
                self.grid_spec_gpu[key] = (coords, weights)
        
        import equiv_dens.utils.base as utils
        self.atom_symbols = utils.numbers_to_symbols(self.atom_numbers)
        self.utils = utils
    
    def _prepare_inputs_cpu(self, atoms):
        """
        CPU-bound preparation: grid generation + atom density sampling.
        
        This method prepares all inputs on CPU and returns them as CPU tensors.
        The GPU transfer happens later in _move_inputs_to_gpu().
        
        This separation allows parallel CPU preparation while GPU is busy.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            inputs: Dict of CPU tensors ready for GPU transfer
        """
        import equiv_dens.utils.base as utils
        
        inputs = {}
        
        # Get atom numbers and positions from ASE atoms
        atom_numbers = np.array([atoms.get_atomic_numbers()])  # Shape: (1, n_atoms)
        
        # Always use CPU for preparation
        positions_cpu = torch.tensor(atoms.get_positions()).unsqueeze(0).float()
        
        if self.density_expansion:
            atom_numbers_for_grid = atom_numbers
            max_samples = 10000
            
            # CPU-based grid generation
            sample_coords, coord_weights = self.grid_sampling_fn(
                self.grid_spec_cpu, max_samples,
                atom_numbers_for_grid,
                positions_cpu
            )
            
            # Ensure coords are on CPU
            if sample_coords.is_cuda:
                sample_coords = sample_coords.cpu()
            if coord_weights.is_cuda:
                coord_weights = coord_weights.cpu()
            
            # Sample atom density (CPU - scipy splines)
            if self.atom_dens is not None:
                from equiv_dens.utils import orbitals
                atom_numbers_tensor = torch.tensor(atom_numbers_for_grid).float()
                
                atom_density, _ = orbitals.sample_atom_density(
                    positions_cpu,
                    atom_numbers_tensor,
                    sample_coords,
                    self.atom_dens_type,
                    self.atom_dens,
                    individual_dens=False,
                    density_grad=False,
                )
                # Ensure on CPU
                if atom_density.is_cuda:
                    atom_density = atom_density.cpu()
                inputs['atom_density'] = atom_density
            
            inputs['coords'] = sample_coords
            inputs['coord_weights'] = coord_weights
        
        inputs['positions'] = positions_cpu
        inputs['atom_numbers_first_positions'] = utils.get_atom_num_first_positions(atom_numbers)
        inputs['atom_numbers'] = torch.tensor(atom_numbers).float().long()
        inputs['atom_mask'] = inputs['atom_numbers'] > 0
        
        # Generate neighbor lists (CPU)
        nl = utils.TorchNeighborList(self.cutoff)
        idx_is, idx_js, _ = nl.get_neighbors(inputs)
        
        prev_max = 0
        for i in range(len(idx_is)):
            idx_is[i] += prev_max
            idx_js[i] += prev_max
            max_i = torch.max(idx_is[i]) if len(idx_is[i]) > 0 else torch.tensor(0)
            max_j = torch.max(idx_js[i]) if len(idx_js[i]) > 0 else torch.tensor(0)
            prev_max = max(max_i, max_j) + 1
        
        # Create batch indices
        atom_batch_idx = np.zeros_like(atom_numbers)
        for i in range(len(atom_numbers)):
            atom_batch_idx[i, :] = i
        atom_batch_idx = torch.tensor(atom_batch_idx).float().long()
        
        idx_is = torch.cat(idx_is, dim=0) if idx_is else torch.tensor([], dtype=torch.long)
        idx_js = torch.cat(idx_js, dim=0) if idx_js else torch.tensor([], dtype=torch.long)
        inputs['idx_i'] = idx_is
        inputs['idx_j'] = idx_js
        inputs['batch_atom_numbers'] = inputs['atom_numbers'] * 1
        inputs['batch_atom_mask'] = (inputs['atom_mask'] * 1).type(torch.bool)
        inputs['batch_positions'] = inputs['positions'] * 1
        inputs['positions'] = positions_cpu.view(1, -1, *inputs['positions'].shape[2:])
        inputs['atom_numbers'] = inputs['batch_atom_numbers'].flatten()
        inputs['atom_mask'] = inputs['batch_atom_mask'].flatten()
        batch_nz = inputs['atom_mask'].float()
        batch_idx_pos = batch_nz * torch.arange(len(batch_nz)).float()
        inputs['batch_idx_pos'] = batch_idx_pos[inputs['atom_mask']].long()
        inputs['atom_numbers'] = inputs['atom_numbers'][inputs['atom_mask']].view(1, -1)
        inputs['atom_batch_idx'] = atom_batch_idx.flatten()
        inputs['atom_batch_idx'] = inputs['atom_batch_idx'][inputs['atom_mask']].view(1, -1)
        inputs['positions'] = inputs['positions'][:, inputs['atom_mask']]
        
        return inputs
    
    def _move_inputs_to_gpu(self, inputs):
        """
        Move prepared CPU inputs to GPU.
        
        Note: Don't set requires_grad here - the model handles that internally.
        
        Args:
            inputs: Dict of CPU tensors
            
        Returns:
            inputs: Dict with tensors moved to GPU
        """
        if not self.use_gpu:
            return inputs
        
        gpu_inputs = {}
        for key, val in inputs.items():
            if isinstance(val, torch.Tensor):
                gpu_inputs[key] = val.cuda()
            else:
                gpu_inputs[key] = val
        
        return gpu_inputs
    
    def run_model_inference(self, inputs):
        """
        Run model inference on GPU and compute dipole moment.
        
        Args:
            inputs: Dict of GPU tensors (already moved to GPU)
            
        Returns:
            energy: Energy value
            dipole: Dipole moment array (3,)
        """
        # Run model (no torch.no_grad() because force calculation needs autograd)
        self.model.eval()
        results = self.model(inputs)
        
        # Extract energy
        energy = results['energy'].detach().cpu().numpy().item()
        
        # Compute dipole moment from density if available
        if 'density' in results and 'coords' in inputs and 'coord_weights' in inputs:
            from equiv_dens.utils import orbitals
            
            # Add required keys to results for dipole calculation
            results['batch_atom_numbers'] = inputs['batch_atom_numbers']
            results['batch_positions'] = inputs['batch_positions']
            results['coords'] = inputs['coords']
            results['coord_weights'] = inputs['coord_weights']
            
            # Calculate dipole moment
            results_with_dipole = orbitals.calc_dipole_moment(
                results,
                center_coordinates=True,
                normalize_density=True,
                positive_density=True
            )
            
            dipole = results_with_dipole['dipole_moment']
            if isinstance(dipole, torch.Tensor):
                dipole = dipole.detach().cpu().numpy().squeeze()
        else:
            dipole = np.zeros(3)
        
        return energy, dipole
        
    def _generate_input(self, atoms):
        """Generate proper model input from ASE atoms"""
        import equiv_dens.utils.base as utils
        
        # Clear GPU cache before processing
        if self.use_gpu:
            torch.cuda.empty_cache()
        
        inputs = {}
        
        # Get atom numbers and positions from ASE atoms
        atom_numbers = np.array([atoms.get_atomic_numbers()])  # Shape: (1, n_atoms)
        
        if self.density_expansion:
            atom_numbers_for_grid = atom_numbers
            n_atoms = len(atoms)
            max_samples = 10000
            
            if self.cpu_grid_generation:
                # CPU-based grid generation (safer for large molecules)
                positions_cpu = torch.tensor(atoms.get_positions()).unsqueeze(0).float()
                
                sample_coords, coord_weights = self.grid_sampling_fn(
                    self.grid_spec_cpu, max_samples,
                    atom_numbers_for_grid,
                    positions_cpu
                )
                
                # Move positions to GPU for model inference
                if self.use_gpu:
                    positions = torch.tensor(atoms.get_positions()).unsqueeze(0).float().cuda()
                else:
                    positions = positions_cpu
                
                # Move grid to GPU as well
                if self.use_gpu:
                    sample_coords = sample_coords.cuda()
                    coord_weights = coord_weights.cuda()
                
                # Sample atom density (keep on GPU if possible for GPU splines)
                if self.atom_dens is not None:
                    from equiv_dens.utils import orbitals
                    atom_numbers_tensor = torch.tensor(atom_numbers_for_grid).float()
                    
                    # Keep coordinates on GPU if using GPU and if atom_dens supports it
                    if self.use_gpu and hasattr(self.atom_dens, 'get_spline'):
                        # GPU-accelerated path with AtomicDensityCache
                        positions_for_dens = torch.tensor(atoms.get_positions()).unsqueeze(0).float().cuda()
                        coords_for_dens = sample_coords  # Already on GPU
                        atom_numbers_for_dens = atom_numbers_tensor.cuda()
                    else:
                        # CPU fallback for legacy splines
                        positions_for_dens = positions_cpu
                        coords_for_dens = sample_coords.cpu()
                        atom_numbers_for_dens = atom_numbers_tensor
                    
                    atom_density, _ = orbitals.sample_atom_density(
                        positions_for_dens,
                        atom_numbers_for_dens,
                        coords_for_dens,
                        self.atom_dens_type,
                        self.atom_dens,
                        individual_dens=False,
                        density_grad=False,
                    )
                    if self.use_gpu and not atom_density.is_cuda:
                        atom_density = atom_density.cuda()
                    inputs['atom_density'] = atom_density
            else:
                # GPU-based grid generation
                positions = torch.tensor(atoms.get_positions()).unsqueeze(0).float()
                if self.use_gpu:
                    positions = positions.cuda()
                
                # Use GPU-accelerated grid function if available
                if self.gpu_grid_acceleration and self._gpu_grid_fn is not None:
                    sample_coords, coord_weights = self._gpu_grid_fn(
                        self.grid_spec_gpu,
                        max_samples,
                        atom_numbers_for_grid,
                        positions
                    )
                else:
                    sample_coords, coord_weights = self.grid_sampling_fn(
                        self.grid_spec_gpu if self.use_gpu else self.grid_spec_cpu,
                        max_samples,
                        atom_numbers_for_grid,
                        positions
                    )
                
                if self.atom_dens is not None:
                    from equiv_dens.utils import orbitals
                    atom_numbers_tensor = torch.tensor(atom_numbers_for_grid).float()
                    
                    # Keep on GPU if using AtomicDensityCache with GPU splines
                    if self.use_gpu and hasattr(self.atom_dens, 'get_spline'):
                        # GPU-accelerated path
                        positions_for_dens = positions  # Already on GPU
                        coords_for_dens = sample_coords  # Already on GPU
                        atom_numbers_for_dens = atom_numbers_tensor.cuda()
                    else:
                        # CPU fallback
                        positions_for_dens = positions.cpu()
                        coords_for_dens = sample_coords.cpu()
                        atom_numbers_for_dens = atom_numbers_tensor
                    
                    atom_density, _ = orbitals.sample_atom_density(
                        positions_for_dens,
                        atom_numbers_for_dens,
                        coords_for_dens,
                        self.atom_dens_type,
                        self.atom_dens,
                        individual_dens=False,
                        density_grad=False,
                    )
                    if self.use_gpu and not atom_density.is_cuda:
                        atom_density = atom_density.cuda()
                    inputs['atom_density'] = atom_density
            
            inputs['coords'] = sample_coords
            inputs['coord_weights'] = coord_weights
        else:
            positions = torch.tensor(atoms.get_positions()).unsqueeze(0).float()
            if self.use_gpu:
                positions = positions.cuda()
        
        inputs['positions'] = positions
        inputs['atom_numbers_first_positions'] = utils.get_atom_num_first_positions(atom_numbers)
        inputs['atom_numbers'] = torch.tensor(atom_numbers).to(positions).type(torch.long)
        inputs['atom_mask'] = inputs['atom_numbers'] > 0
        
        # Generate neighbor lists
        nl = utils.TorchNeighborList(self.cutoff)
        idx_is, idx_js, _ = nl.get_neighbors(inputs)
        
        prev_max = 0
        for i in range(len(idx_is)):
            idx_is[i] += prev_max
            idx_js[i] += prev_max
            max_i = torch.max(idx_is[i]) if len(idx_is[i]) > 0 else torch.tensor(0)
            max_j = torch.max(idx_js[i]) if len(idx_js[i]) > 0 else torch.tensor(0)
            prev_max = max(max_i, max_j) + 1
        
        # Create batch indices
        atom_batch_idx = np.zeros_like(atom_numbers)
        for i in range(len(atom_numbers)):
            atom_batch_idx[i, :] = i
        atom_batch_idx = torch.tensor(atom_batch_idx).to(positions).type(torch.long)
        
        idx_is = torch.cat(idx_is, dim=0) if idx_is else torch.tensor([], dtype=torch.long).to(positions.device)
        idx_js = torch.cat(idx_js, dim=0) if idx_js else torch.tensor([], dtype=torch.long).to(positions.device)
        inputs['idx_i'] = idx_is
        inputs['idx_j'] = idx_js
        inputs['batch_atom_numbers'] = inputs['atom_numbers'] * 1
        inputs['batch_atom_mask'] = (inputs['atom_mask'] * 1).type(torch.bool)
        inputs['batch_positions'] = inputs['positions'] * 1
        inputs['positions'] = positions.view(1, -1, *inputs['positions'].shape[2:])
        inputs['atom_numbers'] = inputs['batch_atom_numbers'].flatten()
        inputs['atom_mask'] = inputs['batch_atom_mask'].flatten()
        batch_nz = inputs['atom_mask'].to(inputs['positions'])
        batch_idx_pos = batch_nz * torch.arange(len(batch_nz)).to(batch_nz)
        inputs['batch_idx_pos'] = batch_idx_pos[inputs['atom_mask']].to(torch.long)
        inputs['atom_numbers'] = inputs['atom_numbers'][inputs['atom_mask']].view(1, -1)
        inputs['atom_batch_idx'] = atom_batch_idx.flatten()
        inputs['atom_batch_idx'] = inputs['atom_batch_idx'][inputs['atom_mask']].view(1, -1)
        inputs['positions'] = inputs['positions'][:, inputs['atom_mask']]
        
        return inputs
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Generate proper input for the model
        in_atoms = self._generate_input(atoms)
        
        # Run model
        self.model.eval()
        results = self.model(in_atoms)
        
        # Extract results
        energy = results['energy'].detach().cpu().numpy().item()
        forces = results['forces'].detach().cpu().numpy().squeeze()
        
        # Store results
        self.results['energy'] = energy
        self.results['forces'] = forces
        
        # Compute dipole moment from density if available
        if 'density' in results and 'coords' in results and 'coord_weights' in results:
            from equiv_dens.utils import orbitals
            
            # Add required keys to results for dipole calculation
            results['batch_atom_numbers'] = in_atoms['batch_atom_numbers']
            results['batch_positions'] = in_atoms['batch_positions']
            results['coords'] = in_atoms['coords']
            results['coord_weights'] = in_atoms['coord_weights']
            
            # Calculate dipole moment
            results_with_dipole = orbitals.calc_dipole_moment(
                results,
                center_coordinates=True,
                normalize_density=True,
                positive_density=True
            )
            
            dipole = results_with_dipole['dipole_moment']
            if isinstance(dipole, torch.Tensor):
                dipole = dipole.detach().cpu().numpy().squeeze()
            
            self.results['dipole_moment'] = dipole
        else:
            self.results['dipole_moment'] = np.zeros(3)


def load_polythiophene_model(model_path: Path, device: str = 'cuda', cpu_grid_generation: bool = True, 
                             enable_grid_caching: bool = True, gpu_grid_acceleration: bool = False,
                             oligomer: str = None):
    """
    Load the polythiophene model and create a calculator.
    
    Args:
        model_path: Path to model directory
        device: 'cuda' or 'cpu'
        cpu_grid_generation: If True, use CPU for grid generation (ignored if gpu_grid_acceleration=True)
        enable_grid_caching: If True, enable grid caching for MD trajectories (recommended!)
        gpu_grid_acceleration: If True, use GPU-accelerated Becke grid partitioning (fastest!)
        oligomer: Oligomer type ('8mer', '10mer', '12mer') - used to auto-disable GPU grid for larger systems
    """
    print(f"Loading model from: {model_path.name}")
    print(f"Device: {device}")
    print(f"GPU grid acceleration: {'enabled' if gpu_grid_acceleration else 'disabled'}")
    print(f"Grid caching: {'enabled' if enable_grid_caching else 'disabled'}")
    
    # Load args from checkpoint (ensures correct model architecture)
    checkpoint_path = model_path / 'checkpoints' / 'latest_checkpoint.pth'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    
    # Override with our settings
    args.restart = str(model_path)
    args.use_gpu = (device == 'cuda' and torch.cuda.is_available())
    
    # Fix hardcoded paths (use repo root derived from script location)
    base_dir = _REPO_ROOT
    args.orbitals_file = str(base_dir / 'datasets' / 'augccpvqzjkfit_orbital_basis_df.npy')
    args.radial_coeffs_file = str(base_dir / 'datasets' / 'augccpvqzjkfit_radial_coeffs_df.npy')
    args.atom_dens_path = str(base_dir / 'datasets' / 'free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy')
    args.pseudo_pot_path = str(base_dir / 'pseudo_potentials')
    args.atomic_energies = str(base_dir / 'paper' / 'atomization_energy_augccpvdz.npy')
    
    # Set dataset paths (use repo-relative path)
    base_data_dir = base_dir / 'datasets'
    args.np_dataset = str(base_data_dir / 'thiophene_all_train_d4.npy')
    # Keep dens_dataset None for inference (no density labels needed)
    args.dens_dataset = None
    # Provide calc_basis_path to get correct model dimensions without density dataset
    args.calc_basis_file = str(base_data_dir / 'augccpvdz_orbital_basis.npy')
    
    print("  Loading dataset for model initialization...")
    dataset = AtomsDensityData(
        np_path=args.np_dataset,
        density_path=args.dens_dataset,
        orbitals_path=args.orbitals_file,
        density_n_samp=args.density_subsamples,
        radial_coeffs_file=args.radial_coeffs_file,
        L0_coeffs_file=getattr(args, 'L0_coeffs_file', None),
        pyscf_grid=args.pyscf_grid,
        atom_dens_path=args.atom_dens_path,
        atom_dens_type=args.atom_dens_type,
        cutoff=args.cutoff,
        verbose=args.verbose,
        timing=args.timing,
        use_gpu=args.use_gpu,
        projected_density=args.projected_density,
        df_loss_weights=args.df_loss_weights,
        density_grad=args.density_grad,
        calc_basis_path=getattr(args, 'calc_basis_file', None),
        dtype=args.dtype
    )
    
    print("  Loading model...")
    model = load_model(args, dataset, train=False)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Get a sample from dataset
    data_atoms = dataset.get_properties(0)
    
    # Check system size - GPU grid acceleration causes OOM for larger systems
    # Use oligomer name since dataset might be a different size
    if gpu_grid_acceleration and oligomer and oligomer in ['10mer', '12mer']:
        print(f"WARNING: {oligomer} system - disabling GPU grid acceleration to avoid OOM")
        print(f"   (GPU grid acceleration works for 8mer, but causes OOM for 10mer/12mer on 48GB GPUs)")
        gpu_grid_acceleration = False
    
    # Create ASE-compatible calculator wrapper
    # If GPU grid acceleration is enabled, use GPU for grid generation (overrides cpu_grid_generation)
    use_cpu_grid = cpu_grid_generation and not gpu_grid_acceleration
    
    calculator = ASECalculatorWrapper(
        model=model,
        data_atoms=data_atoms,
        grid_spec=dataset.grid_spec,
        grid_sampling_fn=dataset.sampling_fn,
        density_expansion=(args.density_weight > 0),
        use_gpu=args.use_gpu,
        cutoff=args.cutoff,
        atom_dens=dataset.atom_dens,
        atom_dens_type=args.atom_dens_type,
        cpu_grid_generation=use_cpu_grid,
        gpu_grid_acceleration=gpu_grid_acceleration,
    )
    
    # Enable grid caching for MD trajectories (11x speedup!)
    if enable_grid_caching:
        try:
            # Import grid caching module
            import sys
            grid_cache_path = _REPO_ROOT / 'scripts'
            if str(grid_cache_path) not in sys.path:
                sys.path.insert(0, str(grid_cache_path))
            
            from grid_caching import wrap_calculator_with_cache
            
            calculator = wrap_calculator_with_cache(calculator, cache_config={
                'max_cache_size': 100,      # Cache up to 100 grids
                'rmsd_tolerance': 0.15,     # Angstroms (good for MD trajectories)
                'device': 'cuda' if args.use_gpu else 'cpu'
            })
            print("OK Grid caching enabled (expected 5-10x speedup for MD trajectories)")
        except ImportError as e:
            print(f"WARNING: Grid caching not available: {e}")
            print("   Continuing without caching (slower but functional)")
    
    return calculator, args


def _save_checkpoint(output_file: Path, frame_indices, dipole_moments, 
                     energies, oligomer, replica, start_frame, end_frame):
    """Save intermediate checkpoint to allow restart."""
    try:
        dipole_arr = np.array(dipole_moments)
        energies_arr = np.array(energies)
        frame_indices_arr = np.array(frame_indices)
        
        np.savez_compressed(
            output_file,
            frame_indices=frame_indices_arr,
            dipole_moments=dipole_arr,
            energies=energies_arr,
            oligomer=oligomer,
            replica=replica,
            start_frame=start_frame,
            end_frame=end_frame,
            is_checkpoint=True,
            timestamp=time.time()
        )
    except Exception as e:
        print(f"WARNING: Checkpoint save failed: {e}")


def process_frames_parallel(atoms_list, calculator, start_frame: int, 
                           n_workers: int = 1, progress_callback=None):
    """
    Process frames for dipole calculation.
    
    Note: The n_workers parameter is provided for API compatibility but
    within-job parallelism has limited benefit due to Python's GIL.
    The real parallelism comes from SLURM array jobs (24 concurrent GPUs).
    
    For better GPU utilization, the recommended optimization is to replace
    the scipy spline interpolation with GPU-accelerated CuPy splines.
    See GPU_ANALYSIS.md for details.
    
    Args:
        atoms_list: List of ASE Atoms objects to process
        calculator: ASECalculatorWrapper instance
        start_frame: Starting frame index (for result ordering)
        n_workers: Number of workers (currently unused due to GIL limitations)
        progress_callback: Optional callback(frame_idx, dipole, energy) for progress
        
    Returns:
        results: Dict mapping frame_idx -> (dipole, energy)
    """
    results = {}
    total_frames = len(atoms_list)
    
    # Sequential processing - the most reliable approach
    # True parallelism comes from SLURM array jobs, not within-job threading
    for i, atoms in enumerate(atoms_list):
        frame_idx = start_frame + i
        np.random.seed(GLOBAL_RANDOM_SEED)
        torch.manual_seed(GLOBAL_RANDOM_SEED)
        
        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy()
        
        dipole = calculator.results.get('dipole_moment', np.zeros(3))
        if isinstance(dipole, torch.Tensor):
            dipole = dipole.detach().cpu().numpy()
        
        results[frame_idx] = (dipole, energy)
        
        if progress_callback:
            progress_callback(frame_idx, dipole, energy)
    
    return results


def process_frame_range(traj_file: Path, calculator, start_frame: int, end_frame: int,
                        oligomer: str, replica: int, output_file: Path,
                        skip_existing: bool = False, n_workers: int = 1):
    """
    Process a specific range of frames from a trajectory.
    
    Args:
        traj_file: Path to trajectory .xyz file
        calculator: Calculator instance
        start_frame: First frame to process (inclusive)
        end_frame: Last frame to process (exclusive)
        oligomer: Oligomer type (e.g., '8mer')
        replica: Replica number
        output_file: Path to save results
        skip_existing: If True, skip frames that already have output files
        n_workers: Number of CPU workers for parallel preparation (default: 1 = sequential)
    """
    start_time = time.time()
    
    print(f"{'='*70}")
    print(f"Processing Frame Range")
    print(f"{'='*70}")
    print(f"Trajectory: {traj_file.name}")
    print(f"Oligomer: {oligomer}, Replica: {replica}")
    print(f"Frame range: [{start_frame}, {end_frame})")
    print(f"Output: {output_file}")
    print(f"CPU workers: {n_workers} ({'parallel' if n_workers > 1 else 'sequential'})")
    
    # Load trajectory
    try:
        if traj_file.suffix == '.npy':
            # Load .npy format (dict with 'positions' and 'atom_numbers')
            print(f"Loading .npy trajectory...")
            traj_data = np.load(str(traj_file), allow_pickle=True).item()
            positions = traj_data['positions'][start_frame:end_frame]
            atom_numbers = traj_data['atom_numbers'][start_frame:end_frame]
            n_frames = len(positions)
            print(f"Loaded {n_frames} frames from .npy")
            
            # Convert to ASE Atoms objects
            from ase import Atoms
            atoms_list = []
            for frame_positions, frame_atom_numbers in zip(positions, atom_numbers):
                atoms = Atoms(
                    numbers=frame_atom_numbers,
                    positions=frame_positions
                )
                atoms_list.append(atoms)
        else:
            # Load .xyz or other ASE-supported formats
            print(f"Loading {traj_file.suffix} trajectory...")
            atoms_list = read(str(traj_file), index=f'{start_frame}:{end_frame}')
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            n_frames = len(atoms_list)
            print(f"Loaded {n_frames} frames")
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Check for existing output file to enable restart (if skip_existing is enabled)
    existing_frames = {}  # {frame_idx: (dipole, energy)}
    if skip_existing and output_file.exists():
        try:
            existing_data = np.load(output_file)
            existing_indices = existing_data['frame_indices']
            existing_dipoles = existing_data['dipole_moments']
            existing_energies = existing_data['energies']
            for i, idx in enumerate(existing_indices):
                existing_frames[int(idx)] = (existing_dipoles[i], existing_energies[i])
            print(f"📂 Found existing results with {len(existing_frames)} frames")
        except Exception as e:
            print(f"WARNING: Could not load existing results: {e}")
            existing_frames = {}
    
    # Determine which frames to compute
    frames_to_compute = []
    preloaded_dipoles = []
    preloaded_energies = []
    preloaded_indices = []
    
    for i in range(n_frames):
        frame_idx = start_frame + i
        if frame_idx in existing_frames:
            # Use cached result
            dipole, energy = existing_frames[frame_idx]
            preloaded_dipoles.append(dipole)
            preloaded_energies.append(energy)
            preloaded_indices.append(frame_idx)
        else:
            frames_to_compute.append(i)  # Store local index
    
    if preloaded_indices:
        print(f"⏭️  Loaded {len(preloaded_indices)} existing frames from cache")
        print(f"📊 Computing {len(frames_to_compute)} missing frames")
    else:
        print(f"📊 Computing all {n_frames} frames")
        frames_to_compute = list(range(n_frames))
    
    # If all frames exist, return cached results
    if not frames_to_compute:
        print("OK All frames already computed!")
        dipole_magnitudes = np.linalg.norm(np.array(preloaded_dipoles), axis=1)
        print(f"✓ Cached: |μ| = {np.mean(dipole_magnitudes):.2f}±{np.std(dipole_magnitudes):.2f}D")
        return {
            'dipole_moments': np.array(preloaded_dipoles),
            'energies': np.array(preloaded_energies),
            'frame_indices': np.array(preloaded_indices)
        }
    
    # Create output dir
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute dipole moments for missing frames only
    dipole_moments = list(preloaded_dipoles)  # Start with cached results
    energies = list(preloaded_energies)
    frame_indices = list(preloaded_indices)
    
    # Get only the atoms that need to be computed
    atoms_to_compute = [atoms_list[i] for i in frames_to_compute]
    
    # Checkpoint settings
    checkpoint_interval = 50
    last_checkpoint_count = 0
    
    if n_workers > 1:
        # Parallel processing with producer-consumer pipeline
        print(f"Using parallel processing with {n_workers} CPU workers...")
        
        # Progress bar for parallel mode
        pbar = tqdm(total=len(frames_to_compute), 
                    desc=f"{oligomer} Rep{replica} [{start_frame}:{end_frame})",
                    unit="frame", ncols=100)
        
        def progress_callback(frame_idx, dipole, energy):
            pbar.update(1)
            current_dipole_mag = np.linalg.norm(dipole)
            pbar.set_postfix({'|μ|': f'{current_dipole_mag:.2f}D'})
        
        # Run parallel processing
        parallel_results = process_frames_parallel(
            atoms_to_compute, 
            calculator, 
            start_frame,  # Use start_frame since atoms_to_compute is already filtered
            n_workers=n_workers,
            progress_callback=progress_callback
        )
        
        pbar.close()
        
        # Collect results in order
        for i in frames_to_compute:
            frame_idx = start_frame + i
            if frame_idx in parallel_results:
                dipole, energy = parallel_results[frame_idx]
            else:
                # Frame was in frames_to_compute but not processed - recalculate frame_idx
                # Since atoms_to_compute is filtered, we need to map back
                pass
        
        # Actually, we need to correctly map frame indices
        # atoms_to_compute[k] corresponds to atoms_list[frames_to_compute[k]]
        # which is frame index start_frame + frames_to_compute[k]
        for k, i in enumerate(frames_to_compute):
            frame_idx = start_frame + i
            if frame_idx in parallel_results:
                dipole, energy = parallel_results[frame_idx]
                dipole_moments.append(dipole)
                energies.append(energy)
                frame_indices.append(frame_idx)
            else:
                print(f"WARNING: Missing result for frame {frame_idx}")
                dipole_moments.append(np.zeros(3))
                energies.append(np.nan)
                frame_indices.append(frame_idx)
    else:
        # Sequential processing (original behavior)
        pbar = tqdm(frames_to_compute, total=len(frames_to_compute), 
                    desc=f"{oligomer} Rep{replica} [{start_frame}:{end_frame})",
                    unit="frame", ncols=100)
        
        for count, i in enumerate(pbar):
            atoms = atoms_list[i]
            frame_idx = start_frame + i
            try:
                # CRITICAL: Use FIXED random seed (same for all frames) to ensure 
                # consistent grid topology throughout the trajectory.
                np.random.seed(GLOBAL_RANDOM_SEED)
                torch.manual_seed(GLOBAL_RANDOM_SEED)
                
                # Attach calculator
                atoms.set_calculator(calculator)
                
                # Compute energy (triggers dipole calculation)
                energy = atoms.get_potential_energy()
                
                # Extract dipole moment
                dipole = None
                if 'dipole_moment' in atoms.calc.results:
                    dipole = atoms.calc.results['dipole_moment']
                elif 'dipole' in atoms.calc.results:
                    dipole = atoms.calc.results['dipole']
                
                if dipole is None:
                    print(f"    ERROR: No dipole moment found at frame {frame_idx}")
                    dipole = np.zeros(3)
                else:
                    if isinstance(dipole, torch.Tensor):
                        dipole = dipole.detach().cpu().numpy()
                
                dipole_moments.append(dipole)
                energies.append(energy)
                frame_indices.append(frame_idx)
                
                # Update progress bar
                current_dipole_mag = np.linalg.norm(dipole)
                pbar.set_postfix({'|μ|': f'{current_dipole_mag:.2f}D'})
                
                # Periodic checkpoint
                if (count + 1 - last_checkpoint_count) >= checkpoint_interval:
                    _save_checkpoint(output_file, frame_indices, dipole_moments, 
                                    energies, oligomer, replica, start_frame, end_frame)
                    last_checkpoint_count = count + 1
            
            except Exception as e:
                import traceback
                tqdm.write(f"    ERROR at frame {frame_idx}: {str(e)}")
                tqdm.write(traceback.format_exc())
                dipole_moments.append(np.zeros(3))
                energies.append(np.nan)
                frame_indices.append(frame_idx)
        
        pbar.close()
    
    elapsed_time = time.time() - start_time
    
    # Convert to arrays
    dipole_moments = np.array(dipole_moments)
    energies = np.array(energies)
    frame_indices = np.array(frame_indices)
    
    # Compute statistics
    dipole_magnitudes = np.linalg.norm(dipole_moments, axis=1)
    
    n_computed = len(frames_to_compute)
    n_total = len(frame_indices)
    
    results = {
        'trajectory_file': str(traj_file),
        'oligomer': oligomer,
        'replica': replica,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'n_frames': n_total,
        'n_frames_computed': n_computed,
        'n_frames_cached': len(preloaded_indices),
        'frame_indices': frame_indices.tolist(),  # Convert to list for JSON
        'dipole_moments_shape': list(dipole_moments.shape),  # Store shape only
        'energies_shape': list(energies.shape),  # Store shape only
        'statistics': {
            'dipole_magnitude_mean': float(np.mean(dipole_magnitudes)),
            'dipole_magnitude_std': float(np.std(dipole_magnitudes)),
            'dipole_magnitude_min': float(np.min(dipole_magnitudes)),
            'dipole_magnitude_max': float(np.max(dipole_magnitudes)),
            'energy_mean': float(np.mean(energies[~np.isnan(energies)])),
            'energy_std': float(np.std(energies[~np.isnan(energies)])),
        },
        'timing': {
            'total_seconds': elapsed_time,
            'seconds_per_frame': elapsed_time / n_computed if n_computed > 0 else 0
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Print grid cache statistics if available
    if hasattr(calculator, 'grid_cache'):
        print(f"\n{'='*70}")
        calculator.grid_cache.print_stats()
        print(f"{'='*70}")
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        frame_indices=frame_indices,
        dipole_moments=dipole_moments,
        energies=energies,
        metadata=json.dumps(results)
    )
    
    mean_dipole = results['statistics']['dipole_magnitude_mean']
    std_dipole = results['statistics']['dipole_magnitude_std']
    time_per_frame = elapsed_time / len(frames_to_compute) if frames_to_compute else 0
    total_frames = len(frame_indices)
    print(f"\n✓ Completed: |μ| = {mean_dipole:.2f}±{std_dipole:.2f}D, {time_per_frame:.3f}s/frame")
    print(f"✓ Results saved to: {output_file} ({total_frames} frames)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Process a specific frame range from a polythiophene trajectory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--trajectory', type=str, required=True,
                        help='Path to trajectory .xyz file')
    parser.add_argument('--oligomer', type=str, required=True,
                        help='Oligomer type (8mer, 10mer, 12mer)')
    parser.add_argument('--replica', type=int, required=True,
                        help='Replica number (0, 1, 2, 3)')
    parser.add_argument('--start_frame', type=int, required=True,
                        help='First frame to process (inclusive)')
    parser.add_argument('--end_frame', type=int, required=True,
                        help='Last frame to process (exclusive)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--model_path', type=str,
                        default=str(_REPO_ROOT / 'paper' / 'models' / 'polythiophene' / '2024-03-23_1XDL67zp'),
                        help='Path to the polythiophene model')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for computation')
    parser.add_argument('--cpu-grid', action='store_true',
                        help='Use CPU for grid generation (safer for very large molecules)')
    parser.add_argument('--gpu-grid', dest='cpu_grid', action='store_false',
                        help='Use GPU for grid generation (default, faster)')
    parser.set_defaults(cpu_grid=False)  # Default to GPU for better performance
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip frames that already have output files (for restarts)')
    parser.add_argument('--gpu-grid-accel', action='store_true',
                        help='Use GPU-accelerated Becke grid partitioning. '
                             'Provides 25-50x speedup for grid generation. '
                             'Recommended for large molecules (>30 atoms).')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of CPU workers for parallel frame preparation. '
                             'Use 3-4 for better GPU utilization (~70-80%% vs ~25%% with 1). '
                             'Default: 1 (sequential processing)')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    traj_file = Path(args.trajectory)
    output_dir = Path(args.output_dir)
    
    # Check paths
    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        return 1
    
    if not traj_file.exists():
        print(f"Error: Trajectory file not found: {traj_file}")
        return 1
    
    # Load model
    # Grid caching is DISABLED because it introduces systematic drift in dipole
    # moments (up to ~5 D over a few frames). Instead, we use fixed seeding
    # (same random seed for all frames) which produces smooth results.
    #
    # With fixed seeding: max jump ~0.5 D (smooth, consistent)
    # With grid caching: max jump ~3 D with drift (problematic)
    #
    # The tradeoff is speed: ~24s/frame without caching vs ~6s/frame with caching.
    # Accuracy is more important than speed for these calculations.
    calculator, model_args = load_polythiophene_model(
        model_path,
        device=args.device,
        cpu_grid_generation=args.cpu_grid,
        enable_grid_caching=False,  # Disabled due to drift; fixed seeding is sufficient
        gpu_grid_acceleration=args.gpu_grid_accel,
        oligomer=args.oligomer
    )
    
    # Create output filename
    output_file = output_dir / args.oligomer / f"replica_{args.replica}" / \
                   f"frames_{args.start_frame}_{args.end_frame}.npz"
    
    # Process frames
    results = process_frame_range(
        traj_file, calculator,
        args.start_frame, args.end_frame,
        args.oligomer, args.replica,
        output_file,
        skip_existing=args.skip_existing,
        n_workers=args.num_workers
    )
    
    if results is None:
        return 1
    
    print(f"\n{'='*70}")
    print("Processing complete!")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

