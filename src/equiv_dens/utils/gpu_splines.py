#!/usr/bin/env python3
"""Spline evaluation on GPU (CuPy) with scipy fallback on CPU.

Called from free-atom density loading during MD; see grids.py.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional

# Try to import CuPy
try:
    import cupy as cp
    from cupyx.scipy.interpolate import make_interp_spline as cp_make_interp_spline
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Fallback to scipy
from scipy.interpolate import make_interp_spline as scipy_make_interp_spline

# Import parameter constants
try:
    from equiv_dens.utils import param
except ImportError:
    # Fallback if param module not available
    class param:
        BOHR = 0.529177210903  # Angstrom to Bohr conversion


class GPUSpline:
    """CuPy spline with scipy fallback if GPU init fails."""
    
    def __init__(self, x, y, k=7, use_gpu=True):
        """k=7 matches the radial basis order used in atom density tables."""
        self.k = k
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        # Convert to appropriate format
        if self.use_gpu:
            # Convert to CuPy for GPU computation
            if isinstance(x, torch.Tensor):
                if x.is_cuda:
                    x_gpu = cp.asarray(x.detach())
                    y_gpu = cp.asarray(y.detach())
                else:
                    x_gpu = cp.asarray(x.detach().numpy())
                    y_gpu = cp.asarray(y.detach().numpy())
            elif isinstance(x, np.ndarray):
                x_gpu = cp.asarray(x)
                y_gpu = cp.asarray(y)
            else:
                # Already CuPy
                x_gpu = x
                y_gpu = y
            
            try:
                self.spl = cp_make_interp_spline(x_gpu, y_gpu, k=k)
                self.is_gpu = True
            except Exception as e:
                print(f"Warning: CuPy spline creation failed ({e}), falling back to scipy")
                self.use_gpu = False
                self.is_gpu = False
                # Fallback to scipy
                x_cpu = cp.asnumpy(x_gpu) if hasattr(x_gpu, 'get') else x
                y_cpu = cp.asnumpy(y_gpu) if hasattr(y_gpu, 'get') else y
                self.spl = scipy_make_interp_spline(x_cpu, y_cpu, k=k)
        else:
            # Use CPU scipy
            if isinstance(x, torch.Tensor):
                x_cpu = x.detach().cpu().numpy()
                y_cpu = y.detach().cpu().numpy()
            elif hasattr(x, 'get'):  # CuPy array
                x_cpu = cp.asnumpy(x)
                y_cpu = cp.asnumpy(y)
            else:
                x_cpu = x
                y_cpu = y
            
            self.spl = scipy_make_interp_spline(x_cpu, y_cpu, k=k)
            self.is_gpu = False
    
    def __call__(self, x):
        """
        Evaluate spline at given points.
        
        Args:
            x: Points to evaluate (numpy/torch/cupy array)
            
        Returns:
            Interpolated values in same format as input
        """
        input_type = type(x)
        was_torch = isinstance(x, torch.Tensor)
        was_cuda = was_torch and x.is_cuda
        
        if self.is_gpu:
            # GPU evaluation
            if isinstance(x, torch.Tensor):
                if x.is_cuda:
                    x_eval = cp.asarray(x.detach())
                else:
                    x_eval = cp.asarray(x.detach().numpy())
            elif isinstance(x, np.ndarray):
                x_eval = cp.asarray(x)
            else:
                x_eval = x
            
            result = self.spl(x_eval)
            
            # Convert back to original format
            if was_torch:
                if was_cuda:
                    result = torch.as_tensor(result, device='cuda')
                else:
                    result = torch.from_numpy(cp.asnumpy(result))
            elif isinstance(x, np.ndarray):
                result = cp.asnumpy(result)
            
            return result
        else:
            # CPU evaluation
            if isinstance(x, torch.Tensor):
                x_eval = x.detach().cpu().numpy()
            elif hasattr(x, 'get'):  # CuPy array
                x_eval = cp.asnumpy(x)
            else:
                x_eval = x
            
            result = self.spl(x_eval)
            
            # Convert back to original format
            if was_torch:
                if was_cuda:
                    result = torch.from_numpy(result).cuda()
                else:
                    result = torch.from_numpy(result)
            
            return result
    
    def derivative(self, n=1):
        """
        Return the derivative of the spline.
        
        Args:
            n: Order of derivative
            
        Returns:
            GPUSpline object representing the derivative
        """
        deriv_spl = GPUSpline.__new__(GPUSpline)
        deriv_spl.k = self.k
        deriv_spl.use_gpu = self.use_gpu
        deriv_spl.is_gpu = self.is_gpu
        deriv_spl.spl = self.spl.derivative(n)
        return deriv_spl


def spline_radial_gpu(x, y, k=7, use_gpu=True):
    """
    Create GPU-accelerated radial spline in log space.
    
    Args:
        x: Radial coordinates
        y: Values at coordinates
        k: Spline order (default: 7)
        use_gpu: Whether to use GPU if available
        
    Returns:
        GPUSpline object
    """
    if use_gpu and CUPY_AVAILABLE:
        # Convert to CuPy if needed
        if isinstance(x, torch.Tensor):
            x_gpu = cp.asarray(x.detach().cpu().numpy() if not x.is_cuda else x.detach())
            y_gpu = cp.asarray(y.detach().cpu().numpy() if not y.is_cuda else y.detach())
        elif isinstance(x, np.ndarray):
            x_gpu = cp.asarray(x)
            y_gpu = cp.asarray(y)
        else:
            x_gpu = x
            y_gpu = y
        
        log_x = cp.log(x_gpu)
        return GPUSpline(log_x, y_gpu, k=k, use_gpu=True)
    else:
        # CPU fallback
        if isinstance(x, torch.Tensor):
            x_cpu = x.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
        elif hasattr(x, 'get'):
            x_cpu = cp.asnumpy(x)
            y_cpu = cp.asnumpy(y)
        else:
            x_cpu = x
            y_cpu = y
        
        log_x = np.log(x_cpu)
        return GPUSpline(log_x, y_cpu, k=k, use_gpu=False)


def eval_spline_density_gpu(spl, coords, density_grad=False, use_gpu=True):
    """
    GPU-accelerated version of eval_spline_density.
    
    Evaluates atomic density using spline interpolation, keeping computation
    on GPU when possible.
    
    Args:
        spl: Spline object (GPUSpline or scipy spline)
        coords: Coordinates to evaluate density at (torch.Tensor)
        density_grad: Whether to compute density gradient
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Density values (and gradients if requested) as torch.Tensor
    """
    is_cuda = isinstance(coords, torch.Tensor) and coords.is_cuda
    
    if use_gpu and CUPY_AVAILABLE and is_cuda:
        # GPU computation
        # Compute radial distances
        x_in = torch.norm(coords, dim=-1) / param.BOHR
        
        # Convert to CuPy for spline evaluation
        x_in_cp = cp.asarray(x_in.detach())
        log_x_in_cp = cp.log(x_in_cp)
        
        # Evaluate spline
        if isinstance(spl, GPUSpline):
            y_out_cp = spl(log_x_in_cp)
        else:
            # Fallback to CPU evaluation
            log_x_in_cpu = cp.asnumpy(log_x_in_cp)
            y_out_cpu = spl(log_x_in_cpu)
            y_out_cp = cp.asarray(y_out_cpu)
        
        # Clamp negative values
        y_out_cp = cp.maximum(y_out_cp, 0)
        
        # Convert back to torch
        y_out = torch.as_tensor(y_out_cp, device='cuda')
        
        if density_grad:
            # Compute gradient
            deriv = spl.derivative()
            if isinstance(deriv, GPUSpline):
                spl_deriv_cp = deriv(log_x_in_cp)
            else:
                spl_deriv_cpu = deriv(cp.asnumpy(log_x_in_cp))
                spl_deriv_cp = cp.asarray(spl_deriv_cpu)
            
            spl_deriv = torch.as_tensor(spl_deriv_cp, device='cuda')
            
            # Gradient in Cartesian coordinates
            y_deriv = spl_deriv.unsqueeze(-1) * (1 / x_in).unsqueeze(-1) * \
                      (coords / x_in.unsqueeze(-1)) / param.BOHR
            y_out = torch.cat([y_out.unsqueeze(-1), y_deriv], dim=-1)
        
        return y_out
    else:
        # CPU fallback (original implementation)
        x_in = torch.norm(coords, dim=-1) / param.BOHR
        
        # Move to CPU for evaluation
        if isinstance(x_in, torch.Tensor):
            x_in_cpu = x_in.detach().cpu().numpy()
        else:
            x_in_cpu = x_in
        
        y_out = spl(np.log(x_in_cpu))
        y_out[y_out < 0] = 0
        y_out = torch.from_numpy(y_out)
        
        if is_cuda:
            y_out = y_out.cuda()
        
        if density_grad:
            deriv = spl.derivative()
            spl_deriv_cpu = deriv(np.log(x_in_cpu))
            spl_deriv = torch.from_numpy(spl_deriv_cpu)
            if is_cuda:
                spl_deriv = spl_deriv.cuda()
                x_in = x_in.cuda() if not isinstance(x_in, torch.Tensor) or not x_in.is_cuda else x_in
            
            y_deriv = spl_deriv.unsqueeze(-1) * (1 / x_in).unsqueeze(-1) * \
                      (coords / x_in.unsqueeze(-1)) / param.BOHR
            y_out = torch.cat([y_out.unsqueeze(-1), y_deriv], dim=-1)
        
        return y_out


def is_gpu_available():
    """Check if GPU acceleration is available."""
    return CUPY_AVAILABLE and cp.cuda.is_available()


def get_gpu_info():
    """Get GPU information for debugging."""
    if not CUPY_AVAILABLE:
        return "CuPy not available"
    
    try:
        device = cp.cuda.Device()
        return {
            'cupy_version': cp.__version__,
            'cuda_available': cp.cuda.is_available(),
            'device_id': device.id,
            'device_name': device.compute_capability,
            'memory_total': device.mem_info[1] / (1024**3),  # GB
            'memory_free': device.mem_info[0] / (1024**3),   # GB
        }
    except Exception as e:
        return f"Error getting GPU info: {e}"


class AtomicDensityCache:
    """
    Cache for GPU-accelerated atomic density splines.
    
    This class manages atomic density representations and creates GPU-compatible
    spline objects on-the-fly for fast GPU computation.
    
    Supports two modes:
    1. Legacy mode: Works with pre-computed scipy splines (from .npy file)
    2. GPU mode: Creates GPU splines from raw data (from .npz file)
    """
    
    def __init__(self, atom_dens_data, use_gpu=True, verbose=False):
        """
        Initialize atomic density cache.
        
        Args:
            atom_dens_data: Either dict (legacy scipy splines) or loaded .npz file (raw data)
            use_gpu: Whether to create GPU splines
            verbose: Print debug information
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.verbose = verbose
        self.gpu_splines = {}  # {atom_num: GPUSpline}
        self.legacy_splines = {}  # {atom_num: scipy spline}
        self.atom_data = {}  # {atom_num: full data dict}
        
        if verbose:
            print(f"Initializing AtomicDensityCache (GPU: {self.use_gpu})")
        
        # Determine input format
        if isinstance(atom_dens_data, dict):
            # Legacy format: pre-computed scipy splines
            self._load_legacy_format(atom_dens_data)
        else:
            # New format: raw data from .npz
            self._load_raw_format(atom_dens_data)
    
    def _load_legacy_format(self, atom_dens_dict):
        """Load from legacy scipy spline format."""
        if self.verbose:
            print(f"  Loading legacy format ({len(atom_dens_dict)} atoms)")
        
        for atom_num, atom_data in atom_dens_dict.items():
            self.atom_data[atom_num] = atom_data
            
            if 'spline_interp' in atom_data:
                scipy_spl = atom_data['spline_interp']
                self.legacy_splines[atom_num] = scipy_spl
                
                # Optionally convert to GPU spline
                if self.use_gpu:
                    try:
                        # Extract data from scipy spline and create GPU version
                        # We'll use the spline's knots and coefficients
                        t = scipy_spl.t
                        c = scipy_spl.c
                        k = scipy_spl.k
                        
                        # Create a GPU spline using the same knots/coefficients
                        # Note: This requires CuPy's make_interp_spline to support
                        # the same interface, which it does
                        if CUPY_AVAILABLE:
                            t_gpu = cp.asarray(t)
                            c_gpu = cp.asarray(c)
                            from cupyx.scipy.interpolate import BSpline
                            gpu_spl_obj = BSpline(t_gpu, c_gpu, k)
                            
                            # Wrap in our GPUSpline class
                            gpu_spl = GPUSpline.__new__(GPUSpline)
                            gpu_spl.k = k
                            gpu_spl.use_gpu = True
                            gpu_spl.is_gpu = True
                            gpu_spl.spl = gpu_spl_obj
                            
                            self.gpu_splines[atom_num] = gpu_spl
                            
                            if self.verbose:
                                print(f"    ✓ Atom {atom_num}: converted to GPU spline")
                    except Exception as e:
                        if self.verbose:
                            print(f"    ⚠ Atom {atom_num}: GPU conversion failed ({e}), using CPU")
                        self.gpu_splines[atom_num] = None
    
    def _load_raw_format(self, npz_data):
        """Load from raw data format (.npz file)."""
        if self.verbose:
            print(f"  Loading raw format")
        
        # Extract atom data
        for key in npz_data.files:
            if key.startswith('atom_'):
                atom_num = int(key.split('_')[1])
                atom_dict = npz_data[key].item()
                
                self.atom_data[atom_num] = atom_dict
                
                if atom_dict.get('type') == 'spline':
                    # Create GPU or CPU spline from raw data
                    x_samples = atom_dict['x_samples']
                    y_samples = atom_dict['y_samples']
                    k = atom_dict['spline_order']
                    
                    # Create spline using log(x)
                    log_x = np.log(x_samples)
                    
                    if self.use_gpu and CUPY_AVAILABLE:
                        try:
                            gpu_spl = GPUSpline(log_x, y_samples, k=k, use_gpu=True)
                            self.gpu_splines[atom_num] = gpu_spl
                            if self.verbose:
                                print(f"    ✓ Atom {atom_num}: created GPU spline")
                        except Exception as e:
                            if self.verbose:
                                print(f"    ⚠ Atom {atom_num}: GPU spline failed ({e}), using CPU")
                            # Fallback to CPU
                            from scipy.interpolate import make_interp_spline
                            cpu_spl = make_interp_spline(log_x, y_samples, k=k)
                            self.legacy_splines[atom_num] = cpu_spl
                    else:
                        # CPU mode
                        from scipy.interpolate import make_interp_spline
                        cpu_spl = make_interp_spline(log_x, y_samples, k=k)
                        self.legacy_splines[atom_num] = cpu_spl
                        if self.verbose:
                            print(f"    ✓ Atom {atom_num}: created CPU spline")
    
    def get_spline(self, atom_num):
        """
        Get spline for a given atom number.
        
        Args:
            atom_num: Atomic number
            
        Returns:
            GPUSpline or scipy spline object
        """
        # Try GPU first
        if atom_num in self.gpu_splines and self.gpu_splines[atom_num] is not None:
            return self.gpu_splines[atom_num]
        
        # Fallback to legacy
        if atom_num in self.legacy_splines:
            return self.legacy_splines[atom_num]
        
        # Try to get from atom_data if we have spline_interp
        if atom_num in self.atom_data and 'spline_interp' in self.atom_data[atom_num]:
            return self.atom_data[atom_num]['spline_interp']
        
        raise KeyError(f"No spline found for atom {atom_num}")
    
    def __getitem__(self, atom_num):
        """Dictionary-like access to atom data."""
        return self.atom_data[atom_num]
    
    def __contains__(self, atom_num):
        """Check if atom is in cache."""
        return atom_num in self.atom_data
    
    def keys(self):
        """Get all atom numbers."""
        return self.atom_data.keys()
    
    def items(self):
        """Iterate over atom data."""
        return self.atom_data.items()


