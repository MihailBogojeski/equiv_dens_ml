"""
GPU inference tests: correctness, numerical equivalence, and GPU profiling.

Tests require:
- CUDA GPU
- Real ethanol checkpoint at paper/models/ethanol/2024-03-22_96w7KyGG

All tests are skipped if either requirement is missing.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

import equiv_dens.compat  # noqa: F401
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.inference import FastInferenceWrapper, InferenceSettings
from equiv_dens.training.model_loader import load_model
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.utils import orbitals
from equiv_dens.utils.base import get_atom_num_first_positions
from equiv_dens.utils.grids import spherical_grid, spherical_radial_sampling

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = _repo_root / "paper" / "models" / "ethanol" / "2024-03-22_96w7KyGG"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoints" / "latest_checkpoint.pth"
CONFIG_FILE = _repo_root / "config" / "md" / "nn" / "ethanol.txt"

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required"
)
_needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(), reason=f"Checkpoint not found: {CHECKPOINT_PATH}"
)


# ---------------------------------------------------------------------------
# Helper: ensure npy has energy/forces
# ---------------------------------------------------------------------------
def _ensure_npy_has_forces(npy_path: Path) -> None:
    """Add placeholder energy/forces to npy if missing."""
    data = np.load(str(npy_path), allow_pickle=True).item()
    n = data["positions"].shape[0]
    changed = False
    if "forces" not in data:
        data["forces"] = np.zeros_like(data["positions"])
        changed = True
    if "energy" not in data:
        data["energy"] = np.zeros((n, 1), dtype=np.float64)
        changed = True
    elif data["energy"].ndim == 1:
        data["energy"] = data["energy"].reshape(-1, 1)
        changed = True
    if changed:
        np.save(str(npy_path), data, allow_pickle=True)


# ---------------------------------------------------------------------------
# GPU timing utility
# ---------------------------------------------------------------------------
def gpu_timer(fn, n_warmup: int = 5, n_runs: int = 20) -> tuple[float, float]:
    """
    Time a GPU function using torch.cuda.Event for accurate GPU-side timing.

    Args:
        fn: Callable to time.
        n_warmup: Number of warmup iterations.
        n_runs: Number of timed iterations.

    Returns:
        Tuple of (mean_ms, std_ms).
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(n_runs):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    return float(np.mean(times_ms)), float(np.std(times_ms))


# ---------------------------------------------------------------------------
# Fixture: load model + single-molecule input
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_and_inputs():
    """
    Load ethanol model and prepare single-molecule GPU input.

    Skips the entire module if the checkpoint is missing.
    """
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")

    # Parse config
    saved_argv = sys.argv
    sys.argv = ["test", f"@{CONFIG_FILE}"]
    try:
        args, hyperparam_args = parse_command_line_arguments()
    finally:
        sys.argv = saved_argv

    args.use_gpu = True

    # Restore hyperparams from checkpoint (skip path args)
    path_args = {
        "np_dataset", "np_dataset_test", "np_dataset_valid",
        "dens_dataset", "dens_dataset_test", "dens_dataset_valid",
        "orbitals_file", "radial_coeffs_file", "atom_dens_path",
        "log_dir", "restart",
    }
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    for arg in vars(checkpoint["args"]):
        if arg in path_args:
            continue
        setattr(args, arg, getattr(checkpoint["args"], arg))

    # Resolve dataset paths relative to repo root
    np_path = Path(args.np_dataset)
    if not np_path.is_absolute():
        np_path = _repo_root / np_path
    _ensure_npy_has_forces(np_path)

    if not Path(args.orbitals_file).is_absolute():
        args.orbitals_file = str(_repo_root / args.orbitals_file)

    grid_fn = partial(spherical_grid, level=getattr(args, "spherical_grid_level", 3))
    sampling_fn = partial(spherical_radial_sampling, rotate=False)

    dataset = AtomsDensityData(
        np_path=str(np_path),
        density_path=None,
        orbitals_path=args.orbitals_file,
        density_n_samp=10000000000,
        required_properties=["energy", "forces"],
        center_positions=False,
        radial_coeffs_file=getattr(args, "radial_coeffs_file", None),
        dtype=args.dtype,
        grid_fn=grid_fn,
        pyscf_grid=getattr(args, "pyscf_grid", False),
        sampling_fn=sampling_fn,
        cutoff=args.cutoff,
        atom_dens_path=getattr(args, "atom_dens_path", None),
        atom_dens_type=getattr(args, "atom_dens_type", "spline"),
    )
    model = load_model(args, dataset, train=False)
    model.eval()
    model = model.cuda()

    # Build single-molecule input
    sample = dataset.get_properties([0])
    atoms = {
        "positions": sample["batch_positions"],
        "atom_numbers": sample["batch_atom_numbers"],
    }
    if atoms["atom_numbers"].ndim == 1:
        atoms["atom_numbers"] = atoms["atom_numbers"][None, :]
    if atoms["positions"].ndim == 2:
        atoms["positions"] = atoms["positions"][None, :, :]

    has_density = "density" in getattr(model, "property_models", {})
    inputs = orbitals.model_input_from_atoms(
        atoms,
        density_expansion=has_density,
        pyscf_grid=getattr(dataset, "pyscf_grid", False) and has_density,
        grid_spec=dataset.grid_spec if has_density else None,
        grid_sampling_fn=dataset.sampling_fn if has_density else None,
        center_coords=False,
        cutoff=args.cutoff,
        dtype=torch.float32,
        free_atom_densities=dataset.atom_dens if has_density else None,
    )
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.cuda()

    compute_forces = getattr(model, "calculate_forces", False)

    return model, inputs, compute_forces


def _clone_inputs(inputs: dict) -> dict:
    """Clone tensor values for a fresh inference run."""
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = dict(v)
        else:
            out[k] = v
    return out


def _run_baseline(model, inputs, compute_forces):
    """Run a single baseline forward pass."""
    data = _clone_inputs(inputs)
    data = model.transform_input(data)
    with torch.enable_grad():
        return model(data)


# ===================================================================
# Part 1: Baseline Correctness
# ===================================================================
@_needs_cuda
@_needs_checkpoint
class TestBaselineInference:
    """Verify the baseline model produces valid outputs."""

    def test_baseline_runs_without_error(self, model_and_inputs):
        """Model forward pass completes without raising."""
        model, inputs, compute_forces = model_and_inputs
        result = _run_baseline(model, inputs, compute_forces)
        assert isinstance(result, dict)

    def test_baseline_produces_energy(self, model_and_inputs):
        """Model output contains 'energy' key."""
        model, inputs, compute_forces = model_and_inputs
        result = _run_baseline(model, inputs, compute_forces)
        assert "energy" in result, f"Expected 'energy' in result, got keys: {list(result.keys())}"

    def test_baseline_energy_is_finite(self, model_and_inputs):
        """Model energy is finite (no NaN or Inf)."""
        model, inputs, compute_forces = model_and_inputs
        result = _run_baseline(model, inputs, compute_forces)
        energy = result["energy"]
        assert torch.isfinite(energy).all(), f"Energy contains non-finite values: {energy}"

    def test_baseline_forces_shape(self, model_and_inputs):
        """If model computes forces, they have the correct shape."""
        model, inputs, compute_forces = model_and_inputs
        if not compute_forces:
            pytest.skip("Model does not compute forces")
        result = _run_baseline(model, inputs, compute_forces)
        assert "forces" in result, "Expected 'forces' in result"
        forces = result["forces"]
        positions = inputs["positions"]
        assert forces.shape == positions.shape, (
            f"Forces shape {forces.shape} != positions shape {positions.shape}"
        )

    def test_baseline_forces_are_finite(self, model_and_inputs):
        """Forces contain no NaN or Inf."""
        model, inputs, compute_forces = model_and_inputs
        if not compute_forces:
            pytest.skip("Model does not compute forces")
        result = _run_baseline(model, inputs, compute_forces)
        forces = result["forces"]
        assert torch.isfinite(forces).all(), f"Forces contain non-finite values"


# ===================================================================
# Part 2: TF32 Numerical Equivalence
# ===================================================================
@_needs_cuda
@_needs_checkpoint
class TestTF32Inference:
    """TF32 should produce energy close to baseline (within ~1e-3)."""

    def test_tf32_produces_close_energy(self, model_and_inputs):
        """TF32 energy matches baseline within tolerance."""
        model, inputs, compute_forces = model_and_inputs

        # Baseline
        baseline = _run_baseline(model, inputs, compute_forces)
        baseline_energy = baseline["energy"].detach().clone()

        # TF32
        wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=True, compile=False, cuda_graph=False)
        ).cuda()
        data = _clone_inputs(inputs)
        data = model.transform_input(data)
        tf32_result = wrapper(data, compute_forces=compute_forces)
        tf32_energy = tf32_result["energy"].detach()

        # TF32 uses lower-precision matmul; allow 1e-3 relative tolerance
        torch.testing.assert_close(
            tf32_energy, baseline_energy, rtol=1e-3, atol=1e-3,
        )


# ===================================================================
# Part 3: torch.compile Numerical Equivalence
# ===================================================================
@_needs_cuda
@_needs_checkpoint
class TestCompiledInference:
    """torch.compile should be numerically identical (same precision)."""

    @pytest.mark.timeout(0)  # disable timeout: torch.compile takes 5+ min for first compilation
    def test_compiled_produces_same_energy(self, model_and_inputs):
        """Compiled model energy matches baseline."""
        model, inputs, compute_forces = model_and_inputs

        # Baseline
        baseline = _run_baseline(model, inputs, compute_forces)
        baseline_energy = baseline["energy"].detach().clone()

        # Compiled (no TF32 to ensure same precision)
        wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=False, compile=True, cuda_graph=False)
        ).cuda()

        # Warmup compile
        for _ in range(3):
            data = _clone_inputs(inputs)
            data = model.transform_input(data)
            wrapper(data, compute_forces=compute_forces)

        data = _clone_inputs(inputs)
        data = model.transform_input(data)
        compiled_result = wrapper(data, compute_forces=compute_forces)
        compiled_energy = compiled_result["energy"].detach()

        # Should be nearly identical (same precision path)
        torch.testing.assert_close(
            compiled_energy, baseline_energy, rtol=1e-4, atol=1e-4,
        )


# ===================================================================
# Part 4: CUDA Graph Inference
# ===================================================================
@_needs_cuda
@_needs_checkpoint
class TestCUDAGraphInference:
    """
    CUDA graph capture tests.

    CUDA graph capture may fail if the model's forward path contains operations
    not permitted during stream capture (e.g., CPU-GPU transfers via .to()).
    The FastInferenceWrapper handles this gracefully by falling back to eager mode.
    """

    def test_cuda_graph_wrapper_does_not_crash(self, model_and_inputs):
        """CUDA graph wrapper runs without crashing (may fall back to eager)."""
        model, inputs, _ = model_and_inputs

        wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=False, compile=False, cuda_graph=True)
        ).cuda()
        data = _clone_inputs(inputs)
        data = model.transform_input(data)
        result = wrapper(data, compute_forces=False)

        assert "energy" in result
        energy = result["energy"]
        assert torch.isfinite(energy).all(), f"Energy is not finite: {energy}"

    def test_cuda_graph_or_fallback_matches_baseline(self, model_and_inputs):
        """Whether captured or fallen back, result should match baseline."""
        model, inputs, _ = model_and_inputs

        # Baseline (energy-only, via wrapper with no optimizations)
        baseline_wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=False, compile=False, cuda_graph=False)
        ).cuda()
        data = _clone_inputs(inputs)
        data = model.transform_input(data)
        baseline = baseline_wrapper(data, compute_forces=False)
        baseline_energy = baseline["energy"].detach().clone()

        # CUDA graph attempt
        wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=False, compile=False, cuda_graph=True)
        ).cuda()
        data = _clone_inputs(inputs)
        data = model.transform_input(data)
        result = wrapper(data, compute_forces=False)
        cg_energy = result["energy"].detach()

        torch.testing.assert_close(
            cg_energy, baseline_energy, rtol=1e-5, atol=1e-5,
        )

    def test_cuda_graph_replay_consistency(self, model_and_inputs):
        """Multiple calls produce consistent results (whether graph or eager)."""
        model, inputs, _ = model_and_inputs

        wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=False, compile=False, cuda_graph=True)
        ).cuda()

        energies = []
        for _ in range(5):
            data = _clone_inputs(inputs)
            data = model.transform_input(data)
            result = wrapper(data, compute_forces=False)
            energies.append(result["energy"].detach().clone())

        # All calls should produce identical results
        for i in range(1, len(energies)):
            assert torch.equal(energies[0], energies[i]), (
                f"Call {i} differs from call 0: {energies[i]} vs {energies[0]}"
            )

    def test_cuda_graph_skipped_when_forces(self, model_and_inputs):
        """CUDA graph falls back to eager when compute_forces=True."""
        model, inputs, compute_forces = model_and_inputs
        if not compute_forces:
            pytest.skip("Model does not compute forces; cannot test fallback")

        wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=False, compile=False, cuda_graph=True)
        ).cuda()
        data = _clone_inputs(inputs)
        data = model.transform_input(data)

        # Should not raise, should fall back to eager
        result = wrapper(data, compute_forces=True)
        assert "energy" in result
        assert "forces" in result

    def test_cuda_graph_failed_flag(self, model_and_inputs):
        """If capture fails, _cuda_graph_failed is set and no retries occur."""
        model, inputs, _ = model_and_inputs

        wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=False, compile=False, cuda_graph=True)
        ).cuda()

        # Run once to trigger capture attempt
        data = _clone_inputs(inputs)
        data = model.transform_input(data)
        result = wrapper(data, compute_forces=False)
        assert "energy" in result

        if wrapper._cuda_graph_failed:
            # Capture failed; verify it doesn't retry
            assert wrapper._cuda_graph is None
            # Run again -- should use eager without retrying capture
            data = _clone_inputs(inputs)
            data = model.transform_input(data)
            result2 = wrapper(data, compute_forces=False)
            assert "energy" in result2
        else:
            # Capture succeeded; verify graph is populated
            assert wrapper._cuda_graph is not None


# ===================================================================
# Part 5: GPU Profiling
# ===================================================================
@_needs_cuda
@_needs_checkpoint
class TestGPUProfiling:
    """GPU profiling tests using torch.cuda.Event timing."""

    def test_cuda_events_timing_baseline(self, model_and_inputs):
        """Baseline inference can be timed with CUDA events."""
        model, inputs, compute_forces = model_and_inputs

        def runner():
            data = _clone_inputs(inputs)
            data = model.transform_input(data)
            with torch.enable_grad():
                return model(data)

        mean_ms, std_ms = gpu_timer(runner, n_warmup=3, n_runs=10)
        assert mean_ms > 0, "Mean timing must be positive"
        assert std_ms >= 0, "Std timing must be non-negative"
        print(f"\n  Baseline: {mean_ms:.2f} +/- {std_ms:.2f} ms")

    def test_cuda_events_timing_tf32(self, model_and_inputs):
        """TF32 inference timing with CUDA events."""
        model, inputs, compute_forces = model_and_inputs

        wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=True, compile=False, cuda_graph=False)
        ).cuda()

        def runner():
            data = _clone_inputs(inputs)
            data = model.transform_input(data)
            return wrapper(data, compute_forces=compute_forces)

        mean_ms, std_ms = gpu_timer(runner, n_warmup=3, n_runs=10)
        assert mean_ms > 0
        print(f"\n  TF32: {mean_ms:.2f} +/- {std_ms:.2f} ms")

    def test_cuda_events_timing_cuda_graph(self, model_and_inputs):
        """CUDA graph (or fallback) inference timing with CUDA events."""
        model, inputs, _ = model_and_inputs

        wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=False, compile=False, cuda_graph=True)
        ).cuda()

        def runner():
            data = _clone_inputs(inputs)
            data = model.transform_input(data)
            return wrapper(data, compute_forces=False)

        mean_ms, std_ms = gpu_timer(runner, n_warmup=3, n_runs=10)
        assert mean_ms > 0
        mode = "eager fallback" if wrapper._cuda_graph_failed else "CUDA Graph"
        print(f"\n  {mode}: {mean_ms:.2f} +/- {std_ms:.2f} ms")

    def test_memory_usage(self, model_and_inputs):
        """Check peak GPU memory allocation during inference."""
        model, inputs, compute_forces = model_and_inputs

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

        data = _clone_inputs(inputs)
        data = model.transform_input(data)
        with torch.enable_grad():
            _ = model(data)

        peak_mem = torch.cuda.max_memory_allocated()
        mem_used = peak_mem - mem_before
        print(f"\n  Peak memory: {peak_mem / 1024**2:.1f} MB")
        print(f"  Memory used by inference: {mem_used / 1024**2:.1f} MB")

        # Sanity: peak memory should be positive and reasonable
        assert peak_mem > 0, "Peak memory should be positive"
        assert peak_mem < 8 * 1024**3, "Peak memory > 8 GB seems unreasonable for one molecule"

    def test_speedup_tf32(self, model_and_inputs):
        """TF32 should not be slower than baseline (>= 0.9x)."""
        model, inputs, compute_forces = model_and_inputs

        def baseline_runner():
            data = _clone_inputs(inputs)
            data = model.transform_input(data)
            with torch.enable_grad():
                return model(data)

        wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=True, compile=False, cuda_graph=False)
        ).cuda()

        def tf32_runner():
            data = _clone_inputs(inputs)
            data = model.transform_input(data)
            return wrapper(data, compute_forces=compute_forces)

        baseline_mean, _ = gpu_timer(baseline_runner, n_warmup=5, n_runs=20)
        tf32_mean, _ = gpu_timer(tf32_runner, n_warmup=5, n_runs=20)

        speedup = baseline_mean / tf32_mean if tf32_mean > 0 else 0
        print(f"\n  TF32 speedup: {speedup:.2f}x ({baseline_mean:.2f} -> {tf32_mean:.2f} ms)")
        assert speedup >= 0.9, f"TF32 too slow: {speedup:.2f}x (expected >= 0.9x)"

    def test_speedup_cuda_graph(self, model_and_inputs):
        """CUDA graph (or fallback) should not be slower than baseline (>= 0.9x)."""
        model, inputs, _ = model_and_inputs

        baseline_wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=False, compile=False, cuda_graph=False)
        ).cuda()

        def baseline_runner():
            data = _clone_inputs(inputs)
            data = model.transform_input(data)
            return baseline_wrapper(data, compute_forces=False)

        cg_wrapper = FastInferenceWrapper(
            model, InferenceSettings(tf32=False, compile=False, cuda_graph=True)
        ).cuda()

        def cg_runner():
            data = _clone_inputs(inputs)
            data = model.transform_input(data)
            return cg_wrapper(data, compute_forces=False)

        baseline_mean, _ = gpu_timer(baseline_runner, n_warmup=5, n_runs=20)
        cg_mean, _ = gpu_timer(cg_runner, n_warmup=5, n_runs=20)

        speedup = baseline_mean / cg_mean if cg_mean > 0 else 0
        mode = "eager fallback" if cg_wrapper._cuda_graph_failed else "CUDA Graph"
        print(f"\n  {mode} speedup: {speedup:.2f}x ({baseline_mean:.2f} -> {cg_mean:.2f} ms)")
        # Fallback to eager should be at worst ~1.0x; real CUDA graph should be faster
        assert speedup >= 0.9, f"{mode} too slow: {speedup:.2f}x (expected >= 0.9x)"


# ===================================================================
# Part 6: atom_numbers_first_positions guard fix
# ===================================================================
class TestAtomNumbersFirstPositions:
    """Verify get_atom_num_first_positions returns Python int keys."""

    def test_numpy_input_returns_python_ints(self):
        """Numpy array input produces dict with Python int keys."""
        atom_numbers = np.array([6, 1, 1, 8, 1])
        result = get_atom_num_first_positions(atom_numbers)
        for key in result:
            assert type(key) is int, f"Key {key} is {type(key)}, expected int"
        assert result == {6: 0, 1: 1, 8: 3}

    def test_torch_input_returns_python_ints(self):
        """Torch tensor input produces dict with Python int keys."""
        atom_numbers = torch.tensor([6, 1, 1, 8, 1], dtype=torch.long)
        result = get_atom_num_first_positions(atom_numbers)
        for key in result:
            assert type(key) is int, f"Key {key} is {type(key)}, expected int"
        assert result == {6: 0, 1: 1, 8: 3}

    def test_2d_numpy_input(self):
        """2D numpy array (batch) is reduced and produces Python int keys."""
        atom_numbers = np.array([[6, 1, 1, 8], [6, 1, 1, 8]])
        result = get_atom_num_first_positions(atom_numbers)
        for key in result:
            assert type(key) is int, f"Key {key} is {type(key)}, expected int"

    def test_2d_torch_input(self):
        """2D torch tensor (batch) is reduced and produces Python int keys."""
        atom_numbers = torch.tensor([[6, 1, 1, 8], [6, 1, 1, 8]], dtype=torch.long)
        result = get_atom_num_first_positions(atom_numbers)
        for key in result:
            assert type(key) is int, f"Key {key} is {type(key)}, expected int"

    def test_preserves_first_index(self):
        """First occurrence index is correctly tracked."""
        atom_numbers = np.array([8, 6, 1, 6, 8, 1])
        result = get_atom_num_first_positions(atom_numbers)
        assert result == {8: 0, 6: 1, 1: 2}
