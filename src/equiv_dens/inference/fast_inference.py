"""
Fast inference wrapper for DFTNetwork models.

Applies TF32, torch.compile, and optionally CUDA graphs for faster single-point inference.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import nullcontext

import torch

from equiv_dens.inference.utils import InferenceSettings, tf32_context_manager

logger = logging.getLogger(__name__)


class FastInferenceWrapper(torch.nn.Module):
    """
    Wraps a DFTNetwork model with inference optimizations.

    Applies TF32 (TensorFloat32) and torch.compile following fairchem patterns.
    CUDA graph support captures the full DFTNetwork.forward for energy-only inference
    (forces are incompatible because they require autograd).

    If CUDA graph capture fails (e.g. due to host-device transfers in the model's
    forward path), the wrapper automatically falls back to eager mode with a warning.

    Usage:
        wrapper = FastInferenceWrapper(model, InferenceSettings(tf32=True, compile=True))
        result = wrapper(data, compute_forces=False)
    """

    def __init__(self, model: torch.nn.Module, settings: InferenceSettings | None = None):
        """
        Args:
            model: DFTNetwork or compatible model.
            settings: Inference settings. If None, uses defaults (all optimizations off).
        """
        super().__init__()
        self.model = model
        self.settings = settings or InferenceSettings()
        self._compiled = False

        # CUDA graph state
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._static_input: dict | None = None
        self._static_output: dict | None = None
        self._cuda_graph_failed: bool = False  # True if capture failed; skip future attempts

    def _maybe_compile(self) -> None:
        """Lazy compilation on first inference."""
        if self.settings.compile and not self._compiled:
            # fullgraph=False allows graph breaks (dict iteration, atom_numbers_first_positions).
            # suppress_errors falls back to eager when guards fail, avoiding AssertionError.
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            self.model = torch.compile(
                self.model,
                dynamic=True,
                fullgraph=False,
            )
            self._compiled = True

    def _capture_cuda_graph(self, data: dict) -> bool:
        """
        Attempt to capture a CUDA graph for the model forward pass.

        Runs two warmup passes to fill CUDA caches, then captures the third run.
        All input/output tensors are stored as static buffers; subsequent calls
        copy new data into these buffers and replay the graph.

        During capture, ``model.calculate_forces`` is temporarily set to False
        so that no ``requires_grad_`` (in-place metadata) calls occur inside the
        captured region.  Forces are incompatible with CUDA graph replay anyway
        because autograd cannot be replayed.

        Args:
            data: Model input dictionary. All tensors must be on CUDA.

        Returns:
            True if capture succeeded, False if it failed and eager fallback
            should be used.
        """
        self.model.eval()

        # Temporarily disable forces on the model AND all property sub-models
        # so that (a) ``requires_grad_`` in-place metadata ops are skipped and
        # (b) ``torch.autograd.grad`` is not called inside the captured region.
        saved_flags = self._set_forces_flag(False)

        try:
            # Warmup pass 1: fill CUDA caches (allocator, cuDNN, etc.)
            with torch.no_grad():
                _ = self.model(data)
            torch.cuda.synchronize()

            # Warmup pass 2: allocate static input/output buffers
            self._static_input = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    self._static_input[k] = v.clone()
                elif isinstance(v, dict):
                    self._static_input[k] = dict(v)
                else:
                    self._static_input[k] = v

            with torch.no_grad():
                self._static_output = self.model(self._static_input)
            torch.cuda.synchronize()

            # Capture -- use no_grad to prevent any autograd bookkeeping
            self._cuda_graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(self._cuda_graph):
                self._static_output = self.model(self._static_input)
            return True

        except RuntimeError as exc:
            # CUDA graph capture can fail if the model's forward path contains
            # operations not permitted during stream capture (e.g. CPU-GPU
            # transfers via .to(), torch.tensor().to(device), synchronize, etc.).
            # Fall back to eager mode gracefully.
            logger.warning(
                "CUDA graph capture failed: %s. Falling back to eager mode. "
                "To fix, replace torch.zeros(...).to(tensor) with "
                "torch.zeros(..., device=tensor.device, dtype=tensor.dtype) "
                "throughout the model forward path.",
                exc,
            )
            warnings.warn(
                f"CUDA graph capture failed ({exc}). Using eager mode instead.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._cuda_graph = None
            self._static_input = None
            self._static_output = None
            self._cuda_graph_failed = True
            # Reset CUDA state after failed capture
            torch.cuda.synchronize()
            return False

        finally:
            # Restore original forces settings on all sub-models
            self._restore_forces_flag(saved_flags)

    def _replay_cuda_graph(self, data: dict) -> dict:
        """
        Copy new data into static input buffers and replay the captured graph.

        Args:
            data: Model input dictionary with the same shapes as the capture data.

        Returns:
            Dictionary of predicted properties. Output tensors are cloned to avoid
            aliasing the static output buffer.
        """
        # Copy tensor data into static buffers (same memory, new values)
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and k in self._static_input:
                if isinstance(self._static_input[k], torch.Tensor):
                    self._static_input[k].copy_(v)

        self._cuda_graph.replay()
        torch.cuda.synchronize()

        # Clone outputs so caller gets independent tensors
        return {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in self._static_output.items()
        }

    def _set_forces_flag(self, enable: bool) -> list[tuple[torch.nn.Module, bool]]:
        """
        Set ``calculate_forces`` on the model and all sub-models.

        Returns:
            List of (module, original_flag) tuples for restoring later.
        """
        saved: list[tuple[torch.nn.Module, bool]] = []
        for mod in self.model.modules():
            if hasattr(mod, "calculate_forces"):
                saved.append((mod, bool(mod.calculate_forces)))
                mod.calculate_forces = enable
        return saved

    @staticmethod
    def _restore_forces_flag(saved: list[tuple[torch.nn.Module, bool]]) -> None:
        """Restore ``calculate_forces`` flags from saved list."""
        for mod, flag in saved:
            mod.calculate_forces = flag

    def forward(self, data: dict, compute_forces: bool = False) -> dict:
        """
        Run model forward with inference optimizations.

        Args:
            data: Model input dictionary (positions, atom_numbers, idx_i, idx_j, etc.).
            compute_forces: If True, enable autograd for force computation. Slower but
                required for forces. CUDA graphs are automatically disabled when True.

        Returns:
            Dictionary of predicted properties (energy, forces, etc.).
        """
        self.model.eval()

        tf32_ctx = (
            tf32_context_manager() if self.settings.tf32 else nullcontext()
        )

        with tf32_ctx:
            # CUDA graph path: energy-only, no forces (autograd incompatible)
            if (
                self.settings.cuda_graph
                and not compute_forces
                and not self._cuda_graph_failed
            ):
                if self._cuda_graph is None:
                    success = self._capture_cuda_graph(data)
                    if not success:
                        # Capture failed; fall through to eager
                        pass
                    else:
                        return self._replay_cuda_graph(data)
                else:
                    return self._replay_cuda_graph(data)

            # torch.compile path (skip if using CUDA graphs successfully)
            if self._cuda_graph is None and not self.settings.cuda_graph:
                self._maybe_compile()

            # When compute_forces=False, disable forces on the model to prevent
            # requires_grad_ and autograd.grad calls inside no_grad context.
            saved_flags = None
            if not compute_forces:
                saved_flags = self._set_forces_flag(False)

            try:
                grad_ctx = nullcontext() if compute_forces else torch.no_grad()
                with grad_ctx:
                    return self.model(data)
            finally:
                if saved_flags is not None:
                    self._restore_forces_flag(saved_flags)
