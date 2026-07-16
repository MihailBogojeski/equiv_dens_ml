"""
Inference utilities for fast single-point evaluation.

Provides TF32 context manager and inference settings following fairchem patterns.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@contextmanager
def tf32_context_manager():
    """
    Context manager that enables TF32 for matmul and cuDNN, then restores original settings.

    TF32 (TensorFloat32) speeds up matrix multiplication on Ampere+ GPUs with
    minimal precision impact for most inference workloads. Use within inference
    forward passes for faster computation.

    Yields:
        None
    """
    original_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    original_allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
    original_float32_matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_allow_tf32_matmul
        torch.backends.cudnn.allow_tf32 = original_allow_tf32_cudnn
        torch.set_float32_matmul_precision(original_float32_matmul_precision)


@dataclass
class InferenceSettings:
    """
    Configuration for fast inference optimizations.

    Attributes:
        tf32: Enable TensorFloat32 for faster matmul on Ampere+ GPUs.
        compile: Use torch.compile for JIT optimization.
        cuda_graph: Use CUDA graphs for equivariant representation (experimental).
    """

    tf32: bool = False
    compile: bool = False
    cuda_graph: bool = False
