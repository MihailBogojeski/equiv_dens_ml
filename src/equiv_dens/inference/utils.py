"""TF32 context manager and InferenceSettings dataclass."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@contextmanager
def tf32_context_manager():
    """Enable TF32 matmul/cuDNN for the block, then restore prior settings."""
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
    """Flags for FastInferenceWrapper (all default False)."""

    tf32: bool = False
    compile: bool = False
    cuda_graph: bool = False
