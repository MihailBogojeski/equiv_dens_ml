"""P1 contracts use full precision independent of unrelated test imports."""
import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def reference_source():
    from benchmarks.fixtures import ensure_reference_source
    ensure_reference_source()


@pytest.fixture(autouse=True)
def full_precision_contracts():
    from benchmarks.precision import use_full_precision
    original = (torch.get_float32_matmul_precision(),
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    use_full_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(original[0])
        torch.backends.cuda.matmul.allow_tf32 = original[1]
        torch.backends.cudnn.allow_tf32 = original[2]
