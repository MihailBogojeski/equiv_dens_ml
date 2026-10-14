"""Primitive contracts protecting first and force-loss derivative modes."""
import pytest
import torch

from benchmarks.kernels import derivative_snapshot, make_case
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from benchmarks.experimental_kernels import PairMixing


def compare_tree(actual, reference, atol, rtol):
    if isinstance(reference, (tuple, list)):
        assert len(actual) == len(reference)
        for left, right in zip(actual, reference):
            compare_tree(left, right, atol, rtol)
    elif reference is None:
        assert actual is None
    else:
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_cuda_real_kernel_and_higher_derivatives(dtype):
    reference, actual, inputs = make_case('pair', 2, width=8, rows=4,
                                        dtype=dtype, device='cuda')
    atol, rtol = (1e-8, 1e-6) if dtype == torch.float64 else (1e-4, 1e-3)
    result, expected = derivative_snapshot(actual, inputs), derivative_snapshot(reference, inputs)
    value_atol, value_rtol = (1e-9, 1e-7) if dtype == torch.float64 else (1e-5, 1e-4)
    compare_tree(result[0], expected[0], value_atol, value_rtol)
    compare_tree(result[1:], expected[1:], atol, rtol)
