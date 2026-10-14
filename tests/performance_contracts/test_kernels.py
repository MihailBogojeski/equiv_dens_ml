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


@pytest.mark.parametrize('kind', ['self', 'pair'])
@pytest.mark.parametrize('order', [1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_same_weights_values_first_hvp_and_mixed(kind, order, dtype):
    reference, actual, inputs = make_case(kind, order, width=4, rows=3, dtype=dtype)
    expected = derivative_snapshot(reference, inputs)
    result = derivative_snapshot(actual, inputs)
    atol, rtol = (1e-8, 1e-6) if dtype == torch.float64 else (1e-4, 1e-3)
    value_atol, value_rtol = (1e-9, 1e-7) if dtype == torch.float64 else (1e-5, 1e-4)
    compare_tree(result[0], expected[0], value_atol, value_rtol)
    compare_tree(result[1:], expected[1:], atol, rtol)
    assert result[1][0].abs().max() > 1e-8
    assert any(gradient is not None and gradient.abs().max() > 1e-8 for gradient in result[3])
    assert set(reference.state_dict()) == set(actual.state_dict())


@pytest.mark.parametrize('kind', ['self', 'pair'])
def test_multiple_leading_dimensions_and_old_module_attribute(kind):
    reference, actual, inputs = make_case(kind, 2, width=4, leading_shape=(2, 3))
    compare_tree(actual(*inputs), reference(*inputs), 1e-9, 1e-7)
    del actual.contraction_backend  # Legacy pickled modules lack the selector.
    compare_tree(actual(*inputs), reference(*inputs), 1e-9, 1e-7)


def test_scalar_coupling_independent_value_and_mixed_derivative():
    layer = PairMixing(0, 0, 0, 1, 1, ClebschGordanMatrix(0),
                       contraction_backend='matmul').double()
    with torch.no_grad():
        layer.coeff(0, 0, 0).weight.fill_(0.7)
    first = torch.tensor([[[1.3]]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[[0.9]]], dtype=torch.float64, requires_grad=True)
    radial = torch.tensor([[[1.1]]], dtype=torch.float64, requires_grad=True)
    result = layer([first], [second], radial)[0]
    torch.testing.assert_close(result, 0.7*radial*first*second)
    dx = torch.autograd.grad(result.sum(), first, create_graph=True)[0]
    dw = torch.autograd.grad(dx.sum(), layer.coeff(0, 0, 0).weight)[0]
    torch.testing.assert_close(dw, (radial*second).sum().reshape(1, 1))


@pytest.mark.parametrize('kind', ['self', 'pair'])
def test_empty_leading_dimension(kind):
    reference, actual, inputs = make_case(kind, 1, width=4, rows=0)
    compare_tree(actual(*inputs), reference(*inputs), 1e-9, 1e-7)


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
