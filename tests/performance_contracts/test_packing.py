"""Packing values and derivatives against immutable D source and independent order."""
import pytest
import torch

from benchmarks.packing import (basis_fixture, coefficient_loss, inputs_from_positions,
                                load_reference, make_case)
from equiv_dens.utils import orbitals


def assert_close(actual, expected, derivative=False):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_close(actual[key], expected[key], derivative)
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_close(a, b, derivative)
    elif actual is None:
        assert expected is None
    else:
        fp32 = actual.dtype == torch.float32
        atol = (1e-4 if derivative else 1e-5) if fp32 else (1e-8 if derivative else 1e-9)
        rtol = (1e-3 if derivative else 1e-4) if fp32 else (1e-6 if derivative else 1e-7)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize('convert', [False, True])
@pytest.mark.parametrize('radial', [False, True])
def test_independent_vector_order_reference_and_roundtrip(convert, radial):
    ref_orbitals, _ = load_reference()
    basis, _ = basis_fixture()
    nums = torch.tensor([[1, 6], [0, 6]])
    data = {key: [] for key in ('spherical_coeffs', 'radial_width', 'radial_scale', 'coeff_weights')}
    expected = {key: [] for key in data}
    serial = 0
    for atom, z in enumerate((1, 6)):
        for key in data:
            data[key].append({})
        shells_by_degree = {L: [orb for orb in basis[z] if orb[2] == L]
                            for L in {orb[2] for orb in basis[z]}}
        for L, shells in shells_by_degree.items():
            key = (z, L)
            for kind in data:
                m = 1 if kind in ('radial_width', 'radial_scale') else 2 * L + 1
                tensor = torch.arange(serial + 1, serial + 1 + 2 * m * len(shells),
                                      dtype=torch.float64).reshape(2, 1, m, len(shells))
                serial += tensor.numel()
                data[kind][atom][key] = tensor.requires_grad_()
        counts = {}
        for _, _, L in basis[z]:
            shell = counts.get(L, 0)
            counts[L] = shell + 1
            for kind in data:
                value = data[kind][atom][(z, L)][..., shell]
                if kind == 'spherical_coeffs' and convert:
                    if L == 1:
                        value = value[..., [2, 0, 1]]
                    if L % 2:
                        value = -value
                expected[kind].append(value)
    relevant = ['spherical_coeffs', 'coeff_weights']
    if radial:
        relevant += ['radial_width', 'radial_scale']
    expected = {k: torch.cat(expected[k], dim=2).squeeze(1) for k in relevant}
    kwargs = dict(radial_coeffs=radial, coeff_weighting=True, convert_to_pyscf=convert)
    reference = ref_orbitals.coeffs_dict_to_vector(data, basis, nums, **kwargs)
    result = orbitals.coeffs_dict_to_vector(data, basis, nums, **kwargs)
    assert_close(result, expected)
    assert_close(result, reference)
    targets = tuple(t for kind in relevant for atom in data[kind] for t in atom.values())
    weights = {k: torch.linspace(.1, 1., v.numel()).reshape_as(v) for k, v in result.items()}
    gradients = [torch.autograd.grad(sum((v * weights[k]).square().sum() for k, v in arm.items()),
                                     targets, retain_graph=True) for arm in (reference, result)]
    assert_close(*gradients, derivative=True)
    inverse = orbitals.vector_to_coeffs_dict(result, basis, nums, radial_coeffs=radial,
                                             convert_to_equiv_dens=convert)
    for kind in inverse:
        assert_close(inverse[kind], data[kind])

