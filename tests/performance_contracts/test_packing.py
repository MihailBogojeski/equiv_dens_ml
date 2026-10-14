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


@pytest.mark.parametrize('learned,positive,constraint,remove', [
    (False, False, None, False), (True, False, None, False),
    (True, True, None, False), (True, True, 'coeffs_in_coeff_net', False),
    (True, False, 'coeffs_in_coeff_net', True)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_head_state_values_and_mixed_derivatives(learned, positive, constraint, remove, dtype):
    reference, candidate, inputs, _, _ = make_case(
        dtype=dtype, learned=learned, positive=positive,
        constraint=constraint, remove_atom_density=remove)
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    reference.load_state_dict(candidate.state_dict(), strict=True)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    outputs, derivatives = [], []
    for model in (reference, candidate):
        fresh = inputs_from_positions(inputs['batch_atom_numbers'], inputs['batch_positions'], model.num_features)
        result = model(fresh)
        outputs.append({k: result[k] for k in ('df_coeffs', 'spherical_coeffs',
                                              'radial_width', 'radial_scale')})
        assert result['df_coeffs'].abs().max() > 0
        loss = coefficient_loss(result)
        parameters = tuple(model.parameters())
        gradients = torch.autograd.grad(loss, (inputs['batch_positions'],) + parameters,
                                        create_graph=True, allow_unused=True)
        assert gradients[0].abs().max() > 0
        assert any(g is not None and g.abs().max() > 0 for g in gradients[1:])
        mixed = torch.autograd.grad(gradients[0].square().sum(), parameters,
                                    allow_unused=True)
        assert any(g is not None and g.abs().max() > 0 for g in mixed)
        derivatives.append((gradients, mixed))
    assert_close(*outputs)
    assert_close(*derivatives, derivative=True)


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


def test_head_position_derivative_finite_difference_sweep():
    _, candidate, inputs, _, _ = make_case()
    positions = inputs['batch_positions']
    value = coefficient_loss(candidate(dict(inputs)))
    gradient = torch.autograd.grad(value, positions)[0][0, 0, 0]
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        losses = []
        for sign in (-1, 1):
            shifted = positions.detach().clone()
            shifted[0, 0, 0] += sign * step
            new_inputs = inputs_from_positions(inputs['batch_atom_numbers'], shifted, candidate.num_features)
            losses.append(coefficient_loss(candidate(new_inputs)).detach())
        errors.append(abs((losses[1] - losses[0]) / (2 * step) - gradient).item())
    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-6


def test_head_legacy_pickled_module_without_derived_maps():
    _, candidate, inputs, _, _ = make_case()
    expected = candidate(dict(inputs))['df_coeffs']
    for name in list(candidate._buffers):
        if name.startswith('_radial_pack_'):
            delattr(candidate, name)
    actual = candidate(dict(inputs))['df_coeffs']
    assert_close(actual, expected)


@pytest.mark.parametrize('remove', [False, True])
def test_reference_constraint_ragged_padding_is_nonfinite(remove):
    # Record an existing unsupported combination, never count NaN parity as
    # successful derivative validation or a performance capability.
    reference, candidate, inputs, _, _ = make_case(
        positive=not remove, constraint='coeffs_in_coeff_net',
        remove_atom_density=remove, constraint_safe=False)
    for model in (reference, candidate):
        assert not torch.isfinite(model(dict(inputs))['df_coeffs']).all()


@pytest.mark.parametrize('remove', [False, True])
def test_supported_constraint_independent_charge(remove):
    _, candidate, inputs, _, _ = make_case(
        positive=not remove, constraint='coeffs_in_coeff_net', remove_atom_density=remove)
    output = candidate(dict(inputs))
    charge = torch.zeros(inputs['batch_atom_numbers'].shape[0], dtype=torch.float64)
    for i, atom in enumerate(output['spherical_coeffs']):
        for key, coeff in atom.items():
            if key[1] == 0:
                widths = output['radial_width'][i][key]
                scales = output['radial_scale'][i][key]
                # Integral of the code's normalized L=0 Gaussian is
                # pi**1.5 / width**1.5 / (2 sqrt(pi)); no project helper used.
                integral = torch.pi ** 1.5 / widths ** 1.5 / (2 * torch.pi ** .5)
                charge += (coeff * scales * integral).flatten(1).sum(1)
    expected = torch.zeros_like(charge) if remove else inputs['batch_atom_numbers'].sum(1).double()
    assert_close(charge, expected)


def test_force_surrogate_parameter_derivative_finite_difference_sweep():
    _, candidate, inputs, _, _ = make_case()
    parameter = candidate.spherical_output.linear[0].weight
    def objective():
        fresh = inputs_from_positions(inputs['batch_atom_numbers'], inputs['batch_positions'], candidate.num_features)
        loss = coefficient_loss(candidate(fresh))
        coordinate_gradient = torch.autograd.grad(loss, inputs['batch_positions'], create_graph=True)[0]
        return coordinate_gradient.square().sum()
    derivative = torch.autograd.grad(objective(), parameter)[0][0, 0]
    assert derivative.abs() > 0
    initial = parameter[0, 0].detach().clone()
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        values = []
        for sign in (-1, 1):
            with torch.no_grad():
                parameter[0, 0] = initial + sign * step
            values.append(objective().detach())
        errors.append(abs((values[1] - values[0]) / (2 * step) - derivative).item())
    with torch.no_grad():
        parameter[0, 0] = initial
    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-5


@pytest.mark.parametrize('mode', ['codec', 'head', 'head-inference', 'head-backward', 'head-mixed'])
def test_benchmark_callables_smoke_without_timing(mode):
    from benchmarks.packing import benchmark_case
    reference, candidate, _ = benchmark_case(mode, repetitions=2, batch_size=3)
    for _ in range(2):
        before, after = reference(), candidate()
        if mode in ('head', 'head-inference'):
            before, after = before['df_coeffs'], after['df_coeffs']
        assert_close(after, before, derivative=mode in ('head-backward', 'head-mixed'))


def test_coefficient_weighted_head_reference():
    reference, candidate, inputs, _, _ = make_case(weighting=True)
    before, after = reference(dict(inputs)), candidate(dict(inputs))
    for key in ('df_coeffs', 'df_weights', 'coeff_weights'):
        assert_close(after[key], before[key])
