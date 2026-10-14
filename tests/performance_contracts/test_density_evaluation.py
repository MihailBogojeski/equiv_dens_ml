"""H4-C1 exact density/derivative contracts, no timing assertions."""
import pytest
import torch
from pyscf.lib import param

BOHR = param.BOHR

from benchmarks.density_evaluation import (benchmark_case, differentiable_inputs,
    evaluation_loss, reference_evaluator, supplied_fixture, validate_case)
from benchmarks.experimental_density import (
    ExperimentalDensityExpansion,
    eval_gto_contracted,
    eval_gto_grad_contracted,
    _needs_full_precision_reduction,
    install_experimental_density_evaluator,
)
from benchmarks.fixtures import build_model, fixture_spec, tensor_hash
from equiv_dens.nn.property_output.density import DensityExpansion


def close(a, b, derivative=False):
    if a is None:
        assert b is None
        return
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y, derivative)
        return
    fp32 = a.dtype == torch.float32
    atol, rtol = ((1e-4, 1e-3) if derivative else (1e-5, 1e-4)) if fp32 else ((1e-8, 1e-6) if derivative else (1e-9, 1e-7))
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('gradient', [False, True])
@pytest.mark.parametrize('partial,core', [(False, False), (True, False), (False, True)])
def test_evaluator_values_first_and_mixed_derivatives(dtype, gradient, partial, core):
    basis, data = supplied_fixture(dtype=dtype, core=core)
    evaluators = [reference_evaluator()(basis, density_grad=gradient),
                  ExperimentalDensityExpansion(basis, density_grad=gradient, evaluation_backend='contracted')]
    options = dict(eval_atoms=[0, 2], eval_L=[0, 1]) if partial else {}
    outputs, derivatives = [], []
    initial_hash = tensor_hash(data)
    for evaluator in evaluators:
        inputs, targets = differentiable_inputs(data)
        result = evaluator(inputs, **options)
        outputs.append([result['density']] + ([result['density_grad']] if gradient else []))
        assert all(value.abs().max() > 1e-12 for value in outputs[-1])
        grads = torch.autograd.grad(evaluation_loss(result), targets, create_graph=True, allow_unused=True)
        assert all(g is not None and g.abs().max() > 1e-12 for g in grads[:2])
        count = len(targets[2:]) // 3
        for start, end in ((2, 2 + count), (2 + count, 2 + 2 * count),
                           (2 + 2 * count, len(targets))):
            assert any(g is not None and g.abs().max() > 1e-12 for g in grads[start:end])
        mixed = torch.autograd.grad(grads[1].square().sum(), targets, allow_unused=True)
        assert any(g is not None and g.abs().max() > 1e-12 for g in mixed[2:])
        derivatives.append((grads, mixed))
    close(*outputs)
    close(*derivatives, derivative=True)
    assert tensor_hash(data) == initial_hash


@pytest.mark.parametrize('constraint', ['sq', 'abs', 'sp', 'grid'])
def test_postprocessing_matches_existing_gradient_convention(constraint):
    basis, data = supplied_fixture()
    options = dict(expansion_constraint=constraint if constraint != 'grid' else None,
                   integral_constraint='grid' if constraint == 'grid' else None,
                   density_grad=True, grid_scaling_factor=constraint == 'grid')
    reference, candidate = reference_evaluator()(basis, **options), ExperimentalDensityExpansion(basis, evaluation_backend='contracted', **options)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    results = []
    for evaluator in (reference, candidate):
        inputs, targets = differentiable_inputs(data)
        out = evaluator(inputs)
        grads = torch.autograd.grad(evaluation_loss(out), targets)
        results.append((out['density'], out['density_grad'], grads))
    close(results[0][:2], results[1][:2])
    close(results[0][2], results[1][2], derivative=True)
    # Analytic density_grad is the pre-postprocessing gradient in the reference.
    # Preserve this convention rather than introducing a correctness change.
    plain = DensityExpansion(basis, density_grad=True)(dict(data))
    close(results[1][1], plain['density_grad'])


def test_independent_single_s_gaussian_and_spatial_derivative():
    basis = {1: [(1, 2, 0)]}
    coords = torch.tensor([[[.19, -.31, .27], [.41, .13, -.29]]], dtype=torch.float64, requires_grad=True)
    position = torch.tensor([[[.03, -.05, .02]]], dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor([[[[.73]]]], dtype=torch.float64, requires_grad=True)
    widths = torch.tensor([[[[.61], [1.32]]]], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[[[.28], [.57]]]], dtype=torch.float64, requires_grad=True)
    data = dict(batch_atom_numbers=torch.tensor([[1]]), batch_positions=position, coords=coords,
                spherical_coeffs=[{(1, 0): coefficient}], radial_width=[{(1, 0): widths}],
                radial_scale=[{(1, 0): scales}], coord_weights=torch.ones(1, 2, dtype=torch.float64))
    out = ExperimentalDensityExpansion(basis, density_grad=True, evaluation_backend='contracted')(data)
    displacement = (coords - position) / BOHR
    radius2 = displacement.square().sum(-1)
    primitive = scales.reshape(2) / (2 * torch.pi ** .5) * torch.exp(-radius2[..., None] * widths.reshape(2))
    expected = coefficient.squeeze() * primitive.sum(-1)
    expected_grad = coefficient.squeeze() * (-2 * primitive * widths.reshape(2))[..., None].mul(displacement[..., None, :]).sum(-2)
    close(out['density'], expected)
    close(out['density_grad'], expected_grad)
    # Returned spatial gradient uses derivative per bohr; grid inputs are angstrom.
    coordinate_grad = torch.autograd.grad(out['density'].sum(), coords, retain_graph=True)[0]
    close(coordinate_grad * BOHR, expected_grad, derivative=True)


def test_uneven_partial_grid_blocks_preserve_all_points():
    basis, data = supplied_fixture(grid_points=31)
    evaluator = ExperimentalDensityExpansion(basis, density_grad=True, evaluation_backend='contracted')
    whole = evaluator(dict(data))
    blocks = []
    for start in range(0, 31, 7):
        part = dict(data, coords=data['coords'][:, start:start + 7],
                    coord_weights=data['coord_weights'][:, start:start + 7])
        blocks.append(evaluator(part))
    for key in ('density', 'density_grad'):
        close(torch.cat([b[key] for b in blocks], 1), whole[key])


def test_near_center_values_derivatives_and_exact_center_exclusion():
    basis, data = supplied_fixture(grid_points=3)
    data['coords'] = data['batch_positions'][:, :1] + torch.tensor(
        [[[1e-4, -2e-4, 3e-4], [2e-5, 1e-5, -1e-5], [1e-6, -2e-6, 3e-6]]], dtype=torch.float64)
    results = []
    for evaluator in (reference_evaluator()(basis, density_grad=True),
                      ExperimentalDensityExpansion(basis, density_grad=True, evaluation_backend='contracted')):
        inputs, targets = differentiable_inputs(data)
        out = evaluator(inputs)
        grads = torch.autograd.grad(evaluation_loss(out), targets)
        results.append((out['density'], out['density_grad'], grads))
    close(results[0][:2], results[1][:2])
    close(results[0][2], results[1][2], derivative=True)
    # At an exact center legacy forward is sanitized, but differentiating its
    # d=0/u=0/0 operations is unsupported. Explicitly record this limitation.
    data['coords'] = data['batch_positions'][:, :1].repeat(1, 3, 1)
    for evaluator in (reference_evaluator()(basis, density_grad=True),
                      ExperimentalDensityExpansion(basis, density_grad=True, evaluation_backend='contracted')):
        inputs, targets = differentiable_inputs(data)
        out = evaluator(inputs)
        assert torch.isfinite(out['density']).all()
        assert torch.isfinite(out['density_grad']).all()
        grads = torch.autograd.grad(evaluation_loss(out), targets)
        assert any(not torch.isfinite(g).all() for g in grads)


def test_gradient_parameter_finite_difference_step_sweep():
    basis, data = supplied_fixture(grid_points=7)
    evaluator = ExperimentalDensityExpansion(basis, density_grad=True, evaluation_backend='contracted')
    data, targets = differentiable_inputs(data)
    width = data['radial_width'][0][(1, 0)]
    derivative = torch.autograd.grad(evaluation_loss(evaluator(dict(data))), width)[0][0, 0, 0, 0]
    assert derivative.abs() > 1e-12
    original = width[0, 0, 0, 0].detach().clone()
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        values = []
        for sign in (-1, 1):
            with torch.no_grad():
                width[0, 0, 0, 0] = original + sign * step
            values.append(evaluation_loss(evaluator(dict(data))).detach())
        errors.append(abs((values[1] - values[0]) / (2 * step) - derivative).item())
    with torch.no_grad():
        width[0, 0, 0, 0] = original
    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-8


def test_default_state_legacy_module_and_invalid_selector():
    basis, data = supplied_fixture()
    evaluator = DensityExpansion(basis, density_grad=True)
    expected = evaluator(dict(data))
    actual = evaluator(dict(data))
    close(expected['density_grad'], actual['density_grad'])
    adapter = ExperimentalDensityExpansion(basis, density_grad=True)
    assert adapter.evaluation_backend == 'legacy'
    del adapter.evaluation_backend
    close(expected['density_grad'], adapter(dict(data))['density_grad'])
    assert evaluator.state_dict().keys() == ExperimentalDensityExpansion(basis, evaluation_backend='contracted').state_dict().keys()
    with pytest.raises(ValueError, match='backend'):
        ExperimentalDensityExpansion(basis, evaluation_backend='unknown')


def test_fixture_installer_keeps_state_and_production_api_is_legacy():
    import inspect
    basis, data = supplied_fixture()
    assert 'evaluation_backend' not in inspect.signature(DensityExpansion).parameters
    spec = fixture_spec('tiny', torch.float64)
    legacy = build_model(spec, 'density')
    model = build_model(spec, 'density', density_backend='contracted')
    assert isinstance(model.property_models['density'], ExperimentalDensityExpansion)
    assert model.property_models['density'].evaluation_backend == 'contracted'
    assert model.state_dict().keys() == legacy.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[key])
    with pytest.raises(ValueError, match='backend'):
        install_experimental_density_evaluator(model, 'unknown')
    assert torch.isfinite(DensityExpansion(basis)(dict(data))['density']).all()


def test_spatial_gradient_finite_difference_step_sweep():
    basis, data = supplied_fixture(grid_points=7)
    evaluator = ExperimentalDensityExpansion(basis, density_grad=True, evaluation_backend='contracted')
    expected = evaluator(dict(data))['density_grad'][0, 0, 1]
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        values = []
        for sign in (-1, 1):
            coords = data['coords'].clone()
            coords[0, 0, 1] += sign * step
            values.append(evaluator(dict(data, coords=coords))['density'][0, 0])
        errors.append(abs((values[1] - values[0]) / (2 * step) * BOHR - expected).item())
    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-7


def test_configured_sad_addition_at_composition_boundary():
    from equiv_dens.nn.dft_network import DFTNetwork
    basis, data = supplied_fixture(grid_points=7)
    displacement = (data['coords'] - data['batch_positions'][:, :1]) / BOHR
    sad = .11 * torch.exp(-.3 * displacement.square().sum(-1))
    sad_grad = -.6 * displacement * sad[..., None]
    data.update(positions=data['batch_positions'].clone(), atom_density=sad,
                atom_density_grad=sad_grad)
    plain = reference_evaluator()(basis, density_grad=True)(dict(data))
    values = []
    for evaluator in (reference_evaluator()(basis, density_grad=True),
                      ExperimentalDensityExpansion(basis, density_grad=True, evaluation_backend='contracted')):
        model = DFTNetwork(torch.nn.Identity(), {'density': evaluator}, remove_atom_density=True)
        model.train()
        output = model(data)
        close(output['density'], plain['density'] + sad)
        close(output['density_grad'], plain['density_grad'] + sad_grad)
        values.append((output['density'], output['density_grad']))
    close(*values)


@pytest.mark.parametrize('learned', [False, True])
def test_density_training_radial_modes_one_matched_update(learned):
    reference, candidate, _ = benchmark_case('density-training', grid_points=7,
                                             dtype=torch.float64, learned_radial=learned)
    before = {key: value.clone() for key, value in reference.model.state_dict().items()}
    close(reference(), candidate())
    assert any(not torch.equal(value, before[key]) for key, value in candidate.model.state_dict().items())
    for key, value in reference.model.state_dict().items():
        close(value, candidate.model.state_dict()[key])


@pytest.mark.parametrize('L', [0, 1, 2, 3, 4, 5])
def test_all_supported_harmonic_degrees_contraction(L):
    from benchmarks.packing import load_reference
    from equiv_dens.utils import orbitals
    from equiv_dens.utils.base import calculate_distances_and_directions
    from equiv_dens.utils.spherical_harmonics import spherical_harmonics
    from equiv_dens.utils.spherical_harmonics_deriv import spherical_harmonics_deriv
    reference, _ = load_reference()
    coords = torch.tensor([[[.2, -.3, .4], [.7, -.5, -.3]]], dtype=torch.float64)
    center = torch.tensor([[[.03, -.04, .07]]], dtype=torch.float64)
    d, u = calculate_distances_and_directions(coords, center=center)
    s, ds = spherical_harmonics(L, u)[L], spherical_harmonics_deriv(L, u)[L]
    coefficients = torch.linspace(-.2, .7, (2 * L + 1) * 2, dtype=torch.float64).reshape(1, 1, 2 * L + 1, 2).requires_grad_()
    widths = torch.tensor([[[[.5, 1.3], [1.1, .8]]]], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[[[.2, .7], [.4, .6]]]], dtype=torch.float64, requires_grad=True)
    outputs = []
    for evaluate, gradient in ((reference.eval_gto, reference.eval_gto_grad),
                               (reference.eval_gto_einsum, reference.eval_gto_grad_einsum),
                               (eval_gto_contracted, eval_gto_grad_contracted)):
        rho, angular, radial = evaluate(s, d, L, coefficients, widths, scales)
        spatial = gradient(ds, d, L, coords - center, coefficients, widths, scales, angular, radial)
        derivatives = torch.autograd.grad(rho.square().sum() + spatial.square().sum(),
                                          (coefficients, widths, scales))
        outputs.append((rho, spatial, derivatives))
    for output in outputs[1:]:
        close(output[:2], outputs[0][:2])
        close(output[2], outputs[0][2], derivative=True)


def test_untimed_benchmark_gate_rejects_wrong_actual_grid_and_restores_state():
    reference, candidate, manifest = benchmark_case('density-training', grid_points=7,
                                                  dtype=torch.float64)
    initial = tensor_hash(reference.model.state_dict())
    manifest['grid_points'] = 8
    with pytest.raises(AssertionError, match='actual grid shape'):
        validate_case(reference, candidate, manifest)
    assert tensor_hash(reference.model.state_dict()) == initial
    assert tensor_hash(candidate.model.state_dict()) == initial


def _wide_gto_contract_snapshots(device='cpu', L=5, shells=17, grid_points=257):
    from benchmarks.packing import load_reference
    from equiv_dens.utils import orbitals
    from equiv_dens.utils.base import calculate_distances_and_directions
    from equiv_dens.utils.spherical_harmonics import spherical_harmonics
    from equiv_dens.utils.spherical_harmonics_deriv import spherical_harmonics_deriv
    reference, _ = load_reference()
    dtype = torch.float32
    torch.manual_seed(8341)
    source = [torch.randn(2, 1, 3, dtype=dtype, device=device) * .03,
              torch.randn(2, grid_points, 3, dtype=dtype, device=device) * .2 + .31,
              torch.randn(2, 1, 2 * L + 1, shells, dtype=dtype, device=device) * .2,
              torch.rand(2, 1, 3, shells, dtype=dtype, device=device) + .5,
              torch.rand(2, 1, 3, shells, dtype=dtype, device=device) * .4 + .1]
    results = []
    for evaluate, gradient in ((reference.eval_gto, reference.eval_gto_grad),
                               (eval_gto_contracted, eval_gto_grad_contracted)):
        center, coords, coefficients, widths, scales = [v.detach().clone().requires_grad_() for v in source]
        targets = (center, coords, coefficients, widths, scales)
        d, u = calculate_distances_and_directions(coords, center=center)
        s = spherical_harmonics(L, u)[L]
        ds = spherical_harmonics_deriv(L, u)[L]
        rho, angular, radial = evaluate(s, d, L, coefficients, widths, scales)
        spatial = gradient(ds, d, L, coords - center, coefficients, widths, scales, angular, radial)
        derivatives = torch.autograd.grad(rho.square().mean() + .03 * spatial.square().mean(),
                                          targets, create_graph=True)
        mixed = torch.autograd.grad(derivatives[1].square().mean(), targets[2:])
        assert all(value.abs().max() > 1e-12 for value in derivatives)
        assert any(value.abs().max() > 1e-12 for value in mixed)
        results.append((rho, spatial, derivatives, mixed))
    close(results[0][:2], results[1][:2])
    close(results[0][2:], results[1][2:], derivative=True)


def test_full_precision_elementwise_fallback_cpu(monkeypatch):
    from benchmarks.precision import precision_metadata
    before = precision_metadata()
    monkeypatch.setattr('benchmarks.experimental_density._needs_full_precision_reduction',
                        lambda _tensor: True)
    _wide_gto_contract_snapshots(grid_points=31)
    assert precision_metadata() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
@pytest.mark.parametrize('tf32', [False, True])
def test_cuda_fp32_large_magnetic_shells_precision_and_mixed_derivatives(tf32):
    from benchmarks.precision import precision_metadata
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    before = precision_metadata()
    probe = torch.ones(1, dtype=torch.float32, device='cuda')
    assert _needs_full_precision_reduction(probe) == tf32
    _wide_gto_contract_snapshots(device='cuda')
    assert precision_metadata() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
@pytest.mark.parametrize('tf32', [False, True])
def test_cuda_fp32_evaluator_actual_grid_values_and_derivatives(tf32):
    from benchmarks.precision import precision_metadata
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    torch.backends.cuda.matmul.allow_tf32 = tf32
    before = precision_metadata()
    for mode in ('rho', 'rho-grad', 'backward', 'mixed'):
        reference, candidate, manifest = benchmark_case(mode, grid_points=1000,
            dtype=torch.float32, device='cuda')
        assert validate_case(reference, candidate, manifest)['passed']
    assert precision_metadata() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
@pytest.mark.parametrize('tf32', [False, True])
@pytest.mark.parametrize('learned', [False, True])
def test_cuda_fp32_same_state_density_training_precision(tf32, learned):
    from benchmarks.precision import precision_metadata
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    torch.backends.cuda.matmul.allow_tf32 = tf32
    before = precision_metadata()
    reference, candidate, manifest = benchmark_case('density-training', grid_points=1000,
        dtype=torch.float32, device='cuda', learned_radial=learned)
    assert validate_case(reference, candidate, manifest)['passed']
    for _ in range(2):
        close(reference(), candidate())
    for key, value in reference.model.state_dict().items():
        close(value, candidate.model.state_dict()[key])
    assert precision_metadata() == before


@pytest.mark.parametrize('mode', ['rho', 'rho-grad', 'backward', 'mixed', 'density-training'])
def test_benchmark_calls_without_timing(mode):
    reference, candidate, manifest = benchmark_case(mode, grid_points=11, dtype=torch.float64)
    assert validate_case(reference, candidate, manifest)['passed']
    for _ in range(2):
        before, after = reference(), candidate()
        if mode in ('rho', 'rho-grad'):
            close(after['density'], before['density'])
            if mode == 'rho-grad':
                close(after['density_grad'], before['density_grad'])
        else:
            close(after, before, derivative=True)
    if mode == 'density-training':
        for key, value in reference.model.state_dict().items():
            close(candidate.model.state_dict()[key], value)
