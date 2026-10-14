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

