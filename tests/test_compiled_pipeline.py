"""Original eager model is the independent numerical compiler reference."""
import pytest
import torch
import os

from benchmarks.fixtures import build_model, fixture_spec, make_atoms, clone_tree
from equiv_dens.inference.compiled import FunctionalDenSNet


@pytest.fixture(autouse=True)
def isolate_independent_compiler_cases():
    # Each test constructs an independent model/closure. Keep graph caches for
    # reuse checks within each test and release them between unrelated layouts.
    with torch._dynamo.config.patch(suppress_errors=False):
        torch._dynamo.reset()
        yield
        torch._dynamo.reset()


def fixture(mode):
    model = build_model(fixture_spec(), 'force_training' if mode == 'energy_force_update' else 'ef')
    if mode.startswith('density'):
        model.force_props, model.no_force_props = [], ['density']
    model.train()
    return model, make_atoms()


def eager_reference(model, inputs, mode, targets=None):
    output = model(clone_tree(inputs))
    keys = ('density', 'density_grad') if mode.startswith('density') else ('energy', 'forces')
    result = {key: output[key] for key in keys}
    if not mode.endswith('update'):
        return result
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if mode == 'density_update':
        names = [name for name in names if name.startswith('density_repr_model.')]
    params = dict(model.named_parameters())
    loss = sum((result[key] - targets[key]).square().mean() for key in keys)
    gradient_values = torch.autograd.grad(loss, [params[name] for name in names], allow_unused=True)
    gradients = {name: torch.zeros_like(params[name]) if grad is None else grad for name, grad in zip(names, gradient_values)}
    updated = {name: value.detach() - 1e-5 * gradients[name] if name in gradients else value.detach() for name, value in params.items()}
    return dict(result, loss=loss, gradients=gradients, parameters=updated)


def assert_tree(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if isinstance(actual[name], dict):
            assert_tree(actual[name], expected[name])
        else:
            torch.testing.assert_close(actual[name], expected[name], atol=1e-8, rtol=1e-8, msg=lambda msg: name + ': ' + msg)


@pytest.mark.parametrize('mode', FunctionalDenSNet.MODES)
def test_fullgraph_original_outputs_gradients_and_sgd(mode):
    model, inputs = fixture(mode)
    adapter = FunctionalDenSNet(model, inputs)
    params = adapter.parameters()
    initial = eager_reference(model, inputs, 'density' if mode.startswith('density') else 'ef')
    targets = {name: value.detach() + (0.031 if name in ('energy', 'density') else -0.019) for name, value in initial.items()}
    expected = eager_reference(model, inputs, mode, targets)
    graphs = []
    def backend(graph, example):
        graphs.append(graph)
        return graph.forward
    operation = adapter.operation(mode, backend=backend)
    actual = operation(params, inputs, targets)
    assert_tree(actual, expected)
    assert len(graphs) == 1, 'Strict application must compile one complete graph'
    if mode.endswith('update'):
        assert actual['loss'] > 0
        assert any(value.abs().max() > 0 for value in actual['gradients'].values())
        for name in adapter.density_names if mode == 'energy_force_update' else adapter.energy_names:
            assert torch.equal(actual['parameters'][name], params[name])
        # Re-use identical shapes after a complete update and changed geometry.
        params = actual['parameters']
        with torch.no_grad():
            for name, value in model.named_parameters():
                value.copy_(params[name])
    changed = clone_tree(inputs)
    changed['positions'][0, 0, 0] += 0.027
    changed['batch_positions'][0, 0, 0] += 0.027
    actual_changed = operation(params, changed, targets)
    assert_tree(actual_changed, eager_reference(model, changed, mode, targets))
    assert len(graphs) == 1, 'Coordinates and parameter values must not trigger recompilation'
    assert not torch.equal(actual_changed[next(iter(initial))], actual[next(iter(initial))])


def test_frozen_density_remains_geometry_dependent_and_force_loss_has_second_derivatives():
    model, inputs = fixture('energy_force_update')
    adapter = FunctionalDenSNet(model, inputs)
    params = adapter.parameters()
    outputs = adapter.predict(params, inputs)
    # Independent central energy difference includes geometry-dependent frozen density.
    step = 1e-5
    energies = []
    for sign in (1, -1):
        moved = clone_tree(inputs)
        moved['positions'][0, 0, 0] += sign * step
        energies.append(adapter.predict(params, moved)['energy'].sum())
    torch.testing.assert_close(outputs['forces'][0, 0, 0], -(energies[0] - energies[1]) / (2 * step), atol=1e-8, rtol=1e-8)
    # A uniform force offset cancels by translational invariance; use a spatial
    # pattern so this specifically exercises mixed coordinate/parameter derivatives.
    offset = torch.linspace(-0.02, 0.03, outputs['forces'].numel()).reshape_as(outputs['forces'])
    targets = {'energy': outputs['energy'].detach(), 'forces': outputs['forces'].detach() + offset}
    result = adapter.operation('energy_force_update', backend='eager')(params, inputs, targets)
    assert result['loss'] > 0
    assert any(g.abs().max() > 1e-10 for g in result['gradients'].values())
    assert_tree(result, eager_reference(model, inputs, 'energy_force_update', targets))

    def stopped_density_energy(positions):
        atoms = adapter.module.density_repr_model(dict(inputs, positions=positions))
        for key in ('spherical_coeffs', 'radial_width', 'radial_scale'):
            atoms[key] = [{shell: value.detach() for shell, value in atom.items()} for atom in atoms[key]]
        return adapter.module.property_models['energy'](atoms)['energy'].sum()
    partial_forces = -torch.func.grad(stopped_density_energy)(inputs['positions'])
    assert (partial_forces - outputs['forces']).abs().max() > 1e-10


def test_no_silent_compiler_fallback():
    model, inputs = fixture('ef')
    adapter = FunctionalDenSNet(model, inputs)
    with torch._dynamo.config.patch(suppress_errors=True):
        with pytest.raises(RuntimeError, match='suppress_errors=False'):
            adapter.operation('ef')


def test_changed_topology_is_rejected_before_compilation():
    model, inputs = fixture('ef')
    adapter = FunctionalDenSNet(model, inputs)
    changed = clone_tree(inputs)
    changed['batch_atom_numbers'][0, 0] = 8
    with pytest.raises(ValueError, match='Fixed topology changed'):
        adapter.operation('ef')(adapter.parameters(), changed)


def test_l0_start_outputs_force_gradients_and_global_zero_variance():
    from equiv_dens.inference.compiled import _layer_norm
    from equiv_dens.nn.modules.network_blocks import layer_norm
    model, inputs = fixture('energy_force_update')
    # Rebuild only the energy head using the actual paper L0-start default.
    from equiv_dens.nn.property_output.energy import SphericalHarmonicsEnergyNetwork
    spec = fixture_spec()
    model.property_models['energy'] = SphericalHarmonicsEnergyNetwork(
        orbital_basis=spec['orbital_basis'], order=[1], num_modules=1,
        num_features=4, num_basis_functions=4, cutoff=5., normalize=0,
        pred_radial_coeffs=True, L0_start=True, calculate_forces=True,
        clebsch_gordan=model.density_repr_model[0].clebsch_gordan).double()
    adapter = FunctionalDenSNet(model, inputs)
    initial = adapter.predict(adapter.parameters(), inputs)
    targets = {'energy': initial['energy'].detach() + .03,
               'forces': initial['forces'].detach() + torch.linspace(-.02, .03, 9).reshape_as(initial['forces'])}
    actual = adapter.operation('energy_force_update', backend='eager')(adapter.parameters(), inputs, targets)
    assert_tree(actual, eager_reference(model, inputs, 'energy_force_update', targets))
    for value in (torch.ones(1, 3, 1, 4, dtype=torch.float64), torch.arange(12, dtype=torch.float64).reshape(1, 3, 1, 4)):
        x = value.clone().requires_grad_()
        expected, got = layer_norm(x, (-3, -2, -1)), _layer_norm(x)
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
        torch.testing.assert_close(torch.autograd.grad(got.sum(), x, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), x)[0], atol=0, rtol=0)


def test_harmonic_scalar_conversion_preserves_fp64_outputs_and_gradients(monkeypatch):
    import importlib
    import numpy as np
    harmonics = importlib.import_module('equiv_dens.utils.spherical_harmonics')
    derivatives = importlib.import_module('equiv_dens.utils.spherical_harmonics_deriv')
    coords = torch.tensor([[.1, .2, .3], [-.7, .4, .2]], dtype=torch.float64, requires_grad=True)
    direction = coords / coords.norm(dim=-1, keepdim=True)
    def evaluate():
        result = harmonics.spherical_harmonics(5, direction) + derivatives.spherical_harmonics_deriv(5, direction)
        grad = torch.autograd.grad(sum(x.square().sum() for x in result), coords, retain_graph=True)[0]
        return result, grad
    actual, grad_actual = evaluate()
    for module in (harmonics, derivatives):
        for name, value in vars(module).copy().items():
            if name.startswith('sqrt'):
                assert type(value) is float
                original = np.float64(value)
                assert original.tobytes() == np.float64(float(original)).tobytes()
                monkeypatch.setattr(module, name, original)
    original, grad_original = evaluate()
    for x, y in zip(actual, original):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    torch.testing.assert_close(grad_actual, grad_original, atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['density_update', 'energy_force_update'])
def test_ethanol_auxiliary_basis_l0_start_fullgraph(mode):
    import numpy as np
    from benchmarks.application_fixtures import application_atoms, application_spec, build_application_model
    spec = application_spec('ethanol-small')
    atoms, _ = application_atoms(grid_points=9)
    inputs = {key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value for key, value in atoms.items()}
    model = build_application_model(spec, mode)
    if mode == 'energy_force_update':
        for parameter in model.density_repr_model.parameters():
            parameter.requires_grad_(False)
    adapter = FunctionalDenSNet(model, inputs)
    params = adapter.parameters()
    keys = ('density', 'density_grad') if mode.startswith('density') else ('energy', 'forces')
    initial = eager_reference(model, inputs, 'density' if mode.startswith('density') else 'ef')
    targets = {key: initial[key].detach() + torch.linspace(.011, .029, initial[key].numel()).reshape_as(initial[key]) for key in keys}
    actual = adapter.operation(mode, backend='eager')(params, inputs, targets)
    assert_tree(actual, eager_reference(model, inputs, mode, targets))


def test_paper_degree_schedule_reduced_width_force_update():
    import numpy as np
    from benchmarks.application_fixtures import application_atoms, application_spec, build_application_model
    spec = application_spec('ethanol-paper')
    # Preserve all six paper degree transitions and the actual HCO auxiliary
    # basis, bounding numerical qualification by reducing feature/radial width.
    spec['num_features'] = spec['num_basis_functions'] = 4
    atoms, _ = application_atoms(grid_points=9)
    inputs = {key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value for key, value in atoms.items()}
    model = build_application_model(spec, 'energy_force_update')
    for parameter in model.density_repr_model.parameters():
        parameter.requires_grad_(False)
    adapter = FunctionalDenSNet(model, inputs)
    params = adapter.parameters()
    initial = eager_reference(model, inputs, 'ef')
    targets = {key: value.detach() + torch.linspace(.011, .029, value.numel()).reshape_as(value) for key, value in initial.items()}
    weights = {'energy': .2, 'forces': .3}
    actual = adapter.operation('energy_force_update', backend='eager', loss_weights=weights)(params, inputs, targets)
    original = model(clone_tree(inputs))
    names = adapter.energy_names
    original_params = dict(model.named_parameters())
    loss = sum(weights[key] * (original[key] - targets[key]).square().mean() for key in weights)
    values = torch.autograd.grad(loss, [original_params[name] for name in names], allow_unused=True)
    gradients = {name: torch.zeros_like(original_params[name]) if value is None else value for name, value in zip(names, values)}
    updated = {name: value.detach() - 1e-5 * gradients[name] if name in gradients else value.detach() for name, value in original_params.items()}
    assert_tree(actual, dict(energy=original['energy'], forces=original['forces'], loss=loss, gradients=gradients, parameters=updated))


def test_partial_stage_freeze_preserves_parameters_and_excludes_gradients():
    model, inputs = fixture('energy_force_update')
    frozen_name = next(name for name, parameter in model.named_parameters()
                       if name.startswith('property_models.energy.') and parameter.requires_grad)
    original_params = dict(model.named_parameters())
    original_params[frozen_name].requires_grad_(False)
    adapter = FunctionalDenSNet(model, inputs)
    params = adapter.parameters()
    assert frozen_name in adapter.energy_names
    assert frozen_name not in adapter.trainable_names
    initial = eager_reference(model, inputs, 'ef')
    targets = {key: value.detach() + torch.linspace(.011, .029, value.numel()).reshape_as(value)
               for key, value in initial.items()}
    actual = adapter.operation('energy_force_update', backend='eager')(params, inputs, targets)
    assert_tree(actual, eager_reference(model, inputs, 'energy_force_update', targets))
    assert frozen_name not in actual['gradients']
    assert torch.equal(actual['parameters'][frozen_name], params[frozen_name])
    for parameter in model.property_models['energy'].parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match='No trainable parameters'):
        FunctionalDenSNet(model, inputs).operation('energy_force_update')


@pytest.mark.skipif(os.environ.get('DENSNET_TEST_INDUCTOR') != '1', reason='Bounded actual compiler qualification is opt-in')
def test_actual_cpu_inductor_force_update():
    """Actual generated code correctness; this test makes no timing claim."""
    import hashlib
    import json
    import resource
    import socket
    import time
    from pathlib import Path
    torch._inductor.config.compile_threads = 1
    model, inputs = fixture('energy_force_update')
    adapter = FunctionalDenSNet(model, inputs)
    params = adapter.parameters()
    initial = eager_reference(model, inputs, 'ef')
    targets = {key: value.detach() + torch.linspace(.011, .029, value.numel()).reshape_as(value)
               for key, value in initial.items()}
    expected = eager_reference(model, inputs, 'energy_force_update', targets)
    operation = adapter.operation('energy_force_update', backend='inductor')
    start = time.monotonic()
    actual = operation(params, inputs, targets)
    first_call_seconds = time.monotonic() - start
    assert_tree(actual, expected)
    assert_tree(operation(actual['parameters'], inputs, targets),
                adapter.operation('energy_force_update', compile=False)(actual['parameters'], inputs, targets))
    artifact = os.environ.get('DENSNET_INDUCTOR_ARTIFACT')
    if artifact:
        import equiv_dens.inference.compiled as compiled_module
        report = {'purpose': 'CPU Inductor generated-code numerical qualification only',
                  'status': 'passed', 'mode': 'energy_force_update', 'backend': 'inductor',
                  'fullgraph': True, 'suppress_errors': bool(torch._dynamo.config.suppress_errors),
                  'dtype': 'float64', 'atol': 1e-8, 'rtol': 1e-8,
                  'compiler_threads': torch._inductor.config.compile_threads,
                  'numerical_threads': torch.get_num_threads(), 'torch_version': torch.__version__,
                  'host': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                  'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'first_call_seconds': first_call_seconds,
                  'adapter_sha256': hashlib.sha256(Path(compiled_module.__file__).read_bytes()).hexdigest(),
                  'gates': ['original eager outputs', 'force-loss parameter gradients',
                            'complete SGD update', 'second update versus functional adapter'],
                  'fixture': 'tiny HHO, F4, density order1, energy order1, nine grid points'}
        Path(artifact).write_text(json.dumps(report, indent=2) + '\n')
