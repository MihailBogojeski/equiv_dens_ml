"""Independent original-model gates for two-region density-stage training."""
import numpy as np
import pytest
import torch

from benchmarks.fixtures import build_model, clone_tree, fixture_spec, make_atoms
from equiv_dens.inference.regional import RegionalDensityDenSNet


@pytest.fixture(autouse=True)
def isolate_compiler_cases():
    with torch._dynamo.config.patch(suppress_errors=False):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        yield
        torch._dynamo.reset()


def tiny_model():
    model = build_model(fixture_spec(), 'density')
    model.force_props, model.no_force_props = [], ['density']
    model.train()
    return model, make_atoms()


def reference(model, inputs, targets, weights):
    output = model(clone_tree(inputs))
    output = {key: output[key] for key in ('density', 'density_grad')}
    loss = sum(weights[key] * (output[key] - targets[key]).square().mean() for key in output)
    params = dict(model.named_parameters())
    names = tuple(name for name, value in params.items()
                  if name.startswith('density_repr_model.') and value.requires_grad)
    grads = torch.autograd.grad(loss, tuple(params[name] for name in names), allow_unused=True)
    gradients = {name: torch.zeros_like(params[name]) if value is None else value
                 for name, value in zip(names, grads)}
    updated = {name: value.detach() - 1e-5 * gradients[name] if name in gradients else value.detach()
               for name, value in params.items()}
    return dict(output, loss=loss, gradients=gradients, parameters=updated)


def targets_from_original(model, inputs):
    output = model(clone_tree(inputs))
    return {key: output[key].detach() + torch.linspace(.011, .029, output[key].numel(),
                                                     dtype=output[key].dtype).reshape_as(output[key])
            for key in ('density', 'density_grad')}


def assert_tree(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if isinstance(actual[name], dict):
            assert_tree(actual[name], expected[name])
        else:
            torch.testing.assert_close(actual[name], expected[name], atol=1e-8, rtol=1e-8,
                                       msg=lambda message: name + ': ' + message)


def replace_model_params(model, params):
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(params[name])


def test_two_regions_preserve_original_gradients_radial_updates_and_two_sgd_steps():
    model, inputs = tiny_model()
    targets = targets_from_original(model, inputs)
    weights = {'density': 1., 'density_grad': .1}
    adapter = RegionalDensityDenSNet(model, inputs)
    params = adapter.parameters()
    originals = clone_tree(params)
    graphs = []
    def backend(graph, example):
        graphs.append(graph)
        return graph.forward
    operation = adapter.operation(backend=backend, loss_weights=weights)
    expected = reference(model, inputs, targets, weights)
    actual = operation(params, inputs, targets)
    assert_tree(actual, expected)
    assert len(graphs) == 2
    assert operation.compilation['kind'] == 'regional'
    assert not operation.compilation['full_application_fullgraph']
    for kind in ('radial_width', 'radial_scale'):
        names = tuple(name for name in actual['gradients'] if '.' + kind + '.' in name)
        assert names
        assert any(actual['gradients'][name].abs().max() > 1e-12 for name in names)
        assert any(not torch.equal(actual['parameters'][name], params[name]) for name in names)
    for name in adapter.energy_names:
        assert torch.equal(actual['parameters'][name], params[name])
    for name in params:
        assert torch.equal(params[name], originals[name])
        assert params[name].grad is None
    replace_model_params(model, expected['parameters'])
    changed = clone_tree(inputs)
    changed['positions'][0, 0, 0] += .027
    changed['batch_positions'][0, 0, 0] += .027
    changed['coords'][0, 0, 1] += .019
    second = operation.unchecked(actual['parameters'], changed, targets)
    assert_tree(second, reference(model, changed, targets, weights))
    assert len(graphs) == 2, 'Parameter/coordinate values must reuse both fullgraph regions'
    assert not torch.equal(second['density'], actual['density'])
    assert not torch._dynamo.utils.counters.get('graph_break')
    assert torch._dynamo.utils.counters['stats']['unique_graphs'] == 2
    assert not second['loss'].requires_grad
    assert all(not value.requires_grad for value in second['parameters'].values())


def test_partial_freeze_and_unused_gradient_preserve_complete_parameter_schema():
    model, inputs = tiny_model()
    model.density_repr_model.register_parameter('unused', torch.nn.Parameter(torch.ones(3, dtype=torch.float64)))
    frozen = next(name for name, _ in model.named_parameters() if name.startswith('density_repr_model.0.'))
    dict(model.named_parameters())[frozen].requires_grad_(False)
    adapter = RegionalDensityDenSNet(model, inputs)
    params = adapter.parameters()
    targets = targets_from_original(model, inputs)
    weights = {'density': 1., 'density_grad': 1.}
    result = adapter.operation(backend='eager')(params, inputs, targets)
    assert_tree(result, reference(model, inputs, targets, weights))
    assert frozen in adapter.density_names and frozen not in result['gradients']
    assert torch.equal(result['parameters'][frozen], params[frozen])
    assert torch.equal(result['gradients']['density_repr_model.unused'], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(result['parameters']['density_repr_model.unused'], params['density_repr_model.unused'])
    for name in adapter.energy_names:
        assert torch.equal(result['parameters'][name], params[name])


def test_aot_backward_regions_match_original_on_repeated_parameter_updates():
    model, inputs = tiny_model()
    adapter = RegionalDensityDenSNet(model, inputs)
    operation = adapter.operation(backend='aot_eager', loss_weights={'density_grad': .1})
    params, targets = adapter.parameters(), targets_from_original(model, inputs)
    weights = {'density': 1., 'density_grad': .1}
    for _ in range(2):
        expected = reference(model, inputs, targets, weights)
        actual = operation(params, inputs, targets)
        assert_tree(actual, expected)
        replace_model_params(model, expected['parameters'])
        params = actual['parameters']
    assert torch._dynamo.utils.counters['stats']['unique_graphs'] == 2
    assert not torch._dynamo.utils.counters.get('graph_break')


def test_strict_region_policy_and_topology_checks():
    model, inputs = tiny_model()
    adapter = RegionalDensityDenSNet(model, inputs)
    for mode in ('ef', 'density', 'energy_force_update'):
        with pytest.raises(ValueError, match='density_update only'):
            adapter.operation(mode)
    with torch._dynamo.config.patch(suppress_errors=True):
        with pytest.raises(RuntimeError, match='suppress_errors=False'):
            adapter.operation()
    changed = clone_tree(inputs)
    changed['batch_atom_numbers'][0, 0] = 8
    with pytest.raises(ValueError, match='Fixed topology changed'):
        adapter.operation(backend='eager')(adapter.parameters(), changed, {})
    for parameter in model.density_repr_model.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match='No trainable parameters'):
        RegionalDensityDenSNet(model, inputs).operation()


def test_exact_paper_density_orders_and_hco_auxiliary_basis_reduced_width():
    from benchmarks.application_fixtures import application_atoms, application_spec, build_application_model
    spec = application_spec('ethanol-paper')
    spec['num_features'] = spec['num_basis_functions'] = 4
    atoms, _ = application_atoms(grid_points=5)
    inputs = {key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value
              for key, value in atoms.items()}
    model = build_application_model(spec, 'density_update')
    adapter = RegionalDensityDenSNet(model, inputs)
    params, targets = adapter.parameters(), targets_from_original(model, inputs)
    weights = {'density': 1., 'density_grad': .1}
    operation = adapter.operation(backend='eager', loss_weights=weights)
    for _ in range(2):
        expected = reference(model, inputs, targets, weights)
        actual = operation(params, inputs, targets)
        assert_tree(actual, expected)
        replace_model_params(model, expected['parameters'])
        params = actual['parameters']
    assert torch._dynamo.utils.counters['stats']['unique_graphs'] == 2
    assert not torch._dynamo.utils.counters.get('graph_break')
