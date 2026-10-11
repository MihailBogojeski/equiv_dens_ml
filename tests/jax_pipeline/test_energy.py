"""Native JAX energy/moment parity against separately generated Torch arrays.

Generate references with generate_energy_reference.py in the Torch image first.
No Torch module or source evaluator is imported by this suite.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equiv_dens_jax import energy, observables
from equiv_dens_jax.serialization import load_tree

jax.config.update("jax_enable_x64", True)

REFERENCE = Path(os.environ.get(
    "DENSNET_JAX_ENERGY_REFERENCE",
    str(Path(__file__).resolve().parents[2] / "scratch/jax-pipeline/energy-reference"),
))
ENERGY_CASES = ("spherical", "spherical_norm2", "spherical_L0", "spherical_orders",
                "spherical_gaussian", "spherical_bernstein", "spherical_exp_gaussian",
                "embedding", "embedding_norm2", "linear", "linear_fixed",
                "linear_orders", "representation", "representation_ssp", "spherical_SAD")
DERIVATIVE_CASES = ("spherical_norm2", "spherical_L0", "spherical_SAD", "embedding_norm2",
                    "linear_fixed", "linear_orders", "representation_ssp")


@pytest.fixture(scope="module")
def references():
    if not (REFERENCE / "manifest.json").exists():
        pytest.fail(f"Missing independent Torch energy reference at {REFERENCE}; "
                    "run generate_energy_reference.py in the Torch runtime")
    return load_tree(REFERENCE)


@pytest.fixture(autouse=True)
def bounded_compilation_cache():
    yield
    jax.clear_caches()
    gc.collect()


def arrays(tree):
    return jax.tree_util.tree_map(
        lambda value: jnp.asarray(value) if isinstance(value, np.ndarray) else value, tree
    )


def assert_tree(actual, expected, *, atol=1e-8, rtol=1e-7):
    actual_leaves, actual_shape = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_shape = jax.tree_util.tree_flatten(expected)
    assert actual_shape == expected_shape
    for observed, reference in zip(actual_leaves, expected_leaves):
        assert np.isfinite(np.asarray(observed)).all()
        assert np.isfinite(np.asarray(reference)).all()
        np.testing.assert_allclose(np.asarray(observed), reference, atol=atol, rtol=rtol)


def energy_function(case):
    node = case["node"]
    layout = energy.prepare_layout(node, case["atoms"])
    base = arrays(case["atoms"])
    buffers = arrays(case["state"]["buffers"])

    def evaluate(params, variables):
        atoms = dict(base)
        atoms.update(variables)
        return energy.apply_energy(node, {"params": params, "buffers": buffers}, atoms,
                                   layout=layout)["energy"]
    return evaluate


def scalar(function, params, variables):
    output = function(params, variables)
    return jnp.sum(output**2) + 0.13 * jnp.sum(jnp.cos(output))


@pytest.mark.parametrize("name", ENERGY_CASES)
def test_energy_eager_jit_parity_and_changed_inputs(references, name):
    case = references["energy"][name]
    function = energy_function(case)
    params, variables = arrays(case["state"]["params"]), arrays(case["variables"])
    assert_tree(function(params, variables), case["output"], atol=1e-9, rtol=1e-8)
    compiled = jax.jit(function)
    assert_tree(compiled(params, variables), case["output"], atol=1e-9, rtol=1e-8)
    changed = jax.tree_util.tree_map(lambda value, direction: value + 0.03 * direction,
                                     variables, arrays(case["directions"]))
    assert_tree(compiled(params, changed), function(params, changed), atol=1e-9, rtol=1e-8)
    assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)


@pytest.mark.parametrize("name", DERIVATIVE_CASES)
def test_energy_first_hvp_and_input_gradient_loss_parameter_derivatives(references, name):
    case = references["energy"][name]
    function = energy_function(case)
    loss = lambda params, variables: scalar(function, params, variables)
    params, variables = arrays(case["state"]["params"]), arrays(case["variables"])
    parameter_gradient, input_gradient = jax.grad(loss, argnums=(0, 1))(params, variables)
    assert_tree(parameter_gradient, case["parameter_gradient"])
    assert_tree(input_gradient, case["input_gradient"])
    derivative = lambda values: jax.grad(loss, argnums=1)(params, values)
    _, hvp = jax.jvp(derivative, (variables,), (arrays(case["directions"]),))
    assert_tree(hvp, case["input_hvp"])

    def gradient_loss(values):
        gradients = jax.grad(loss, argnums=1)(values, variables)
        return sum(jnp.sum(value**2) for value in jax.tree_util.tree_leaves(gradients))
    mixed = jax.grad(gradient_loss)(params)
    assert_tree(mixed, case["input_gradient_loss_parameter_gradient"])
    # Independently corroborate a nonzero directional derivative away from cuts.
    direction = arrays(case["directions"])
    def shifted(amount):
        return jax.tree_util.tree_map(lambda value, tangent: value + amount * tangent,
                                     variables, direction)
    step = 1e-5
    finite_difference = (loss(params, shifted(step)) - loss(params, shifted(-step))) / (2 * step)
    directional = sum(jnp.sum(value * tangent) for value, tangent in zip(
        jax.tree_util.tree_leaves(input_gradient), jax.tree_util.tree_leaves(direction)))
    assert abs(float(directional)) > 1e-10
    np.testing.assert_allclose(finite_difference, directional, atol=1e-7, rtol=1e-5)


def test_current_fixed_radial_spherical_source_limitation_is_explicit(references):
    case = references["energy"]["spherical"]
    node = {**case["node"], "attrs": {**case["node"]["attrs"], "pred_radial_coeffs": False}}
    layout = energy.prepare_layout(node, case["atoms"])
    with pytest.raises(NotImplementedError, match="empty radial lists"):
        energy.apply_energy(node, arrays(case["state"]), arrays(case["atoms"]), layout=layout)


def observable_function(case):
    node = case["node"]
    base = arrays(case["atoms"])
    if "atom_dipole_moment" in case:
        base["atom_dipole_moment"] = arrays(case["atom_dipole_moment"])
    layout = observables.prepare_layout(node, case["atoms"])
    integrals = arrays(case["integrals"]) if "integrals" in case else None

    def evaluate(params, variables):
        atoms = dict(base)
        atoms.update(variables)
        return observables.apply_observable(node, {"params": params, "buffers": {}}, atoms,
                                            layout=layout, integrals=integrals,
                                            center_coordinates=case["center_coordinates"])["dipole_moment"]
    return evaluate


@pytest.mark.parametrize("name", ["grid_True", "grid_False", "analytic_sp",
                                  "analytic_free_df", "analytic_free_mo"])
def test_dipole_values_and_derivative_contracts(references, name):
    case = references["moments"][name]
    function = observable_function(case)
    variables = arrays(case["variables"])
    assert_tree(function({}, variables), case["output"], atol=1e-9, rtol=1e-8)
    assert_tree(jax.jit(function)({}, variables), case["output"], atol=1e-9, rtol=1e-8)
    loss = lambda values: scalar(function, {}, values)
    assert_tree(jax.grad(loss)(variables), case["input_gradient"])
    _, hvp = jax.jvp(jax.grad(loss), (variables,), (arrays(case["directions"]),))
    assert_tree(hvp, case["input_hvp"])
    if name.startswith("analytic"):
        actual = observables.prepare_integrals(case["node"], case["atoms"])
        assert_tree(actual, case["integrals"], atol=1e-10, rtol=1e-9)
        if "atom_dipole_moment" in case:
            actual = observables.prepare_free_atom_dipole(case["atoms"])
            assert_tree(actual, case["atom_dipole_moment"], atol=1e-10, rtol=1e-9)


def test_analytic_integrals_are_explicit_constants(references):
    case = references["moments"]["analytic_sp"]
    layout = observables.prepare_layout(case["node"], case["atoms"])
    atoms = arrays(case["atoms"])
    with pytest.raises(ValueError, match="explicit current/frozen"):
        observables.apply_observable(case["node"], {}, atoms, layout=layout)
    function = lambda values: observables.apply_observable(
        case["node"], {}, atoms, layout=layout, integrals=values)["dipole_moment"].sum()
    np.testing.assert_array_equal(jax.grad(function)(arrays(case["integrals"])), 0.0)


def test_host_analytic_cache_parity_across_geometry_coefficients_and_radial_changes(references):
    case = references["moments"]["analytic_cache"]
    cache = observables.HostDipoleCache()
    layout = observables.prepare_layout(case["node"], case["sequence"][0]["atoms"])
    expected_actions = ("prepared_integrals", "reused_electronic", "reused_integrals",
                        "reused_electronic", "prepared_integrals")
    for step, action in zip(case["sequence"], expected_actions):
        result = observables.apply_host_observable(case["node"], {}, arrays(step["atoms"]),
                                                   layout=layout, cache=cache)
        assert_tree(result["dipole_moment"], step["output"], atol=1e-9, rtol=1e-8)
        assert cache.last_action == action
