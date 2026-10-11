"""Native LDA/Ewald parity using independently generated current Torch arrays."""
from __future__ import annotations

import gc
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equiv_dens_jax import functionals
from equiv_dens_jax.serialization import load_tree

jax.config.update("jax_enable_x64", True)
REFERENCE = Path(os.environ.get("DENSNET_JAX_FUNCTIONAL_REFERENCE", str(
    Path(__file__).resolve().parents[2] / "scratch/jax-pipeline/functional-reference")))
CASES = ("orthogonal", "triclinic", "minimum_only")


@pytest.fixture(scope="module")
def references():
    if not (REFERENCE / "manifest.json").exists():
        pytest.fail(f"Missing independent Torch functional reference at {REFERENCE}; generate it in the Torch runtime")
    return load_tree(REFERENCE)


@pytest.fixture(autouse=True)
def bounded_cache():
    yield
    jax.clear_caches()
    gc.collect()


def arrays(tree):
    return jax.tree.map(lambda value: jnp.asarray(value) if isinstance(value, np.ndarray) else value, tree)


def assert_tree(actual, expected, *, atol=1e-8, rtol=1e-7):
    observed_leaves, observed_structure = jax.tree.flatten(actual)
    expected_leaves, expected_structure = jax.tree.flatten(expected)
    assert observed_structure == expected_structure
    for observed, reference in zip(observed_leaves, expected_leaves):
        assert np.isfinite(np.asarray(observed)).all()
        assert np.isfinite(np.asarray(reference)).all()
        np.testing.assert_allclose(observed, reference, atol=atol, rtol=rtol)


def evaluator(case):
    node = case["node"]
    grid = arrays(functionals.prepare_grid(case["grid_metadata"], ewald_node=node["children"]["ewald"]))
    base = dict(arrays(case["atoms"]), functional_grid=grid)
    buffers = arrays(case["state"]["buffers"])

    def evaluate(params, variables):
        return functionals.apply_functional(node, {"params": params, "buffers": buffers},
                                            dict(base, **variables))
    return evaluate, grid


@pytest.mark.parametrize("name", CASES)
def test_individual_terms_and_eager_jit_changed_inputs(references, name):
    case = references["cases"][name]
    evaluate, grid = evaluator(case)
    variables, params = arrays(case["variables"]), arrays(case["state"]["params"])
    rho = variables["density"]
    terms = functionals.ewald_terms(case["node"]["children"]["ewald"], grid,
                                   variables["shifted_positions"] / functionals.BOHR)
    terms.update(tf=functionals.thomas_fermi_en(rho, grid), vw=functionals.von_weizsacker_en(rho, grid),
                 lda=functionals.lda_en(rho, grid), hartree=functionals.hartree_en(rho, grid),
                 pseudo=functionals.pseudo_en(rho, grid, arrays(case["atoms"]["v_real"])))
    assert_tree(terms, case["terms"], atol=1e-9, rtol=1e-8)
    def outputs(params, inputs):
        result = evaluate(params, inputs)
        return {key: result[key] for key in ("energy_min", "energy") if key in result}
    expected = {"energy_min": case["energy_min"]}
    if case["node"]["attrs"]["store_energy"]:
        expected["energy"] = case["output"]
    assert_tree(outputs(params, variables), expected, atol=1e-9, rtol=1e-8)
    compiled = jax.jit(outputs)
    assert_tree(compiled(params, variables), expected, atol=1e-9, rtol=1e-8)
    changed = jax.tree.map(lambda value, tangent: value + 0.02 * tangent,
                           variables, arrays(case["directions"]))
    assert_tree(compiled(params, changed), outputs(params, changed), atol=1e-9, rtol=1e-8)
    assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)


@pytest.mark.parametrize("name", CASES)
def test_total_first_derivative_hvp_and_finite_difference(references, name):
    case = references["cases"][name]
    evaluate, _ = evaluator(case)
    params, variables = arrays(case["state"]["params"]), arrays(case["variables"])
    output_key = "energy" if case["node"]["attrs"]["store_energy"] else "energy_min"
    def loss(params, inputs):
        output = evaluate(params, inputs)[output_key]
        return jnp.sum(output**2) + 0.13 * jnp.sum(jnp.cos(output))
    parameter_grad, input_grad = jax.grad(loss, argnums=(0, 1))(params, variables)
    assert_tree(parameter_grad, case["parameter_gradient"])
    assert_tree(input_grad, case["input_gradient"])
    direction = arrays(case["directions"])
    step = 1e-5
    shift = lambda amount: jax.tree.map(lambda value, tangent: value + amount * tangent, variables, direction)
    numeric = (loss(params, shift(step)) - loss(params, shift(-step))) / (2 * step)
    analytic = sum(jnp.sum(gradient * tangent) for gradient, tangent in zip(jax.tree.leaves(input_grad), jax.tree.leaves(direction)))
    assert abs(float(analytic)) > 1e-6
    np.testing.assert_allclose(numeric, analytic, atol=1e-6, rtol=1e-5)
    _, hvp = jax.jvp(lambda inputs: jax.grad(loss, argnums=1)(params, inputs), (variables,), (direction,))
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(hvp))
    if case["input_hvp"] is not None:
        assert_tree(hvp, case["input_hvp"])
    else:
        assert case["limitations"], "A source derivative restriction must be recorded explicitly"
        # Corroborate JAX's smooth extension by a centered gradient difference;
        # this is not a claim of source cdist higher derivative parity.
        numeric_hvp = jax.tree.map(lambda plus, minus: (plus - minus) / (2 * step),
            jax.grad(loss, argnums=1)(params, shift(step)), jax.grad(loss, argnums=1)(params, shift(-step)))
        assert_tree(hvp, numeric_hvp, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("name", CASES)
def test_density_branch_first_and_higher_derivatives(references, name):
    case = references["cases"][name]
    _, grid = evaluator(case)
    potential = arrays(case["atoms"]["v_real"])
    def smooth_density_loss(rho):
        return jnp.sum(functionals.thomas_fermi_en(rho, grid) + functionals.von_weizsacker_en(rho, grid)
                       + functionals.lda_en(rho, grid) + functionals.hartree_en(rho, grid)
                       + functionals.pseudo_en(rho, grid, potential))
    rho = arrays(case["variables"]["density"])
    gradient = jax.grad(smooth_density_loss)(rho)
    _, hvp = jax.jvp(jax.grad(smooth_density_loss), (rho,), (arrays(case["directions"]["density"]),))
    assert_tree(gradient, case["density_branch_gradient"])
    assert_tree(hvp, case["density_branch_hvp"])


def test_host_cold_eta_and_unsupported_grid_domain(references):
    case = references["cases"]["orthogonal"]
    node = case["node"]["children"]["ewald"]
    cold = {**node, "attrs": {**node["attrs"], "eta": None}}
    prepared = functionals.prepare_grid(case["grid_metadata"], ewald_node=cold, initial_eta_dtype=np.float64)
    np.testing.assert_allclose(prepared["eta"], node["attrs"]["eta"], atol=1e-14, rtol=1e-14)
    metadata = {**case["grid_metadata"], "shape": np.asarray([4, 4, 5])}
    with pytest.raises(ValueError, match="even final axis"):
        functionals.prepare_grid(metadata, ewald_node=node)


@pytest.mark.parametrize("name", CASES)
def test_native_host_grid_metadata_matches_actual_source_grid(references, name):
    expected = references["cases"][name]["grid_metadata"]
    actual = functionals.make_grid_metadata(expected["shape"], expected["lattice"])
    np.testing.assert_array_equal(actual["reciprocal_mask"], expected["reciprocal_mask"])
    for key in ("gg", "reciprocal_coords", "volume", "point_volume"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-13, rtol=1e-13)


def test_potential_is_explicit_and_detached(references):
    case = references["cases"]["orthogonal"]
    evaluate, grid = evaluator(case)
    rho, potential = arrays(case["variables"]["density"]), arrays(case["atoms"]["v_real"])
    gradient = jax.grad(lambda field: jnp.sum(functionals.pseudo_en(rho, grid, field)))(potential)
    np.testing.assert_array_equal(gradient, np.zeros_like(potential))
    node = case["node"]
    with pytest.raises(ValueError, match="explicitly prepared"):
        functionals.apply_functional(node, arrays(case["state"]), arrays(case["atoms"]))


def test_host_potential_callback_selection_and_shape():
    class Potential:
        def __init__(self):
            self.calls = []
        def restart(self, *, grid, ions):
            self.current = ions
            self.calls.append(("restart", grid, ions))
        def __call__(self, density):
            self.calls.append(("evaluate", density.copy()))
            self.vreal = density * self.current
    potential = Potential()
    density = np.arange(32, dtype=float).reshape(2, 16) + 1
    atoms = {"density": density, "pseudo_pot": potential, "dftpy_grid": "host-grid",
             "idx": [1, 0], "ions": [2, 3]}
    actual = functionals.prepare_pseudopotential(atoms, shape=(2, 2, 4))
    expected = np.stack([density[1].reshape(2, 2, 4) * 3, density[0].reshape(2, 2, 4) * 2])
    np.testing.assert_array_equal(actual, expected)
    assert potential.calls[0] == ("restart", "host-grid", 3)
    np.testing.assert_array_equal(potential.calls[1][1], density[1].reshape(2, 2, 4))
    assert potential.calls[2] == ("restart", "host-grid", 2)
