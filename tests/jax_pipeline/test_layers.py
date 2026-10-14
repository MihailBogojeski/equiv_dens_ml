"""Cross-framework references are generated in a separate Torch process."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from equiv_dens_jax.layers import apply, cutoff_function, layer_norm
from equiv_dens_jax.representation import apply_representation
from equiv_dens_jax.harmonics import spherical_harmonics, spherical_harmonics_deriv

jax.config.update("jax_enable_x64",True)
FIXTURES = Path(os.environ.get("DENSNET_JAX_LAYER_FIXTURES","scratch/jax_pipeline/layers-production"))
CASES = sorted(FIXTURES.glob("*.json"))


def unpack(data, arrays):
    if "array" in data:
        return arrays[data["array"]]
    if "list" in data:
        return [unpack(v,arrays) for v in data["list"]]
    return {k: unpack(v,arrays) for k,v in data["dict"].items()}


def leaves(tree):
    if isinstance(tree,dict):
        return [v for item in tree.values() for v in leaves(item)]
    if isinstance(tree,(list,tuple)):
        return [v for item in tree for v in leaves(item)]
    return [tree]


def replace_floats(tree, floating):
    values = iter(floating)
    def visit(value):
        if isinstance(value,dict):
            return {k: visit(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):
            return [visit(v) for v in value]
        return next(values) if np.issubdtype(value.dtype,np.floating) else jnp.asarray(value)
    return visit(tree)


@pytest.mark.parametrize("path",CASES,ids=lambda p:p.stem)
def test_exact_reference_outputs_and_higher_derivatives(path):
    spec = json.loads(path.read_text())
    assert spec.get("schema") == 1, f"Regenerate the layer reference fixture {path}"
    with np.load(path.with_suffix(".npz")) as arrays:
        state = {g: {k:jnp.asarray(arrays[f"state/{g}/{k}"]) for k in keys} for g,keys in spec["state_keys"].items()}
        inputs = unpack(spec["inputs"],arrays)
        floating = tuple(jnp.asarray(v) for v in leaves(inputs) if np.issubdtype(v.dtype,np.floating))
        node = spec["node"]
        if path.stem == "representation_shared_params":
            def alpha_key(index):
                return node["children"]["module"]["children"][str(index)]["children"]["residual_post_x"]["children"]["stack"]["children"]["0"]["children"]["activation_pre"]["params"]["alpha"]
            assert alpha_key(0) == alpha_key(1)
        def evaluate(values):
            current = replace_floats(inputs,values)
            if spec["representation"]:
                return apply_representation(node,state,current[0])["sph_repr"]
            return apply(node,state,*current)
        expected = unpack(spec["outputs"],arrays)
        actual = evaluate(floating)
        compiled = jax.jit(evaluate)(floating)
        assert len(leaves(expected)) == len(leaves(actual)) == len(leaves(compiled))
        for ref,eager,jitted in zip(leaves(expected),leaves(actual),leaves(compiled)):
            assert eager.shape == ref.shape == jitted.shape
            np.testing.assert_allclose(eager,ref,atol=1e-9,rtol=1e-8)
            np.testing.assert_allclose(jitted,ref,atol=1e-9,rtol=1e-8)
        changed = tuple(v+0.005*jnp.sin(jnp.arange(v.size).reshape(v.shape)+1) for v in floating)
        for eager,jitted in zip(leaves(evaluate(changed)),leaves(jax.jit(evaluate)(changed))):
            np.testing.assert_allclose(eager,jitted,atol=1e-9,rtol=1e-8)
        if "grads" in spec:
            loss = lambda values: sum(jnp.sum(v*v) for v in leaves(evaluate(values)))
            gradient = jax.jit(jax.grad(loss))
            actual_grad = gradient(floating)
            expected_grad = unpack(spec["grads"],arrays)
            assert len(actual_grad) == len(expected_grad)
            for result,ref in zip(actual_grad,expected_grad):
                np.testing.assert_allclose(result,ref,atol=1e-8,rtol=1e-7)
            directions = tuple(jnp.asarray(v) for v in unpack(spec["directions"],arrays))
            hess_times_direction = jax.jvp(gradient,(floating,),(directions,))[1]
            second = sum(jnp.sum(h*d) for h,d in zip(hess_times_direction,directions))
            np.testing.assert_allclose(second,arrays["second"],atol=1e-8,rtol=1e-7)
        if "parameter_grads" in spec:
            keys = tuple(state["params"])
            weights = tuple(state["params"][k] for k in keys)
            def parameter_loss(weights,values):
                current_state = {"params":dict(zip(keys,weights)),"buffers":state["buffers"]}
                current = replace_floats(inputs,values)
                result = (apply_representation(node,current_state,current[0])["sph_repr"]
                          if spec["representation"] else apply(node,current_state,*current))
                return sum(jnp.sum(v*v) for v in leaves(result))
            parameter_gradient = jax.jit(jax.grad(parameter_loss,argnums=0))
            actual_param = parameter_gradient(weights,floating)
            expected_param = unpack(spec["parameter_grads"],arrays)
            assert len(actual_param) == len(expected_param)
            for result,ref in zip(actual_param,expected_param):
                np.testing.assert_allclose(result,ref,atol=1e-8,rtol=1e-7)
            directions = tuple(jnp.asarray(v) for v in unpack(spec["directions"],arrays))
            mixed = jax.jvp(lambda values:parameter_gradient(weights,values),(floating,),(directions,))[1]
            expected_mixed = unpack(spec["mixed_grads"],arrays)
            assert len(mixed) == len(expected_mixed)
            for result,ref in zip(mixed,expected_mixed):
                np.testing.assert_allclose(result,ref,atol=1e-8,rtol=1e-7)
        # Release per-tree compilations; these are qualification, not timing tests.
        jax.clear_caches()


def test_required_reference_fixtures_present():
    assert CASES, (
        f"No Torch layer references found in {FIXTURES}; run "
        "tests/jax_pipeline/generate_layer_fixtures.py in the separate Torch runtime."
    )
    assert (FIXTURES/"harmonics.npz").is_file()
    for path in CASES:
        spec = json.loads(path.read_text())
        assert spec.get("schema") == 1, f"Unsupported/missing reference schema: {path}"
        assert path.with_suffix(".npz").is_file(), f"Missing reference arrays: {path}"
        assert set(spec["state_keys"]) == {"params","buffers"}
        assert {"type","path","attrs","children","params","buffers"} <= set(spec["node"])
        assert {"inputs","outputs","grads","directions"} <= set(spec)


def test_harmonic_orders_zero_through_ten():
    with np.load(FIXTURES/"harmonics.npz") as arrays:
        values = jax.jit(lambda u: spherical_harmonics(10,u))(jnp.asarray(arrays["u"]))
        for L,value in enumerate(values):
            np.testing.assert_allclose(value,arrays[f"Y{L}"],atol=1e-9,rtol=1e-8)
        if "dY0" in arrays:
            gradients = jax.jit(lambda u:spherical_harmonics_deriv(5,u))(jnp.asarray(arrays["u"]))
            for L,value in enumerate(gradients):
                np.testing.assert_allclose(value,arrays[f"dY{L}"],atol=1e-9,rtol=1e-8)


def test_zero_normalization_and_cutoff_derivatives_are_finite():
    zeros = jnp.zeros((2,3,1,4))
    np.testing.assert_array_equal(jax.jit(layer_norm)(zeros),zeros)
    assert jnp.all(jnp.isfinite(jax.grad(lambda x:jnp.sum(layer_norm(x)))(zeros)))
    fn = lambda x:cutoff_function(x,3.)
    for x in (0.,2.9,3.,3.5):
        assert np.isfinite(jax.grad(fn)(x))
        assert np.isfinite(jax.grad(jax.grad(fn))(x))


def test_shifted_softplus_zero_beta_and_large_inputs():
    node = {"type":"ShiftedSoftplus","path":"probe","attrs":{},"children":{},
            "params":{"alpha":"alpha","beta":"beta"},"buffers":{}}
    state = {"params":{"alpha":jnp.array([2.,1.,1.]),"beta":jnp.array([0.,1.,1.])},"buffers":{}}
    x = jnp.array([3.,1000.,-1000.])
    result = jax.jit(lambda x:apply(node,state,x))(x)
    np.testing.assert_allclose(result,jnp.array([3.,1000.-np.log(2),-np.log(2)]),atol=1e-12)
    np.testing.assert_allclose(jax.grad(lambda x:jnp.sum(apply(node,state,x)))(x),jnp.array([1.,1.,0.]),atol=1e-12)


def test_jax_execution_does_not_import_torch():
    assert "torch" not in sys.modules
