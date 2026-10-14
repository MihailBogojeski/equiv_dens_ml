"""Torch None means skipped; a connected zero gradient still receives decay."""
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from equiv_dens_jax.parameter_usage import used_parameters
from equiv_dens_jax.serialization import load_tree
from equiv_dens_jax.pipeline import Pipeline
from equiv_dens_jax.losses import ErrorDict
from equiv_dens_jax.training import Optimizer, _update

jax.config.update("jax_enable_x64",True)


def test_unused_stopped_and_zero_connected_parameters():
    params = {name:jnp.ones((2,)) for name in ("used","unused","stopped","zero_connected","literal_zero")}
    objective = lambda p,data:jnp.sum(p["used"]**2+p["zero_connected"]*data+p["literal_zero"]*0+ jax.lax.stop_gradient(p["stopped"]))
    assert set(used_parameters(objective,params,jnp.asarray(0.))) == {"used","zero_connected","literal_zero"}
    assert set(used_parameters(objective,params,jnp.asarray(1.))) == {"used","zero_connected","literal_zero"}


def test_nested_jit_keeps_distinct_output_connectivity():
    nested = jax.jit(lambda used,unused:(used**2,unused*3))
    params = {"used":jnp.ones((2,)),"unused":jnp.ones((2,))}
    objective = lambda p,data:jnp.sum(nested(p["used"],p["unused"])[0])
    assert used_parameters(objective,params,{}) == ("used",)


def test_dependency_trace_inside_outer_jit_and_connected_zero_decay():
    params = {"unused":jnp.array([2.]),"connected_zero":jnp.array([3.])}
    objective = lambda p,data:jnp.sum(p["connected_zero"]*0)
    @jax.jit
    def step(parameters,data):
        names = used_parameters(objective,parameters,data)
        selected = {name:parameters[name] for name in names}
        gradients = jax.grad(lambda p:objective(p,data))(parameters)
        gradients = {name:gradients[name] for name in names}
        zeros = jax.tree.map(jnp.zeros_like,selected)
        state = {"count":jnp.asarray(0),"m":zeros,"v":zeros,"vmax":zeros,"slow":selected}
        updated,_ = _update(selected,gradients,state,Optimizer(name="sgd",learning_rate=.1,
                           weight_decay=.2,selective_weight_decay=False,lookahead_k=0))
        result = dict(parameters)
        result.update(updated)
        return result
    result = step(params,{})
    assert result["unused"][0] == params["unused"][0]
    assert result["connected_zero"][0] == params["connected_zero"][0]*(1-.1*.2)


def test_complete_pipeline_matches_all_torch_none_gradients():
    path = Path(os.environ.get("DENSNET_JAX_PIPELINE_REFERENCE","scratch/jax_pipeline/pipeline-tiny"))
    if not (path/"manifest.json").is_file():
        pytest.fail("Generate independent complete Torch parameter-gradient references first")
    reference = load_tree(path)
    pipeline = Pipeline(reference["model"],reference["state"])
    atoms = pipeline.prepare(reference["inputs"])
    loss = ErrorDict(**reference["loss_config"])
    def objective(params,data):
        prediction = pipeline.forward({"params":params,"buffers":pipeline.state["buffers"]},data,
                                      training=True,properties=("energy","density"),forces=True)
        return loss.compute(prediction,data)["loss"]
    actual = set(used_parameters(objective,pipeline.state["params"],atoms))
    expected = {name for name,gradient in reference["gradients"].items() if gradient is not None}
    assert actual == expected, {"false_includes":actual-expected,"false_excludes":expected-actual}
