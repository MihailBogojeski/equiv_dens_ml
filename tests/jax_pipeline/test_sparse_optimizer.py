"""Sparse optimizer state agrees with separate Torch None-gradient traces."""
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equiv_dens_jax.serialization import load_tree
from equiv_dens_jax.training import Optimizer,_update,active_parameters

jax.config.update("jax_enable_x64",True)


@pytest.mark.parametrize("index",range(8))
def test_sparse_optimizer_seven_updates(index):
    path = Path(os.environ.get("DENSNET_JAX_SPARSE_OPTIMIZER_REFERENCE","scratch/jax_pipeline/sparse-optimizers"))
    if not (path/"manifest.json").exists():
        pytest.fail("Generate independent sparse Torch optimizer references first")
    reference = load_tree(path)
    case = reference["cases"][index]
    params = jax.tree.map(jnp.asarray,reference["initial"])
    zeros = jax.tree.map(jnp.zeros_like,params)
    state = {"count":jnp.asarray(0),"param_count":{name:jnp.asarray(0) for name in params},
             "m":zeros,"v":zeros,"vmax":zeros,"slow":params}
    config = Optimizer(**case["config"])
    update = jax.jit(lambda p,g,s:_update(p,g,s,config))
    for position,expected in enumerate(case["updates"]):
        previous = state
        updated,state = update(params,jax.tree.map(jnp.asarray,expected["gradients"]),state)
        params = {**params,**updated}
        assert int(state["count"]) == position+1
        for group in ("param_count","m","v","vmax","slow"):
            assert state[group].keys() == reference["initial"].keys()
        for name,value in expected["params"].items():
            np.testing.assert_allclose(params[name],value,atol=1e-12,rtol=1e-12,err_msg=f"{config} update{position} {name}")
            fast = expected["fast_state"][name]
            if name not in expected["gradients"]:
                for group in ("param_count","m","v","vmax"):
                    np.testing.assert_array_equal(state[group][name],previous[group][name])
            if config.name != "sgd":
                assert int(state["param_count"][name]) == int(fast.get("step",0))
            for native,source in (("m","momentum_buffer" if config.name=="sgd" else "exp_avg"),
                                  ("v","exp_avg_sq"),("vmax","max_exp_avg_sq")):
                if source in fast:
                    np.testing.assert_allclose(state[native][name],fast[source],atol=1e-12,rtol=1e-12)
            if config.lookahead_k:
                np.testing.assert_allclose(state["slow"][name],expected["slow"][name],atol=1e-12,rtol=1e-12)
    assert "torch" not in sys.modules
    jax.clear_caches()


def test_core_refinement_freezes_source_density_and_explicit_metadata():
    params = {"density_repr_model.weight":jnp.ones(1),"property_models.core_density.weight":jnp.ones(1),
              "property_models.energy.weight":jnp.ones(1)}
    core = {"type":"DensityCoeffsNetwork","attrs":{"core_basis_ratio":.5},"children":{}}
    model = {"children":{"property_models":{"children":{"core_density":core}}},"attrs":{}}
    assert active_parameters(params,"density",model=model) == ("property_models.core_density.weight",)
    assert set(active_parameters(params,"joint",model=model)) == {"property_models.core_density.weight","property_models.energy.weight"}
    core["attrs"]["core_basis_ratio"] = 0
    model["children"]["property_models"]["children"]["energy"] = {
        "type":"Linear","attrs":{},"children":{},"params":{"weight":"property_models.energy.weight"},
        "trainable":{"weight":False}}
    assert set(active_parameters(params,"joint",model=model)) == {"density_repr_model.weight","property_models.core_density.weight"}
