"""Loss equations and multistep optimizer trajectories against Torch."""
from pathlib import Path
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equiv_dens_jax.serialization import load_tree
from equiv_dens_jax.losses import ErrorDict
from equiv_dens_jax.training import Optimizer,_update

jax.config.update("jax_enable_x64",True)


@pytest.fixture(scope="module")
def reference():
    path=Path(os.environ.get("DENSNET_JAX_TRAINING_REFERENCE","scratch/jax_pipeline/training"))
    if not (path/"manifest.json").exists():
        pytest.fail("Generate independent Torch training references first")
    return jax.tree.map(lambda x:jnp.asarray(x) if isinstance(x,np.ndarray) else x,load_tree(path))


def test_all_loss_components(reference):
    for case in reference["loss_cases"]:
        loss=ErrorDict(**case["config"])
        def evaluate(predictions):
            errors=loss.compute(predictions,reference["data"],pair_indices=reference["pairs"])
            return errors["loss"],errors
        (_,actual),grads=jax.jit(jax.value_and_grad(evaluate,has_aux=True))(reference["predictions"])
        for name,expected in case["errors"].items():
            np.testing.assert_allclose(actual[name],expected,atol=1e-9,rtol=1e-8,err_msg=str(case["config"]))
        for name,expected in case["gradients"].items():
            np.testing.assert_allclose(grads[name],jnp.zeros_like(grads[name]) if expected is None else expected,
                                       atol=1e-8,rtol=1e-7,err_msg=f"{case['config']} {name}")


@pytest.mark.parametrize("case_index",range(8))
def test_optimizer_trajectory(reference,case_index):
    case=reference["optimizer_cases"][case_index]
    params=reference["initial"]
    zeros=jax.tree.map(jnp.zeros_like,params)
    state={"count":jnp.asarray(0),"m":zeros,"v":zeros,"vmax":zeros,"slow":params}
    config=Optimizer(**case["config"])
    update=jax.jit(lambda p,g,s:_update(p,g,s,config))
    for expected in case["updates"]:
        params,state=update(params,expected["gradients"],state)
        for name in params:
            np.testing.assert_allclose(params[name],expected["params"][name],atol=1e-12,rtol=1e-12,
                                       err_msg=f"{case['config']} {name}")
