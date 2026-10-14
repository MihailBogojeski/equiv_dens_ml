"""Independent source conventions at exact matches and zero vectors.

Torch abs(0) and vector_norm(0) have first derivative zero. RMSE expressed
as sqrt(MSE) retains its source undefined zero derivative; it is not repaired.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equiv_dens_jax.losses import ErrorDict
from equiv_dens_jax.training import Optimizer, _update

jax.config.update("jax_enable_x64",True)


FIELDS = {"density":(2,4),"density_grad":(2,4,3),"energy":(2,1),
          "forces":(1,5,3),"dipole_moment":(2,3),"df_coeffs":(2,5)}


@pytest.mark.parametrize("field",FIELDS)
def test_perfect_match_mae_has_zero_gradient_and_compiled_update(field):
    target = jnp.arange(np.prod(FIELDS[field]),dtype=jnp.float64).reshape(FIELDS[field])+1
    data = {"positions":jnp.zeros((1,5,3)),field:target,"coord_weights":jnp.ones((2,4)),
            "coords":jnp.arange(24,dtype=jnp.float64).reshape(2,4,3)}
    loss = ErrorDict({field:1.},loss_comp={field:["mae"]})
    objective = lambda value:loss.compute({field:value},data)["loss"]
    value,gradient = jax.jit(jax.value_and_grad(objective))(target)
    np.testing.assert_array_equal(value,0)
    np.testing.assert_array_equal(gradient,jnp.zeros_like(target))
    zeros = {"value":jnp.zeros_like(target)}
    state = {"count":jnp.asarray(0),"m":zeros,"v":zeros,"vmax":zeros,
             "slow":{"value":target}}
    optimizer = Optimizer(name="sgd",learning_rate=.2,lookahead_k=0)
    update = jax.jit(lambda parameter,opt_state:_update(
        {"value":parameter},{"value":jax.grad(objective)(parameter)},opt_state,optimizer))
    updated,_ = update(target,state)
    np.testing.assert_array_equal(updated["value"],target)
    assert "torch" not in sys.modules


@pytest.mark.parametrize("component",["norm_int","perc_norm_int","kinetic_vw"])
def test_zero_density_gradient_vector_has_zero_first_derivative(component):
    zero = jnp.zeros((2,4,3))
    data = {"positions":jnp.zeros((1,5,3)),"density_grad":zero,
            "density":jnp.ones((2,4)),"coord_weights":jnp.ones((2,4))}
    loss = ErrorDict({"density_grad":1.},loss_comp={"density_grad":[component]})
    objective = lambda value:loss.compute({"density_grad":value,"density":data["density"]},data)["loss"]
    value,gradient = jax.jit(jax.value_and_grad(objective))(zero)
    np.testing.assert_array_equal(value,0)
    np.testing.assert_array_equal(gradient,zero)


def test_energy_zero_decay_override_uses_source_common_decay():
    parameter = jnp.array([[2.,-3.]])
    params = {"property_models.energy.weight":parameter}
    zeros = jax.tree.map(jnp.zeros_like,params)
    state = {"count":jnp.asarray(0),"m":zeros,"v":zeros,"vmax":zeros,"slow":params}
    config = Optimizer(name="sgd",learning_rate=.1,weight_decay=.2,
                       energy_weight_decay=0.,lookahead_k=0)
    updated,_ = jax.jit(lambda p,s:_update(p,zeros,s,config))(params,state)
    # Torch routes energy weights to its common group when en_weight_decay==0.
    np.testing.assert_allclose(updated["property_models.energy.weight"],parameter*(1-.1*.2),atol=1e-15)


def test_abs_nan_propagates_instead_of_masking_invalid_loss():
    data = {"positions":jnp.zeros((1,1,3)),"energy":jnp.zeros((1,1))}
    loss = ErrorDict({"energy":1.},loss_comp={"energy":["mae"]})
    assert jnp.isnan(loss.compute({"energy":jnp.full((1,1),jnp.nan)},data)["loss"])


@pytest.mark.parametrize("component",["dpm_loss","dpm_abs_loss","coulomb","lda_mae","hartree_mae"])
def test_perfect_density_match_absolute_moment_losses(component):
    target = jnp.array([[1.,2.,3.,4.]])
    data = {"positions":jnp.zeros((1,1,3)),"density":target,
            "coord_weights":jnp.ones((1,4)),
            "coords":jnp.arange(12,dtype=jnp.float64).reshape(1,4,3)}
    loss = ErrorDict({"density":1.},loss_comp={"density":[component]})
    pairs = (jnp.array([0,1,2,3]),jnp.array([1,2,3,0]))
    # Keep both reductions dynamic: constant-folded vs runtime sqrt reduction
    # can differ by one ULP and would test a nonzero difference instead.
    objective = lambda value,reference:loss.compute(
        {"density":value},dict(data,density=reference),pair_indices=pairs)["loss"]
    value,gradient = jax.jit(jax.value_and_grad(objective,argnums=0))(target,target)
    np.testing.assert_array_equal(value,0)
    np.testing.assert_array_equal(gradient,jnp.zeros_like(target))


def test_source_sqrt_mse_zero_derivative_remains_undefined():
    zero = jnp.zeros((1,1))
    data = {"positions":jnp.zeros((1,1,3)),"energy":zero}
    loss = ErrorDict({"energy":1.},loss_comp={"energy":["rmse"]})
    gradient = jax.jit(jax.grad(lambda value:loss.compute({"energy":value},data)["loss"]))(zero)
    assert np.isnan(np.asarray(gradient)).all()
