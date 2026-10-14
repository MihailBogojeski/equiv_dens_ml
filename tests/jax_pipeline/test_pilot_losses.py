"""Batch-independent integrated gradient objective against explicit quadrature."""
import jax
import jax.numpy as jnp
import numpy as np
from equiv_dens_jax.losses import ErrorDict


def test_mean_integrated_gradient_error_is_batch_independent():
    data={"positions":jnp.zeros((1,2,3)),"coord_weights":jnp.array([[.4,.7]]),
          "density":jnp.array([[2.,3.]]),"density_grad":jnp.zeros((1,2,3))}
    prediction={"density_grad":jnp.array([[[3.,4.,0.],[0.,0.,2.]]])}
    loss=ErrorDict({"density_grad":.02},loss_comp={"density_grad":["mean_perc_norm_int"]})
    expected=(5*.4+2*.7)/(2*.4+3*.7)
    for count in (1,5,10):
        batch=jax.tree.map(lambda value:jnp.repeat(value,count,axis=0),data)
        pred=jax.tree.map(lambda value:jnp.repeat(value,count,axis=0),prediction)
        result=loss.compute(pred,batch)
        np.testing.assert_allclose(result["density_grad_mean_perc_norm_int"],expected,rtol=1e-6)
        np.testing.assert_allclose(result["loss"],.02*expected,rtol=1e-6)
        gradient=jax.grad(lambda values:loss.compute({"density_grad":values},batch)["loss"])(pred["density_grad"])
        assert np.isfinite(gradient).all()
