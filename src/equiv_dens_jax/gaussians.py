"""JAX versions of the Gaussian density basis functions used by DenSnet."""
from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln

BOHR_ANGSTROM = 0.529177210903
PYSCF_GTO_FACTOR = 2.0 * 3.141592653589793**0.5


def gto_norm(order, width):
    """Historical (not square-root) Gaussian normalization convention."""
    order = jnp.asarray(order, dtype=width.dtype)
    return (jnp.exp(gammaln(order / 2 + 1)) * 2**order
            * width ** ((order + 3) / 2)
            / (jnp.pi**1.5 * jnp.exp(gammaln(order + 2))))


def gaussian_rbf(r, width, scale, order, normalize=False):
    scale_calc = scale * gto_norm(order, width) if normalize else scale / PYSCF_GTO_FACTOR
    r_bohr = r / BOHR_ANGSTROM
    return jnp.sum(scale_calc * r_bohr**order * jnp.exp(-width * r_bohr**2), axis=-2, keepdims=True)


def gaussian_rbf_deriv(r, width, scale, order, normalize=False):
    scale_calc = scale * gto_norm(order, width) if normalize else scale / PYSCF_GTO_FACTOR
    r_bohr = r / BOHR_ANGSTROM
    deriv = (-scale_calc * r_bohr**(order - 1)
             * (2 * width * r_bohr**2 - order) * jnp.exp(-width * r_bohr**2)
             / BOHR_ANGSTROM)
    return jnp.sum(deriv, axis=-2, keepdims=True)
