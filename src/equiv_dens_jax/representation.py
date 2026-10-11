"""Current DenSnet spherical representation, with native JAX derivatives."""
import math
import jax.numpy as jnp
from .common import attr, child
from .layers import apply
from .harmonics import spherical_harmonics


def apply_representation(node, state, atoms, **kwargs):
    if node["type"] != "EquivariantSphericalHarmonics":
        raise NotImplementedError(node["type"])
    atoms = dict(atoms)
    R, idx_i, idx_j = atoms["positions"], atoms["idx_i"], atoms["idx_j"]
    displacement = jnp.take(R,idx_j,axis=-2)-jnp.take(R,idx_i,axis=-2)
    distances = jnp.linalg.norm(displacement,axis=-1,keepdims=True)
    directions = displacement/distances
    rbf = apply(child(node,"radial_basis_functions"),state,distances)[...,None,:]
    sph = [y[...,None] for y in spherical_harmonics(attr(node,"order_max"),directions)]
    atoms.update(distances=distances,directions=directions,sph=sph)
    xs = apply(child(node,"embedding"),state,atoms["atom_numbers"])
    fs = [jnp.zeros_like(x) for x in xs]
    order_changes = child(node,"order_change")["children"]
    for i,module in enumerate(child(node,"module")["children"].values()):
        xs = apply(order_changes[str(i)],state,xs)
        xs,ys = apply(module,state,xs,rbf,sph,idx_i,idx_j,neighbor_mask=1)
        for L in range(attr(node,"order")[i]+1):
            # The original checks fs twice, rather than checking ys.
            scale = jnp.where(jnp.mean(fs[L]**2) == 0,1,math.sqrt(0.5)) if attr(node,"normalize",0) else 1
            fs[L] = ys[L]*scale+fs[L]*scale
    if attr(node,"nonmixing_interaction",False):
        fs = apply(child(node,"nonmixing_interaction_block"),state,fs,rbf,sph,idx_i,idx_j,neighbor_mask=1)
    atoms["sph_repr"] = fs
    return atoms
