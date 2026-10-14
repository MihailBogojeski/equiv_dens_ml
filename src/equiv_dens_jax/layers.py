"""Native JAX evaluation of the current DenSnet numerical modules.

Static module trees are closed over by compiled functions. Historical parity,
normalization, and nonmixing scale conventions deliberately match Torch.
"""
from __future__ import annotations

import math
import jax.nn as jnn
import jax.numpy as jnp
from .common import attr, child, parameter, buffer, linear


def softplus(x):
    # Torch F.softplus uses its linear branch above the default threshold 20.
    return jnp.where(x > 20, x, jnn.softplus(x))


def shifted_softplus(x):
    return softplus(x)-math.log(2)


def softplus_inverse(x):
    x = jnp.asarray(x)
    return x+jnp.log(-jnp.expm1(-x))


def cutoff_function(x, cutoff):
    inside = x < cutoff
    safe = jnp.where(inside, x, 0)
    return jnp.where(inside, jnp.exp(-safe**2 / ((cutoff-safe)*(cutoff+safe))), 0)


def switch_function(x, cuton, cutoff):
    x = (x-cuton)/(cutoff-cuton)
    component = lambda v: jnp.where(v <= 0, 0, jnp.exp(-1 / jnp.where(v <= 0, 1, v)))
    fp, fm = component(x), component(1-x)
    return jnp.where(x <= 0, 1, jnp.where(x >= 1, 0, fm/(fp+fm)))


def layer_norm(x, dims=(-3, -2, -1)):
    mean = jnp.mean(x, axis=dims, keepdims=True)
    var = jnp.var(x, axis=dims, keepdims=True, ddof=1)
    # Keep the global Torch zero-variance branch and avoid sqrt(0) AD NaNs.
    nonzero = jnp.sum(var) != 0
    std = jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, var, 1)), 0)
    return (x-mean)/(std+1e-8)


def residual_scale(x, y, normalize):
    if not normalize:
        return 1
    return jnp.where((jnp.mean(x**2) == 0) | (jnp.mean(y**2) == 0), 1, math.sqrt(0.5))


def _zeros(x, L):
    return jnp.zeros((*x.shape[:-2], 2*L+1, x.shape[-1]), dtype=x.dtype)


def _cg(node, state, masked=False, dtype=None):
    if "clebsch_gordan" not in node["children"]:
        return jnp.ones((1, 1, 1), dtype=dtype)
    return buffer(child(node, "clebsch_gordan"), state, "cg_masked" if masked else "cg_matrix")


def _contract(cg, x1, x2, l1, l2, L):
    block = cg[l1*l1:(l1+1)**2, l2*l2:(l2+1)**2, L*L:(L+1)**2]
    # Same product and sequential sums as Torch, retaining its reduction order.
    tp = x1[..., :, None, None, :] * x2[..., None, :, None, :]
    return jnp.sum(jnp.sum(tp * block[..., None], axis=-3), axis=-3)


def _self_mixing(node, state, xs):
    oi, oo = attr(node,"order_in"), attr(node,"order_out")
    ys = [parameter(node,state,f"keepcoeff_{L}")*xs[L] if L <= oi else _zeros(xs[0],L) for L in range(oo+1)]
    cg = _cg(node,state,masked=True,dtype=xs[0].dtype)
    for l1 in range(oi+1):
        for l2 in range(l1+1,oi+1):
            for L in range(abs(l1-l2),min(l1+l2,oo)+1):
                if attr(node,"parity",False) and l1 + l2 % 2 != L % 2:
                    continue
                ys[L] = ys[L] + parameter(node,state,f"mixcoeff_{l1}_{l2}_{L}") * _contract(cg,xs[l1],xs[l2],l1,l2,L)
    return ys


def _pair(node,state,x1s,x2s,rbf):
    nonmix = node["type"] == "PairInteraction"
    oo = attr(node,"order") if nonmix else attr(node,"order_out")
    ys = [_zeros(x1s[0],L) for L in range(oo+1)]
    cg = _cg(node,state,dtype=x1s[0].dtype)
    if nonmix:
        triples = [(L,L,L) for L in range(oo+1)]
    else:
        triples = [(l1,l2,L) for l1 in range(attr(node,"order_in1")+1)
                   for l2 in range(attr(node,"order_in2")+1)
                   for L in range(abs(l1-l2),min(l1+l2,oo)+1)
                   if not (attr(node,"parity",False) and l1 + l2 % 2 != L % 2)]
    counts = attr(node,"L_count")
    for l1,l2,L in triples:
        norm = math.sqrt(L+1)*math.sqrt(attr(node,"num_features")/attr(node,"num_basis_functions")) if attr(node,"normalize",0) else 1
        coeff = child(node,f"coeff_{L}" if nonmix else f"coeff_{l1}_{l2}_{L}")
        count = counts[L] if counts is not None else sum(t[2] == L for t in triples)
        ys[L] = ys[L] + apply(coeff,state,rbf)*norm*(_contract(cg,x1s[l1],x2s[l2],l1,l2,L)/math.sqrt(count))
    return ys


def _radial(node,state,r):
    kind = node["type"]
    cutoff = buffer(node,state,"cutoff")
    if kind in ("GaussianRadialBasisFunctions","ExponentialGaussianRadialBasisFunctions"):
        x = r if kind == "GaussianRadialBasisFunctions" else jnp.exp(-softplus(parameter(node,state,"_alpha"))*r)
        return cutoff_function(r,cutoff)*jnp.exp(-buffer(node,state,"width")*(x-buffer(node,state,"center"))**2)
    if kind == "BernsteinRadialBasisFunctions":
        # Preserve the original flattening, including its batch-layout limitation.
        x = jnp.log(r.reshape((-1,1))/cutoff)
    elif kind == "ExponentialBernsteinRadialBasisFunctions":
        x = -softplus(parameter(node,state,"_alpha"))*r
    elif kind == "OverlapBernsteinRadialBasisFunctions":
        ar = softplus(parameter(node,state,"_alpha"))*r
        x = jnp.log1p(ar)-ar
    else:
        raise NotImplementedError(kind)
    exponent = buffer(node,state,"logc") + buffer(node,state,"n")*x + buffer(node,state,"v")*jnp.log(-jnp.expm1(x))
    return cutoff_function(r,cutoff)*jnp.exp(exponent)*attr(node,"norm_factor",1)


def _interaction(node,state,xs,rbf,sph,idx_i,idx_j,neighbor_mask=1):
    xs = list(xs)
    run = lambda name,*x: apply(child(node,name),state,*x)
    nonmix = node["type"] == "NonmixingInteractionBlock"
    norm = attr(node,"normalize",0)
    if nonmix:
        yi = [jnp.zeros_like(x) for x in xs]
        mixing_order = attr(node,"order")
    else:
        yi = run("residual_pre_vi",xs)
        yi[0] = run("activation_i",yi[0])
        if norm > 1:
            yi = [layer_norm(y) for y in yi]
        yi = run("linear_i",yi)
        mixing_order = attr(node,"mixing_order")
        yi.extend(_zeros(yi[0],L) for L in range(len(yi),mixing_order+1))
    yj = run("residual_pre_vj",xs)
    yj[0] = run("activation_j",yj[0])
    if norm == 1:
        yj[0] = layer_norm(yj[0])
    elif norm > 1:
        yj = [layer_norm(y) for y in yj]
    yj = run("linear_j",yj)
    gather_order = attr(node,"order") if nonmix else min(2*attr(node,"input_order"),attr(node,"order"))
    for L in range(gather_order+1):
        yj[L] = jnp.take(yj[L],idx_j,axis=1)*neighbor_mask
    vs = run("mixing",yj,run("angular_fn1",sph),rbf)
    angular = run("angular_fn2",sph)
    radial = child(node,"radial_fn")["children"]
    norm_rbf = math.sqrt(attr(node,"num_features")/attr(node,"num_basis_functions")) if norm else 1
    for L in range(mixing_order+1):
        scale = residual_scale(yi[L],vs[L],norm)
        message = scale*(vs[L]+apply(radial[str(L)],state,rbf)*norm_rbf*angular[L]*yj[0])
        vs[L] = (yi[L]*scale).at[:,idx_i,:,:].add(message)
        if norm:
            vs[L] = vs[L]*math.sqrt(mixing_order+1)/attr(node,"num_neighbours",1)
    if nonmix:
        vs = run("linear_v",vs)
        if not attr(node,"residual",True):
            return vs
    else:
        if norm > 1:
            vs = [layer_norm(v) for v in vs]
        if mixing_order != attr(node,"order"):
            vs = run("linear_contract",vs)
    vs = run("residual_post_v",vs)
    vs[0] = run("activation_v",vs[0])
    vs = run("linear_v",vs)
    xs.extend(jnp.zeros_like(vs[L]) for L in range(len(xs),len(vs)))
    return [x+v for x,v in zip(xs,vs)]


def apply(node, state, *inputs, **kwargs):
    """Evaluate a converted module without importing the Torch package."""
    kind = node["type"]
    if kind == "Linear":
        return linear(node,state,inputs[0])
    if kind == "Identity":
        return inputs[0]
    if kind in ("Sequential","ModuleList"):
        x = inputs[0]
        for sub in node["children"].values():
            x = apply(sub,state,x)
        return x
    if kind == "Swish":
        x = inputs[0]
        return parameter(node,state,"alpha")*x*jnn.sigmoid(parameter(node,state,"beta")*x)
    if kind == "ShiftedSoftplus":
        x = inputs[0]
        beta = parameter(node,state,"beta")
        safe_beta = jnp.where(beta != 0,beta,1)
        return parameter(node,state,"alpha")*jnp.where(beta != 0,(softplus(beta*x)-math.log(2))/safe_beta,0.5*x)
    if kind.endswith("RadialBasisFunctions"):
        return _radial(node,state,inputs[0])
    if kind == "Embedding":
        config = apply(child(node,"config_linear"),state,buffer(node,state,"electron_config"))
        return (parameter(node,state,"element_embedding")+config)[inputs[0]]
    if kind == "SphericalEmbedding":
        Z = inputs[0]
        scalar = apply(child(node,"embedding"),state,Z)[...,None,:]
        return [scalar]+[_zeros(scalar,L) for L in range(1,attr(node,"order")+1)]
    if kind == "ClebschGordanMatrix":
        slices = tuple(slice(None,(int(L)+1)**2) for L in inputs)
        return buffer(node,state,"cg_matrix")[slices],buffer(node,state,"cg_masked")[slices]
    if kind == "ClebschGordan":
        return buffer(node,state,"cg_{}_{}_{}".format(*inputs))
    if kind == "SelfMixing":
        return _self_mixing(node,state,inputs[0])
    if kind in ("PairMixing","PairInteraction"):
        return _pair(node,state,*inputs)
    if kind == "SphericalLinear":
        xs = inputs[0]
        if attr(node,"mix_orders",True):
            xs = apply(child(node,"mixing"),state,xs)
            scale = 1
        else:
            scale = math.sqrt(attr(node,"num_out"))
        return [apply(sub,state,x)*scale for x,sub in zip(xs,child(node,"linear")["children"].values())]
    if kind == "ResidualStack":
        xs = list(inputs[0])
        for block in child(node,"stack")["children"].values():
            xs = apply(block,state,xs)
        return xs
    if kind == "ResidualBlock":
        xs = list(inputs[0])
        ys = list(xs)
        ys[0] = apply(child(node,"activation_pre"),state,ys[0])
        ys = apply(child(node,"linear1"),state,ys)
        ys[0] = apply(child(node,"activation_post"),state,ys[0])
        if attr(node,"normalize",0) > 1:
            ys = [layer_norm(y) for y in ys]
        ys = apply(child(node,"linear2"),state,ys)
        xs.extend(jnp.zeros_like(ys[L]) for L in range(len(xs),len(ys)))
        return [(x+y)*residual_scale(x,y,attr(node,"normalize",0)) for x,y in zip(xs,ys)]
    if kind in ("InteractionBlock","NonmixingInteractionBlock"):
        return _interaction(node,state,*inputs,**kwargs)
    if kind == "ModularBlock":
        xs,*rest = inputs
        xs = apply(child(node,"residual_pre_x"),state,xs)
        xs = apply(child(node,"interaction"),state,xs,*rest,**kwargs)
        xs = apply(child(node,"residual_post_x"),state,xs)
        return xs,apply(child(node,"residual_out"),state,xs)
    raise NotImplementedError(f"Unsupported native JAX layer {kind!r} at {node.get('path','')}")
