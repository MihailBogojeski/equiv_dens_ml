"""Fresh native JAX parameters from a portable current-architecture template.

Only parameter shapes/dtypes and scientific buffers are taken from the template;
trained parameter values are never reused. Source initialization distributions
and parent overrides are preserved, with independently seeded JAX randomness.
"""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp
from .common import attr, child


class InitializationRuleConflict(ValueError):
    """A tied parameter has incompatible source initialization rules."""


def _merge(target, additions):
    for key, rule in additions.items():
        if key in target and target[key] != rule:
            raise InitializationRuleConflict(
                f"Shared parameter {key!r} has incompatible initialization rules "
                f"{target[key]!r} and {rule!r}; specify an architecture with an "
                "unambiguous tied-parameter initialization.")
        target[key] = rule


def _rules(node, template):
    rules = {}
    for sub in node["children"].values():
        _merge(rules, _rules(sub, template))
    kind = node["type"]
    params = node["params"]
    local = {}
    if kind == "Linear":
        fan_in = template["params"][params["weight"]].shape[1]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        local = {key:("uniform",-bound,bound) for key in params.values()}
    elif kind in ("Swish","ShiftedSoftplus"):
        _merge(local,{params["alpha"]:("constant",attr(node,"initial_alpha",1.))})
        _merge(local,{params["beta"]:("constant",attr(node,"initial_beta",1.702 if kind == "Swish" else 1.))})
    elif kind == "Embedding":
        local[params["element_embedding"]] = ("uniform",-math.sqrt(3),math.sqrt(3))
        rules[child(node,"config_linear")["params"]["weight"]] = ("orthogonal",)
    elif kind == "SelfMixing":
        oi, oo = attr(node,"order_in"), attr(node,"order_out")
        triples = [(l1,l2,L) for l1 in range(oi+1) for l2 in range(l1+1,oi+1)
                   for L in range(abs(l1-l2),min(l1+l2,oo)+1)
                   if not (attr(node,"parity",False) and l1 + l2 % 2 != L % 2)]
        counts = [1+sum(t[2] == L for t in triples) for L in range(oo+1)]
        for name, key in params.items():
            if not name.startswith(("mixcoeff_","keepcoeff_")):
                raise ValueError(f"Unsupported SelfMixing parameter {name!r}")
            L = int(name.rsplit("_",1)[1])
            norm = L+1 if attr(node,"normalize",0) else 1
            bound = math.sqrt(3*norm/counts[L])
            _merge(local,{key:("uniform",-bound,bound)})
    elif kind in ("ExponentialGaussianRadialBasisFunctions",
                  "ExponentialBernsteinRadialBasisFunctions",
                  "OverlapBernsteinRadialBasisFunctions"):
        alpha = attr(node,"ini_alpha",0.5)
        if alpha <= 0:
            raise ValueError("Current positive radial alpha initialization requires ini_alpha > 0")
        local[params["_alpha"]] = ("constant",alpha+math.log(-math.expm1(-alpha)))
    elif kind == "CoeffsIntegralConstraint":
        if "integral_scale" in params:
            local[params["integral_scale"]] = ("constant",1.)
    elif kind == "LDAFunctional":
        for name,key in params.items():
            if name != "en_offset":
                raise ValueError(f"Unsupported LDA parameter {name!r} at {node['path']}")
            local[key] = ("constant",0.)
    elif kind in ("SphericalHarmonicsEnergyNetwork","SphericalLinearEnergyNetwork",
                  "SphericalHarmonicsEmbeddingEnergyNetwork"):
        for name,key in params.items():
            if not name.startswith(("radial_scale_filters_","radial_width_filters_")):
                raise ValueError(f"Unsupported energy parameter {name!r} at {node['path']}")
            local[key] = ("constant",1.)
    elif params:
        raise ValueError(f"No source initialization contract for {kind!r} at {node['path']}")
    _merge(rules,local)
    # Parents reset selected child parameters after the children's construction.
    if kind == "SphericalLinear":
        for linear in child(node,"linear")["children"].values():
            rules[linear["params"]["weight"]] = ("constant",0.) if attr(node,"zero_init",False) else ("orthogonal",)
        if attr(node,"bias",True):
            bias = child(child(node,"linear"),"0")["params"].get("bias")
            if bias is not None:
                rules[bias] = ("constant",0.)
    elif kind in ("PairMixing","PairInteraction"):
        for name,sub in node["children"].items():
            if name.startswith("coeff_"):
                rules[sub["params"]["weight"]] = ("orthogonal",)
    elif kind in ("InteractionBlock","NonmixingInteractionBlock"):
        radial = child(node,"radial_fn")["children"]
        # The mixing block resets only through `order`, not `mixing_order`.
        for L in range(attr(node,"order")+1):
            rules[radial[str(L)]["params"]["weight"]] = ("orthogonal",)
    elif kind == "DensityCoeffsNetwork" and attr(node,"pred_radial_coeffs",False):
        for name in ("radial_width","radial_scale"):
            for linear in child(node,name)["children"].values():
                rules[linear["params"]["weight"]] = ("constant",0.)
                # Linear's random bias remains; the source does not zero it.
    return rules


def initialization_rules(model, template_state):
    """Return the effective source rule for every canonical parameter key."""
    rules = _rules(model,template_state)
    missing = set(template_state["params"])-set(rules)
    extra = set(rules)-set(template_state["params"])
    if missing or extra:
        raise ValueError(f"Architecture/state initialization mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return rules


def initialize_parameters(model, template_state, seed=0):
    """Initialize every shared parameter once; preserve scientific buffers.

    Enable JAX's x64 setting before calling when the template requires FP64.
    This is host setup from a fixed architecture, rather than a traced operation.
    """
    rules = initialization_rules(model,template_state)
    keys = sorted(rules)
    random_keys = jax.random.split(jax.random.PRNGKey(seed),len(keys))
    params = {}
    for key,random_key in zip(keys,random_keys):
        reference = template_state["params"][key]
        shape, dtype = tuple(reference.shape), jnp.dtype(reference.dtype)
        rule = rules[key]
        if rule[0] == "constant":
            value = jnp.full(shape,rule[1],dtype=dtype)
        elif rule[0] == "uniform":
            value = jax.random.uniform(random_key,shape,dtype=dtype,minval=rule[1],maxval=rule[2])
        elif rule[0] == "orthogonal":
            if len(shape) < 2:
                raise ValueError(f"Orthogonal initialization requires a matrix: {key!r} has {shape}")
            value = jax.nn.initializers.orthogonal(dtype=dtype)(random_key,shape,dtype)
        else:
            raise AssertionError(rule)
        params[key] = value
    return {"params":params,"buffers":{key:jnp.asarray(value) for key,value in template_state["buffers"].items()}}
