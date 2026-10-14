"""Static differentiated connectivity, preserving connected zero gradients.

Torch optimizers skip a parameter whose gradient is None, but decay a connected
parameter whose gradient is numerically zero. Native JAX gradients alone cannot
distinguish these cases. Trace scalar JVP tangent flow before compiler folding.
"""
from __future__ import annotations

import jax


def _dependencies(jaxpr, inputs):
    graph = getattr(jaxpr,"jaxpr",jaxpr)
    environment = dict(zip(graph.invars,inputs))
    def dependencies(variable):
        return set() if hasattr(variable,"val") else environment.get(variable,set())
    for equation in graph.eqns:
        incoming = [dependencies(variable) for variable in equation.invars]
        nested = equation.params.get("jaxpr",equation.params.get("call_jaxpr"))
        if nested is not None and len(getattr(nested,"jaxpr",nested).invars) == len(incoming):
            outgoing = _dependencies(nested,incoming)
        elif equation.primitive.name == "cond":
            branches = [_dependencies(branch,incoming[1:]) for branch in equation.params["branches"]]
            outgoing = [set().union(*(branch[index] for branch in branches))
                        for index in range(len(equation.outvars))]
        else:
            # For opaque primitives retain all tangent dependencies rather than
            # accidentally freezing a parameter. Current traces are tested against
            # independently recorded Torch None gradients for the complete model.
            combined = set().union(*incoming)
            outgoing = [combined for _ in equation.outvars]
        for variable,dependency in zip(equation.outvars,outgoing):
            environment[variable] = dependency
    return [dependencies(variable) for variable in graph.outvars]


def used_parameters(objective, parameters, data):
    """Return canonical names connected to scalar ``objective(params,data)``.

    Run once on the host for each static loss/layout before constructing an
    optimizer update. Trace only: no numerical differentiation is executed.
    Parameter values and zero gradients do not determine this selection.
    """
    def tangent(params,inputs,tangents):
        return jax.jvp(lambda p:objective(p,inputs),(params,),(tangents,))[1]
    graph = jax.make_jaxpr(tangent)(parameters,data,parameters).jaxpr
    parameter_leaves,definition = jax.tree.flatten(parameters)
    data_leaves = jax.tree.leaves(data)
    count = len(parameter_leaves)
    incoming = [set() for _ in range(count+len(data_leaves))]+[{index} for index in range(count)]
    dependencies = set().union(*_dependencies(graph,incoming))
    names = jax.tree.unflatten(definition,[index in dependencies for index in range(count)])
    if any(not isinstance(value,bool) for value in names.values()):
        raise ValueError("Expected a flat canonical parameter mapping")
    return tuple(name for name in parameters if names[name])
