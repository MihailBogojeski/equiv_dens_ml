"""Shared, deliberately small interface for converted module trees."""
from __future__ import annotations

import jax.numpy as jnp


def attr(node, name, default=None):
    return node["attrs"].get(name, default)


def child(node, name):
    return node["children"][str(name)]


def parameter(node, state, name):
    return state["params"][node["params"][name]]


def buffer(node, state, name):
    return state["buffers"][node["buffers"][name]]


def linear(node, state, value):
    out = jnp.matmul(value, jnp.swapaxes(parameter(node, state, "weight"), -1, -2))
    if "bias" in node["params"]:
        out = out + parameter(node, state, "bias")
    return out
