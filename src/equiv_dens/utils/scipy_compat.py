"""Read SAD priors pickled against older SciPy releases.

The free-atom density files that ship with the published models store
``scipy.interpolate.BSpline`` objects. Those were pickled when ``BSpline`` had
no custom ``__getstate__``, so the payload is the instance ``__dict__``:
``{"t", "c", "k", "extrapolate", "axis"}``.

SciPy 1.15 moved ``BSpline`` onto an array-API delegate and added a
``__getstate__``/``__setstate__`` pair that exchanges a two-element tuple
``(delegate, sentinel_array)``. Unpickling an old file against a new SciPy
therefore fails inside ``__setstate__`` with

    ValueError: too many values to unpack (expected 2)

which is what stopped ``paper/models/ethanol/2024-03-22_96w7KyGG`` from being
evaluated at all: its prior is ``atom_dens_type=spline``, and without the prior
there is no delta-learning target to add the predicted correction back onto.

Rebuilding the spline through the public constructor rather than writing the
old attributes straight onto the instance keeps whatever internal
representation the installed SciPy expects.
"""

from __future__ import annotations

from scipy.interpolate import BSpline

_PATCHED_FLAG = "_equiv_dens_legacy_setstate"


def enable_legacy_bspline_unpickling() -> None:
    """Teach ``BSpline`` to accept the pre-1.15 pickle format.

    Idempotent, and modern two-tuple payloads still take the original path, so
    this is safe to call unconditionally before loading any prior.
    """
    original = BSpline.__setstate__
    if getattr(original, _PATCHED_FLAG, False):
        return

    def __setstate__(self, state):
        if isinstance(state, dict) and {"t", "c", "k"} <= set(state):
            rebuilt = BSpline(
                state["t"],
                state["c"],
                state["k"],
                extrapolate=state.get("extrapolate", True),
                axis=state.get("axis", 0),
            )
            self.__dict__.update(rebuilt.__dict__)
            return
        original(self, state)

    setattr(__setstate__, _PATCHED_FLAG, True)
    BSpline.__setstate__ = __setstate__
