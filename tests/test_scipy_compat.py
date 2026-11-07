"""The legacy-BSpline shim must work on every SciPy: patch when BSpline defines __setstate__, do nothing otherwise."""
import pickle

import numpy as np
from scipy.interpolate import BSpline

from equiv_dens.utils import scipy_compat


def _legacy_payload():
    t = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    c = np.array([1.0, 2.0, 3.0, 4.0])
    return {"t": t, "c": c, "k": 2, "extrapolate": True, "axis": 0}


def test_noop_when_class_has_no_setstate():
    class Plain:
        pass

    scipy_compat.enable_legacy_bspline_unpickling(cls=Plain)
    assert "__setstate__" not in Plain.__dict__


def test_patch_is_idempotent_and_accepts_dict_payload():
    scipy_compat.enable_legacy_bspline_unpickling()
    scipy_compat.enable_legacy_bspline_unpickling()
    if "__setstate__" not in BSpline.__dict__:
        return  # SciPy without a custom __setstate__: nothing to patch, pickle sets __dict__ itself
    spline = BSpline.__new__(BSpline)
    spline.__setstate__(_legacy_payload())
    assert np.isclose(spline(1.0), BSpline(_legacy_payload()["t"], _legacy_payload()["c"], 2)(1.0))
    # a modern pickle of a real spline still round-trips
    ref = BSpline(_legacy_payload()["t"], _legacy_payload()["c"], 2)
    assert np.isclose(pickle.loads(pickle.dumps(ref))(0.5), ref(0.5))
