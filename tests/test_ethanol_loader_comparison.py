import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "experiments/jacs_revision/analysis/ethanol_loader_comparison.py"
_spec = importlib.util.spec_from_file_location("ethanol_loader_comparison", _PATH)
elc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(elc)


def test_frame_selection_is_deterministic_and_sorted():
    a = elc.select_frames(900, 20, 0)
    b = elc.select_frames(900, 20, 0)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0) and len(set(a.tolist())) == 20


def test_force_stats_scale_invariance_and_slope():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(5, 9, 3))
    s = elc.force_stats(27.0 * ref, ref)
    assert abs(s["pearson_r"] - 1.0) < 1e-12
    assert abs(s["slope_through_origin"] - 27.0) < 1e-9
    assert s["mae_after_slope"] < 1e-12
    noisy = elc.force_stats(ref + rng.normal(scale=1.0, size=ref.shape), ref)
    assert 0.0 < noisy["pearson_r"] < 0.9
