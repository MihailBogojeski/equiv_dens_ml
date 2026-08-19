"""The evaluation forward pass must work for models that predict forces.

This exists because of a bug that no existing test could have caught. The
evaluation ran the model under `torch.no_grad()`, which is correct for the CSH
models -- they predict density only -- and fatal for every model in the water
campaign, which predicts forces by differentiating the energy with respect to
the positions. Nothing would have noticed until the labels were finished and the
watchdog ran the analysis for the first time, days in, with the traceback going
to a log nobody was reading.

The model is stubbed rather than built from a config so the test costs
milliseconds and needs no checkpoint, dataset or GPU. The stub reproduces the
two things about `DFTNetwork.forward` that the caller has to respect, both of
which the real model does at the top and bottom of its forward:

  1. it clones the inputs and turns on `requires_grad` for the positions, then
     differentiates through them, so the caller must not disable grad;
  2. it turns `requires_grad` off again afterwards, which only works on a leaf,
     so the caller must not hand it positions that already require grad.

Those two pull in opposite directions, which is exactly why the first attempt at
a fix (pre-marking the positions) traded one traceback for another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "revision"))

from csh_evaluate import (  # noqa: E402
    absolute_fractional_error,
    batch_absolute_fractional_error,
)


class ForcePredictingStub(torch.nn.Module):
    """Mimics DFTNetwork's handling of `positions` for a model with force output."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, data):
        atoms = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
        atoms["positions"].requires_grad_(True)

        energy = (atoms["positions"] ** 2).sum()
        atoms["forces"] = -torch.autograd.grad(energy, atoms["positions"], create_graph=False)[0]

        # Only legal on a leaf; fails if the caller pre-marked the positions.
        atoms["positions"].requires_grad_(False)

        atoms["density"] = data["density"] * self.scale
        return atoms


def make_props(n_structures: int = 3, n_atoms: int = 4, n_points: int = 16):
    torch.manual_seed(0)
    return {
        "positions": torch.randn(n_structures, n_atoms, 3),
        "density": torch.rand(n_structures, n_points) + 0.5,
        "coord_weights": torch.rand(n_structures, n_points) + 0.5,
        "batch_atom_numbers": torch.full((n_structures, n_atoms), 8),
    }


def test_runs_for_a_model_that_differentiates_through_positions():
    """The regression itself: this raised before the fix."""
    afe = batch_absolute_fractional_error(ForcePredictingStub(), make_props())
    assert afe.shape == (3,)
    assert np.all(np.isfinite(afe))


def test_no_grad_around_the_forward_is_what_broke_it():
    """Pins the cause, so re-introducing `no_grad` fails here rather than in a campaign."""
    with pytest.raises(RuntimeError, match="does not require grad"):
        with torch.no_grad():
            batch_absolute_fractional_error(ForcePredictingStub(), make_props())


def test_pre_marking_positions_is_also_wrong():
    """The obvious fix for the above, which trades one traceback for another."""
    props = make_props()
    props["positions"] = props["positions"].detach().requires_grad_(True)
    with pytest.raises(RuntimeError, match="leaf variables"):
        batch_absolute_fractional_error(ForcePredictingStub(), props)


def test_error_is_zero_when_the_prediction_matches():
    afe = batch_absolute_fractional_error(ForcePredictingStub(scale=1.0), make_props())
    assert np.allclose(afe, 0.0, atol=1e-6)


def test_error_is_the_fractional_density_difference():
    """A uniform 10 % overprediction must read as 0.1, whatever the weights are."""
    afe = batch_absolute_fractional_error(ForcePredictingStub(scale=1.1), make_props())
    assert np.allclose(afe, 0.1, atol=1e-5)


def test_matches_the_underlying_metric():
    props = make_props()
    pred = props["density"] * 1.3
    expected = absolute_fractional_error(pred, props["density"], props["coord_weights"]).numpy()
    got = batch_absolute_fractional_error(ForcePredictingStub(scale=1.3), props)
    assert np.allclose(got, expected)


def test_returns_a_detached_array():
    """The caller writes these to JSON; a tensor still attached to the graph would
    keep a batch of activations alive for every structure evaluated."""
    afe = batch_absolute_fractional_error(ForcePredictingStub(scale=1.2), make_props())
    assert isinstance(afe, np.ndarray)
