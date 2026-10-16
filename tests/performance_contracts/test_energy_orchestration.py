"""Exact accumulator policy and full source-isolated H5-E1 contracts."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from benchmarks.fixtures import REFERENCE_ROOT, build_model, clone_tree, fixture_spec, load_fixture, tensor_hash
from benchmarks.pipeline import compare_snapshots

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = REFERENCE_ROOT


def accumulator():
    from benchmarks.experimental_energy import _accumulate_energy_features
    return _accumulate_energy_features


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("normalize", [0, 2])
@pytest.mark.parametrize("case", ["nonzero", "previous_zero", "update_zero", "both_zero", "scalar_zero", "underflow", "update_underflow"])
def test_independent_accumulator_policy(dtype, normalize, case):
    previous = torch.tensor([0.3, -0.7, 1.2], dtype=dtype)
    update = torch.tensor([-0.1, 0.9, 0.4], dtype=dtype)
    if case in ("previous_zero", "both_zero"):
        previous.zero_()
    if case in ("update_zero", "both_zero"):
        update.zero_()
    if case == "scalar_zero":
        previous = torch.tensor(0., dtype=dtype)
    if case in ("underflow", "update_underflow"):
        value = previous if case == "underflow" else update
        value.fill_(1e-170 if dtype == torch.float64 else 1e-24)
        assert torch.count_nonzero(value) == 3
        assert torch.mean(value**2) == 0
    # Expected policy is independently stated rather than delegated to either
    # implementation. In particular, nonzero entries can have zero mean square.
    scale = 1. if not normalize or previous.square().mean() == 0 or update.square().mean() == 0 else 2.**-0.5
    expected = update * scale + previous * scale
    for backend in ("legacy", "device"):
        differentiable_previous = previous.clone().requires_grad_()
        differentiable_update = update.clone().requires_grad_()
        actual = accumulator()(differentiable_previous, differentiable_update, normalize, backend)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.tensor([0.2, 0.5, 0.8], dtype=dtype)
        previous_gradient, update_gradient = torch.autograd.grad((actual * weights).sum(),
                                                                  (differentiable_previous, differentiable_update))
        expected_gradient = weights * scale
        torch.testing.assert_close(update_gradient, expected_gradient, rtol=0, atol=0)
        torch.testing.assert_close(previous_gradient,
                                   expected_gradient.sum() if previous.ndim == 0 else expected_gradient,
                                   rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("normalize", [0, 2])
def test_nonzero_first_and_mixed_derivatives(dtype, normalize):
    results = []
    for backend in ("legacy", "device"):
        position = torch.tensor([0.3, -0.7, 1.2], dtype=dtype, requires_grad=True)
        parameter = torch.tensor(0.83, dtype=dtype, requires_grad=True)
        previous = (position * parameter).sin()
        update = position.square() * parameter + 0.2
        energy = accumulator()(previous, update, normalize, backend).square().sum()
        force, parameter_gradient = torch.autograd.grad(energy, (position, parameter), create_graph=True)
        mixed = torch.autograd.grad((force - 0.17).square().mean(), parameter)[0]
        assert parameter_gradient.abs() > 1e-6 and mixed.abs() > 1e-6
        results.append((energy, force, parameter_gradient, mixed))
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("normalize", [0, 2])
def test_device_accumulator_never_converts_tensor_to_python_bool(monkeypatch, normalize):
    function = accumulator()
    previous, update = torch.tensor([0.3, 0.8]), torch.tensor([0.5, 0.9])
    def forbidden(*_):
        raise AssertionError("Device accumulation converted a tensor to Python bool")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        output = function(previous, update, normalize, "device")
    assert torch.isfinite(output).all()


def run_module(module, *args):
    completed = subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr

