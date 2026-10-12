"""Exact accumulator policy and full source-isolated H5-E1 contracts."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from benchmarks.fixtures import build_model, clone_tree, fixture_spec, load_fixture, tensor_hash
from benchmarks.pipeline import compare_snapshots

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(os.environ.get("DENSNET_REFERENCE_ROOT", ROOT.parent / "equiv_dens_ml-paper-reference"))


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


def test_default_old_attribute_fallback_and_state_schema():
    spec = fixture_spec("tiny", torch.float64, True, num_modules=2, energy_normalize=2)
    legacy = build_model(spec)
    energy = legacy.property_models["energy"]
    assert not hasattr(energy, "accumulation_backend")
    import inspect
    assert "accumulation_backend" not in inspect.signature(type(energy)).parameters
    device = build_model(spec, energy_backend="device")
    assert device.property_models["energy"].accumulation_backend == "device"
    assert legacy.state_dict().keys() == device.state_dict().keys()
    device.load_state_dict(legacy.state_dict(), strict=True)
    assert tensor_hash(legacy.state_dict()) == tensor_hash(device.state_dict())
    from benchmarks.fixtures import make_atoms
    atoms = make_atoms("tiny", torch.float64)
    baseline = legacy(clone_tree(atoms))
    candidate = device(clone_tree(atoms))
    del device.property_models["energy"].accumulation_backend
    fallback = device(clone_tree(atoms))
    for field in ("energy", "forces"):
        torch.testing.assert_close(baseline[field], fallback[field], rtol=0, atol=0)
        torch.testing.assert_close(baseline[field], candidate[field], rtol=0, atol=0)
    with pytest.raises(ValueError, match="backend"):
        build_model(spec, energy_backend="unknown")


def run_module(module, *args):
    completed = subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("case,dtype,normalize", [
    ("tiny", "float64", 0), ("tiny", "float64", 2),
    ("tiny", "float32", 0), ("tiny", "float32", 2),
    ("batch", "float64", 2),
])
def test_full_density_mediated_energy_reference(tmp_path, case, dtype, normalize):
    fixture_path = tmp_path / "fixture.pt"
    run_module("benchmarks.pipeline", "--source-root", REFERENCE, "generate", "--case", case,
               "--dtype", dtype, "--modules", 2, "--energy-normalize", normalize, "--output", fixture_path)
    fixture = load_fixture(fixture_path)
    reference_path = tmp_path / "reference.pt"
    run_module("benchmarks.energy_orchestration", "--source-root", REFERENCE,
               "--fixture", fixture_path, "--output", reference_path)
    reference = torch.load(reference_path, weights_only=False)
    for backend in ("legacy", "device"):
        output = tmp_path / f"candidate-{backend}.pt"
        options = ["--finite-differences"] if case == "tiny" and dtype == "float64" and normalize == 2 and backend == "device" else []
        run_module("benchmarks.energy_orchestration", "--source-root", ROOT, "--fixture", fixture_path,
                   "--backend", backend, "--output", output, *options)
        candidate = torch.load(output, weights_only=False)
        comparison = compare_snapshots(reference, candidate, dtype)
        assert comparison["passed"], comparison["failures"]
        assert candidate["state_hash"] == fixture["state_hash"]
        training = candidate["contracts"]["force_training/train"]
        assert training["forces_graph"]
        assert all(gradient is None for name, gradient in training["gradients"].items() if name.startswith("density_repr_model."))
        assert training["changed_parameters"]
        assert all(name.startswith("property_models.energy.") for name in training["changed_parameters"])
        for snapshot, source in ((reference, REFERENCE), (candidate, ROOT)):
            assert all(Path(module["path"]).is_relative_to(source) for module in snapshot["provenance"]["imports"].values())
