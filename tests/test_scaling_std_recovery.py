"""Resume/load must tolerate checkpoints that predate the scaling.std buffer."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from equiv_dens.training.model_loader import _recover_scaling_std, _scaling_std_recoverable
from equiv_dens.utils.scaling import VarianceScaling


def test_scaling_std_recoverable_when_forces_present():
    args = SimpleNamespace(output_scaling=True)
    dataset = SimpleNamespace(required_properties=['energy', 'forces'], forces=[[0.0, 1.0]])
    assert _scaling_std_recoverable(args, dataset)


def test_scaling_std_not_recoverable_without_forces():
    args = SimpleNamespace(output_scaling=True)
    dataset = SimpleNamespace(required_properties=['energy'], forces=None)
    assert not _scaling_std_recoverable(args, dataset)


def test_recover_scaling_std_copies_buffer():
    model_scaling = VarianceScaling()
    source_scaling = VarianceScaling([1.0, 3.0, 5.0])
    model = SimpleNamespace(scaling=model_scaling)
    _recover_scaling_std(model, source_scaling)
    assert torch.allclose(model_scaling.std, source_scaling.std)
