"""Energy errors must pair each prediction with its own label.

The model returns energies as (B, 1). The revision label files store them as
(N,), and (B, 1) - (B,) broadcasts to (B, B): the loss scored every prediction
against every label in the batch, so no revision energy head could learn
energies (training MAE stuck at ~32 kcal/mol while forces trained).
"""

from __future__ import annotations

import pytest
import torch

from equiv_dens.training.errors import ErrorDict


def _error_dict() -> ErrorDict:
    return ErrorDict(
        {'energy': 1.0, 'forces': 0.0, 'density': 0.0},
        loss_comp={'energy': ['mae']},
        loss_comp_weights={'energy': {'mae': 1.0}},
    )


@pytest.mark.parametrize("label_shape", [(4,), (4, 1)])
def test_energy_mae_is_per_structure(label_shape):
    pred = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    labels = torch.tensor([1.5, 2.5, 3.5, 4.5]).reshape(label_shape)
    errors = _error_dict().compute({'energy': pred}, {'energy': labels, 'positions': torch.zeros(1, 4, 3)})
    assert errors['energy_mae'].item() == pytest.approx(0.5)


def test_energy_size_mismatch_raises():
    with pytest.raises(ValueError):
        _error_dict().compute({'energy': torch.zeros(4, 1)},
                              {'energy': torch.zeros(3), 'positions': torch.zeros(1, 4, 3)})
