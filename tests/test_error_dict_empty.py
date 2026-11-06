"""Energy test eval must accept every key ``ErrorDict.compute`` emits.

``train_only_en.py`` initialises the running average from ``empty()`` and then
adds each ``compute()`` entry. ``empty()`` omitted ``*_mse``, so every
direct-water energy job died at ``Starting test evaluation!!!`` with
``KeyError: energy_mse`` after the LR stop (e.g. job 16812650).
"""

from __future__ import annotations

import torch

from equiv_dens.training.errors import ErrorDict, accumulate_running_errors


def _energy_error_dict() -> ErrorDict:
    return ErrorDict(
        {'energy': 1.0, 'forces': 0.99, 'density': 0.0},
        loss_comp={'energy': ['mae'], 'forces': ['mae']},
        loss_comp_weights={
            'energy': {'mae': 1.0},
            'forces': {'mae': 1.0},
        },
    )


def _batch(error_dict: ErrorDict) -> dict:
    pred = {
        'energy': torch.zeros(2, 1),
        'forces': torch.zeros(2, 3, 3),
    }
    data = {
        'energy': torch.ones(2, 1),
        'forces': torch.ones(2, 3, 3),
        'positions': torch.zeros(2, 3, 3),
    }
    return error_dict.compute(pred, data)


def test_empty_includes_every_compute_key_for_energy_and_forces():
    error_dict = _energy_error_dict()
    empty = error_dict.empty()
    computed = _batch(error_dict)
    missing = set(computed) - set(empty)
    assert missing == set()


def test_accumulate_running_errors_accepts_keys_absent_from_empty():
    running = {'loss': 0.0, 'energy_mae': 0.0}
    batch = {'loss': torch.tensor(4.0), 'energy_mae': torch.tensor(2.0), 'energy_mse': torch.tensor(8.0)}
    accumulate_running_errors(running, batch, n=1)
    assert running['energy_mse'] == 8.0
    accumulate_running_errors(
        running,
        {'loss': torch.tensor(2.0), 'energy_mae': torch.tensor(0.0), 'energy_mse': torch.tensor(0.0)},
        n=2,
    )
    assert running['energy_mse'] == 4.0
