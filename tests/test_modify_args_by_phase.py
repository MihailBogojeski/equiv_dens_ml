"""Phase-specific training args for multiphase DenSNet (density then energy)."""

from __future__ import annotations

import copy
from argparse import Namespace

import pytest

from equiv_dens.training.utils import modify_args_by_phase


def _base_args(**overrides) -> Namespace:
    defaults = dict(
        learning_rate=1e-3,
        stop_at_learning_rate=1e-5,
        validation_interval=500,
        decay_patience=5,
        max_steps=300_000,
        fast_df=True,
        df_weight=1.0,
        density_weight=1.0,
        dipole_moment_weight=0.0,
        energy_weight=0.0,
        forces_weight=0.0,
        core_density_basis=0.0,
        fine_tuning_lr_factor=0.1,
        fine_tuning_stop_lr_factor=0.1,
        density_loss_comp=['perc_mae'],
        density_loss_comp_weights=[1.0],
        density_loss_comp_ft=['perc_mae'],
        density_loss_comp_ft_weights=[1.0],
        remove_atom_density=True,
        integral_constraint=None,
        positive_coeffs=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.mark.parametrize(
    "phase,expect",
    [
        ('density', {'density_weight': 1.0, 'energy_weight': 0.0, 'integral_constraint': None}),
        (
            'density_fine_tuning',
            {
                'density_weight': 1.0,
                'energy_weight': 0.0,
                'integral_constraint': 'coeffs_in_coeffs_net',
                'positive_coeffs': False,
                'learning_rate': 1e-4,
            },
        ),
        ('energy', {'density_weight': 0.0, 'energy_weight': 1.0, 'forces_weight': 0.99}),
    ],
)
def test_modify_args_by_phase(phase, expect):
    orig = _base_args(energy_weight=1.0, forces_weight=0.99)
    args = modify_args_by_phase(copy.deepcopy(orig), orig, phase)
    for key, value in expect.items():
        assert getattr(args, key) == value, f"{phase}: {key}"


def test_direct_fine_tune_enables_positive_coeffs():
    """Total-density models may wrap L0 in softplus; SAD residuals may not."""
    orig = _base_args(remove_atom_density=False)
    args = modify_args_by_phase(copy.deepcopy(orig), orig, 'density_fine_tuning')
    assert args.positive_coeffs is True
    assert args.integral_constraint == 'coeffs_in_coeffs_net'
