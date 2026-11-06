"""Density-phase checkpoints must load into the constraint fine-tune model.

The paper recipe (SI S1.3) trains unconstrained density first, then turns on
charge conservation. That adds ``CoeffsIntegralConstraint.integral_scale``,
which is not in the density-phase state dict. ``load_state_dict(strict=False)``
already initialises it to ones; treating the missing key as fatal is what
killed water_direct and cutoff_6 after density early-stopped.
"""

from __future__ import annotations

from types import SimpleNamespace

from equiv_dens.training.model_loader import _is_fatal_missing_key, _is_fatal_unexpected_key


def _density_phase_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        ignore_missing_keywords=False,
        df_weight=0.0,
        density_weight=1.0,
        core_density_basis=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_constraint_scale_is_not_fatal_when_entering_fine_tune():
    args = _density_phase_args()
    key = 'density_repr_model.1.integral_constraint.integral_scale'
    assert not _is_fatal_missing_key(key, args)


def test_unrelated_missing_repr_weight_is_still_fatal():
    args = _density_phase_args()
    key = 'density_repr_model.0.embedding.weight'
    assert _is_fatal_missing_key(key, args)


def test_init_buffers_remain_allowed():
    args = _density_phase_args()
    key = 'density_repr_model.1.init_coeffs'
    assert not _is_fatal_missing_key(key, args)


def test_constraint_scale_unexpected_is_not_fatal_when_evaluating_fine_tuned_ckpt():
    """Reverse of test_constraint_scale_is_not_fatal_when_entering_fine_tune.

    A checkpoint that *did* go through density_fine_tuning (SI S1.3) carries
    ``integral_constraint.integral_scale``. If the eval harness reconstructs
    the model without ``args.integral_constraint`` set (the flag is set only
    transiently during that training phase and is not persisted to
    args.txt), the key comes back on the *unexpected* side instead. That must
    stay non-fatal too, or every cutoff/curve checkpoint that passed through
    fine-tuning becomes unloadable for evaluation.
    """
    args = _density_phase_args()
    key = 'density_repr_model.1.integral_constraint.integral_scale'
    assert not _is_fatal_unexpected_key(key, args)


def test_unrelated_unexpected_density_key_is_still_fatal():
    args = _density_phase_args()
    key = 'density_repr_model.1.some_new_untracked_submodule.weight'
    assert _is_fatal_unexpected_key(key, args)


def test_unexpected_key_fatal_for_energy_only_model_missing_energy_prefix():
    args = _density_phase_args(density_weight=0.0, df_weight=0.0, energy_weight=1.0, forces_weight=0.99)
    key = 'property_models.some_unrelated_head.weight'
    assert _is_fatal_unexpected_key(key, args)


def test_unexpected_key_allowed_for_energy_model_density_extra():
    args = _density_phase_args(density_weight=0.0, df_weight=0.0, energy_weight=1.0, forces_weight=0.99)
    key = 'property_models.density.init_coeffs'
    assert not _is_fatal_unexpected_key(key, args)
