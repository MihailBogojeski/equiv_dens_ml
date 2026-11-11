"""SAD fine-tune must not NaN when the charge-projection denominator vanishes.

``CoeffsIntegralConstraint`` with ``remove_atom_density=True`` subtracts
``(v·w / w·w) w``. After density early-stop, water/cutoff jobs died with
``Nans found in predicted density`` on the first fine-tune batch whose L0
scales (or widths) made ``w·w == 0``.
"""

from __future__ import annotations

import torch

from equiv_dens.nn.property_output.density import CoeffsIntegralConstraint


def _l0_atoms(
    scale: torch.Tensor,
    width: torch.Tensor | None = None,
    sph: torch.Tensor | None = None,
) -> dict:
    """Packed L=0 coefficients: coeff [B, S0, 1], width / scale [B, S0, P] (one primitive per shell here)."""
    batch, n_basis = scale.shape[0], scale.shape[-1]
    if width is None:
        width = torch.ones_like(scale)
    if sph is None:
        sph = torch.ones_like(scale)
    packed = {
        'coeff': sph.reshape(batch, n_basis, 1),
        'width': width.reshape(batch, n_basis, 1),
        'scale': scale.reshape(batch, n_basis, 1),
    }
    return {
        'batch_atom_numbers': torch.tensor([[8.0, 1.0, 1.0]]).expand(batch, 3),
        'packed_coeffs': [packed],
    }


def test_sad_constraint_is_finite_when_all_l0_scales_are_zero():
    scale = torch.zeros(2, 1, 1, 3)
    atoms = _l0_atoms(scale)
    out = CoeffsIntegralConstraint(remove_atom_density=True)(atoms)
    coeffs = out['packed_coeffs'][0]['coeff']
    assert torch.isfinite(coeffs).all()


def test_sad_constraint_is_finite_on_mixed_zero_and_nonzero_scales():
    scale = torch.tensor(
        [
            [[[0.0, 0.0, 0.0]]],
            [[[1.0, 0.5, 0.25]]],
        ]
    )
    atoms = _l0_atoms(scale)
    out = CoeffsIntegralConstraint(remove_atom_density=True)(atoms)
    coeffs = out['packed_coeffs'][0]['coeff']
    assert torch.isfinite(coeffs).all()


def test_direct_constraint_is_finite_when_coeff_sum_is_zero():
    scale = torch.ones(2, 1, 1, 3)
    sph = torch.zeros(2, 1, 1, 3)
    atoms = _l0_atoms(scale, sph=sph)
    out = CoeffsIntegralConstraint(remove_atom_density=False)(atoms)
    coeffs = out['packed_coeffs'][0]['coeff']
    assert torch.isfinite(coeffs).all()
