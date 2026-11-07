"""coeffs_dict_to_tensors must honour global sph_dict layout widths.

The energy head sizes ``radial_scale_filters_L`` to the density network's
global DF layout (sum of features over every Z in the dataset).  ``sph_dict``
stores the corresponding global index ranges.  A molecule that only contains a
subset of those elements still has to be packed into that full width so

    scale_fs[L] * radial_scale_filters(L)

broadcasts.  Sizing by ``sum(n_coeffs for Z in molecule)`` instead yields the
QM7x energy failure ``38 vs 66`` on non-singleton dim 3.
"""

from __future__ import annotations

import torch

from equiv_dens.utils.orbitals import coeffs_dict_to_tensors


def _qm7x_like_sph_dict_l0() -> dict[tuple[int, int], torch.Tensor]:
    """Global L=0 layout for H,C,N,O,S,Cl matching aug-cc-pVQZ-JKFIT DF counts."""
    widths = {1: 5, 6: 11, 7: 11, 8: 11, 16: 14, 17: 14}
    sph_dict: dict[tuple[int, int], torch.Tensor] = {}
    cursor = 0
    for z, n in widths.items():
        sph_dict[(z, 0)] = torch.arange(cursor, cursor + n)
        cursor += n
    assert cursor == 66
    return sph_dict


def test_coeffs_dict_to_tensors_uses_global_sph_dict_width() -> None:
    sph_dict = _qm7x_like_sph_dict_l0()
    # Molecule with only H/C/N/O: local sum is 38, global layout is 66.
    present = (1, 6, 7, 8)
    batch = 2
    n_radial = 4
    spherical_coeffs = []
    radial_width = []
    radial_scale = []
    for z in present:
        key = (z, 0)
        n = int(sph_dict[key].numel())
        spherical_coeffs.append(
            {key: torch.randn(batch, 1, 1, n)}
        )
        radial_width.append(
            {key: torch.randn(batch, 1, n_radial, n)}
        )
        radial_scale.append(
            {key: torch.randn(batch, 1, n_radial, n)}
        )

    sph, scale, width = coeffs_dict_to_tensors(
        {
            "spherical_coeffs": spherical_coeffs,
            "radial_width": radial_width,
            "radial_scale": radial_scale,
            "sph_dict": sph_dict,
        }
    )

    assert scale[0].shape[-1] == 66, (
        f"expected global DF width 66, got {scale[0].shape[-1]} "
        "(molecule-local packing breaks energy radial filters)"
    )
    assert sph[0].shape[-1] == 66
    assert width[0].shape[-1] == 66

    # The multiply that failed in tre-qm7x-energy must succeed.
    filters = torch.ones(1, 66)
    out = scale[0] * filters
    assert out.shape == scale[0].shape
