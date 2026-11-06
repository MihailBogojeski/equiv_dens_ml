"""Energy/force inference must not expand the density on a real-space grid.

``DFTNetwork.forward`` used to run every trained property head, including
``density``, even when the caller only wanted energy and forces. The density
head indexes ``atoms['coords']``, so ASE BFGS either crashed (no grid) or
spent minutes per step expanding ρ on a DFT quadrature (full grid). The
energy head already consumes the equivariant features from
``density_repr_model``; the grid expansion is only needed when the caller
asks for ρ. Skipping it when ``coords`` is absent is the paper two-stage
MD path and is what makes ``relax_water_clusters.py`` tractable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from equiv_dens.nn.dft_network import DFTNetwork
from equiv_dens.utils.scaling import UnitConversion, VarianceScaling


class _PassthroughRepr(nn.Module):
    def forward(self, atoms):
        return atoms


class _EnergyHead(nn.Module):
    def forward(self, atoms):
        n = int(atoms["positions"].shape[1])
        atoms["energy"] = torch.tensor([float(n)], dtype=torch.float32)
        atoms["forces"] = torch.zeros(1, n, 3, dtype=torch.float32)
        return atoms


class _DensityHead(nn.Module):
    def forward(self, atoms):
        _ = atoms["coords"]
        atoms["density"] = torch.ones(1, 4)
        return atoms


def _energy_density_net():
    identity = UnitConversion(en_conversion_func=lambda x: x, dist_conversion_func=lambda x: x)
    return DFTNetwork(
        _PassthroughRepr(),
        {"energy": _EnergyHead(), "density": _DensityHead()},
        calculate_forces_dict={"energy": True, "density": False},
        conversions_in=identity,
        conversions_out=identity,
        scaling=VarianceScaling(),
    )


def test_forward_returns_energy_without_coords():
    model = _energy_density_net()
    model.eval()
    out = model({"positions": torch.zeros(1, 2, 3), "batch_atom_numbers": torch.ones(1, 2)})
    assert "energy" in out
    assert torch.isfinite(out["energy"]).all()
    assert "density" not in out


def test_forward_still_runs_density_when_coords_are_present():
    model = _energy_density_net()
    model.eval()
    out = model(
        {
            "positions": torch.zeros(1, 2, 3),
            "batch_atom_numbers": torch.ones(1, 2),
            "coords": torch.zeros(1, 4, 3),
        }
    )
    assert "density" in out
    assert out["density"].shape == (1, 4)
