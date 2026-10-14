"""Rejected H4-C1/C2 evaluator retained for reproducible benchmarks only.

The production evaluator remains the pinned broadcast implementation. This
module subclasses it so benchmark fixtures keep the identical parameters and
postprocessing while selecting the measured magnetic-first contraction.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from equiv_dens.nn.property_output.density import DensityExpansion
from equiv_dens.utils.orbitals import gaussian_rbf, gaussian_rbf_deriv
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
import equiv_dens.utils.spherical_harmonics_deriv as sph_deriv
from equiv_dens.utils.base import (
    calculate_distances_and_directions,
    bohr_to_angstrom,
)


def _needs_full_precision_reduction(tensor):
    # The measured C2 fallback preserves the FP32 arithmetic of the original
    # elementwise products under the caller's inherited CUDA precision policy.
    return (tensor.is_cuda and tensor.dtype == torch.float32
            and (torch.backends.cuda.matmul.allow_tf32
                 or torch.get_float32_matmul_precision() != "highest"))


def eval_gto_contracted(s, d, L, sph_coeff, width, scale):
    """Measured C1/C2 angular contraction, including its original return shape."""
    if _needs_full_precision_reduction(s):
        angular = (s.unsqueeze(-1) * sph_coeff).sum(-2)
    else:
        angular = torch.matmul(s, sph_coeff.squeeze(1))
    rbf = gaussian_rbf(d.unsqueeze(-1), width, scale, L)
    return (rbf.squeeze(-2) * angular).sum(-1), angular, rbf


def eval_gto_grad_contracted(s_deriv, d, L, coords, sph_coeff, width, scale,
                             angular, rbf):
    """Measured C1/C2 spatial derivative with the same precision fallback."""
    full_precision_reduction = _needs_full_precision_reduction(s_deriv)
    if full_precision_reduction:
        angular_direction = (s_deriv.unsqueeze(-1) * sph_coeff.unsqueeze(-2)).sum(-3)
    else:
        angular_direction = torch.einsum(
            "bgmc,bmr->bgcr", s_deriv, sph_coeff.squeeze(1))
    identity = torch.eye(3, device=coords.device, dtype=coords.dtype)
    u_grad = -(identity / d.unsqueeze(-1)
               - coords.unsqueeze(-2) * coords.unsqueeze(-1)
               / d.unsqueeze(-1) ** 3)
    if full_precision_reduction:
        angular_cartesian = (
            angular_direction.unsqueeze(-3) * u_grad.unsqueeze(-1)).sum(-2)
    else:
        angular_cartesian = torch.einsum(
            "bgcr,bgdc->bgdr", angular_direction, u_grad)
    angular_cartesian = torch.where(
        torch.isnan(angular_cartesian), torch.zeros_like(angular_cartesian),
        angular_cartesian)
    radial_gradient = gaussian_rbf_deriv(
        d.unsqueeze(-1), width, scale, L) * (coords / d).unsqueeze(-1)
    radial_gradient = torch.where(
        torch.isnan(radial_gradient), torch.zeros_like(radial_gradient),
        radial_gradient)
    return (radial_gradient * angular.unsqueeze(-2)
            + rbf * angular_cartesian).sum(-1)


class ExperimentalDensityExpansion(DensityExpansion):
    """Benchmark-only subclass of the production expansion.

    ``evaluation_backend`` deliberately exists only here. ``legacy`` delegates
    to the production implementation; ``contracted`` copies the measured C
    forward path exactly, including its precision-safe C2 fallback.
    """

    def __init__(self, orbital_basis, *, evaluation_backend="legacy", **kwargs):
        if evaluation_backend not in ("legacy", "contracted"):
            raise ValueError("Density evaluation backend must be legacy or contracted")
        self.evaluation_backend = evaluation_backend
        super().__init__(orbital_basis, **kwargs)

    def forward(self, atoms, eval_atoms=None, eval_L=None, density_grad=None):
        if getattr(self, "evaluation_backend", "legacy") == "legacy":
            return super().forward(atoms, eval_atoms=eval_atoms, eval_L=eval_L,
                                   density_grad=density_grad)
        return self._forward_contracted(atoms, eval_atoms, eval_L, density_grad)

    def _forward_contracted(self, atoms, eval_atoms=None, eval_L=None,
                            density_grad=None):
        # This is the measured DensityExpansion.forward body with only its
        # evaluator selection moved into this benchmark-only class.
        if density_grad is None:
            density_grad = self.density_grad
        n_eval = len(atoms["spherical_coeffs"])
        if self.timing:
            start = time.time()
        if eval_atoms is None:
            eval_atoms = list(range(n_eval))
        if eval_L is None:
            eval_L = list(range(self.orbitals_max_order + 1))
        atoms["density"] = 0
        if density_grad:
            atoms["density_grad"] = 0
        from equiv_dens.utils.orbitals import get_n_electrons
        n_electrons = get_n_electrons(atoms["batch_atom_numbers"])
        for i in range(n_eval):
            if self.verbose > 1 and self.memory:
                print("Atom", i)
                print("density density expansion:")
                print("Memory allocated", torch.cuda.memory_allocated() / 1024**2)
                print("Memory cached", torch.cuda.memory_cached() / 1024**2)
            z = int(max(atoms["batch_atom_numbers"][:, i]))
            if z == 0:
                continue
            pos = atoms["batch_positions"][:, [i]]
            d, u = calculate_distances_and_directions(atoms["coords"], center=pos)
            s = spherical_harmonics(self.orbitals_max_order_dict[z], u)
            if density_grad:
                s_deriv = sph_deriv.spherical_harmonics_deriv(
                    self.orbitals_max_order_dict[z], u)
            for L in range(len(s)):
                s[L] = torch.where(torch.isnan(s[L]), torch.zeros_like(s[L]), s[L])
            for j in range(len(self.spherical_spec[z])):
                orb = self.spherical_spec[z][j]
                L = orb[2]
                key = (z, L)
                width = atoms["radial_width"][i][key]
                scale = atoms["radial_scale"][i][key]
                sph_coeff = atoms["spherical_coeffs"][i][key]
                dim_diff = s[L].dim() - sph_coeff.dim()
                if dim_diff >= 0:
                    sph_coeff = sph_coeff.reshape(
                        sph_coeff.shape[:2] + (1,) * (dim_diff + 1)
                        + sph_coeff.shape[2:])
                    width = width.reshape(
                        width.shape[:2] + (1,) * (dim_diff + 1) + width.shape[2:])
                    scale = scale.reshape(
                        scale.shape[:2] + (1,) * (dim_diff + 1) + scale.shape[2:])
                if i in eval_atoms and L in eval_L:
                    density_part, angular, rbf = eval_gto_contracted(
                        s[L], d, L, sph_coeff, width, scale)
                    if self.verbose > 2:
                        print("L", L)
                        print("rbf integral", torch.sum(
                            rbf * atoms["coord_weights"].unsqueeze(-1).unsqueeze(-1),
                            dim=(-2, -3)))
                        diagnostic_sph = s[L].unsqueeze(-1) * sph_coeff
                        print("abs sph", torch.sum(
                            torch.abs(diagnostic_sph)
                            * atoms["coord_weights"].unsqueeze(-1).unsqueeze(-1),
                            dim=(-2, -3)))
                    atoms["density"] += density_part
                    if density_grad:
                        density_part_grad = eval_gto_grad_contracted(
                            s_deriv[L], d, L, atoms["coords"] - pos,
                            sph_coeff, width, scale, angular, rbf)
                        atoms["density_grad"] += bohr_to_angstrom(density_part_grad)

        if self.verbose > 0:
            print("Density shape", atoms["density"].shape)
            print("sum neg integrals", torch.sum(
                (atoms["density"] * atoms["coord_weights"])
                [atoms["density"] < 0], dim=-1))
        if self.expansion_constraint == "sq":
            atoms["density"] = atoms["density"]**2
        if self.expansion_constraint == "abs":
            atoms["density"] = torch.sqrt(atoms["density"]**2)
        if self.expansion_constraint == "sp":
            atoms["density"] = F.softplus(atoms["density"], beta=100000000) + 1e-30
        if self.integral_constraint == "grid":
            grid_scaling = n_electrons / torch.sum(
                atoms["density"] * atoms["coord_weights"], dim=1, keepdim=True)
            if self.verbose > 0:
                print("grid_scaling", grid_scaling)
                print("grid_scaling_factor", (1 - self.grid_scaling_factor))
            grid_scaling = (grid_scaling * (1 - self.grid_scaling_factor)
                            + self.grid_scaling_factor)
            atoms["density"] = atoms["density"] * grid_scaling
        if self.timing:
            print("density expansion time:", time.time() - start)
        return atoms


def install_experimental_density_evaluator(model, backend="contracted"):
    """Replace the model's density module with the benchmark adapter in-place."""
    if backend not in ("legacy", "contracted"):
        raise ValueError("Density evaluation backend must be legacy or contracted")
    previous = model.property_models["density"]
    if isinstance(previous, ExperimentalDensityExpansion):
        previous.evaluation_backend = backend
        return model
    adapter = ExperimentalDensityExpansion(
        previous.orbital_basis, evaluation_backend=backend,
        expansion_constraint=previous.expansion_constraint,
        integral_constraint=previous.integral_constraint,
        verbose=previous.verbose, timing=previous.timing, memory=previous.memory,
        grid_scaling_factor=hasattr(previous, "grid_scaling_factor")
        and isinstance(previous.grid_scaling_factor, torch.Tensor),
        remove_atom_density=previous.remove_atom_density,
        density_grad=previous.density_grad)
    adapter.load_state_dict(previous.state_dict(), strict=True)
    model_parameter = next(model.parameters())
    adapter.to(device=model_parameter.device, dtype=model_parameter.dtype)
    adapter.train(previous.training)
    model.property_models["density"] = adapter
    return model
