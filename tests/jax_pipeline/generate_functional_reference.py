"""Independent actual Torch LDA/Ewald references; no JAX import or evaluator.

The potential is a prescribed numeric field: external DFTpy pseudopotential
assets are not bundled in this fixture. The actual LDA host callback runs, then
the exported field is held detached as in production. Real ReciprocalGrid
constructs the mask; a grid facade avoids CubicalGrid's removed numpy.int alias.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np
import torch

import equiv_dens.compat  # noqa: F401
from equiv_dens.density_functionals import LDA
from equiv_dens.utils.grids import ReciprocalGrid
from equiv_dens_jax.convert import export_module, to_numpy_tree
from equiv_dens_jax.serialization import save_tree


class Grid:
    def __init__(self, shape, lattice):
        self.shape = shape
        self.lattice = torch.tensor(lattice)
        self.volume = torch.prod(torch.diag(self.lattice))
        self.point_volume = self.volume / np.prod(shape)
        self.reciprocal = ReciprocalGrid(np.asarray(shape), (2 * np.pi * torch.inverse(self.lattice)).T)

    def get_reciprocal_grid(self):
        return self.reciprocal


class PrescribedPotential:
    def __init__(self, fields):
        self.fields = fields
        self.calls = []

    def restart(self, *, grid, ions):
        self.index = ions
        self.calls.append(("restart", ions))

    def __call__(self, density):
        self.vreal = self.fields[self.index].copy()
        self.calls.append(("evaluate", density.shape))


def reference(shape, lattice, *, offset, store_energy):
    grid = Grid(shape, lattice)
    phase = torch.linspace(0.1, 5.7, 2 * np.prod(shape)).reshape(2, *shape)
    density = 0.025 + 0.37 * (0.5 + 0.5 * torch.sin(phase))
    potential = -0.13 + 0.07 * torch.cos(phase * 1.7)
    positions = torch.tensor([[[0.35, 0.51, 0.27], [1.32, 0.73, 0.81]],
                              [[0.48, 0.62, 0.34], [1.15, 0.97, 0.92]]])
    density.requires_grad_(True)
    positions.requires_grad_(True)
    pseudo = PrescribedPotential(potential.numpy())
    atoms = {"density": density.reshape(2, -1), "shifted_positions": positions,
             "grid": grid, "dftpy_grid": object(), "idx": [0, 1],
             "ions": [0, 1], "pseudo_pot": pseudo}
    module = LDA.LDAFunctional([1.0, 2.0], energy_offset=offset, store_energy=store_energy).double()
    with torch.no_grad():
        module.en_offset.fill_(0.037)
    result = module(atoms)
    rho, v_real, _, pos_bohr = LDA.prepare_functional_vars(atoms)
    terms = {"tf": LDA.thomas_fermi_en(rho, grid), "vw": LDA.von_weizsacker_en(rho, grid),
             "lda": LDA.lda_en(rho, grid), "hartree": LDA.hartree_en(rho, grid),
             "pseudo": LDA.pseudo_en(rho, grid, v_real),
             "real": module.ewald.real_energy(grid, pos_bohr),
             "correction": module.ewald.corr_energy(grid),
             "reciprocal": module.ewald.rec_energy(grid, pos_bohr)}
    output = result["energy"] if store_energy else result["energy_min"]
    loss = output.square().sum() + 0.13 * output.cos().sum()
    parameters = dict(module.named_parameters())
    values = [density, positions, *parameters.values()]
    gradients = torch.autograd.grad(loss, values, create_graph=True, retain_graph=True)
    directions = [torch.linspace(0.11, 0.31, value.numel()).reshape(value.shape) for value in values[:2]]
    directional = sum((grad * direction).sum() for grad, direction in zip(gradients, directions))
    limitations = []
    hessian = None
    try:
        hvp = torch.autograd.grad(directional, values[:2], retain_graph=True)
        if all(torch.isfinite(value).all() for value in hvp):
            hessian = hvp
        else:
            limitations.append("Torch Ewald/total second derivatives returned nonfinite values")
    except (RuntimeError, NotImplementedError) as error:
        limitations.append(f"Torch Ewald/total second derivative unsupported: {error}")
    # Independently probe the smooth density branch even if cdist restricts HVP.
    density_only = sum(terms[key] for key in ("tf", "vw", "lda", "hartree", "pseudo"))
    density_grad = torch.autograd.grad(density_only.sum(), density, create_graph=True, retain_graph=True)[0]
    density_hvp = torch.autograd.grad((density_grad * directions[0]).sum(), density)[0]
    node, state, skipped = export_module(module)
    rec = grid.reciprocal
    grid_metadata = {"shape": np.asarray(shape), "lattice": grid.lattice,
                     "volume": grid.volume, "point_volume": grid.point_volume,
                     "gg": rec.gg, "reciprocal_coords": rec.coords, "reciprocal_mask": rec.mask}
    assert all(torch.isfinite(value).all() for value in [output, *gradients, density_grad, density_hvp])
    return to_numpy_tree({"node": node, "state": state, "grid_metadata": grid_metadata,
                          "atoms": {"density": density.detach().reshape(2, -1),
                                    "shifted_positions": positions.detach(), "v_real": v_real},
                          "variables": {"density": density.detach(), "shifted_positions": positions.detach()},
                          "directions": dict(zip(("density", "shifted_positions"), directions)),
                          "output": output, "energy_min": result["energy_min"], "terms": terms,
                          "input_gradient": dict(zip(("density", "shifted_positions"), gradients[:2])),
                          "parameter_gradient": dict(zip(parameters, gradients[2:])),
                          "input_hvp": None if hessian is None else dict(zip(("density", "shifted_positions"), hessian)),
                          "density_branch_gradient": density_grad, "density_branch_hvp": density_hvp,
                          "limitations": limitations, "skipped_metadata": skipped,
                          "host_potential_scope": "Prescribed field; no external DFTpy pseudopotential assets"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    cases = {}
    for name, shape, lattice, offset, store in (
        ("orthogonal", (4, 4, 4), [[5.0, 0, 0], [0, 5.3, 0], [0, 0, 5.7]], True, True),
        ("triclinic", (4, 5, 4), [[5.0, 0.2, 0.1], [0.3, 5.3, 0.2], [0.1, 0.4, 5.7]], False, True),
        ("minimum_only", (3, 4, 4), [[4.6, 0, 0], [0, 4.9, 0], [0, 0, 5.1]], False, False),
    ):
        cases[name] = reference(shape, lattice, offset=offset, store_energy=store)
        print(f"Generated {name}: {cases[name]['limitations']}", flush=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save_tree(args.output, {"cases": cases, "provenance": {"torch": torch.__version__,
              "source_revision": revision, "dtype": "float64", "external_pseudopotential_assets": False}})


if __name__ == "__main__":
    main()
