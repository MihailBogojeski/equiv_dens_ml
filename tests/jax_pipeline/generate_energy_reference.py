"""Generate exact portable energy/moment references in the separate Torch runtime.

This is an explicit preparation command, never imported by the JAX test suite.
Run: tools/runtime/run.sh torch python tests/jax_pipeline/generate_energy_reference.py --output PATH
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np
import torch

import equiv_dens.compat  # noqa: F401
from benchmarks.fixtures import build_model, fixture_spec, make_atoms, clone_tree
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.property_output.energy import (
    SphericalHarmonicsEnergyNetwork, SphericalHarmonicsEmbeddingEnergyNetwork,
    SphericalLinearEnergyNetwork, RepresentationEnergyNetwork,
)
from equiv_dens.nn.property_output.dipole_moment import DipoleMomentCalc, DipoleMomentIntorCalc
from equiv_dens_jax.convert import export_module, to_numpy_tree
from equiv_dens_jax.serialization import save_tree


def _fixed_radial_workaround():
    """SphericalLinearEnergyNetwork.forward indexes scale_fs/width_fs even for pred_radial_coeffs=False, where
    coeffs_dict_to_tensors now returns empty lists. Feed placeholders (never used by the network) so the fixed-radial
    case still runs, and append the atom-density channels on the dict route; the Torch source is left unchanged."""
    from equiv_dens.nn.property_output import energy
    from equiv_dens.utils.orbitals import append_atom_df
    original = energy.coeffs_dict_to_tensors

    def patched(coeffs, radial_coeffs=True, atom_df=None, **kwargs):
        # The dict route (differentiable per-atom inputs, as the JAX tests feed them) ignores atom_df; append the
        # free-atom channels here exactly as the packed route does.
        sph, scale, width = original(coeffs, radial_coeffs=radial_coeffs, **kwargs)
        if atom_df is not None:
            l0_map, c, w, sc = atom_df
            Z = coeffs['batch_atom_numbers'].reshape(-1)
            sph[0] = append_atom_df(sph[0], Z, l0_map, c)
            if radial_coeffs:
                width[0] = append_atom_df(width[0], Z, l0_map, w)
                scale[0] = append_atom_df(scale[0], Z, l0_map, sc)
        return sph, scale or list(sph), width or list(sph)
    energy.coeffs_dict_to_tensors = patched


def leaves(tree):
    if isinstance(tree, dict):
        return [v for value in tree.values() for v in leaves(value)]
    if isinstance(tree, (list, tuple)):
        return [v for value in tree for v in leaves(value)]
    return [tree]


def rebuild(tree, iterator):
    if isinstance(tree, dict):
        return {key: rebuild(value, iterator) for key, value in tree.items()}
    if isinstance(tree, list):
        return [rebuild(value, iterator) for value in tree]
    if isinstance(tree, tuple):
        return tuple(rebuild(value, iterator) for value in tree)
    return next(iterator)


def reference(module, atoms, fields, *, center=True):
    module = module.double().eval()
    # Export after dtype conversion, then match flat canonical identities.
    node, state, skipped = export_module(module)
    parameters = dict(module.named_parameters())
    assert tuple(parameters) == tuple(state["params"])
    variables = clone_tree({key: atoms[key] for key in fields})
    inputs = leaves(variables)
    for value in inputs:
        value.requires_grad_(True)
    direction = rebuild(variables, iter([
        torch.linspace(0.17, 0.43, value.numel(), dtype=value.dtype).reshape(value.shape)
        for value in inputs
    ]))
    prepared = clone_tree(atoms)
    # RepresentationEnergyNetwork replaces fs[0] in-place. Give the evaluator
    # fresh containers with the same differentiable tensor identities so saved
    # inputs still describe the inputs at which its derivatives were evaluated.
    prepared.update(rebuild(variables, iter(inputs)))
    if isinstance(prepared.get("sph"), list):
        prepared["sph"] = torch.cat(prepared["sph"], dim=-2)
    if isinstance(module, DipoleMomentCalc):
        output = module(prepared, center_coordinates=center)["dipole_moment"]
    elif isinstance(module, DipoleMomentIntorCalc):
        output = module(prepared)["dipole_moment"]
    else:
        output = module(prepared)["energy"]
    scalar = output.square().sum() + 0.13 * output.cos().sum()
    all_values = inputs + list(parameters.values())
    derivatives = torch.autograd.grad(scalar, all_values, create_graph=True,
                                      retain_graph=True, allow_unused=True)
    derivatives = [torch.zeros_like(value) if gradient is None else gradient
                   for value, gradient in zip(all_values, derivatives)]
    input_grad = derivatives[:len(inputs)]
    directional = sum((gradient * tangent).sum()
                      for gradient, tangent in zip(input_grad, leaves(direction)))
    hessian = torch.autograd.grad(directional, inputs, retain_graph=True, allow_unused=True)
    hessian = [torch.zeros_like(value) if gradient is None else gradient
               for value, gradient in zip(inputs, hessian)]
    mixed_loss = sum(gradient.square().sum() for gradient in input_grad)
    if parameters:
        mixed = torch.autograd.grad(mixed_loss, list(parameters.values()), allow_unused=True)
        mixed = {key: torch.zeros_like(value) if gradient is None else gradient
                 for (key, value), gradient in zip(parameters.items(), mixed)}
    else:
        mixed = {}
    payload = {"node": node, "state": state, "atoms": to_numpy_tree(atoms),
               "fields": tuple(fields), "variables": to_numpy_tree(variables),
               "directions": to_numpy_tree(direction), "output": to_numpy_tree(output),
               "input_gradient": to_numpy_tree(rebuild(variables, iter(input_grad))),
               "input_hvp": to_numpy_tree(rebuild(variables, iter(hessian))),
               "parameter_gradient": to_numpy_tree(dict(zip(parameters, derivatives[len(inputs):]))),
               "input_gradient_loss_parameter_gradient": to_numpy_tree(mixed),
               "center_coordinates": center, "skipped_metadata": skipped}
    if isinstance(module, DipoleMomentIntorCalc):
        payload["integrals"] = np.stack([value.numpy() for value in module._cached_int1e_r])
        if module.remove_atom_density:
            payload["atom_dipole_moment"] = to_numpy_tree(prepared["atom_dipole_moment"])
    return payload


def energy_cases():
    _fixed_radial_workaround()
    torch.manual_seed(8021)
    torch.set_default_dtype(torch.float64)
    # Geometry supplies harmonics through L=2 for the changing-order energy case.
    spec = fixture_spec("tiny", torch.float64, order=2)
    model = build_model(spec, mode="coefficients")
    inputs = {}
    for case in ("tiny", "batch"):
        atoms = make_atoms(case, torch.float64)
        out = model.density_repr_model(atoms)
        out["spherical_coeffs"]  # CoeffsDict builds the per-atom dicts lazily from the packed coefficients
        out = dict(out)
        # saved format: per-L spherical harmonics [..., 2L+1, 1]; reference() packs them for the Torch call
        order = int(round(out["sph"].shape[-2] ** 0.5)) - 1
        out["sph"] = list(torch.split(out["sph"], [2 * L + 1 for L in range(order + 1)], dim=-2))
        # drop the packed form so the energy head takes the differentiable per-atom dict route
        out.pop("packed_coeffs"), out.pop("packed_chan")
        out["sph_dict"] = model.density_repr_model[1].sph_dict  # per-(z, L) channel index, a head attribute now
        inputs[case] = clone_tree(out)
    configurations = [
        ("spherical", "tiny", {"L0_start": False, "normalize": 0}),
        ("spherical_norm2", "batch", {"L0_start": False, "normalize": 2, "num_modules": 2}),
        ("spherical_L0", "tiny", {"L0_start": True, "normalize": 0}),
        ("spherical_orders", "tiny", {"L0_start": False, "order": [1, 2], "num_modules": 2, "normalize": 1}),
        ("spherical_gaussian", "tiny", {"L0_start": False, "basis_functions": "gaussian"}),
        ("spherical_bernstein", "tiny", {"L0_start": False, "basis_functions": "bernstein"}),
        ("spherical_exp_gaussian", "tiny", {"L0_start": False, "basis_functions": "exp-gaussian"}),
        ("embedding", "tiny", {}),
        ("embedding_norm2", "batch", {"normalize": 2}),
        ("linear", "tiny", {}),  # Torch SphericalLinearEnergyNetwork now sums the energy over the whole batch
        ("linear_fixed", "tiny", {"pred_radial_coeffs": False}),
        ("linear_orders", "tiny", {"order": [1, 0], "num_modules": 2, "normalize": 2}),
        ("representation", "batch", {}),
        ("representation_ssp", "tiny", {"activation": "ssp"}),
    ]
    atom_dens = {z: {"df_basis": {z: [[0, [0.6, 1.0]], [0, [1.4, 1.0]]]},
                     "df_coeffs": torch.tensor([[0.9, 0.1]])} for z in (1, 8)}
    configurations.append(("spherical_SAD", "tiny", {"L0_start": False, "atom_dens": atom_dens}))
    output = {}
    fields = ("spherical_coeffs", "radial_width", "radial_scale", "distances", "sph")
    for name, case, overrides in configurations:
        common = dict(orbital_basis=spec["orbital_basis"], order=1, num_features=4,
                      num_modules=1, num_basis_functions=4, cutoff=5.0,
                      clebsch_gordan=ClebschGordanMatrix(order_max=2))
        common.update(overrides)
        if name.startswith("representation"):
            module = RepresentationEnergyNetwork(order=1, num_features=4,
                                                  activation=overrides.get("activation", "swish"),
                                                  clebsch_gordan=common["clebsch_gordan"])
            selected = ("sph_repr",)
        elif name.startswith("linear"):
            common.pop("num_basis_functions")
            common.pop("cutoff")
            module = SphericalLinearEnergyNetwork(**common)
            selected = fields[:3]
        elif name.startswith("embedding"):
            module = SphericalHarmonicsEmbeddingEnergyNetwork(**common)
            selected = fields
        else:
            module = SphericalHarmonicsEnergyNetwork(**common)
            selected = fields
        for parameter_name, value in module.named_parameters():
            if parameter_name.endswith(("alpha", "beta")):
                with torch.no_grad():
                    value.uniform_(0.7, 1.25)
        output[name] = reference(module, inputs[case], selected)
        print(f"Generated {name}", flush=True)
    return output


def moment_cases():
    grid = {"batch_atom_numbers": torch.tensor([[8, 1, 0], [1, 1, 0]]),
            "batch_positions": torch.tensor([[[0.1, -0.2, 0.3], [0.8, 0.2, -0.1], [0, 0, 0]],
                                               [[0.2, 0.1, 0.4], [-0.3, 0.7, 0.1], [0, 0, 0]]]),
            "density": torch.tensor([[0.8, -0.1, 0.3, 0.6], [0.2, 0.7, -0.2, 0.3]]),
            "coords": torch.tensor([[[0.2, 0.1, -0.3], [-0.4, 0.3, 0.8], [0.6, -0.1, 0.2], [0.4, 0.5, 0.1]]]).repeat(2, 1, 1),
            "coord_weights": torch.tensor([[0.12, 0.2, 0.08, 0.3], [0.1, 0.07, 0.2, 0.4]])}
    fields = ("batch_positions", "density", "coords", "coord_weights")
    result = {f"grid_{center}": reference(DipoleMomentCalc(), grid, fields, center=center)
              for center in (True, False)}
    analytic = {"batch_atom_numbers": torch.tensor([[8]]),
                "batch_positions": torch.tensor([[[0.1, 0.2, 0.3]]]),
                "spherical_coeffs": [{(8, 0): torch.tensor([[[[0.7, 0.3]]]]),
                                      (8, 1): torch.tensor([[[[0.1, -0.2], [0.3, 0.05], [-0.1, 0.2]]]])}],
                "radial_width": [{(8, 0): torch.tensor([[[[0.8, 1.2]]]]),
                                  (8, 1): torch.tensor([[[[0.9, 1.3]]]])}],
                "radial_scale": [{(8, 0): torch.tensor([[[[0.9, 0.5]]]]),
                                  (8, 1): torch.tensor([[[[0.7, -0.4]]]])}]}
    basis = {8: [(8, 1, 0), (8, 1, 0), (8, 1, 1), (8, 1, 1)]}
    result["analytic_sp"] = reference(DipoleMomentIntorCalc(basis), analytic,
                                      ("batch_positions", "spherical_coeffs"))
    free_df = clone_tree(analytic)
    free_df.update(atom_df_coeffs=torch.tensor([[0.6, 0.4]]),
                   atom_df_coeffs_basis={8: [[0, [0.8, 1.0]], [0, [1.2, 1.0]]]})
    result["analytic_free_df"] = reference(DipoleMomentIntorCalc(basis, remove_atom_density=True),
                                           free_df, ("batch_positions", "spherical_coeffs"))
    free_mo = clone_tree(analytic)
    free_mo.update(atom_mo_coeffs=torch.eye(5)[None], atom_mo_coeffs_occ=torch.tensor([[2., 2., 2., 2., 0.]]),
                   atom_mo_coeffs_basis="sto-3g", positions=analytic["batch_positions"].clone())
    result["analytic_free_mo"] = reference(DipoleMomentIntorCalc(basis, remove_atom_density=True),
                                           free_mo, ("batch_positions", "spherical_coeffs"))
    cached = DipoleMomentIntorCalc(basis, dipole_every_n_steps=2)
    node, state, _ = export_module(cached)
    sequence = []
    for step, shift in enumerate((0.0, 0.01, 0.02, 0.8, 0.81)):
        atoms = clone_tree(analytic)
        atoms["batch_positions"] += shift
        for key in atoms["spherical_coeffs"][0]:
            atoms["spherical_coeffs"][0][key] *= 1 + 0.1 * step
            atoms["radial_width"][0][key] *= 1 + 0.05 * step
        observed = cached(clone_tree(atoms))["dipole_moment"]
        sequence.append({"atoms": to_numpy_tree(atoms), "output": to_numpy_tree(observed)})
    result["analytic_cache"] = {"node": node, "state": state, "sequence": sequence}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    payload = {"energy": energy_cases(), "moments": moment_cases(),
               "provenance": {"torch": torch.__version__, "source_revision": revision,
                              "precision": "float64", "source": str(root / "src/equiv_dens")}}
    save_tree(args.output, payload)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
