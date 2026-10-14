"""Generate a portable reference bundle using the legacy Torch density expansion.

Run only in the project Torch runtime. The JAX tests intentionally do not import
Torch; this script records both source inputs and outputs using save_tree.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from equiv_dens.nn.property_output.density import (
    DensityExpansion, DensityCoeffsNetwork, DFDensityCoeffs, FreeAtomDensityCoeffs,
)
from equiv_dens_jax.serialization import save_tree
from equiv_dens_jax.convert import export_module
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.utils.orbitals import gto_norm, pyscf_gto_factor


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, dict):
        return {_numpy(k): _numpy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_numpy(v) for v in value)
    return value


def _generator_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def make_reference():
    torch.set_default_dtype(torch.float64)
    basis = {1: [(1, 1, 0), (1, 1, 1)]}
    atoms = {
        "batch_atom_numbers": torch.tensor([[1]], dtype=torch.long),
        "batch_positions": torch.tensor([[[0.1, -0.2, 0.05]]]),
        "coords": torch.tensor([[[0.4, 0.3, -0.1], [-0.5, 0.2, 0.7]]]),
        "coord_weights": torch.ones((1, 2)),
        "spherical_coeffs": [{
            (1, 0): torch.tensor([[[[0.7]]]]),
            (1, 1): torch.tensor([[[[0.2], [-0.4], [0.9]]]]),
        }],
        "radial_width": [{
            (1, 0): torch.tensor([[[[0.8]]]]),
            (1, 1): torch.tensor([[[[0.6]]]]),
        }],
        "radial_scale": [{
            (1, 0): torch.tensor([[[[1.1]]]]),
            (1, 1): torch.tensor([[[[0.5]]]]),
        }],
    }
    model = DensityExpansion(basis, density_grad=True)
    result = model({k: v for k, v in atoms.items()})
    components = {}
    for L in (0, 1):
        component = model(copy.deepcopy(atoms), eval_L=[L])
        components[L] = {"density": _numpy(component["density"]),
                         "density_grad": _numpy(component["density_grad"])}
    return {"metadata": {"source": "equiv_dens.nn.property_output.DensityExpansion",
                          "dtype": "float64", "case": "one H with s+p shells"},
            "node": {"type": "DensityExpansion", "attrs": {"orbital_basis": basis,
                      "density_grad": True, "expansion_constraint": None,
                      "integral_constraint": None}, "children": {}, "params": {}, "buffers": {}},
            "atoms": _numpy(atoms),
            "expected": {"density": _numpy(result["density"]),
                         "density_grad": _numpy(result["density_grad"])},
            "components": components}


def make_head_reference(variant="learned"):
    """Small exact head probe, including batched padding and a parameter gradient."""
    torch.set_default_dtype(torch.float64)
    basis = {1: [(1, 2, 0), (1, 1, 0), (1, 1, 1)],
             8: [(8, 2, 0), (8, 1, 0), (8, 1, 1)]}
    radial = {1: [([.7, 1.4], [.8, .3]), ([.5], [.6]), ([.5], [.6])],
              8: [([1.1, 2.0], [.9, .4]), ([.6], [.5]), ([.7], [.5])]}
    if variant == "constraint":
        # The current source constraint includes zero-padded radial entries in
        # its normalization, so ragged shell counts can produce 0 * inf -> NaN.
        # Use one radial function per L0 shell (and positive predicted widths)
        # to exercise the unchanged equation on a finite valid layout.
        basis = {1: [(1, 1, 0), (1, 1, 1)], 8: [(8, 1, 0), (8, 1, 1)]}
        radial = {1: [([.7], [.8]), ([.5], [.6])],
                  8: [([1.1], [.9]), ([.7], [.5])]}
    positive = variant == "positive"
    constraint = "coeffs_in_coeff_net" if variant == "constraint" else None
    options = dict(orbital_basis=basis, init_radial_coeffs=radial, order=1,
                   num_features=3, pred_radial_coeffs=variant != "fixed",
                   positive_coeffs=positive, integral_constraint=constraint,
                   clebsch_gordan=ClebschGordanMatrix(order_max=1),
                   core_basis_ratio=.5 if variant == "core" else 0,
                   coeff_weights={1:[[.4,.8],[.6],[.5]],8:[[.3,.9],[.5],[.7]]} if variant == "weighted" else None)
    torch.manual_seed(411)
    module = DensityCoeffsNetwork(**options).double()
    # Open the initialized radial prediction path so gradients exercise it.
    with torch.no_grad():
        if options["pred_radial_coeffs"]:
            for layer in (*module.radial_width, *module.radial_scale):
                layer.weight.copy_(torch.linspace(-.06,.07,layer.weight.numel()).reshape_as(layer.weight))
            if variant == "constraint":
                # The source constraint integrates Gaussian widths. Its current
                # tanh width head can produce negative exponents, for which the
                # source fractional-power normalization is NaN. Keep this
                # fixture inside the valid positive-width domain while testing
                # the unchanged constraint equation.
                for layer in module.radial_width:
                    layer.weight.zero_()
                    layer.bias.fill_(.2)
    nums = torch.tensor([[1, 8], [1, 0]], dtype=torch.long)
    positions = torch.tensor([[[.1,.2,.3],[-.2,.4,.1]], [[.3,-.1,.2],[0.,0.,0.]]])
    active = (nums.flatten()!=0).nonzero().flatten()
    features = [torch.linspace(-.4,.8,3*1*3).reshape(1,3,1,3).requires_grad_(),
                torch.linspace(.7,-.2,3*3*3).reshape(1,3,3,3).requires_grad_()]
    atoms = {"batch_atom_numbers": nums, "batch_atom_mask": (nums!=0).double(),
             "batch_positions": positions, "positions": positions, "batch_idx_pos": active,
             "sph_repr": features}
    if variant == "delta":
        # Delta path appends existing core coefficients after newly learned ones.
        groups = ({"spherical_coeffs": [], "radial_width": [], "radial_scale": [], "coeff_weights": []})
        for z in nums.max(0).values.tolist():
            sph,width,scale,weights={},{},{},{}
            if z:
                grouped={}
                for _,n,L in basis[z]: grouped.setdefault(L,[]).append(n)
                for L,counts in grouped.items():
                    key=(z,L); nshell=len(counts); rmax=max(counts)
                    sph[key]=torch.full((2,1,2*L+1,nshell),.07)
                    width[key]=torch.ones((2,1,rmax,nshell))
                    scale[key]=torch.full((2,1,rmax,nshell),.2)
                    weights[key]=torch.zeros((1,1,2*L+1,nshell))
            groups["spherical_coeffs"].append(sph); groups["radial_width"].append(width)
            groups["radial_scale"].append(scale); groups["coeff_weights"].append(weights)
        atoms.update(groups)
    reference_inputs = copy.deepcopy(atoms)
    atoms["sph_repr"] = torch.cat(features, dim=-2)  # Torch core takes packed features; the saved inputs stay per-L
    if variant == "delta":
        # The Torch head takes the existing core coefficients packed; the saved inputs keep the per-atom dicts.
        for key in ("spherical_coeffs", "radial_width", "radial_scale", "coeff_weights"):
            del atoms[key]
        lay = module.layout.get(nums.max(0).values.tolist(), positions.device, positions.dtype)
        atoms["packed_coeffs"] = {
            L: {"atom": lay["atom"][L], "coeff": torch.full((2, len(lay["atom"][L]), 2*L+1), .07),
                "width": torch.ones(2, len(lay["atom"][L]), module.layout.P[L]),
                "scale": torch.full((2, len(lay["atom"][L]), module.layout.P[L]), .2)}
            for L in range(len(lay["atom"])) if len(lay["atom"][L])}
    if variant == "constraint":
        constraint_layer = module.integral_constraint
        module.integral_constraint = None
        unconstrained = module(dict(atoms))
        module.integral_constraint = constraint_layer
        integral_terms = []
        for i, values in enumerate(unconstrained["spherical_coeffs"]):
            for key, coeff in values.items():
                if key[1] == 0:
                    width = unconstrained["radial_width"][i][key]
                    scale = unconstrained["radial_scale"][i][key]
                    integral_terms.append((coeff * scale / gto_norm(0, width) / pyscf_gto_factor).sum(-1))
        print("constraint pre-scale integral", torch.cat(integral_terms, dim=1).sum(1).tolist())
    result = module(atoms)
    if variant == "delta":
        # CoeffsDict.to_dicts only views the learned shells; the legacy per-atom layout is learned shells followed by
        # the appended core shells of the same (atom, L), rebuilt here from the concatenated packed coefficients.
        packed, zs = result["packed_coeffs"], nums.max(0).values.tolist()
        start = module.layout.get(zs, positions.device)["start"].tolist()
        sph, width, scale = [], [], []
        for a, z in enumerate(zs):
            sph.append({}), width.append({}), scale.append({})
            for L in range(module.layout.Lmax + 1):
                if z == 0 or (z, L) not in module.layout.nshell:
                    continue
                n, r, S = module.layout.nshell[(z, L)], module.layout.rmax[(z, L)], len(lay["atom"][L])
                rows = lambda name, w: torch.cat([packed[L][name][:, start[L][a]:start[L][a] + n, :w],
                                                  packed[L][name][:, S + start[L][a]:S + start[L][a] + n, :w]], dim=1)
                sph[a][(z, L)] = rows("coeff", 2*L+1).permute(0, 2, 1).unsqueeze(1)
                width[a][(z, L)] = rows("width", r).permute(0, 2, 1).unsqueeze(1)
                scale[a][(z, L)] = rows("scale", r).permute(0, 2, 1).unsqueeze(1)
        result = dict(result, spherical_coeffs=sph, radial_width=width, radial_scale=scale,
                      coeff_weights=reference_inputs["coeff_weights"])
    loss_terms = [result["df_coeffs"].square().sum()] if "df_coeffs" in result else []
    loss_terms.extend(x.square().sum() for group in ("radial_width","radial_scale")
                      for atom in result[group] for x in atom.values())
    loss_terms.extend(x.square().sum() for atom in result["spherical_coeffs"] for x in atom.values())
    loss = sum(loss_terms)
    named = list(module.named_parameters())
    grads = torch.autograd.grad(loss, [*features,*[p for _,p in named]], allow_unused=True)
    feature_grads = [torch.zeros_like(x) if g is None else g for x,g in zip(features,grads[:len(features)])]
    parameter_grads = {name: (torch.zeros_like(param) if grad is None else grad)
                       for (name,param),grad in zip(named,grads[len(features):])}
    node,state,skipped = export_module(module)
    expected = {k:_numpy(result[k]) for k in ("spherical_coeffs","radial_width","radial_scale","coeff_weights","df_coeffs") if k in result}
    serialized_loss = (float(np.square(np.asarray(expected["df_coeffs"])).sum())
                       if "df_coeffs" in expected else 0.0)
    serialized_loss += sum(float(np.square(np.asarray(v)).sum())
                           for group in ("radial_width","radial_scale","spherical_coeffs")
                           for atom in expected[group] for v in atom.values())
    torch_loss = float(_numpy(loss))
    if not np.isclose(serialized_loss, torch_loss, atol=1e-10, rtol=1e-12):
        finite_outputs = {name: all(np.isfinite(np.asarray(v)).all()
                                    for atom in expected[name] for v in atom.values())
                          for name in ("radial_width", "radial_scale", "spherical_coeffs")}
        if "df_coeffs" in expected:
            finite_outputs["df_coeffs"] = bool(np.isfinite(expected["df_coeffs"]).all())
        raise AssertionError(
            f"{variant}: serialized output loss {serialized_loss} != Torch loss {torch_loss}; "
            f"finite_outputs={finite_outputs}"
        )
    return {"metadata":{"source":"DensityCoeffsNetwork","variant":variant,
                         "torch_version":torch.__version__,"skipped_attributes":skipped,
                         "generator_sha256":_generator_sha256(),
                         "loss_definition":("sum_squares(radial_width,radial_scale,spherical_coeffs)"
                                            if variant == "delta" else
                                            "sum_squares(df_coeffs,radial_width,radial_scale,spherical_coeffs)"),
                         "fixture_note":("equal unpadded L0 shells and positive predicted widths avoid the source "
                                         "zero/negative-width normalization NaN; the historical ragged-padding "
                                         "failure is not repaired" if variant == "constraint" else None)},
            "node":node,"state":state,"atoms":_numpy(reference_inputs),
            "expected":expected,
            "loss":torch_loss,"feature_grads":_numpy(feature_grads),
            "parameter_grads":_numpy(parameter_grads),"loss_mode":"head"}


def make_df_reference(kind="df"):
    torch.set_default_dtype(torch.float64)
    from equiv_dens.utils.orbitals import PackedLayout

    def from_df_vector(self, vec, zs, require_grads=False):
        # PackedLayout.from_df_vector detaches the coefficients into new leaves; the reference gradient
        # is w.r.t. the input df_coeffs, so keep the graph through the (linear) packing.
        lay = self.get(zs, vec.device, vec.dtype)
        flat = (vec * lay['sign'])[:, lay['inv']]
        packed, off = {}, 0
        for L in range(self.Lmax + 1):
            S = len(lay['atom'][L])
            coeff = flat[:, off:off + S * (2 * L + 1)].reshape(vec.shape[0], S, 2 * L + 1)
            off += S * (2 * L + 1)
            if S:
                packed[L] = {'atom': lay['atom'][L], 'coeff': coeff,
                             'width': lay['width0'][L].expand(vec.shape[0], -1, -1).clone().requires_grad_(require_grads),
                             'scale': lay['scale0'][L].expand(vec.shape[0], -1, -1).clone().requires_grad_(require_grads)}
        return packed
    PackedLayout.from_df_vector = from_df_vector
    basis={1:[(1,1,0),(1,1,1)]}
    radial={1:[([.8],[1.1]),([.6],[.5])]}
    if kind=="df":
        module=DFDensityCoeffs(basis,radial,torch.float64)
    else:
        module=FreeAtomDensityCoeffs(basis,radial,{1:{"df_coeffs":torch.tensor([.3,.2,-.4,.7])}},torch.float64)
    nums=torch.tensor([[1,1],[1,1]],dtype=torch.long)
    positions=torch.tensor([[[0.,0.,0.],[.8,.1,0.]],[[.1,.2,0.],[.9,.2,.1]]])
    coeff=torch.tensor([[.5,.2,-.3,.4,.7,-.2,.1,.6],[-.4,.6,.3,-.1,.2,.8,-.5,.4]],requires_grad=True)
    if kind=="freeatom":
        # This branch builds the same per-atom vector from static atom_dens entries.
        pass
    atoms={"batch_atom_numbers":nums,"positions":positions,
           "idx_i":torch.tensor([0,1]),"idx_j":torch.tensor([1,0]),
           "df_coeffs":coeff}
    input_atoms={k:v.detach().clone().requires_grad_(v.is_floating_point()) if v.is_floating_point() else v.clone()
                 for k,v in atoms.items()}
    result=module(input_atoms)
    loss=sum(x.square().sum() for group in ("spherical_coeffs","radial_width","radial_scale")
             for atom in result[group] for x in atom.values())
    grad=torch.autograd.grad(loss,input_atoms["df_coeffs"],allow_unused=True)[0]
    if grad is None: grad=torch.zeros_like(coeff)
    node,state,skipped=export_module(module)
    return {"metadata":{"source":type(module).__name__,"variant":kind,"skipped_attributes":skipped},
            "node":node,"state":state,"atoms":_numpy(atoms),
            "expected":{k:_numpy(result[k]) for k in ("spherical_coeffs","radial_width","radial_scale")},
            "loss":_numpy(loss),"input_grad":_numpy(grad),"loss_mode":"df"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("expansion","head","df","freeatom"), default="expansion")
    parser.add_argument("--variant", choices=("learned","fixed","positive","constraint","weighted","core","delta"), default="learned")
    args = parser.parse_args()
    tree = (make_reference() if args.kind == "expansion" else
            make_head_reference(args.variant) if args.kind == "head" else
            make_df_reference("df" if args.kind == "df" else "freeatom"))
    save_tree(args.output, tree)
    print(f"Wrote legacy Torch density reference to {args.output}")


if __name__ == "__main__":
    main()
