"""Generate exact Torch references in the Torch runtime, never during JAX tests."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import numpy as np
import torch
from equiv_dens.nn.modules.activations import Swish, ShiftedSoftplus
from equiv_dens.nn.modules.radial_basis_functions import (
    BernsteinRadialBasisFunctions, GaussianRadialBasisFunctions,
    ExponentialGaussianRadialBasisFunctions, ExponentialBernsteinRadialBasisFunctions,
    OverlapBernsteinRadialBasisFunctions,
)
from equiv_dens.nn.modules.spherical_harmonic_layers import (
    SphericalLinear, SelfMixing, PairMixing, PairInteraction,
)
from equiv_dens.nn.modules.network_blocks import (
    ResidualBlock, InteractionBlock, NonmixingInteractionBlock,
)
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.representation.spherical_harmonic import EquivariantSphericalHarmonics
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
from equiv_dens.utils.spherical_harmonics_deriv import spherical_harmonics_deriv
from equiv_dens_jax.convert import export_module


def export(module):
    """Use the production conversion boundary in the separate Torch process."""
    node, state, _ = export_module(module)
    return node, state


def pack(tree, arrays, prefix):
    if isinstance(tree, dict):
        return {"dict": {k: pack(v,arrays,prefix) for k,v in tree.items()}}
    if isinstance(tree, (tuple,list)):
        return {"list": [pack(v,arrays,prefix) for v in tree]}
    name = f"{prefix}_{len(arrays)}"
    arrays[name] = tree.detach().cpu().numpy() if isinstance(tree,torch.Tensor) else np.asarray(tree)
    return {"array": name}


def to_packed(arg):
    """Per-L list of [..., 2L+1, F] -> packed [..., (L+1)^2, F] (Torch core format)."""
    return torch.cat(arg,dim=-2) if isinstance(arg,list) else arg


def to_per_L(tree):
    """Packed [..., D, F] outputs -> per-L list, the format stored for the JAX tests."""
    if isinstance(tree,tuple):
        return tuple(to_per_L(v) for v in tree)
    order = int(round(tree.shape[-2]**0.5))-1
    return list(torch.split(tree,[2*L+1 for L in range(order+1)],dim=-2))


def save_case(root,name,module,inputs,representation=False,derivatives=True,packed=True):
    module = module.double()
    # Exercise nonzero residual contributions, with exact shared parameter arrays.
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(0.15*torch.randn_like(parameter))
    node,state = export(module)
    arrays = {}
    for group, values in state.items():
        for key,value in values.items():
            arrays[f"state/{group}/{key}"] = value
    data = {"schema":1,"node":node,"state_keys":{g:list(v) for g,v in state.items()},"representation":representation}
    data["inputs"] = pack(inputs,arrays,"input")
    floating = []
    def require(tree):
        if isinstance(tree,dict):
            return {k: require(v) for k,v in tree.items()}
        if isinstance(tree,(tuple,list)):
            return [require(v) for v in tree]
        if tree.is_floating_point():
            leaf = tree.detach().clone().requires_grad_(derivatives)
            floating.append(leaf)
            return leaf
        return tree
    current = require(inputs)
    result = module(current[0]) if representation else module(*[to_packed(a) for a in current])
    outputs = result["sph_repr"] if representation else result
    if packed:
        outputs = to_per_L(outputs)
    data["outputs"] = pack(outputs,arrays,"output")
    if derivatives:
        def leaves(tree):
            if isinstance(tree,(tuple,list)):
                return [v for item in tree for v in leaves(item)]
            return [tree]
        loss = sum(v.square().sum() for v in leaves(outputs))
        parameter_probe = name in {
            "Swish", "ShiftedSoftplus", "ExponentialGaussianRadialBasisFunctions",
            "self_parityTrue", "pair_parityFalse", "residual_norm2",
            "representation_norm1_exp-bernstein", "representation_shared_params",
        }
        named_parameters = dict(module.named_parameters())
        parameter_values = [named_parameters[k] for k in state["params"]] if parameter_probe else []
        all_grads = torch.autograd.grad(loss,floating+parameter_values,create_graph=True,allow_unused=True)
        grad = all_grads[:len(floating)]
        grad = [torch.zeros_like(x) if g is None else g for x,g in zip(floating,grad)]
        directions = [torch.sin(torch.arange(x.numel(),dtype=x.dtype)+1).reshape(x.shape)/max(x.numel(),1) for x in floating]
        first = sum((g*d).sum() for g,d in zip(grad,directions))
        hess = torch.autograd.grad(first,floating,allow_unused=True,retain_graph=parameter_probe)
        second = sum((h*d).sum() for h,d in zip(hess,directions) if h is not None)
        data["grads"] = pack(grad,arrays,"grad")
        data["directions"] = pack(directions,arrays,"direction")
        arrays["second"] = second.detach().numpy()
        if parameter_probe:
            parameter_grads = [torch.zeros_like(x) if g is None else g for x,g in zip(parameter_values,all_grads[len(floating):])]
            data["parameter_grads"] = pack(parameter_grads,arrays,"parameter_grad")
            mixed = torch.autograd.grad(first,parameter_values,allow_unused=True)
            mixed = [torch.zeros_like(x) if g is None else g for x,g in zip(parameter_values,mixed)]
            data["mixed_grads"] = pack(mixed,arrays,"mixed_grad")
    np.savez(root/f"{name}.npz",**arrays)
    (root/f"{name}.json").write_text(json.dumps(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",default="scratch/jax_pipeline/layers-production")
    parser.add_argument("--allocation",default=os.environ.get("SLURM_JOB_ID"))
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    (root/"provenance.txt").write_text(json.dumps({
        "seed":704,"torch":torch.__version__,"numpy":np.__version__,
        "python":platform.python_version(),"hostname":platform.node(),
        "slurm_job":args.allocation,"device":"cpu","dtype":"float64",
        "source_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [Path(__file__),
                             *Path("src/equiv_dens/nn/modules").glob("*.py"),
                             Path("src/equiv_dens/nn/representation/spherical_harmonic.py"),
                             Path("src/equiv_dens/utils/spherical_harmonics.py"),
                             Path("src/equiv_dens/utils/spherical_harmonics_deriv.py"),
                             Path("src/equiv_dens_jax/layers.py"),
                             Path("src/equiv_dens_jax/harmonics.py"),
                             Path("src/equiv_dens_jax/representation.py")]},
        "reference":"current equiv_dens Torch equations, exact exported arrays",
        "output_gate":{"atol":1e-9,"rtol":1e-8},
        "derivative_gate":{"atol":1e-8,"rtol":1e-7},
    },indent=2)+"\n")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(704)
    cg = ClebschGordanMatrix(3)
    xs = lambda order=2: [0.3*torch.randn(2,3,2*L+1,3) for L in range(order+1)]
    pair = lambda order=2: [0.3*torch.randn(2,6,2*L+1,3) for L in range(order+1)]
    rbf = torch.rand(2,6,1,4)
    angular = [torch.randn(2,6,2*L+1,1) for L in range(4)]
    idx_i = torch.tensor([0,0,1,1,2,2])
    idx_j = torch.tensor([1,2,0,2,0,1])
    for cls in (Swish,ShiftedSoftplus):
        save_case(root,cls.__name__,cls(3),[torch.randn(2,3,3)],packed=False)
    radial_classes = (BernsteinRadialBasisFunctions,GaussianRadialBasisFunctions,
                      ExponentialGaussianRadialBasisFunctions,ExponentialBernsteinRadialBasisFunctions,
                      OverlapBernsteinRadialBasisFunctions)
    for cls in radial_classes:
        r = torch.tensor([0.2,0.8,1.7,2.9]).reshape(1,4,1) if cls is BernsteinRadialBasisFunctions else torch.tensor([0.2,0.8,1.7,2.9,3.0,3.5]).reshape(2,3,1)
        save_case(root,cls.__name__,cls(4,3.0),[r],packed=False)
    for parity in (False,True):
        save_case(root,f"self_parity{parity}",SelfMixing(2,3,3,cg,normalize=1,parity=parity),[xs()])
        save_case(root,f"pair_parity{parity}",PairMixing(2,2,3,4,3,cg,normalize=1,parity=parity),[pair(),pair(),rbf])
    save_case(root,"pair_nonmix",PairInteraction(2,4,3,cg,normalize=1),[pair(),pair(),rbf])
    save_case(root,"linear_nonmix",SphericalLinear(2,3,2,4,mix_orders=False),[xs()])
    save_case(root,"linear_mix",SphericalLinear(2,3,3,4,cg),[xs()])
    for norm in (0,1,2):
        save_case(root,f"residual_norm{norm}",ResidualBlock(2,3,cg,normalize=norm,order_out=3),[xs()])
        kwargs = dict(order=2,num_features=3,num_basis_functions=4,num_residual_pre_vi=1,num_residual_pre_vj=1,num_residual_post_v=1,clebsch_gordan=cg,normalize=norm)
        save_case(root,f"interaction_norm{norm}",InteractionBlock(**kwargs,mixing_order=3),[xs(),rbf,angular,idx_i,idx_j])
        for residual in (False,True):
            save_case(root,f"nonmix_norm{norm}_residual{residual}",NonmixingInteractionBlock(**kwargs,residual=residual),[xs(),rbf,angular,idx_i,idx_j])
    atoms = {"positions":torch.tensor([[[0.0,0.0,0.0],[1.0,0.2,0.1],[0.2,1.3,-0.2]],[[0.0,0.0,0.0],[1.1,-0.1,0.2],[-0.2,1.0,0.3]]]),
             "atom_numbers":torch.tensor([[1,8,1],[1,8,0]]),"atom_mask":torch.tensor([[1.,1.,1.],[1.,1.,0.]]),"idx_i":idx_i,"idx_j":idx_j}
    for norm in (0,1,2):
        for basis in ("gaussian","exp-gaussian","exp-bernstein"):
            m = EquivariantSphericalHarmonics(orbital_basis={1:[(1,0,0)],8:[(8,0,0)]},order=[1,2],mixing_order=[2,2],num_modules=2,num_features=3,num_basis_functions=4,cutoff=3.,normalize=norm,clebsch_gordan=cg,basis_functions=basis,nonmixing_interaction=norm==1)
            save_case(root,f"representation_norm{norm}_{basis}",m,[atoms],representation=True)
    # A deliberate tied-weight probe qualifies production alias canonicalization.
    shared = EquivariantSphericalHarmonics(
        orbital_basis={1:[(1,0,0)],8:[(8,0,0)]},order=[1,2],mixing_order=[2,2],
        num_modules=2,num_features=3,num_basis_functions=4,cutoff=3.,normalize=1,
        clebsch_gordan=cg,basis_functions="exp-bernstein")
    shared.module[1].residual_post_x.stack[0].activation_pre.alpha = (
        shared.module[0].residual_post_x.stack[0].activation_pre.alpha)
    save_case(root,"representation_shared_params",shared,[atoms],representation=True)
    u = torch.randn(2,3,3)
    u /= u.norm(dim=-1,keepdim=True)
    arrays = {"u":u.numpy()}
    for L,y in enumerate(spherical_harmonics(10,u)):
        arrays[f"Y{L}"] = y.numpy()
    for L,y in enumerate(spherical_harmonics_deriv(5,u)):
        arrays[f"dY{L}"] = y.numpy()
    np.savez(root/"harmonics.npz",**arrays)
    print(f"Wrote {len(list(root.glob('*.json')))} layer/representation fixtures and harmonic orders 0..10 to {root}")


if __name__ == "__main__":
    main()
