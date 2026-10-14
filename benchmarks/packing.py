"""Exact head/codec fixtures and paired component timing (coordinator holds lock).

The reference loads D's source files from reference_root; neighboring primitives
are shared deliberately to isolate D. Whole-checkout parity belongs to pipeline.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from equiv_dens.nn.property_output import density
from equiv_dens.utils import orbitals
from benchmarks.timing import paired_benchmark

REFERENCE_ROOT = Path('/home/johannes/equiv_dens_ml-paper-reference')


def load_reference(reference_root=REFERENCE_ROOT):
    root = Path(reference_root)
    modules = []
    for name, relative in [('packing_orbitals', 'utils/orbitals.py'),
                           ('packing_density', 'nn/property_output/density.py')]:
        spec = importlib.util.spec_from_file_location(name, root / 'src/equiv_dens' / relative)
        module = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        modules.append(module)
    modules[1].coeffs_dict_to_vector = modules[0].coeffs_dict_to_vector
    return tuple(modules)


def basis_fixture(constraint_safe=False):
    # Intentionally interleave shell degrees and use unequal primitive counts.
    basis = {1: [(1, 2, 0), (1, 1, 1), (1, 1, 0)],
             6: [(6, 3, 0), (6, 2, 1), (6, 1, 0), (6, 2, 2), (6, 1, 1)]}
    if constraint_safe:
        # The legacy constraint evaluates normalization at padded zero widths.
        # Equal L0 primitive counts avoid its pre-existing singularity.
        basis = {z: [(z, 1 if L == 0 else n, L) for z, n, L in shells]
                 for z, shells in basis.items()}
    radial = {z: [(np.linspace(.6, 1.4, orb[1]),
                   np.linspace(.2, .7, orb[1])) for orb in shells]
              for z, shells in basis.items()}
    return basis, radial


def make_case(*, reference_root=REFERENCE_ROOT, dtype=torch.float64,
              device='cpu', learned=True, positive=False, constraint=None,
              remove_atom_density=False, repetitions=1, batch_size=2,
              constraint_safe=True, weighting=False):
    ref_orbitals, ref_density = load_reference(reference_root)
    basis, radial = basis_fixture(constraint_safe=bool(constraint and constraint_safe))
    options = dict(orbital_basis=basis, init_radial_coeffs=radial,
                   order=2, num_features=6, nonmixing=True,
                   pred_radial_coeffs=learned, positive_coeffs=positive,
                   integral_constraint=constraint,
                   remove_atom_density=remove_atom_density)
    if weighting:
        options['coeff_weights'] = {z: [np.linspace(.4, .8, orb[1]) for orb in shells]
                                    for z, shells in basis.items()}
    torch.manual_seed(2841)
    candidate = density.DensityCoeffsNetwork(**options).to(device=device, dtype=dtype)
    # Exercise learned radial dependencies that zero initialization would hide.
    with torch.no_grad():
        if learned:
            for module in (*candidate.radial_width, *candidate.radial_scale):
                module.weight.uniform_(-.07, .07)
    reference = ref_density.DensityCoeffsNetwork(**options).to(device=device, dtype=dtype)
    reference.load_state_dict(candidate.state_dict(), strict=True)
    for model in (reference, candidate):
        for module in model.modules():
            if hasattr(module, 'contraction_backend'):
                module.contraction_backend = 'broadcast'
    nums = torch.tensor([[1, 1, 6, 6], [1, 0, 6, 0]], device=device)
    if constraint and constraint_safe:
        nums[1] = nums[0]
    nums = nums.repeat((batch_size + 1) // 2, repetitions)[:batch_size]
    positions = torch.linspace(.1, 1.9, nums.numel() * 3, device=device,
                               dtype=dtype).reshape(*nums.shape, 3).requires_grad_()
    inputs = inputs_from_positions(nums, positions, candidate.num_features)
    state_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                                       for v in candidate.state_dict().values())).hexdigest()
    manifest = dict(fixture='packing-ragged-interleaved-v1', dtype=str(dtype),
                    device=str(device), learned=learned, positive=positive,
                    constraint=constraint, remove_atom_density=remove_atom_density,
                    constraint_safe=constraint_safe,
                    weighting=weighting, contraction_backend='broadcast',
                    batch_size=batch_size, padded_atoms=nums.shape[1],
                    state_sha256=state_hash, reference_root=str(reference_root),
                    reference_scope='D source; shared neighboring primitives')
    return reference, candidate, inputs, manifest, ref_orbitals


def inputs_from_positions(nums, positions, features):
    idx = nums.flatten().nonzero().flatten()
    scalar = positions.reshape(-1, 3)[idx].sin().sum(-1)
    fs = [(scalar[None, :, None, None] + .3).expand(1, -1, 2 * L + 1, features)
          * torch.linspace(.2, 1.1, (2 * L + 1) * features,
                           dtype=positions.dtype, device=positions.device).reshape(1, 1, 2 * L + 1, features)
          for L in range(3)]
    return dict(batch_atom_numbers=nums, batch_positions=positions,
                batch_atom_mask=(nums != 0).to(positions.dtype),
                batch_idx_pos=idx, sph_repr=fs)


def coefficient_loss(outputs):
    values = [outputs['df_coeffs']]
    values += [tensor for key in ('radial_width', 'radial_scale')
               for atom in outputs[key] for tensor in atom.values()]
    return sum(tensor.square().sum() for tensor in values)


def benchmark_case(mode='codec', **options):
    reference, candidate, inputs, manifest, ref_orbitals = make_case(**options)
    # Precomputed boundary tensors, so fixture generation is outside timing.
    positions = inputs['batch_positions'].detach().requires_grad_()
    for key in ('sph_repr',):
        inputs[key] = [f.detach().requires_grad_() for f in inputs[key]]
    inputs['batch_positions'] = positions
    if mode == 'codec':
        with torch.no_grad():
            supplied = candidate(dict(inputs))
        basis = candidate.orbital_basis
        nums = inputs['batch_atom_numbers']
        def call(codec):
            return codec.coeffs_dict_to_vector(supplied, basis, nums, radial_coeffs=False)
        return lambda: call(ref_orbitals), lambda: call(orbitals), manifest
    if mode not in ('head', 'head-inference', 'head-backward', 'head-mixed'):
        raise ValueError(mode)
    def call(model):
        if mode == 'head-inference':
            with torch.no_grad():
                return model(dict(inputs))
        outputs = model(dict(inputs))
        if mode == 'head':
            return outputs
        loss = coefficient_loss(outputs)
        targets = tuple(model.parameters()) + tuple(inputs['sph_repr'])
        grads = torch.autograd.grad(loss, targets, create_graph=mode == 'head-mixed',
                                    allow_unused=True)
        if mode == 'head-mixed':
            feature_grads = [g for g in grads[-len(inputs['sph_repr']):] if g is not None]
            return torch.autograd.grad(sum(g.square().sum() for g in feature_grads),
                                       tuple(model.parameters()), allow_unused=True)
        return grads
    return lambda: call(reference), lambda: call(candidate), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--mode', choices=('codec', 'head', 'head-inference', 'head-backward', 'head-mixed'), default='codec')
    parser.add_argument('--repetitions', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--fixed-radial', action='store_true')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from benchmarks.precision import use_full_precision
    use_full_precision()
    reference, candidate, manifest = benchmark_case(
        args.mode, reference_root=args.reference_root, device=args.device,
        dtype=getattr(torch, args.dtype), learned=not args.fixed_radial,
        repetitions=args.repetitions, batch_size=args.batch_size)
    result = paired_benchmark(reference, candidate, device=args.device,
                              warmup=args.warmup, iterations=args.iterations,
                              rounds=args.rounds)
    result.update(hypothesis='H2-D1', mode=args.mode, fixture=manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
