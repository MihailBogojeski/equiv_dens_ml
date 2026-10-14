"""Same-state spherical-kernel contrasts against the pinned source module."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import os
import json
from pathlib import Path

import torch

from benchmarks.timing import paired_benchmark
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from benchmarks import experimental_kernels as candidate

REFERENCE_ROOT = Path(os.environ.get('DENSNET_REFERENCE_ROOT', '/home/johannes/equiv_dens_ml-paper-reference'))


def load_reference(root=REFERENCE_ROOT):
    path = Path(root) / 'src/equiv_dens/nn/modules/spherical_harmonic_layers.py'
    spec = importlib.util.spec_from_file_location('_paper_spherical_layers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(kind='pair', order=1, width=8, rows=8, *, dtype=torch.float64,
              device='cpu', reference_root=REFERENCE_ROOT, leading_shape=None):
    reference = load_reference(reference_root)
    torch.manual_seed(6101729)
    cg = ClebschGordanMatrix(order_max=order)
    common = dict(num_features=width, clebsch_gordan=cg, normalize=1, parity=False)
    if kind == 'self':
        options = dict(order_in=order, order_out=order, **common)
        ref = reference.SelfMixing(**options)
        new = candidate.SelfMixing(**{**options, 'clebsch_gordan': copy.deepcopy(cg)},
                                   contraction_backend='matmul')
    elif kind == 'pair':
        options = dict(order_in1=order, order_in2=order, order_out=order,
                       num_basis_functions=width, **common)
        ref = reference.PairMixing(**options)
        new = candidate.PairMixing(**{**options, 'clebsch_gordan': copy.deepcopy(cg)},
                                   contraction_backend='matmul')
    else:
        raise ValueError(kind)
    new.load_state_dict(ref.state_dict(), strict=True)
    ref = ref.to(device=device, dtype=dtype)
    new = new.to(device=device, dtype=dtype)
    shape = tuple(leading_shape or (rows,))
    generator = torch.Generator().manual_seed(1731)
    first = [torch.randn(*shape, 2*degree+1, width, generator=generator,
                         dtype=dtype).to(device).requires_grad_()
             for degree in range(order+1)]
    inputs = [first]
    if kind == 'pair':
        second = [torch.randn(*shape, 2*degree+1, width, generator=generator,
                              dtype=dtype).to(device).requires_grad_()
                  for degree in range(order+1)]
        # PairMixing broadcasts radial coefficients over the magnetic axis.
        radial = torch.randn(*shape, 1, width, generator=generator,
                             dtype=dtype).to(device).requires_grad_()
        inputs.extend((second, radial))
    return ref, new, inputs


def flat_inputs(inputs):
    return [tensor for group in inputs
            for tensor in (group if isinstance(group, list) else [group])]


def derivative_snapshot(layer, inputs):
    values = layer(*inputs)
    loss = sum((degree+1)*value.square().sum() for degree, value in enumerate(values))
    leaves = flat_inputs(inputs)
    parameters = list(layer.parameters())
    first = torch.autograd.grad(loss, leaves+parameters, create_graph=True,
                                allow_unused=True)
    direction = torch.linspace(-0.2, 0.4, leaves[0].numel(),
                               device=leaves[0].device, dtype=leaves[0].dtype).reshape_as(leaves[0])
    hvp = torch.autograd.grad((first[0]*direction).sum(), leaves[0],
                              retain_graph=True)[0]
    mixed_loss = (first[0]-direction).square().mean()
    mixed = torch.autograd.grad(mixed_loss, parameters, allow_unused=True)
    return values, first, hvp, mixed


def operation(layer, inputs, mode):
    if mode == 'forward':
        def call():
            with torch.no_grad():
                return layer(*inputs)
        return call
    leaves = flat_inputs(inputs)
    parameters = list(layer.parameters())
    def call():
        output = layer(*inputs)
        loss = sum((degree+1)*value.square().mean() for degree, value in enumerate(output))
        if mode == 'backward':
            return torch.autograd.grad(loss, leaves+parameters, allow_unused=True)
        first = torch.autograd.grad(loss, leaves[0], create_graph=True)[0]
        return torch.autograd.grad((first-0.125).square().mean(), parameters,
                                   allow_unused=True)
    return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('self', 'pair'), default='pair')
    parser.add_argument('--mode', choices=('forward', 'backward', 'mixed'), default='forward')
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--rows', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from benchmarks.precision import use_full_precision
    use_full_precision()
    torch.set_num_threads(1)
    ref, new, inputs = make_case(args.kind, args.order, args.width, args.rows,
        dtype=getattr(torch, args.dtype), device=args.device, reference_root=args.reference_root)
    old_snapshot = derivative_snapshot(ref, inputs)
    new_snapshot = derivative_snapshot(new, inputs)
    atol, rtol = (1e-8, 1e-6) if args.dtype == 'float64' else (1e-4, 1e-3)
    errors = []
    def compare(old, actual, atol, rtol):
        if isinstance(old, (tuple, list)):
            for left, right in zip(old, actual):
                compare(left, right, atol, rtol)
        elif old is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, old, atol=atol, rtol=rtol)
            errors.append(float((old-actual).detach().abs().max().cpu()))
    value_atol, value_rtol = (1e-9, 1e-7) if args.dtype == 'float64' else (1e-5, 1e-4)
    compare(old_snapshot[0], new_snapshot[0], value_atol, value_rtol)
    compare(old_snapshot[1:], new_snapshot[1:], atol, rtol)
    result = paired_benchmark(operation(ref, inputs, args.mode),
        operation(new, inputs, args.mode), device=args.device,
        warmup=args.warmup, iterations=args.iterations)
    source_path = 'src/equiv_dens/nn/modules/spherical_harmonic_layers.py'
    result.update(hypothesis='H3-B1', case={key: str(value) for key, value in vars(args).items()},
                  numerical_max_abs_error=max(errors),
                  reference_source_sha256=hashlib.sha256((args.reference_root/source_path).read_bytes()).hexdigest(),
                  candidate_source_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'speedup': result['speedup']}))


if __name__ == '__main__':
    main()
