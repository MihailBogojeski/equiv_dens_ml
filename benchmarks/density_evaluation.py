"""H4-C1 supplied-parameter evaluator and full density-training paired cases.

The coordinator holds the measurement lock and qualifies CUDA. The local
contrast uses pinned evaluator/GTO source with shared surrounding model code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
from pathlib import Path

import torch

from benchmarks.packing import REFERENCE_ROOT, basis_fixture, load_reference
from benchmarks.fixtures import (build_model, clone_tree, fixture_spec, make_atoms,
                                  tensor_hash)
from benchmarks.timing import paired_benchmark
from equiv_dens.nn.property_output.density import DensityExpansion
from benchmarks.experimental_density import (
    ExperimentalDensityExpansion,
    install_experimental_density_evaluator,
)


def reference_evaluator(reference_root=REFERENCE_ROOT):
    ref_orbitals, ref_density = load_reference(reference_root)
    ref_density.eval_gto = ref_orbitals.eval_gto
    ref_density.eval_gto_grad = ref_orbitals.eval_gto_grad
    return ref_density.DensityExpansion


def supplied_fixture(grid_points=31, dtype=torch.float64, device='cpu', core=False):
    basis, _ = basis_fixture()
    if core:
        basis = {z: [shells[0]] for z, shells in basis.items()}
    nums = torch.tensor([[1, 1, 6, 6], [1, 0, 6, 0]], device=device)
    positions = torch.linspace(-.8, .9, nums.numel() * 3, device=device,
                               dtype=dtype).reshape(*nums.shape, 3)
    t = torch.arange(grid_points, device=device, dtype=dtype)
    coords = torch.stack((.31 + .16 * (t * .51).sin(),
                          -.17 + .27 * (t * .37).cos(),
                          .44 + .19 * (t * .73).sin()), -1)[None].repeat(2, 1, 1)
    data = dict(batch_atom_numbers=nums, batch_positions=positions, coords=coords,
                coord_weights=torch.full((2, grid_points), .1, device=device, dtype=dtype))
    for kind in ('spherical_coeffs', 'radial_width', 'radial_scale'):
        data[kind] = []
    for i, z in enumerate((1, 1, 6, 6)):
        by_degree = {}
        for _, count, L in basis[z]:
            by_degree.setdefault(L, []).append(count)
        for kind in ('spherical_coeffs', 'radial_width', 'radial_scale'):
            data[kind].append({})
        for L, counts in by_degree.items():
            shells, primitives = len(counts), max(counts)
            data['spherical_coeffs'][i][(z, L)] = torch.linspace(-.13, .63,
                2 * (2 * L + 1) * shells, device=device, dtype=dtype).reshape(2, 1, 2 * L + 1, shells)
            widths = torch.zeros(2, 1, primitives, shells, dtype=dtype, device=device)
            scales = torch.zeros_like(widths)
            for shell, count in enumerate(counts):
                widths[..., :count, shell] = torch.linspace(.5, 1.4, count, device=device, dtype=dtype)
                scales[..., :count, shell] = torch.linspace(.2, .7, count, device=device, dtype=dtype)
            scales *= (nums[:, i] != 0)[:, None, None, None]
            data['radial_width'][i][(z, L)] = widths
            data['radial_scale'][i][(z, L)] = scales
    return basis, data


def differentiable_inputs(data):
    cloned = clone_tree(data)
    targets = [cloned['batch_positions'], cloned['coords']]
    targets += [tensor for kind in ('spherical_coeffs', 'radial_width', 'radial_scale')
                for atom in cloned[kind] for tensor in atom.values()]
    for target in targets:
        target.requires_grad_()
    return cloned, tuple(targets)


def evaluation_loss(output):
    loss = (output['density'] - .17).square().mean()
    if 'density_grad' in output:
        loss = loss + .1 * (output['density_grad'] - .07).square().mean()
    return loss


def benchmark_case(mode='rho-grad', *, grid_points=1000, dtype=torch.float32,
                   device='cpu', reference_root=REFERENCE_ROOT, learned_radial=True):
    if grid_points < 1:
        raise ValueError('Grid size must be positive')
    reference_class = reference_evaluator(reference_root)
    if mode == 'density-training':
        spec = fixture_spec('tiny', dtype, learned_radial)
        reference_model = build_model(spec, 'density').to(device)
        candidate_model = build_model(spec, 'density').to(device)
        reference_model.property_models['density'] = reference_class(spec['orbital_basis'], density_grad=True).to(device=device, dtype=dtype)
        install_experimental_density_evaluator(candidate_model, 'contracted')
        candidate_model.to(device=device, dtype=dtype)
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        for model in (reference_model, candidate_model):
            model.train()
            for module in model.modules():
                if hasattr(module, 'contraction_backend'):
                    module.contraction_backend = 'broadcast'
        data = clone_tree(make_atoms('tiny', dtype, grid_points=grid_points), device)
        initial_hash = tensor_hash(reference_model.state_dict())
        def operation(model):
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            def step():
                optimizer.zero_grad(set_to_none=True)
                output = model(clone_tree(data))
                loss = evaluation_loss(output)
                loss.backward()
                optimizer.step()
                return loss.detach()
            step.model = model
            return step
        before, after = operation(reference_model), operation(candidate_model)
        manifest = dict(fixture='shared-tiny-density-training', state_hash=initial_hash,
                        input_hash=tensor_hash(data), learned_radial=learned_radial,
                        optimizer='SGD', learning_rate=1e-4,
                        timing_boundary='prepared input clone, model rho+gradient, loss, zeroing, backward, optimizer',
                        lifecycle='matched continuous updates across warmup and alternating rounds')
    else:
        if mode not in ('rho', 'rho-grad', 'backward', 'mixed'):
            raise ValueError(mode)
        basis, data = supplied_fixture(grid_points, dtype, device)
        need_gradient = mode != 'rho'
        reference = reference_class(basis, density_grad=need_gradient)
        candidate = ExperimentalDensityExpansion(
            basis, density_grad=need_gradient, evaluation_backend='contracted')
        data, targets = differentiable_inputs(data)
        def operation(evaluator):
            def step():
                if mode in ('rho', 'rho-grad'):
                    with torch.no_grad():
                        return evaluator(dict(data))
                output = evaluator(dict(data))
                gradients = torch.autograd.grad(evaluation_loss(output), targets,
                                                create_graph=mode == 'mixed')
                if mode == 'mixed':
                    return torch.autograd.grad(gradients[1].square().sum(), targets,
                                               allow_unused=True)
                return gradients
            return step
        before, after = operation(reference), operation(candidate)
        manifest = dict(fixture='supplied-ragged-mixed-species-v1', input_hash=tensor_hash(data),
                        timing_boundary='supplied coefficients/radials/positions/grid -> requested outputs/derivatives')
    manifest.update(hypothesis='H4-C1', mode=mode, grid_points=grid_points, dtype=str(dtype),
                    device=str(device), reference_root=str(reference_root),
                    reference_scope='pinned evaluator and GTO source; shared neighbors',
                    evaluation_backends=['legacy', 'contracted'], kernel_backend='broadcast',
                    candidate_implementation='benchmarks.experimental_density.ExperimentalDensityExpansion')
    assert data['coords'].shape[1] == grid_points
    requested_device = torch.device(device)
    actual_device = data['coords'].device
    assert data['coords'].dtype == dtype and actual_device.type == requested_device.type
    assert requested_device.index is None or actual_device.index == requested_device.index
    manifest.update(device=str(actual_device), requested_device=str(device))
    manifest['batch_size'] = data['coords'].shape[0]
    current_root = Path(__file__).resolve().parents[1]
    manifest['source_sha256'] = {
        f'{arm}/{relative}': hashlib.sha256((root / 'src/equiv_dens' / relative).read_bytes()).hexdigest()
        for arm, root in (('reference', Path(reference_root)), ('candidate', current_root))
        for relative in ('utils/orbitals.py', 'nn/property_output/density.py')}
    adapter_path = Path(__file__).with_name('experimental_density.py')
    manifest['source_sha256']['candidate/benchmarks/experimental_density.py'] = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    manifest.update(host=os.uname().nodename,
                    runtime_image=os.environ.get('APPTAINER_CONTAINER'))
    from benchmarks.precision import precision_metadata
    manifest['precision_policy'] = precision_metadata()
    manifest['contracted_arithmetic'] = ('elementwise-full-fp32'
        if data['coords'].is_cuda and dtype == torch.float32
        and (torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != 'highest')
        else 'matmul-einsum')
    return before, after, manifest


def validate_case(reference, candidate, manifest):
    """Untimed actual-shape/device P1 gate before any measurement warmup.

    Training runs one matched update and then restores both initialized states;
    the SGD case has no momentum/state to restore. Hooks exist only in this gate.
    """
    fp32 = manifest['dtype'] == 'torch.float32'
    value_tolerance = (1e-5, 1e-4) if fp32 else (1e-9, 1e-7)
    derivative_tolerance = (1e-4, 1e-3) if fp32 else (1e-8, 1e-6)
    mode = manifest['mode']
    def compare(before, after, derivative=False):
        if isinstance(before, (tuple, list)):
            if len(before) != len(after):
                raise AssertionError('Derivative layout changed')
            for old, new in zip(before, after):
                compare(old, new, derivative)
        elif before is None:
            if after is not None:
                raise AssertionError('Unused derivative contract changed')
        else:
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                raise AssertionError('Nonfinite requested output or derivative')
            atol, rtol = derivative_tolerance if derivative else value_tolerance
            torch.testing.assert_close(after, before, atol=atol, rtol=rtol)
    def nonzero(values):
        if isinstance(values, (tuple, list)):
            return any(nonzero(value) for value in values if value is not None)
        return bool(values.is_floating_point() and values.abs().max() > 1e-12)
    state_before, outputs, train_gradients, states_after = [], [], [], []
    try:
        for operation in (reference, candidate):
            hook_values = []
            hook = None
            if mode == 'density-training':
                state_before.append(clone_tree(operation.model.state_dict()))
                def capture(_module, _args, output):
                    hook_values.append(tuple(output[key].detach().clone()
                                             for key in ('density', 'density_grad')))
                hook = operation.model.property_models['density'].register_forward_hook(capture)
            try:
                result = operation()
            finally:
                if hook is not None:
                    hook.remove()
            if mode in ('rho', 'rho-grad'):
                values = (result['density'],) + ((result['density_grad'],) if mode == 'rho-grad' else ())
            elif mode == 'density-training':
                if len(hook_values) != 1:
                    raise AssertionError('Density-training branch did not execute exactly once')
                values = hook_values[0]
                train_gradients.append(tuple(None if p.grad is None else p.grad.detach().clone()
                                             for p in operation.model.parameters()))
                states_after.append(clone_tree(operation.model.state_dict()))
                if not nonzero(train_gradients[-1]):
                    raise AssertionError('Degenerate density-training parameter gradients')
            else:
                values = result
            if mode in ('rho', 'rho-grad', 'density-training'):
                if tuple(values[0].shape) != (manifest['batch_size'], manifest['grid_points']):
                    raise AssertionError('Requested actual grid shape differs from manifest')
                for value in values:
                    if str(value.dtype) != manifest['dtype'] or value.device != torch.device(manifest['device']):
                        raise AssertionError('Effective requested dtype/device differs from manifest')
                    if not nonzero(value):
                        raise AssertionError('Degenerate requested density/spatial gradient')
                if len(values) > 1 and tuple(values[1].shape) != (manifest['batch_size'], manifest['grid_points'], 3):
                    raise AssertionError('Requested spatial-gradient shape differs from manifest')
            elif not nonzero(values):
                raise AssertionError('Degenerate requested derivatives')
            else:
                if values[1] is None or tuple(values[1].shape) != (manifest['batch_size'], manifest['grid_points'], 3):
                    raise AssertionError('Requested coordinate-derivative grid shape differs from manifest')
                for value in values:
                    if value is not None and (str(value.dtype) != manifest['dtype'] or value.device != torch.device(manifest['device'])):
                        raise AssertionError('Effective derivative dtype/device differs from manifest')
            outputs.append(values)
        if mode in ('rho-grad', 'density-training'):
            compare(outputs[0][0], outputs[1][0])
            compare(outputs[0][1], outputs[1][1], derivative=True)
        else:
            compare(outputs[0], outputs[1], derivative=mode in ('backward', 'mixed'))
        if mode == 'density-training':
            compare(train_gradients[0], train_gradients[1], derivative=True)
            for key, value in states_after[0].items():
                compare(value, states_after[1][key])
            if not any(not torch.equal(value, state_before[0][key]) for key, value in states_after[0].items()):
                raise AssertionError('Density-training step did not update parameters')
        return {'passed': True, 'protocol': 'P1', 'grid_points': manifest['grid_points'],
                'requested_mode': mode, 'state_restored_before_timing': mode == 'density-training'}
    finally:
        if mode == 'density-training':
            for operation, state in zip((reference, candidate), state_before):
                operation.model.load_state_dict(state, strict=True)
                operation.model.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--grid-points', type=int, choices=(1000, 30000, 100000), default=1000)
    parser.add_argument('--mode', choices=('rho', 'rho-grad', 'backward', 'mixed', 'density-training'), default='rho-grad')
    parser.add_argument('--fixed-radial', action='store_true')
    parser.add_argument('--precision-policy', choices=('full', 'tf32', 'inherit'), default='full')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from benchmarks.precision import use_full_precision
    if args.precision_policy == 'full':
        use_full_precision()
    elif args.precision_policy == 'tf32':
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    reference, candidate, manifest = benchmark_case(args.mode, grid_points=args.grid_points,
        dtype=getattr(torch, args.dtype), device=args.device, reference_root=args.reference_root,
        learned_radial=not args.fixed_radial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        validation = validate_case(reference, candidate, manifest)
    except Exception as error:
        args.output.write_text(json.dumps({'passed': False, 'stage': 'untimed_validation',
                                           'error': repr(error), 'fixture': manifest}, indent=2) + '\n')
        raise
    result = paired_benchmark(reference, candidate, device=args.device, rounds=args.rounds,
                              iterations=args.iterations, warmup=args.warmup)
    result['fixture'] = manifest
    result['untimed_validation'] = validation
    result['passed'] = True
    result['host_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.mode == 'density-training':
        atol, rtol = (1e-9, 1e-7) if args.dtype == 'float64' else (1e-5, 1e-4)
        failures = []
        for key, tensor in reference.model.state_dict().items():
            try:
                torch.testing.assert_close(candidate.model.state_dict()[key], tensor, atol=atol, rtol=rtol)
            except AssertionError as error:
                failures.append({'parameter': key, 'error': str(error)})
        result.update(passed=not failures, matched_final_state_pass=not failures, failures=failures)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise RuntimeError('Matched SGD states failed P1; raw timings and failures retained in output JSON')


if __name__ == '__main__':
    main()
