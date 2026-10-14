"""Paired geometry components; run only under the campaign measurement lock.

Example (CPU; coordinator controls allocation and serialized measurements):
DENSNET_NUM_THREADS=1 tools/runtime/run.sh torch python benchmarks/geometry.py \
  --reference-root /home/johannes/equiv_dens_ml-paper-reference \
  --allocation 5051234 --output /tmp/geometry.json
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time

import numpy as np
import torch
import equiv_dens.compat  # noqa: F401
from schnetpack import properties
from equiv_dens.md import dft_network_calculator as candidate
from equiv_dens.utils import base


def load_reference(root):
    root = Path(root)
    modules = []
    for name, path in [('_geometry_reference_base', 'src/equiv_dens/utils/base.py'),
                       ('_geometry_reference_md', 'src/equiv_dens/md/dft_network_calculator.py')]:
        spec = importlib.util.spec_from_file_location(name, root / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    modules[1].utils = modules[0]
    return modules[0], modules[1]


def make_calculator(module=candidate, backend='native', cutoff=2.5):
    # Exercise the actual adapter with generated SchNetPack molecule tensors;
    # constructor/model execution are outside this component boundary.
    calc = module.DFTNetworkCalculator.__new__(module.DFTNetworkCalculator)
    if isinstance(calc, torch.nn.Module):
        torch.nn.Module.__init__(calc)
    calc.use_gpu = False  # supplied inputs already reside on the requested device
    calc.geometry_backend = backend
    calc._geometry_layout = None
    calc._geometry_neighbors = None
    calc.density_expansion = False
    calc.dpm_intor = False
    calc.remove_atom_density = False
    calc.atom_dens = None
    calc.cutoff = cutoff
    calc.static_compile = False
    calc._get_system_molecules = lambda system: system
    return calc


def make_system(positions, numbers):
    return {properties.R: positions, properties.Z: numbers,
            properties.n_atoms: torch.full((positions.shape[0],), positions.shape[1],
                                           dtype=torch.long, device=positions.device)}


def fixture(natoms=8, batch=1, device='cpu', dtype=torch.float32):
    generator = torch.Generator().manual_seed(14031 + natoms + batch)
    positions = torch.randn(batch, natoms, 3, generator=generator, dtype=dtype) * 0.5
    numbers = torch.tensor(([8, 1, 6, 1] * ((natoms + 3) // 4))[:natoms])
    numbers = numbers.expand(batch, -1).clone()
    positions = positions.to(device)
    numbers = numbers.to(device)
    # Alternating shape-preserving moved frames include strict cutoff crossings.
    frames = []
    for frame in range(4):
        moved = positions.clone()
        moved[:, -1, 0] = 1.8 + frame * 0.45
        frames.append(make_system(moved, numbers))
    return frames


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--allocation', required=True)
    parser.add_argument('--container-sha256', default='not supplied; see campaign runtime record')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--shapes', default='1x8,1x32,1x128,4x8')
    parser.add_argument('--validate-only', action='store_true',
                        help='Verify all scripted input frames without collecting timings')
    args = parser.parse_args()
    from benchmarks.precision import use_full_precision
    use_full_precision()
    if args.rounds < 5 or args.iterations < 1 or args.warmup < 1:
        parser.error('Require at least five rounds and positive warmup/iterations')
    device = torch.device(args.device)
    reference_base, reference_md = load_reference(args.reference_root)
    root = Path(__file__).resolve().parents[1]
    provenance = dict(reference_root=str(args.reference_root.resolve()),
                      reference_commit='b15a0ff7c1e6070265c36d80a75706749778e9da',
                      python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__, host=platform.node(), allocation=args.allocation,
                      threads=torch.get_num_threads(), device=str(device),
                      device_name=(torch.cuda.get_device_name(device) if device.type == 'cuda'
                                   else platform.processor()),
                      container_sha256=args.container_sha256,
                      tf32=torch.backends.cuda.matmul.allow_tf32,
                      files={str(path): file_hash(path) for path in
                             [root / 'src/equiv_dens/utils/base.py',
                              root / 'src/equiv_dens/md/dft_network_calculator.py',
                              args.reference_root / 'src/equiv_dens/utils/base.py',
                              args.reference_root / 'src/equiv_dens/md/dft_network_calculator.py',
                              Path(__file__)]})
    cases = []
    for shape in args.shapes.split(','):
        batch, natoms = map(int, shape.split('x'))
        frames = fixture(natoms, batch, device)
        fixture_digest = hashlib.sha256()
        for frame in frames:
            for key in (properties.R, properties.Z):
                fixture_digest.update(frame[key].detach().cpu().numpy().tobytes())
        fixture_sha256 = fixture_digest.hexdigest()
        calcs = {'reference': make_calculator(reference_md, 'reference'),
                 'native': make_calculator(candidate, 'native')}
        # Refuse to time invalid/inequivalent fixtures, including later frames.
        errors = {'max_absolute': 0., 'max_relative_with_floor_1e-12': 0.}
        for frame in frames:
            prepared = {arm: calc._generate_input(frame) for arm, calc in calcs.items()}
            for key in prepared['reference']:
                left, right = prepared['reference'][key], prepared['native'][key]
                if torch.is_tensor(left):
                    torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)
                    if left.numel() and left.is_floating_point():
                        delta = (left - right).abs()
                        errors['max_absolute'] = max(errors['max_absolute'], delta.max().item())
                        errors['max_relative_with_floor_1e-12'] = max(
                            errors['max_relative_with_floor_1e-12'],
                            (delta / left.abs().clamp_min(1e-12)).max().item())
                else:
                    assert left == right, key
        if args.validate_only:
            cases.append({'shape': shape, 'frame_count': len(frames), 'equivalent': True,
                          'errors': errors, 'fixture_sha256': fixture_sha256})
            continue
        neighbors = {'reference': reference_base.TorchNeighborList(2.5),
                     'native': base.TorchNeighborList(2.5, backend='native')}
        packed = {arm: {'positions': item['batch_positions'],
                        'atom_numbers': item['batch_atom_numbers']}
                  for arm, item in prepared.items()}
        for component in ('prepare', 'neighbors', 'unpack'):
            functions = {}
            for arm in calcs:
                if component == 'prepare':
                    functions[arm] = lambda i, arm=arm: calcs[arm]._generate_input(frames[i % len(frames)])
                elif component == 'neighbors':
                    functions[arm] = lambda i, arm=arm: neighbors[arm].get_neighbors(packed[arm])
                else:
                    mod = reference_base if arm == 'reference' else base
                    functions[arm] = lambda i, arm=arm, mod=mod: mod.batch_compressed_atoms(
                        {**prepared[arm], 'forces': torch.ones_like(prepared[arm]['positions'])},
                        ['positions', 'forces'])
            cold = {}
            for arm, function in functions.items():
                # Recreate preparation state so this call includes metadata
                # initialization. Imports and model construction stay excluded.
                if component == 'prepare':
                    calcs[arm] = make_calculator(reference_md if arm == 'reference' else candidate,
                                                'reference' if arm == 'reference' else 'native')
                synchronize(device)
                start = time.perf_counter()
                function(0)
                synchronize(device)
                cold[arm] = time.perf_counter() - start
                for i in range(args.warmup):
                    function(i)
            samples = {'reference': [], 'native': []}
            memory = {'reference': [], 'native': []}
            orders = []
            for round_idx in range(args.rounds):
                order = ['reference', 'native'] if round_idx % 2 == 0 else ['native', 'reference']
                orders.append(order)
                for arm in order:
                    synchronize(device)
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    for i in range(args.iterations):
                        functions[arm](i)
                    synchronize(device)
                    samples[arm].append((time.perf_counter() - start) / args.iterations)
                    memory[arm].append({'allocated': torch.cuda.max_memory_allocated(device),
                                        'reserved': torch.cuda.max_memory_reserved(device)}
                                       if device.type == 'cuda' else {})
            medians = {arm: statistics.median(times) for arm, times in samples.items()}
            boundary = {
                'prepare': 'molecule tensors -> packed model inputs; includes adapter transfers; '
                           'excludes SchNetPack unit conversion and model',
                'neighbors': 'packed atom positions/species -> selected directed edges',
                'unpack': 'flat model outputs -> padded outputs; includes matched dummy force allocation'}
            cases.append(dict(shape=shape, dtype=str(frames[0][properties.R].dtype),
                              component=component, boundary=boundary[component],
                              frame_count=len(frames), orders=orders,
                              fixture_sha256=fixture_sha256,
                              equivalence_errors=errors,
                              cold_component_first_call_seconds=cold, samples_seconds=samples,
                              median_seconds=medians,
                              speedup=medians['reference'] / medians['native'],
                              molecules_per_second={arm: batch / val for arm, val in medians.items()},
                              peak_device_memory_bytes=memory))
    result = {'provenance': provenance, 'protocol': vars(args), 'cases': cases,
              'host_process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'limitations': ['Component measurements cannot establish complete E/F speedup.',
                              'Host RSS is a cumulative process high-water mark, not per-arm.',
                              'Cold component calls exclude imports/process and model initialization.',
                              'Caller owns measurement lock and device idle checks.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(f'Wrote {len(cases)} {"validation" if args.validate_only else "component"} cases to {args.output}')


if __name__ == '__main__':
    main()
