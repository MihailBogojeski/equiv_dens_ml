"""Full SchNetPack calculator E/F comparison isolating geometry preparation.

Both arms share one saved initialized current DenSNet model with broadcast
kernels. The reference arm imports only pinned calculator/base preparation.
Caller owns the campaign measurement lock and idle-device/allocation checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time

import numpy as np
import torch

from benchmarks.fixtures import build_model, load_fixture, tensor_hash
from benchmarks.geometry import load_reference
from benchmarks.timing import synchronize, timed_block
from equiv_dens.md import dft_network_calculator as native_md
from schnetpack.md import System
from ase import Atoms


def make_system(natoms, batch, device, dtype):
    """Generated H/O finite molecule, already in species-compressed atom order."""
    nhydrogen = max(2, (2 * natoms + 2) // 3)
    numbers = [1] * nhydrogen + [8] * (natoms - nhydrogen)
    indices = np.arange(natoms)
    width = 2 if natoms == 8 else 4
    positions = np.column_stack((indices % width,
                                 (indices // width) % width,
                                 indices // (width * width))).astype(float)
    positions += np.column_stack((.013 * np.sin(indices),
                                  .017 * np.cos(indices),
                                  .019 * np.sin(indices * .7)))
    molecules = [Atoms(numbers=numbers, positions=positions + [0, 0, .07 * i])
                 for i in range(batch)]
    system = System()
    system.load_molecules(molecules, n_replicas=1, position_unit_input='Angstrom')
    return system.to(device=device, dtype=dtype)


def make_calculator(module, model, backend=None):
    kwargs = dict(model=model, required_properties=['energy', 'forces'],
                  force_key='forces', energy_key='energy',
                  position_unit='Angstrom', energy_unit='kcal/mol',
                  grid_spec={}, density_expansion=False, remove_atom_density=False,
                  dpm_intor=False, detach=True, cutoff=5.,
                  enable_tf32=False, enable_inference_mode=False,
                  use_fast_inference=False, compile_model=False, cache_grid=False,
                  use_gpu=False, verbose=0)
    if backend is not None:
        kwargs['geometry_backend'] = backend
    return module.DFTNetworkCalculator(**kwargs)


def frames_for(system, position_conversion):
    frames = []
    # Frame4 moves an early H across the cutoff while trailing O atoms remain
    # connected. This keeps the original edge-offset algorithm on its valid
    # domain, so the separate isolated-trailing-atom correction is not timed.
    for displacement in (0., .013, -.021, 5.15):
        positions = system.positions.detach().clone()
        for mol in range(system.n_molecules):
            positions[0, mol * int(system.n_atoms[mol]), 2] += displacement * position_conversion
        frames.append(positions)
    return frames


def output(system):
    return {'energy': system.energy.detach().cpu().clone(),
            'forces': system.forces.detach().cpu().clone(),
            'properties': {key: value.detach().cpu().clone()
                           for key, value in system.properties.items()}}


def assert_same(left, right, dtype):
    derivative_atol, derivative_rtol = ((1e-8, 1e-6) if dtype == torch.float64 else (1e-4, 1e-3))
    value_atol, value_rtol = ((1e-9, 1e-7) if dtype == torch.float64 else (1e-5, 1e-4))
    errors = {}
    def compare(a, b, path):
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for key in a:
                compare(a[key], b[key], f'{path}/{key}')
            return
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), path
        derivative = 'forces' in path
        torch.testing.assert_close(a, b,
                                   atol=derivative_atol if derivative else value_atol,
                                   rtol=derivative_rtol if derivative else value_rtol)
        delta = (a - b).abs()
        errors[path] = {'max_abs': delta.max().item(),
                        'max_relative_floor_1e-12': (delta / a.abs().clamp_min(1e-12)).max().item()}
    compare(left, right, '')
    assert left['energy'].abs().max() > 1e-12
    assert left['forces'].abs().max() > 1e-12
    return errors


def provenance(root, reference_root, args, fixture, model):
    paths = [root / 'src/equiv_dens/utils/base.py',
             root / 'src/equiv_dens/md/dft_network_calculator.py',
             reference_root / 'src/equiv_dens/utils/base.py',
             reference_root / 'src/equiv_dens/md/dft_network_calculator.py',
             Path(__file__), root / 'benchmarks/fixtures.py']
    imported = {}
    import sys
    for name, module in sys.modules.items():
        if name.startswith('equiv_dens.') and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            imported[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return {'model_source_root': str(root), 'reference_preparation_root': str(reference_root),
            'model_revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
            'reference_preparation_revision': subprocess.check_output(
                ['git', '-C', str(reference_root), 'rev-parse', 'HEAD'], text=True).strip(),
            'model_dirty_source_sha256': hashlib.sha256(subprocess.check_output(
                ['git', '-C', str(root), 'diff', '--', 'src'])).hexdigest(),
            'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            'current_model_imports': imported,
            'saved_fixture_hash': fixture['fixture_hash'], 'saved_state_hash': fixture['state_hash'],
            'effective_model_state_hash': tensor_hash(model.state_dict()),
            'fixture_path': str(args.fixture.resolve()), 'kernel_backend': 'broadcast',
            'host': os.uname().nodename, 'allocation': args.allocation,
            'torch': torch.__version__, 'threads': torch.get_num_threads(),
            'device': args.device,
            'device_name': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'cpu',
            'runtime_image': os.environ.get('APPTAINER_CONTAINER'),
            'container_sha256': args.container_sha256,
            'tf32': torch.backends.cuda.matmul.allow_tf32}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allocation', required=True)
    parser.add_argument('--container-sha256', default='see qualified runtime record')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--shapes', default='1x8,1x32,1x128,4x8')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    from benchmarks.precision import use_full_precision
    use_full_precision()
    if args.rounds < 5 or args.iterations < 1 or args.warmup < 1:
        parser.error('Require at least five rounds and positive iterations/warmup')
    root = Path(__file__).resolve().parents[1]
    reference_root = args.reference_root.resolve()
    _, reference_md = load_reference(reference_root)
    fixture = load_fixture(args.fixture)
    spec = dict(fixture['spec'], dtype=args.dtype)
    dtype = getattr(torch, args.dtype)
    model = build_model(spec, mode='ef', kernel_backend='broadcast').to(args.device)
    model.load_state_dict(fixture['state_dict'], strict=True)
    model.eval()
    cases = []
    for shape in args.shapes.split(','):
        batch, natoms = map(int, shape.split('x'))
        systems = {arm: make_system(natoms, batch, args.device, dtype)
                   for arm in ('reference', 'native')}
        calcs = {'reference': make_calculator(reference_md, model),
                 'native': make_calculator(native_md, model, 'native')}
        frames = frames_for(systems['reference'], calcs['reference'].position_conversion)
        before_state = tensor_hash(model.state_dict())
        validation = []
        first_force = None
        for index, frame in enumerate(frames):
            results = {}
            input_edges = {}
            for arm in calcs:
                systems[arm].positions.copy_(frame)
                prepared = calcs[arm]._generate_input(systems[arm])
                input_edges[arm] = tensor_hash({key: prepared[key] for key in ('idx_i', 'idx_j')})
                calcs[arm].calculate(systems[arm])
                results[arm] = output(systems[arm])
            assert input_edges['reference'] == input_edges['native']
            if first_force is None:
                first_force = results['reference']['forces']
            validation.append({'frame': index, 'edge_hash': input_edges['reference'],
                               'errors': assert_same(results['reference'], results['native'], dtype),
                               'force_change_from_first': (results['reference']['forces'] - first_force).abs().max().item()})
        assert validation[-1]['force_change_from_first'] > 1e-12
        assert len({item['edge_hash'] for item in validation}) > 1, 'No scripted cutoff crossing'
        assert tensor_hash(model.state_dict()) == before_state, 'Inference modified model state'
        row = {'shape': shape, 'dtype': args.dtype, 'validation': validation,
               'frame_hash': tensor_hash(frames),
               'state_hash': before_state,
               'boundary': 'actual System -> DFTNetworkCalculator.calculate -> updated System; '
                           'includes scripted frame copy, SchNetPack extraction/units, preparation, model E/F, '
                           'coefficient conversion, force unpacking and system update',
               'system_atom_order': 'species compressed H then O',
               'units': {'positions_input': 'angstrom', 'model_energy_assigned': 'kcal/mol',
                         'system_energy': 'SchNetPack internal units',
                         'system_forces': 'SchNetPack internal energy/length'},
               'model_shape': {key: spec[key] for key in ('num_features', 'num_modules', 'order')}}
        if not args.validate_only:
            cursor = {'reference': 0, 'native': 0}
            def step(arm):
                index = cursor[arm] % len(frames)
                cursor[arm] += 1
                systems[arm].positions.copy_(frames[index])
                calcs[arm].calculate(systems[arm])
                return calcs[arm].results
            cold = {}
            for arm in calcs:
                calcs[arm] = make_calculator(reference_md if arm == 'reference' else native_md,
                                            model, None if arm == 'reference' else 'native')
                synchronize(args.device)
                started = time.perf_counter()
                step(arm)
                synchronize(args.device)
                cold[arm] = time.perf_counter() - started
                for _ in range(args.warmup):
                    step(arm)
            rounds = []
            for index in range(args.rounds):
                order = ('reference', 'native') if index % 2 == 0 else ('native', 'reference')
                sample = {'round': index, 'order': list(order)}
                for arm in order:
                    # Reset scripted position sequence before every arm/block.
                    cursor[arm] = 0
                    sample[arm] = timed_block(lambda arm=arm: step(arm), device=args.device,
                                              iterations=args.iterations)
                rounds.append(sample)
            medians = {arm: statistics.median(r[arm]['seconds_per_call'] for r in rounds)
                       for arm in calcs}
            row.update(cold_first_calculator_call_seconds=cold, rounds=rounds,
                       median_seconds=medians, speedup=medians['reference'] / medians['native'],
                       molecules_per_second={arm: batch / latency for arm, latency in medians.items()})
        cases.append(row)
    result = {'protocol': 'H1-A/P1', 'arguments': vars(args),
              'provenance': provenance(root, reference_root, args, fixture, model),
              'cases': cases, 'host_process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'limitations': ['Same current model is shared by both arms; comparison isolates A, not all-worker speedup.',
                              'Initialized model outputs have no predictive/physical calibration.',
                              'No MD integrator, grid, dipole, logging or checkpoint/file I/O inside measured calls.',
                              'Cold call excludes System/model/calculator construction and imports.',
                              'Caller owns allocation, measurement lock and GPU idle checks.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(f'Wrote {len(cases)} full-calculator {"validation" if args.validate_only else "paired timing"} cases to {args.output}')


if __name__ == '__main__':
    main()
