"""Alternating full-model timings in resident, source-isolated Python workers.

Only one child computes at a time. Each round resets the initialized model and
optimizer outside timing, replays the same warmup/update sequence, then measures
the complete prepared-input operation. Run under the measurement lock.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import resource
import statistics
import time
import traceback
from pathlib import Path

import torch


def compare_diagnostics(reference, candidate, dtype):
    """Gate requested outputs/derivatives, even when inference state is unchanged."""
    from benchmarks.pipeline import compare_snapshots
    return compare_snapshots(reference, candidate, dtype)


def diagnostic_envelope(fixture, mode, contract):
    return {'fixture_hash': fixture['fixture_hash'], 'state_hash': fixture['state_hash'],
            'contracts': {mode: contract}, 'exclusions': {}, 'successive_frames': []}


def worker(connection, root, fixture_path, mode, device, backend, folder, arm, density_backend, energy_backend):
    try:
        from benchmarks.fixtures import select_source, load_fixture, clone_tree, tensor_hash
        select_source(root)
        from benchmarks import pipeline
        from benchmarks.timing import synchronize, timed_block
        from benchmarks.precision import use_full_precision
        use_full_precision()
        pipeline.KERNEL_BACKEND = backend
        pipeline.DENSITY_BACKEND = density_backend
        pipeline.ENERGY_BACKEND = energy_backend
        torch.set_num_threads(1)
        fixture = load_fixture(fixture_path)
        # Both workers finish imports/loading before either arm can compute.
        connection.send({'ok': True, 'stage': 'ready', 'arm': arm})
        while True:
            command = connection.recv()
            if command is None:
                break
            if command.get('action') == 'diagnostic':
                training = mode in ('coefficients_training', 'density_training', 'force_training')
                requested_mode = mode.removesuffix('_training') if mode != 'force_training' else mode
                contract = pipeline.evaluate_contract(fixture, requested_mode, training, device)
                diagnostic = diagnostic_envelope(fixture, mode, contract)
                diagnostic_path = Path(folder)/f'{arm}-diagnostic.pt'
                torch.save(diagnostic, diagnostic_path)
                response = {'ok': True, 'diagnostic_path': str(diagnostic_path),
                            'provenance': pipeline.provenance(Path(root), fixture),
                            'fixture_spec': fixture['spec']}
                del contract, diagnostic
                if torch.device(device).type == 'cuda':
                    torch.cuda.empty_cache()
                    synchronize(device)
                connection.send(response)
                continue
            started = time.perf_counter()
            operation, model = pipeline.prepared_step(fixture, mode, device)
            synchronize(device)
            cold = time.perf_counter()-started
            for _ in range(command['warmup']):
                operation()
            synchronize(device)
            timing = timed_block(operation, device=device, iterations=command['iterations'])
            state = clone_tree(model.state_dict(), 'cpu')
            for tensor in state.values():
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise AssertionError('Nonfinite model state after matched benchmark sequence')
            state_path = Path(folder)/f'{arm}-round-{command["round"]}-state.pt'
            torch.save(state, state_path)
            provenance = pipeline.provenance(Path(root), fixture)
            response = {'ok': True, 'timing': timing,
                'reset_model_seconds': cold, 'final_state_hash': tensor_hash(state),
                'state_path': str(state_path), 'provenance': provenance,
                'fixture_spec': fixture['spec'],
                'input_shapes': {key: list(tensor.shape) for key, tensor in fixture['atoms'].items()
                                 if isinstance(tensor, torch.Tensor)},
                'host_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            del operation, model, state
            # Inactive workers must not reserve an entire activation pool while
            # the other arm runs. Release only outside timing, before its reply.
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
                synchronize(device)
            connection.send(response)
    except BaseException:
        connection.send({'ok': False, 'error': traceback.format_exc()})
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--mode', choices=('coefficients', 'coefficients_training', 'density',
                                          'density_training', 'ef', 'force_training'), required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reference-backend', choices=('broadcast', 'matmul'), default='broadcast')
    parser.add_argument('--candidate-backend', choices=('broadcast', 'matmul'), default='broadcast')
    parser.add_argument('--reference-density-backend', choices=('legacy', 'contracted'), default='legacy')
    parser.add_argument('--candidate-density-backend', choices=('legacy', 'contracted'), default='legacy')
    parser.add_argument('--reference-energy-backend', choices=('legacy', 'device'), default='legacy')
    parser.add_argument('--candidate-energy-backend', choices=('legacy', 'device'), default='legacy')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.reference_root = args.reference_root.resolve()
    args.candidate_root = args.candidate_root.resolve()
    args.fixture = args.fixture.resolve()
    args.output = args.output.resolve()
    torch.set_num_threads(1)
    if args.rounds < 5 or args.iterations < 1 or args.warmup < 1:
        parser.error('Need at least five rounds and positive iteration/warmup counts')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_folder = args.output.parent/(args.output.stem+'-states')
    state_folder.mkdir(exist_ok=True)
    context = mp.get_context('spawn')
    workers, connections = {}, {}
    records = []
    current_row, diagnostics, diagnostic_comparison = {}, {}, None
    stage = 'startup'
    try:
        for arm, root, backend, density_backend, energy_backend in (
            ('reference', args.reference_root, args.reference_backend, args.reference_density_backend, args.reference_energy_backend),
            ('candidate', args.candidate_root, args.candidate_backend, args.candidate_density_backend, args.candidate_energy_backend),
        ):
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child, str(root), str(args.fixture),
                args.mode, args.device, backend, str(state_folder), arm, density_backend, energy_backend))
            try:
                process.start()
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            workers[arm], connections[arm] = process, parent
        for arm in ('reference', 'candidate'):
            response = connections[arm].recv()
            diagnostics[arm] = {'startup': response}
            if not response.get('ok') or response.get('stage') != 'ready':
                raise RuntimeError(f'{arm} startup failed: {response}')
        stage = 'diagnostic'
        for arm in ('reference', 'candidate'):
            connections[arm].send({'action': 'diagnostic'})
            response = connections[arm].recv()
            diagnostics[arm]['response'] = response
            if not response.get('ok'):
                raise RuntimeError(f'{arm} numerical diagnostic failed: {response}')
        reference_diagnostic = torch.load(diagnostics['reference']['response']['diagnostic_path'],
                                          weights_only=False, map_location='cpu')
        candidate_diagnostic = torch.load(diagnostics['candidate']['response']['diagnostic_path'],
                                          weights_only=False, map_location='cpu')
        dtype = diagnostics['reference']['response']['fixture_spec']['dtype']
        diagnostic_comparison = compare_diagnostics(reference_diagnostic, candidate_diagnostic, dtype)
        del reference_diagnostic, candidate_diagnostic
        if not diagnostic_comparison['passed']:
            raise AssertionError(f'Requested-mode numerical gate failed: {diagnostic_comparison["failures"]}')
        stage = 'timing'
        for index in range(args.rounds):
            order = ('reference', 'candidate') if index % 2 == 0 else ('candidate', 'reference')
            row = {'round': index, 'order': list(order)}
            current_row = row
            for arm in order:
                connections[arm].send({'round': index, 'iterations': args.iterations, 'warmup': args.warmup})
                result = connections[arm].recv()
                row[arm] = result
                if not result['ok']:
                    raise RuntimeError(result['error'])
            old = torch.load(row['reference']['state_path'], weights_only=True)
            new = torch.load(row['candidate']['state_path'], weights_only=True)
            if old.keys() != new.keys():
                raise AssertionError('Parameter/state schema changed')
            dtype = row['reference']['fixture_spec']['dtype']
            atol, rtol = (1e-9, 1e-7) if dtype == 'float64' else (1e-5, 1e-4)
            maximum = 0.0
            for key in old:
                torch.testing.assert_close(new[key], old[key], atol=atol, rtol=rtol)
                if old[key].is_floating_point() and old[key].numel():
                    maximum = max(maximum, float((new[key]-old[key]).abs().max()))
            row['matched_final_state_max_abs_error'] = maximum
            row['speedup'] = row['reference']['timing']['seconds_per_call']/row['candidate']['timing']['seconds_per_call']
            records.append(row)
            current_row = {}
    except BaseException:
        # Preserve every startup/worker/schema/numerical failure and partial row.
        failure = {'protocol': 'P1', 'passed': False, 'stage': stage,
                   'error': traceback.format_exc(),
                   'case': {key: str(value) for key, value in vars(args).items()},
                   'diagnostics': diagnostics, 'diagnostic_comparison': diagnostic_comparison,
                   'rounds': records + ([current_row] if current_row else [])}
        args.output.write_text(json.dumps(failure, indent=2)+'\n')
        raise
    finally:
        for arm, connection in connections.items():
            if workers[arm].is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError):
                    pass
            connection.close()
        for process in workers.values():
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    medians = {arm: statistics.median(row[arm]['timing']['seconds_per_call'] for row in records)
               for arm in ('reference', 'candidate')}
    result = {'protocol': 'P1', 'passed': True, 'case': {key: str(value) for key, value in vars(args).items()},
        'diagnostics': diagnostics, 'diagnostic_comparison': diagnostic_comparison,
        'boundary': 'prepared inputs to outputs; training includes zero_grad/loss/backward/Adam update',
        'excludes': 'source import, model construction, fixture loading, per-round reset, host preparation/logging',
        'concurrency': 'two resident isolated processes; compute serialized; inactive CUDA caches released outside timing before reply',
        'rounds': records, 'median_seconds': medians,
        'speedup': medians['reference']/medians['candidate'],
        'paired_speedup_min': min(row['speedup'] for row in records),
        'paired_speedup_max': max(row['speedup'] for row in records)}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'speedup': result['speedup']}))


if __name__ == '__main__':
    main()
