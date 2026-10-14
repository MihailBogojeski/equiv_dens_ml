"""Native FP32 DenSNet density -> frozen-density energy training from an export.

Preserves source step/validation/checkpoint/LR budgets. --qualification-steps
is a visibly bounded two-stage check, not a completed production experiment.
Prepared grids are fixed; source rotated/resampled training remains distinct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

_STOP = False


def _signal(*_args):
    global _STOP
    _STOP = True


def density_digest(params):
    digest = hashlib.sha256()
    for key in sorted(params):
        if key.startswith('density_repr_model.'):
            digest.update(key.encode()); digest.update(np.asarray(params[key]).tobytes())
    return digest.hexdigest()


def save_selected(trainer, path):
    """Keep the portable model and training state on the same selected weights."""
    original_state, original_pipeline = trainer.state, trainer.pipeline.state
    chosen = (original_state['ema'] if trainer.optimizer.ema_decay is not None
              else original_state['params'])
    try:
        trainer.state = dict(original_state, params=dict(chosen), ema=dict(chosen))
        trainer.pipeline.state = dict(original_pipeline, params=dict(chosen))
        trainer.save(path)
    finally:
        trainer.state, trainer.pipeline.state = original_state, original_pipeline


def run(export, data, output, *, qualification_steps=None, jit=True):
    import jax
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    from equiv_dens_jax.data import load_batches
    from equiv_dens_jax.pipeline import Pipeline
    from equiv_dens_jax.trainer import Trainer
    from equiv_dens_jax.training import Optimizer, load_checkpoint
    from equiv_dens_jax.losses import ErrorDict
    from equiv_dens_jax.initialization import initialize_parameters
    from equiv_dens_jax.serialization import save_tree

    export, data, output = Path(export), Path(data), Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True)
    train, valid = load_batches(data / 'train'), load_batches(data / 'valid')
    pipeline = Pipeline.load(export / 'template')
    seed = json.loads((export / 'density.json').read_text())['seed']
    pipeline.state = initialize_parameters(pipeline.node, pipeline.state, seed=seed)
    save_tree(output / 'initial', {'model': pipeline.node, 'state': pipeline.state, 'metadata': pipeline.metadata})
    started = time.monotonic()
    records = output / 'metrics.jsonl'
    terminal = {'scope': 'bounded qualification' if qualification_steps else 'prepared-grid production',
                'source_export': str(export.resolve()), 'data': str(data.resolve()), 'stages': {}}

    def log(row):
        row['elapsed_seconds'] = time.monotonic() - started
        with records.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        for stage in ('density', 'energy'):
            if _STOP:
                break
            config = json.loads((export / f'{stage}.json').read_text())
            controls = config['source_control']
            trainer = Trainer(pipeline, ErrorDict(**config['loss']), Optimizer(**config['optimizer']),
                              stage=stage, seed=config['seed'], jit=jit, convert_inputs=True,
                              scheduler=config['scheduler'], jit_validation=jit)
            frozen = density_digest(trainer.state['params']) if stage == 'energy' else None
            maximum = qualification_steps or int(controls['max_steps'])
            interval = 1 if qualification_steps else int(controls['validation_interval'])
            best = math.inf
            # Validation is in physical output units; training uses source scaling.
            baseline = trainer.validate(valid)
            log({'event': 'initial_validation', 'stage': stage, 'metrics': baseline})
            for index in range(maximum):
                metrics, _ = trainer.step(train, shuffle=config['shuffle'])
                if stage == 'energy' and density_digest(trainer.state['params']) != frozen:
                    raise RuntimeError('Energy update changed frozen density parameters')
                log({'event': 'train_step', 'stage': stage, 'step': index + 1, 'metrics': metrics})
                if (index + 1) % interval == 0 or index + 1 == maximum or _STOP:
                    validation = trainer.validate(valid)
                    score = float(validation['loss'])
                    if not math.isfinite(score):
                        raise FloatingPointError('Nonfinite validation')
                    selected = score < best
                    if selected:
                        best = score
                        save_selected(trainer, output / f'best_{stage}')
                    trainer.schedule(score)
                    trainer.save(output / 'latest')
                    log({'event': 'validation', 'stage': stage, 'step': index + 1,
                         'selected': selected, 'metrics': validation, 'learning_rate': trainer.loop['learning_rate']})
                elif (index + 1) % int(controls['checkpoint_interval']) == 0:
                    trainer.save(output / 'latest')
                if _STOP or trainer.loop['learning_rate'] < controls['stop_at_learning_rate']:
                    break
            terminal['stages'][stage] = {'steps': int(trainer.state['step']), 'best_validation': best,
                                         'density_frozen_verified': stage == 'energy',
                                         'density_sha256': frozen}
            trainer.save(output / 'latest')
            if not _STOP:
                pipeline, _, _, _, _ = load_checkpoint(output / f'best_{stage}')
        terminal['reason'] = 'preempted' if _STOP else 'completed'
        terminal['torch_imported'] = 'torch' in sys.modules
        if terminal['torch_imported']:
            raise RuntimeError('Native training imported Torch')
    except BaseException as error:
        terminal.update(reason='failed', error=repr(error))
        raise
    finally:
        terminal['elapsed_seconds'] = time.monotonic() - started
        (output / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    print(json.dumps(terminal), flush=True)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification-steps', type=int)
    parser.add_argument('--jit', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.qualification_steps is not None and args.qualification_steps <= 0:
        parser.error('Qualification steps must be positive')
    signal.signal(signal.SIGTERM, _signal); signal.signal(signal.SIGINT, _signal)
    run(args.export, args.data, args.output, qualification_steps=args.qualification_steps, jit=args.jit)


if __name__ == '__main__':
    main()
