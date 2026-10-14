"""Export an author-configured current DenSNet and exact native loss settings.

This one-time handoff runs in Torch. Native model execution and target
preparation use numeric bundles and never import Torch. It starts fresh.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.model_loader import load_model
from equiv_dens.training.utils import init_error_dict, modify_args_by_phase
from equiv_dens.utils.orbitals import get_num_basis
from equiv_dens.utils.base import symbols_to_numbers
from equiv_dens_jax.convert import save_model, to_numpy_tree
from equiv_dens_jax.serialization import save_tree


def export(config, output, *, seed=42):
    args, _ = parse_command_line_arguments(arg_file=config)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh export directory')
    output.mkdir(parents=True)
    original_restart = args.restart
    args.restart = args.load_from = None
    args.save_dir = str(output / 'runs')
    args.use_gpu = False
    torch.manual_seed(seed)
    atoms = np.load(args.np_dataset, allow_pickle=True).item()
    numbers = np.broadcast_to(atoms['atom_numbers'], atoms['positions'].shape[:2]).copy()
    atoms['atom_numbers'] = numbers
    # Identical seeded partition to seeded_random_split in the current source.
    order = np.random.RandomState(args.split_seed).permutation(len(numbers))
    train = order[:args.num_train]
    same_pool = args.np_dataset_valid is None or Path(args.np_dataset_valid).resolve() == Path(args.np_dataset).resolve()
    if same_pool:
        valid = order[args.num_train:args.num_train + args.num_valid]
    else:
        valid_atoms = np.load(args.np_dataset_valid, allow_pickle=True).item()
        valid = np.random.RandomState(args.split_seed + 1).permutation(len(valid_atoms['positions']))[:args.num_valid]
    if len(train) != args.num_train or len(valid) != args.num_valid:
        raise ValueError('Requested disjoint split exceeds source pool')
    mean = float(np.asarray(atoms['energy'])[train].mean()) if args.center_energy else 0.
    atoms['energy'] = np.asarray(atoms['energy']).reshape(-1, 1) - mean
    basis = np.load(args.orbitals_file, allow_pickle=True).item()
    radial = np.load(args.radial_coeffs_file, allow_pickle=True).item()
    atomic = np.load(args.atom_dens_path, allow_pickle=True).item()
    species = sorted(set(map(int, numbers.ravel())))
    basis = get_num_basis(basis, species)
    radial = {symbols_to_numbers([symbol])[0]: value for symbol, value in radial.items()
              if symbols_to_numbers([symbol])[0] in species}
    if args.L0_coeffs_file is not None or args.df_loss_weights or args.atomic_energies is not None:
        raise NotImplementedError('This handoff supports the supplied source configs only')
    dataset = SimpleNamespace(atoms=atoms, forces=atoms['forces'], orbital_basis_num=basis,
                              radial_coeffs=radial, atom_dens=atomic, L0_coeffs=None,
                              coeff_weights=None, required_properties=['density', 'energy', 'forces'])
    model = load_model(args, dataset, train=True).cpu().to(args.dtype)
    metadata = {'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'source_config': str(config), 'source_config_sha256': hashlib.sha256(Path(config).read_bytes()).hexdigest(),
                'author_restart_reference': original_restart, 'fresh_export_seed': seed,
                'split_policy': 'disjoint-source-permutation' if same_pool else 'separate-source-files',
                'split_seed': args.split_seed,
                'indices': {'train': train.tolist(), 'valid': valid.tolist()},
                'energy_mean_hartree': mean, 'source_force_scaling': 'whole training source pool, matching load_model',
                'effective_args': to_numpy_tree(vars(args))}
    save_model(output / 'template', model, metadata=metadata)
    atomic_mo = {z: {key: to_numpy_tree(atomic[z][key]) for key in ('mo_basis', 'mo_coeff', 'mo_occ')}
                 for z in species}
    save_tree(output / 'scientific', {'atomic_mo': atomic_mo, 'orbital_basis': basis, 'radial_basis': radial,
                                    'metadata': metadata})
    optimizer = {'name': args.optimizer, 'learning_rate': args.learning_rate, 'beta1': args.beta1,
                 'beta2': args.beta2, 'epsilon': args.epsilon, 'weight_decay': args.weight_decay,
                 'energy_weight_decay': args.en_weight_decay, 'momentum': args.momentum,
                 'clip_norm': args.clip_norm, 'lookahead_k': max(0, args.lookahead_k),
                 'ema_decay': args.ema_decay if args.use_parameter_averaging else None,
                 'ema_start_epoch': args.ema_start_epoch}
    fields = ('loss_weights', 'weights_balance', 'percentage_error', 'weights_decay', 'weights_min',
              'loss_comp', 'loss_comp_weights', 'relative_en', 'df_loss_weights')
    configs = {}
    for phase in ('density', 'energy'):
        stage_args = modify_args_by_phase(copy.deepcopy(args), args, phase)
        errors = init_error_dict(stage_args)
        loss = {field: to_numpy_tree(getattr(errors, field)) for field in fields}
        config_data = {'stage': phase, 'seed': seed, 'shuffle': True, 'jit_validation': True,
                       'optimizer': optimizer, 'scheduler': {'factor': args.decay_factor,
                           'patience': args.decay_patience, 'min_lr': 0.}, 'loss': loss,
                       'source_control': {'max_steps': args.max_steps, 'validation_interval': args.validation_interval,
                           'checkpoint_interval': args.checkpoint_interval, 'stop_at_learning_rate': args.stop_at_learning_rate},
                       'source_metadata': metadata}
        (output / f'{phase}.json').write_text(json.dumps(config_data, indent=2) + '\n')
        configs[phase] = config_data
    (output / 'export.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'export': str(output), 'parameters': sum(p.numel() for p in model.parameters()),
                      'train': len(train), 'valid': len(valid), 'dtype': str(args.dtype)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trusted-legacy', action='store_true', required=True)
    args = parser.parse_args()
    export(args.config, args.output, seed=args.seed)


if __name__ == '__main__':
    main()
