"""Audit supplied source pools and saved split indices; never resume training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np


def arguments(path):
    result = {}
    key = None
    for token in Path(path).read_text().split():
        if token.startswith('--'):
            key, separator, value = token[2:].partition('=')
            result[key] = value if separator else []
        elif key is not None:
            result[key].append(token)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(positions, numbers, *, invariant=False):
    positions = np.asarray(positions, dtype=np.float64)
    numbers = np.asarray(numbers, dtype=np.int64)
    if not invariant:
        return hashlib.sha256(numbers.tobytes() + positions.tobytes()).hexdigest()
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    # Species-labelled distance multisets screen rotation/translation/permutation
    # matches. A matching fingerprint is a candidate, not a uniqueness proof.
    values = []
    for first in sorted(set(numbers)):
        for second in sorted(set(numbers)):
            if second < first:
                continue
            pairs = [(i, j) for i in range(len(numbers)) for j in range(i + 1, len(numbers))
                     if sorted((int(numbers[i]), int(numbers[j]))) == [first, second]]
            values.extend(sorted(round(float(distances[i, j]), 8) for i, j in pairs))
    return hashlib.sha256(numbers[np.argsort(numbers)].tobytes() + np.array(values).tobytes()).hexdigest()


def audit_config(config, *, historical=True):
    args = arguments(config)
    report = {'config': str(config), 'config_sha256': sha256(config), 'pools': {},
              'historical_checkpoints': [], 'scope': 'Pool identities and saved indices; not proof of paper-model identity.'}
    identities = {}
    for split, atom_key, qc_key in [('source', 'np_dataset', 'dens_dataset'),
                                    ('test', 'np_dataset_test', 'dens_dataset_test')]:
        atom_path, qc_path = Path(args[atom_key]), Path(args[qc_key])
        atoms = np.load(atom_path, allow_pickle=True).item()
        positions = np.asarray(atoms['positions'])
        numbers = np.broadcast_to(atoms['atom_numbers'], positions.shape[:2])
        exact = [fingerprint(p, z) for p, z in zip(positions, numbers)]
        invariant = [fingerprint(p, z, invariant=True) for p, z in zip(positions, numbers)]
        identities[split] = exact, invariant
        report['pools'][split] = {'atoms': str(atom_path), 'qc': str(qc_path),
                                  'atom_sha256': sha256(atom_path), 'qc_sha256': sha256(qc_path),
                                  'samples': len(positions), 'unique_exact': len(set(exact)),
                                  'unique_invariant_candidates': len(set(invariant)),
                                  'geometry_sha256': exact, 'distance_fingerprints': invariant,
                                  'all_atom_labels_finite': all(np.isfinite(atoms[k]).all()
                                                              for k in ('positions', 'energy', 'forces'))}
        del atoms
    report['source_test_overlap'] = {
        'exact': len(set(identities['source'][0]) & set(identities['test'][0])),
        'species_distance_candidates_rounded_1e-8_A': len(set(identities['source'][1]) & set(identities['test'][1]))}
    if historical:
        import torch
        for name in ('best_checkpoint_density.pth', 'latest_checkpoint.pth'):
            path = Path(args['restart']) / 'checkpoints' / name
            saved = torch.load(path, map_location='cpu', weights_only=False)
            split_indices = saved.get('data_split_indices')
            row = {'path': str(path), 'sha256': sha256(path), 'keys': sorted(saved),
                   'step': saved.get('step'), 'ID': saved.get('ID'),
                   'training_phases': saved.get('training_phases'),
                   'ongoing_phases': saved.get('ongoing_phases')}
            old_args = saved.get('args')
            if old_args is not None:
                row['args'] = {key: getattr(old_args, key, None) for key in
                               ('np_dataset', 'dens_dataset', 'np_dataset_valid', 'dens_dataset_valid',
                                'np_dataset_test', 'num_train', 'num_valid', 'num_test', 'split_seed')}
            if split_indices is not None:
                indices = {key: [int(x) for x in value] for key, value in split_indices.items()}
                row['indices'] = indices
                same_pool = row['args']['np_dataset'] == row['args']['np_dataset_valid']
                row['same_train_valid_pool'] = same_pool
                row['train_valid_index_overlap'] = (len(set(indices['train']) & set(indices['valid']))
                                                     if same_pool else None)
                row['counts'] = {key: len(value) for key, value in indices.items()}
            else:
                row['indices'] = None
            report['historical_checkpoints'].append(row)
            print(json.dumps({key: value for key, value in row.items() if key != 'indices'}), flush=True)
            del saved
            gc.collect()
    print(json.dumps({'config': str(config), 'source_test_overlap': report['source_test_overlap']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trusted-legacy', action='store_true', required=True)
    parser.add_argument('--no-historical', action='store_true')
    args = parser.parse_args()
    reports = [audit_config(path, historical=not args.no_historical) for path in args.config]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
