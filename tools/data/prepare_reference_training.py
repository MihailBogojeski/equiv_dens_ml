"""Prepare author-configured molecular labels for native current DenSNet.

Uses exported minimized atomic MO buffers, real archived QC targets, source
grid level/count and original subset weights. No Torch imports. Frozen sampled
grids qualify training; they do not reproduce per-step rotated grid resampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import gto, lib

from equiv_dens_jax.preparation import (load_legacy_qc_results, make_pyscf_grid,
                                       sample_qc_results, sample_sad_coeff_density)
from equiv_dens_jax.serialization import load_tree, save_tree


def verify_labels(atoms, records, *, qc_force_factor=1.):
    positions = np.asarray(atoms['positions'])
    numbers = np.broadcast_to(atoms['atom_numbers'], positions.shape[:2])
    if len(positions) != len(records):
        raise ValueError('QC and geometry record counts differ')
    maxima = {'coordinate_max_abs_A': 0., 'energy_max_abs': 0., 'force_max_abs': 0.}
    for index, (metadata, result) in enumerate(records):
        mol = gto.Mole(**dict(metadata)); mol.verbose = 0; mol.build()
        if not np.array_equal(numbers[index], mol.atom_charges()):
            raise ValueError(f'QC/source atom-order mismatch at {index}')
        errors = {'coordinate_max_abs_A': float(np.max(np.abs(positions[index] - mol.atom_coords(unit='Angstrom')))),
                  'energy_max_abs': abs(float(np.asarray(atoms['energy'][index]).ravel()[0]) - float(result['energy'])),
                  'force_max_abs': float(np.max(np.abs(atoms['forces'][index] - np.asarray(result['forces']) * qc_force_factor)))}
        if any(value > 1e-9 for value in errors.values()):
            raise ValueError(f'QC/source labels differ at {index}: {errors}')
        for key, value in errors.items():
            maxima[key] = max(maxima[key], value)
        for key in ('mo_coeff', 'mo_occ', 'forces'):
            if not np.isfinite(result[key]).all():
                raise ValueError(f'Nonfinite QC field {key} at {index}')
    return maxima


def prepare(export_path, output, *, limit=None, count=None, batch_size=None, validation_qc_force_factor=1.):
    export_path, output = Path(export_path), Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh preparation directory')
    scientific = load_tree(export_path / 'scientific')
    metadata = scientific['metadata']; args = metadata['effective_args']
    if not args['pyscf_grid'] or args['cube_grid'] or args['atom_dens_type'] != 'mo_coeffs':
        raise NotImplementedError('Preparation supports supplied PySCF-grid/MO-reference configs')
    output.mkdir(parents=True)
    lib.num_threads(1)
    report = {'scope': 'Configuration-derived prepared-grid training qualification',
              'source_metadata': metadata, 'grid_policy': 'fixed deterministic uniform subsets, original weights',
              'source_deviation': 'Source regenerates/rotates sampled grids on access; this preparation freezes grids.',
              'count': count or args['density_subsamples'], 'limit_per_split': limit,
              'units': {'positions': 'Angstrom', 'coords': 'Angstrom', 'coord_weights': 'Bohr^3',
                        'density': 'electrons/Bohr^3', 'energy': 'Hartree, centered by full training subset',
                        'forces': 'Hartree/Angstrom'}, 'splits': {}}
    for split in ('train', 'valid'):
        atom_key = 'np_dataset' if split == 'train' else 'np_dataset_valid'
        qc_key = 'dens_dataset' if split == 'train' else 'dens_dataset_valid'
        atoms = np.load(args[atom_key] or args['np_dataset'], allow_pickle=True).item()
        records = load_legacy_qc_results(args[qc_key] or args['dens_dataset'], trusted_legacy=True)
        factor = validation_qc_force_factor if split == 'valid' else 1.
        verified = verify_labels(atoms, records, qc_force_factor=factor)
        numbers = np.broadcast_to(atoms['atom_numbers'], atoms['positions'].shape[:2])
        indices = metadata['indices'][split][:limit] if limit else metadata['indices'][split]
        size = batch_size or args['train_batch_size' if split == 'train' else 'valid_batch_size']
        batches, rows = [], []
        for offset, index in enumerate(indices):
            m, c = records[index]
            mol = gto.Mole(**dict(m)); mol.verbose = 0; mol.build()
            pos = np.asarray(atoms['positions'][index], dtype=np.float32)
            coords, weights = make_pyscf_grid(mol, level=args['spherical_grid_level'],
                count=count or args['density_subsamples'], seed=args['split_seed'] + index,
                coords_unit='angstrom')
            # Source evaluates labels on the FP32 stored coordinates.
            coords = coords.astype(np.float32)
            density = sample_qc_results([(m, c)], coords[None])[0]
            sad, _ = sample_sad_coeff_density(pos[None], numbers[index:index+1], coords[None],
                                             scientific['atomic_mo'], coefficient_type='mo_coeffs')
            row = {'positions': pos[None], 'atom_numbers': numbers[index:index+1].copy(),
                   'coords': coords[None], 'coord_weights': weights[None].astype(np.float32),
                   'density': density[None].astype(np.float32), 'atom_density': sad.astype(np.float32),
                   'energy': np.array([[float(np.asarray(atoms['energy'][index]).ravel()[0])
                                        - metadata['energy_mean_hartree']]], dtype=np.float32),
                   'forces': np.asarray(atoms['forces'][index:index+1], dtype=np.float32),
                   'idx': np.array([[index]], dtype=np.int64)}
            if any(not np.isfinite(value).all() for value in row.values()):
                raise ValueError(f'Nonfinite prepared data at {index}')
            rows.append(row)
            if len(rows) == size or offset + 1 == len(indices):
                batches.append({key: np.concatenate([item[key] for item in rows]) for key in row})
                rows = []
            if offset % 25 == 0:
                print(json.dumps({'split': split, 'prepared': offset + 1, 'target': len(indices)}), flush=True)
        save_tree(output / split, {'batches': batches})
        report['splits'][split] = {'indices': indices, 'count': len(indices), 'batches': len(batches), 'batch_size': size,
                                 'source_atoms': args[atom_key], 'source_qc': args[qc_key],
                                 'verified_source_records': len(records), 'QC_source_label_errors': verified,
                                 'QC_force_to_atom_force_factor': factor}
    report['torch_imported'] = 'torch' in sys.modules
    if report['torch_imported']:
        raise RuntimeError('Native preparation imported Torch')
    report['passed'] = True
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'passed': True, 'splits': report['splits']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-per-split', type=int)
    parser.add_argument('--count', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--validation-qc-force-factor', type=float, default=1.,
                        help='Explicit archive-only convention conversion for validation verification; atom force labels stay unchanged')
    parser.add_argument('--trusted-legacy', action='store_true', required=True)
    args = parser.parse_args()
    for value in (args.limit_per_split, args.count, args.batch_size):
        if value is not None and value <= 0:
            parser.error('Limits, count and batch size must be positive')
    if not np.isfinite(args.validation_qc_force_factor) or args.validation_qc_force_factor == 0:
        parser.error('QC force factor must be finite and nonzero')
    prepare(args.export, args.output, limit=args.limit_per_split, count=args.count, batch_size=args.batch_size,
            validation_qc_force_factor=args.validation_qc_force_factor)


if __name__ == '__main__':
    main()
