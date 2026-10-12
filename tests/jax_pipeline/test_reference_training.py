"""Contracts preventing source-pool leakage, label mistakes and wrong transfers."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.data.audit_mihail_sources import fingerprint
from tools.data.prepare_reference_training import verify_labels
from tools.training.train_reference_densnet import density_digest, save_selected


def test_geometry_screen_is_translation_rotation_and_species_permutation_invariant():
    positions = np.array([[.2, .3, -.1], [1.2, -.4, .3], [.1, .8, .2], [-.2, .1, 1.4]])
    numbers = np.array([1, 6, 1, 8])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    permutation = np.array([2, 1, 0, 3])
    moved = (positions @ rotation.T + [2, 3, 4])[permutation]
    assert fingerprint(positions, numbers, invariant=True) == fingerprint(moved, numbers[permutation], invariant=True)
    assert fingerprint(positions, numbers) != fingerprint(moved, numbers[permutation])
    changed = positions.copy(); changed[0, 0] += .01
    assert fingerprint(positions, numbers, invariant=True) != fingerprint(changed, numbers, invariant=True)


def test_label_audit_rejects_force_sign_units_and_misordered_geometries():
    atoms = {'positions': np.array([[[0., 0., 0.], [0., 0., .8]]]),
             'atom_numbers': np.array([1, 1]), 'energy': np.array([-1.]),
             'forces': np.array([[[0., 0., .3], [0., 0., -.3]]])}
    molecule = {'atom': 'H 0 0 0; H 0 0 .8', 'basis': 'sto-3g', 'unit': 'Angstrom'}
    results = {'energy': -1., 'forces': atoms['forces'][0].copy(),
               'mo_coeff': np.eye(2), 'mo_occ': np.array([2., 0.])}
    assert verify_labels(atoms, [(molecule, results)])['force_max_abs'] == 0.
    for scale in (-1., 1.889725989):
        with pytest.raises(ValueError, match='labels differ'):
            verify_labels(atoms, [(molecule, dict(results, forces=results['forces'] * scale))])
    archived_gradient = -results['forces'] / 1.889725989
    verified = verify_labels(atoms, [(molecule, dict(results, forces=archived_gradient))],
                             qc_force_factor=-1.889725989)
    assert verified['force_max_abs'] < 1e-15
    with pytest.raises(ValueError, match='labels differ'):
        verify_labels(dict(atoms, positions=atoms['positions'][:, ::-1]), [(molecule, results)])


def test_selected_ema_is_saved_in_model_and_training_state_and_restored_afterward():
    old = {'density_repr_model.w': np.array([1.]), 'property_models.energy.w': np.array([2.])}
    ema = {'density_repr_model.w': np.array([3.]), 'property_models.energy.w': np.array([4.])}
    captured = {}
    trainer = SimpleNamespace(state={'params': old, 'ema': ema},
        pipeline=SimpleNamespace(state={'params': old, 'buffers': {}}),
        optimizer=SimpleNamespace(ema_decay=.999))
    def save(_path):
        captured['model'] = trainer.pipeline.state['params']
        captured['training'] = trainer.state['params']
    trainer.save = save
    save_selected(trainer, 'unused')
    assert captured['model']['density_repr_model.w'][0] == 3.
    assert captured['training']['density_repr_model.w'][0] == 3.
    assert trainer.state['params'] is old
    assert trainer.pipeline.state['params'] is old
    updated_energy = {**old, 'property_models.energy.w': np.array([7.])}
    assert density_digest(old) == density_digest(updated_energy)
    assert density_digest(old) != density_digest(ema)
