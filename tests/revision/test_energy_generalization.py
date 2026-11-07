import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

path = Path(__file__).resolve().parents[2] / 'scripts/revision/analysis/energy_generalization.py'
spec = importlib.util.spec_from_file_location('energy_generalization', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reference_uses_training_subset_without_test_recentering(tmp_path):
    args = SimpleNamespace(center_energy=True, atomic_energies=None)
    reference = module.baseline_spec(args, {'energy': np.array([-10., -8., 100.])}, [0, 1], tmp_path, 'malonaldehyde')
    assert module.reference_offset(reference, [1, 6, 8]) == -9.
    # The same offset survives a large change in held-out energies/composition.
    assert module.reference_offset(reference, [1]) == -9.
    with pytest.raises(ValueError, match='mixed-size'):
        module.baseline_spec(args, {'energy': np.array([-10.])}, [0], tmp_path, 'water')


def test_composition_reference_scales_with_water_count(tmp_path):
    path = tmp_path / 'atoms.npy'
    np.save(path, {0: 0., 1: -.5, 8: -75.})
    args = SimpleNamespace(center_energy=True, atomic_energies=str(path))
    reference = module.baseline_spec(args, {}, [], tmp_path, 'water')
    assert module.reference_offset(reference, [8, 1, 1, 0]) == -76.
    assert module.reference_offset(reference, [8, 1, 1] * 8) == -608.
    with pytest.raises(KeyError):
        module.reference_offset(reference, [6])


def test_per_water_error_and_signed_bias_are_not_aligned_away():
    rows = [{'error_kcal_mol': 2., 'n_water': 2}, {'error_kcal_mol': 12., 'n_water': 12}]
    result = module.metrics(rows)
    assert result['mae_kcal_mol'] == 7.
    assert result['mae_per_water_kcal_mol'] == 1.
    assert result['bias_kcal_mol'] == 7.


def test_bins_keep_negative_and_positive_proton_transfer_distinct():
    rows = [{'split': 'pt', 'delta_pt_A': x, 'error_kcal_mol': y} for x, y in [(-.1, 2.), (.1, -4.)]]
    result = module.summarize(rows, 'malonaldehyde')
    assert [row['coordinate'] for row in result['by_coordinate']] == pytest.approx([-.1, .1])
    assert [row['n'] for row in result['by_coordinate']] == [1, 1]
    assert result['by_split']['pt']['mae_kcal_mol'] == 3.
