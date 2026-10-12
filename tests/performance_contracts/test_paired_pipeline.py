"""The timing instrument must gate computation, not just unchanged weights."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from benchmarks.paired_pipeline import compare_diagnostics, diagnostic_envelope


@pytest.mark.parametrize('mode,field', [('coefficients', 'df_coeffs'),
                                        ('density', 'density'), ('ef', 'energy')])
@pytest.mark.parametrize('corruption', ['wrong', 'nonfinite'])
def test_unchanged_state_cannot_hide_bad_inference(mode, field, corruption):
    fixture = {'fixture_hash': 'identical-inputs', 'state_hash': 'identical-unchanged-state'}
    contract = {'outputs': {field: torch.tensor([0.2, 0.7], dtype=torch.float64)},
                'mode': mode, 'training': False}
    reference = diagnostic_envelope(fixture, mode, contract)
    candidate = copy.deepcopy(reference)
    candidate['contracts'][mode]['outputs'][field][0] = 0.3 if corruption == 'wrong' else float('nan')
    assert reference['state_hash'] == candidate['state_hash']
    result = compare_diagnostics(reference, candidate, 'float64')
    assert not result['passed']
    assert any(f'outputs/{field}' in failure for failure in result['failures'])


def test_requested_output_uses_stricter_value_gate_than_gradient_gate():
    fixture = {'fixture_hash': 'same-inputs', 'state_hash': 'same-state'}
    reference = diagnostic_envelope(fixture, 'density', {
        'outputs': {'density': torch.tensor([0.], dtype=torch.float64)},
        'gradients': {'weight': torch.tensor([0.], dtype=torch.float64)}})
    candidate = copy.deepcopy(reference)
    candidate['contracts']['density']['outputs']['density'][0] = 5e-9
    candidate['contracts']['density']['gradients']['weight'][0] = 5e-9
    result = compare_diagnostics(reference, candidate, 'float64')
    assert not result['passed']
    assert len(result['failures']) == 1
    assert 'outputs/density' in result['failures'][0]


def test_worker_startup_failure_is_saved_before_any_timing(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path/'failure.json'
    # Invalid source roots fail during source isolation, before scientific
    # construction, numerical work or the timer. This exercises the real IPC
    # and failure writer rather than a mock of the implementation.
    completed = subprocess.run([
        sys.executable, '-m', 'benchmarks.paired_pipeline',
        '--reference-root', str(tmp_path/'missing-reference'),
        '--candidate-root', str(tmp_path/'missing-candidate'),
        '--fixture', str(tmp_path/'unused-fixture.pt'), '--mode', 'ef',
        '--device', 'cpu', '--output', str(output)],
        cwd=root, text=True, capture_output=True, timeout=60)
    assert completed.returncode != 0
    result = json.loads(output.read_text())
    assert result['passed'] is False
    assert result['stage'] == 'startup'
    assert result['rounds'] == []
    assert result['diagnostics']['reference']['startup']['ok'] is False
    assert 'Wrong equiv_dens import' in result['error']
