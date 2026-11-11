import importlib.util
import json
from pathlib import Path
import sys
import pytest

path = Path(__file__).resolve().parents[2] / 'scripts/revision/recover_hybrid_size_ood.py'
spec = importlib.util.spec_from_file_location('hybrid_recovery', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_large_missing_tier_uses_bounded_array_and_dependent_continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['recover', '--repo', str(tmp_path), '--submit'])
    monkeypatch.setattr(module, 'live_task_ids', lambda: set())
    monkeypatch.setattr(module, 'survey', lambda *a: {'n_shards': 3095, 'n_done': 0, 'counts': {'pending': 3095}, 'outstanding': list(range(3095))})
    calls = []
    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'squeue':
            return 'cpu48\n' * 435 if '-r' in argv else ''
        return '1234\n' if len([x for x in calls if x[0] == 'sbatch']) == 1 else '1235\n'
    monkeypatch.setattr(module.subprocess, 'check_output', command)
    module.main()
    submits = [x for x in calls if x[0] == 'sbatch']
    assert '--array=0-1999%200' in submits[0]
    assert any('RETRY_ERRORS=1' in part for part in submits[0])
    assert '--dependency=afterany:1234' in submits[1]
    assert len(submits) == 2


def test_partial_done_marker_is_not_counted_as_complete(tmp_path, monkeypatch):
    marker = tmp_path / 'results/revision/water_orca_md40x/water_ood_size_small/wb97mv_def2tzvpd/shard_0000/shard.done'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({'n_frames': 3, 'n_results': 2}))
    monkeypatch.setattr(sys, 'argv', ['recover', '--repo', str(tmp_path), '--round', '12', '--submit'])
    monkeypatch.setattr(module, 'live_task_ids', lambda: set())
    monkeypatch.setattr(module, 'survey', lambda *a: {'n_shards': 1, 'n_done': 1, 'counts': {'done': 1}, 'outstanding': []})
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw: '')
    with pytest.raises(RuntimeError, match='round limit'):
        module.main()
