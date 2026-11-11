"""Compatibility boundaries for the isolated layout migration.

Set EQUIV_DENS_REFERENCE to a detached original checkout for differential checks.
All executable checks use temporary data and stub scheduler calls.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
MAP = json.loads((ROOT / 'docs/refactor/migration_map.json').read_text())


def reference():
    value = os.environ.get('EQUIV_DENS_REFERENCE')
    if not value:
        pytest.skip('set EQUIV_DENS_REFERENCE for original/refactor comparisons')
    return Path(value)


def test_every_old_path_survives_without_external_symlinks():
    for item in MAP:
        path = ROOT / item['old_path']
        assert path.is_file(), item
        assert path.resolve().is_relative_to(ROOT), item


def test_slurm_files_are_byte_identical_to_baseline():
    base = reference()
    for item in MAP:
        if item['old_path'].endswith('.sbatch'):
            assert (ROOT / item['old_path']).read_bytes() == (base / item['old_path']).read_bytes()


def test_config_aliases_preserve_bytes():
    base = reference()
    for item in MAP:
        if item['old_path'].startswith('config/'):
            assert (ROOT / item['old_path']).read_bytes() == (base / item['old_path']).read_bytes()


def test_workflow_statements_keep_their_original_order():
    base = reference()
    for old, name in [('train_dens.py','density'),('train.py','joint'),
                      ('train_all.py','multiphase'),('train_only_en.py','energy')]:
        before = ast.parse((base / 'scripts/training' / old).read_text())
        after = ast.parse((ROOT / 'src/equiv_dens/training/workflows' / (name+'.py')).read_text())
        body = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == 'main').body
        # The only body changes are explicit argv forwarding and successful return.
        assert isinstance(body[-1], ast.Return)
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'parse_command_line_arguments':
                node.keywords = [k for k in node.keywords if k.arg != 'argv']
        # Moving into a function indents free-standing documentation strings.
        # They are not executable scientific statements.
        before.body = [n for n in before.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
        body = [n for n in body[:-1] if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
        assert ast.dump(before, include_attributes=False) == ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def test_helper_imports_and_monkeypatches_share_the_implementation(monkeypatch):
    old = ROOT / 'scripts/revision/lib/energy_pack.py'
    spec = importlib.util.spec_from_file_location('old_energy_pack_test', old)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    canonical = importlib.import_module('experiments.jacs_revision.lib.energy_pack')
    replacement = object()
    monkeypatch.setattr(module, 'run_pack', replacement)
    assert canonical.run_pack is replacement
    assert sys.modules[spec.name] is canonical


def test_experiment_submission_preserves_environment_options_and_exit_code(monkeypatch):
    from equiv_dens.commands import experiment
    calls=[]
    monkeypatch.setenv('THEORY','pbe_d4_avdz')
    monkeypatch.setenv('RUN','water')
    def call(argv, **kwargs):
        calls.append((argv,kwargs));return 17
    monkeypatch.setattr(experiment.subprocess,'call',call)
    args=SimpleNamespace(action='submit',task='submit_water_train',extra=['--dependency=afterok:123','--partition=a100_chemistry'])
    assert experiment.dispatch(args)==17
    argv,kwargs=calls[0]
    assert argv[:3]==['sbatch','--dependency=afterok:123','--partition=a100_chemistry']
    assert argv[-1]==str(ROOT/'experiments/jacs_revision/slurm/submit_water_train.sbatch')
    assert kwargs['cwd']==ROOT
    assert kwargs['env']['THEORY']=='pbe_d4_avdz'
    assert kwargs['env']['RUN']=='water'


def test_experiment_registry_never_submits_through_run(monkeypatch):
    from equiv_dens.commands import experiment
    monkeypatch.setattr(experiment.subprocess,'call',lambda *a,**k: pytest.fail('unexpected launch'))
    assert experiment.dispatch(SimpleNamespace(action='run',task='submit_water_train',extra=[]))==2
    assert experiment.dispatch(SimpleNamespace(action='submit',task='water/campaign_status',extra=[]))==2
    assert experiment.dispatch(SimpleNamespace(action='run',task='unknown',extra=[]))==2


def test_registry_targets_exist_and_helpers_are_not_commands():
    registry=json.loads((ROOT/'experiments/jacs_revision/tasks.json').read_text())
    assert 'lib/energy_pack' not in registry
    for name,item in registry.items():
        if item['kind']=='python':
            path=ROOT/(item['target'].replace('.','/')+'.py')
        else:path=ROOT/'experiments/jacs_revision'/item['target']
        assert path.is_file(),(name,item)


def _generate_water(repo, output):
    subprocess.run([sys.executable,str(repo/'scripts/revision/generate_water_cluster_geoms.py'),
                    '--output-dir',str(output),'--seed','19','--train-per-size','1',
                    '--val-per-size','1','--id-test-per-size','1','--ood-per-size','1'],
                   cwd=repo,check=True,capture_output=True,text=True)
    return {str(p.relative_to(output)):p.read_text().replace(str(output),'<OUTPUT>')
            for p in output.rglob('*') if p.is_file()}


def test_seeded_geometry_and_manifest_match_baseline(tmp_path):
    base=reference()
    assert _generate_water(base,tmp_path/'baseline')==_generate_water(ROOT,tmp_path/'refactor')


def test_watchdog_profiles_keep_rackers_classification_and_priority(tmp_path):
    script=ROOT/'experiments/jacs_revision/water/watch_water_campaign.sh'
    # Source initialization and function definitions only. No loop or scheduler calls.
    prefix=script.read_text().split('for ((round = 1;',1)[0]
    prefix=prefix.replace('cd "$(revision_repo_root)"', 'cd '+str(ROOT))
    for profile,expected in [('root_compat','small:99'),('water_compat','large:10')]:
        sample=tmp_path/(profile+'.sh')
        sample.write_text(prefix+'\nif is_large water_ood_rackers; then printf large:; else printf small:; fi\nsplit_rank water_ood_rackers\n')
        result=subprocess.run(['bash',str(sample),'--internal-profile',profile],cwd=tmp_path,capture_output=True,text=True,check=True)
        assert result.stdout.strip()==expected


def test_importing_workflows_does_not_start_training(tmp_path):
    code='''
import importlib
from equiv_dens.training import utils
utils.init_training_vars = lambda *a, **k: (_ for _ in ()).throw(AssertionError("training started on import"))
for mode in ("density","joint","multiphase","energy"):
    module=importlib.import_module("equiv_dens.training.workflows."+mode)
    assert callable(module.main)
'''
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'WANDB_MODE':'disabled','CUDA_VISIBLE_DEVICES':''}
    subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mode',['density','joint','multiphase','energy'])
def test_training_engine_and_cross_tree_resume_match_baseline(tmp_path,mode):
    base=reference()
    runner=ROOT/'tests/fixtures/refactor/trainer_parity.py'
    environments={'baseline':base,'refactor':ROOT}
    for name,repo in environments.items():
        env={**os.environ,'PYTHONPATH':str(repo/'src'),'CUDA_VISIBLE_DEVICES':'','WANDB_MODE':'disabled','OMP_NUM_THREADS':'1'}
        python=repo/'.venv/bin/python'
        assert python.is_file(),'create independent reference/development environments'
        subprocess.run([str(python),str(runner),'--mode',mode,'--output',str(tmp_path/name)],
                       cwd=repo,env=env,check=True,capture_output=True,text=True)
    assert json.loads((tmp_path/'baseline/step.json').read_text())==json.loads((tmp_path/'refactor/step.json').read_text())
    # Resume identical baseline checkpoint bytes through each source tree.
    import shutil
    shutil.rmtree(tmp_path/'refactor/run_fixture')
    shutil.copytree(tmp_path/'baseline/run_fixture',tmp_path/'refactor/run_fixture')
    for name,repo in environments.items():
        env={**os.environ,'PYTHONPATH':str(repo/'src'),'CUDA_VISIBLE_DEVICES':'','WANDB_MODE':'disabled','OMP_NUM_THREADS':'1'}
        subprocess.run([str(repo/'.venv/bin/python'),str(runner),'--mode',mode,'--output',str(tmp_path/name),'--resume'],
                       cwd=repo,env=env,check=True,capture_output=True,text=True)
    assert json.loads((tmp_path/'baseline/resumed.json').read_text())==json.loads((tmp_path/'refactor/resumed.json').read_text())


def test_short_cpu_md_matches_baseline(tmp_path):
    base=reference()
    code=r'''
import json, numpy as np, torch
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet
from equiv_dens.md.dft_network_calculator import DenSNetCalculator
class Harmonic(torch.nn.Module):
    def forward(self, inputs):
        positions=inputs['positions']
        return {'energy': (positions**2).sum().reshape(1), 'forces': -2*positions}
atoms=Atoms('H2',positions=[[0.1,0,0],[0.8,0,0]])
atoms.set_momenta(np.zeros((2,3)))
atoms.calc=DenSNetCalculator(Harmonic(),cutoff=6.0,use_gpu=False,energy_unit='eV')
integrator=VelocityVerlet(atoms,timestep=0.2*units.fs)
integrator.run(5)
print(json.dumps({'positions':atoms.positions.tolist(),'momenta':atoms.get_momenta().tolist(),'energy':atoms.get_potential_energy()}))
'''
    results=[]
    for repo in [base,ROOT]:
        env={**os.environ,'PYTHONPATH':str(repo/'src'),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'}
        result=subprocess.run([str(repo/'.venv/bin/python'),'-c',code],cwd=tmp_path,env=env,capture_output=True,text=True,check=True)
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert results[0]==results[1]


def test_wheel_dispatch_does_not_add_caller_source_tree(monkeypatch):
    from equiv_dens import cli
    monkeypatch.setattr(cli,'checkout_root',lambda:None)
    monkeypatch.setenv('PYTHONPATH','/explicit/dependencies')
    assert cli._env_with_src()['PYTHONPATH']=='/explicit/dependencies'


def test_cli_dispatch_preserves_config_order_cwd_and_exit_code(monkeypatch):
    from equiv_dens import cli
    calls=[]
    monkeypatch.setattr(cli.subprocess,'call',lambda argv,**kw: calls.append((argv,kw)) or 13)
    args=SimpleNamespace(config='config/training/water_energy_base.txt',mode='energy',
                         extra=['@config/training/water_energy_pbe.txt','--save_dir=/tmp/test-only'])
    assert cli._cmd_train(args)==13
    argv,kw=calls[0]
    assert argv==[sys.executable,'-m','equiv_dens.training.workflows.energy',
                  '@config/training/water_energy_base.txt',*args.extra]
    assert Path(kw['cwd'])==ROOT
