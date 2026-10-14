"""Actual native commands: fresh init, fit/validation, resume and inference."""
import json
import copy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from equiv_dens_jax.serialization import load_tree,save_tree


def invoke(*arguments):
    result=subprocess.run([sys.executable,"-m","equiv_dens_jax.cli","--platform","cpu",*map(str,arguments)],
                          text=True,capture_output=True,timeout=300)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout


def test_native_commands_and_exact_shuffled_loop_resume(tmp_path):
    reference=Path(os.environ.get("DENSNET_JAX_PIPELINE_REFERENCE","scratch/jax_pipeline/pipeline-tiny")).resolve()
    if not (reference/"manifest.json").exists():
        pytest.fail("Generate complete Torch pipeline fixture before CLI qualification")
    tree=load_tree(reference)
    second=copy.deepcopy(tree["inputs"])
    second["density"]=second["density"]+.007
    second["density_grad"]=second["density_grad"]-.003
    save_tree(tmp_path/"data",{"batches":[tree["inputs"],second]})
    config={"stage":"density","seed":11,"shuffle":True,
            "optimizer":{"name":"sgd","learning_rate":1e-4,"lookahead_k":3,"ema_decay":.8},
            "loss":{"loss_weights":{"density":1.,"density_grad":.1},
                    "loss_comp":{"density":["mse"],"density_grad":["mse"]},
                    "weights_decay":{"density":.5,"density_grad":1.},
                    "weights_min":{"density":.1,"density_grad":.1}},
            "scheduler":{"factor":.5,"patience":0}}
    path=tmp_path/"config.json"
    path.write_text(json.dumps(config))
    fresh=tmp_path/"fresh"
    invoke("init","--template",reference,"--seed",19,"--output",fresh)
    initialized=load_tree(fresh)
    assert any(not np.array_equal(initialized["state"]["params"][key],value)
               for key,value in tree["state"]["params"].items())
    common=("--data",tmp_path/"data","--validation",tmp_path/"data","--config",path)
    invoke("train","--model",reference,*common,"--steps",1,"--output",tmp_path/"one")
    invoke("train","--model",tmp_path/"one","--resume",*common,"--steps",3,"--output",tmp_path/"resumed")
    invoke("train","--model",reference,*common,"--steps",4,"--output",tmp_path/"uninterrupted")
    resumed=load_tree(tmp_path/"resumed")
    expected=load_tree(tmp_path/"uninterrupted")
    assert resumed["training"]["step"]==4
    assert resumed["training"]["loop"]["epoch"]==2
    assert resumed["training"]["loop"]["loss_weights"]["density"]==.0625
    assert resumed["training"]["loop"]["validation_loss_weights"]["density"]==.125
    mismatch=subprocess.run([sys.executable,"-m","equiv_dens_jax.cli","--platform","cpu",
          "--precision","float32","train","--model",str(tmp_path/"one"),"--resume",
          *map(str,common),"--steps","1","--output",str(tmp_path/"invalid")],
          text=True,capture_output=True,timeout=60)
    assert mismatch.returncode!=0
    assert "Exact resume requires checkpoint precision" in mismatch.stderr
    assert not (tmp_path/"invalid").exists()
    def equal(left,right):
        assert type(left)==type(right)
        if isinstance(left,dict):
            assert left.keys()==right.keys()
            for name in left:
                equal(left[name],right[name])
        elif isinstance(left,(list,tuple)):
            assert len(left)==len(right)
            for a,b in zip(left,right):
                equal(a,b)
        elif isinstance(left,np.ndarray):
            np.testing.assert_array_equal(left,right)
        else:
            assert left==right
    equal(resumed["training"],expected["training"])
    invoke("infer","--model",tmp_path/"resumed","--data",tmp_path/"data",
           "--properties","energy","density","--output",tmp_path/"prediction")
    result=load_tree(tmp_path/"prediction")
    assert len(result["predictions"])==2
    for row in result["predictions"]:
        assert {"energy","forces","density","density_grad","df_coeffs"}<=row.keys()
        assert all(np.isfinite(value).all() for value in row.values())
