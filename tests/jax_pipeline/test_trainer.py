"""Source training-policy boundaries and output-unit validation."""
import copy
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from equiv_dens_jax.pipeline import Pipeline
from equiv_dens_jax.serialization import load_tree
from equiv_dens_jax.training import Optimizer,load_checkpoint
from equiv_dens_jax.trainer import Trainer
from equiv_dens_jax.losses import ErrorDict

jax.config.update("jax_enable_x64",True)


@pytest.fixture
def model():
    path=Path(os.environ.get("DENSNET_JAX_PIPELINE_REFERENCE","scratch/jax_pipeline/pipeline-tiny"))
    if not (path/"manifest.json").exists():
        pytest.fail("Generate the independent complete pipeline reference")
    return load_tree(path)


def test_validation_uses_output_units_for_predictions_and_labels(model):
    node=copy.deepcopy(model["model"])
    node["children"]["scaling"]["attrs"]["std"]=3.
    conversion=node["children"]["conversions_out"]
    conversion["children"].pop("en_conversion_func",None)
    conversion["attrs"]["en_conversion_func"]={"function":"kcal_to_hartree"}
    pipeline=Pipeline(node,model["state"])
    loss=ErrorDict({"energy":1.},loss_comp={"energy":["mse"]})
    trainer=Trainer(pipeline,loss,Optimizer(ema_decay=None),stage="energy")
    actual=trainer.validate([model["inputs"]],ema=False)
    difference=(model["expected"]["energy"]-model["inputs"]["energy"])*3/627.5096080305927
    expected=float(np.mean(difference**2))
    np.testing.assert_allclose(actual["loss"],expected,atol=1e-14,rtol=1e-8)
    assert not np.isclose(actual["loss"],float(np.mean((difference*627.5096080305927/3)**2)),rtol=1e-3)


def test_scheduler_reduces_after_patience_and_restores_state(model,tmp_path):
    pipeline=Pipeline(model["model"],model["state"])
    loss=ErrorDict({"energy":1.},loss_comp={"energy":["mse"]})
    optimizer=Optimizer(learning_rate=.01,ema_decay=None)
    trainer=Trainer(pipeline,loss,optimizer,stage="energy",scheduler={"patience":1,"factor":.5,"min_lr":.002})
    for value,expected in [(10.,.01),(10.,.01),(10.,.005),(9.,.005),(9.,.005),(9.,.0025)]:
        trainer.schedule(value)
        assert trainer.loop["learning_rate"]==expected
    trainer.save(tmp_path/"checkpoint")
    restored,state,stage,opt,error=load_checkpoint(tmp_path/"checkpoint")
    continued=Trainer(restored,error,opt,stage=stage,state=state,scheduler=trainer.scheduler)
    continued.schedule(9.)
    continued.schedule(9.)
    assert continued.loop["learning_rate"]==.002


def test_compiled_validation_matches_eager_with_dynamic_state_and_weights(model,tmp_path):
    loss=ErrorDict({"energy":1.,"forces":.2},loss_comp={"energy":["mse"],"forces":["mse"]})
    optimizer=Optimizer(learning_rate=1e-5,ema_decay=None,lookahead_k=0)
    compiled=Trainer(Pipeline(model["model"],model["state"]),loss,optimizer,
                     stage="energy",jit_validation=True)
    eager=Trainer(Pipeline(model["model"],model["state"]),loss,optimizer,stage="energy")
    for updated in (False,True):
        if updated:
            compiled.step([model["inputs"]],shuffle=False)
            eager.state=compiled.state
            compiled.loop["validation_loss_weights"]={"energy":.3,"forces":.7}
            eager.loop["validation_loss_weights"]=dict(compiled.loop["validation_loss_weights"])
        actual=compiled.validate([model["inputs"]],ema=False)
        expected=eager.validate([model["inputs"]],ema=False)
        assert actual.keys()==expected.keys()
        for key in actual:
            np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=1e-10)
    assert len(compiled.validations)==1
    compiled.save(tmp_path/"compiled")
    restored,_,_,_,_=load_checkpoint(tmp_path/"compiled")
    assert restored.metadata["native_loop"]["jit_validation"] is True
    compiled.change_stage("density",loss=ErrorDict({"density":1.},loss_comp={"density":["mse"]}))
    assert not compiled.validations
    assert np.isfinite(compiled.validate([model["inputs"]],ema=False)["loss"])


def test_grid_annealing_changes_current_scalar_and_serializes_buffer(model,tmp_path):
    node=copy.deepcopy(model["model"])
    density=node["children"]["property_models"]["children"]["density"]
    density["attrs"]["integral_constraint"]="grid"
    name=density["path"]+".grid_scaling_factor"
    density["buffers"]["grid_scaling_factor"]=name
    state=copy.deepcopy(model["state"])
    state["buffers"][name]=np.ones(1)
    pipeline=Pipeline(node,state)
    loss=ErrorDict({"density":1.},loss_comp={"density":["mse"]})
    trainer=Trainer(pipeline,loss,Optimizer(learning_rate=1e-6,ema_decay=None),stage="density",
                    grid_scaling_start=0,grid_scaling_annealing=.5)
    trainer.step([model["inputs"]],shuffle=False)
    assert trainer.loop["grid_scaling_factor"]==1.  # strict epoch > start
    trainer.step([model["inputs"]],shuffle=False)
    assert trainer.loop["grid_scaling_factor"]==.5
    assert len(trainer.updates)==1
    trainer.save(tmp_path/"checkpoint")
    restored,_,_,_,_=load_checkpoint(tmp_path/"checkpoint")
    np.testing.assert_array_equal(restored.state["buffers"][name],[.5])
    inferred=restored.predict(model["inputs"],properties=("density",),forces=False)
    data=restored.prepare(model["inputs"])
    explicit=restored.forward(restored.state,dict(data,density_grid_scaling_factor=.5),
                              properties=("density",),forces=False)
    np.testing.assert_allclose(inferred["density"],explicit["density"],atol=1e-12,rtol=1e-12)
