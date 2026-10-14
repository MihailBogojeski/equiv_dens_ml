"""Complete-mode contracts against arrays collected in a separate Torch runtime."""
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equiv_dens_jax.pipeline import Pipeline
from equiv_dens_jax.losses import ErrorDict
from equiv_dens_jax.training import Optimizer, initialize, make_step, save_checkpoint, load_checkpoint
from equiv_dens_jax.serialization import load_tree

jax.config.update("jax_enable_x64",True)
jax.config.update("jax_default_matmul_precision","highest")
REFERENCE=Path(os.environ.get("DENSNET_JAX_PIPELINE_REFERENCE","scratch/jax_pipeline/pipeline-tiny"))


@pytest.fixture(scope="module")
def reference():
    if not (REFERENCE/"manifest.json").exists():
        pytest.fail("Generate the independent Torch pipeline reference before native JAX qualification")
    return load_tree(REFERENCE)


@pytest.fixture(scope="module")
def prepared(reference):
    pipeline=Pipeline(reference["model"],reference["state"])
    atoms=pipeline.prepare(reference["inputs"])
    return pipeline,atoms


def assert_tree(actual,expected,atol=1e-9,rtol=1e-8):
    left,leftdef=jax.tree.flatten(actual)
    right,rightdef=jax.tree.flatten(expected)
    assert leftdef==rightdef
    for a,b in zip(left,right):
        assert np.shape(a)==np.shape(b)
        assert np.isfinite(np.asarray(a)).all()
        np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)


def test_complete_outputs_eager_and_jit(reference,prepared):
    pipeline,atoms=prepared
    def run(state,data):
        return pipeline.forward(state,data,training=True,properties=("energy","density"),forces=True)
    for output in (run(pipeline.state,atoms),jax.jit(run)(pipeline.state,atoms)):
        selected={key:output[key] for key in reference["expected"]}
        assert_tree(selected,reference["expected"])
    assert "torch" not in sys.modules


def test_joint_loss_and_all_parameter_derivatives(reference,prepared):
    pipeline,atoms=prepared
    loss=ErrorDict(**reference["loss_config"])
    def objective(params):
        result=pipeline.forward({"params":params,"buffers":pipeline.state["buffers"]},atoms,
                                training=True,properties=("energy","density"),forces=True)
        return loss.compute(result,atoms)["loss"]
    value,gradient=jax.jit(jax.value_and_grad(objective))(pipeline.state["params"])
    np.testing.assert_allclose(value,reference["errors"]["loss"],atol=1e-9,rtol=1e-8)
    assert set(gradient)==set(reference["gradients"])
    for name,expected in reference["gradients"].items():
        if expected is None:
            np.testing.assert_array_equal(gradient[name],np.zeros_like(gradient[name]))
        else:
            np.testing.assert_allclose(gradient[name],expected,atol=1e-8,rtol=1e-7,err_msg=name)


def test_weight_decay_skips_torch_none_gradients_and_keeps_optimizer_state(reference):
    pipeline=Pipeline(reference["model"],reference["state"])
    atoms=pipeline.prepare(reference["inputs"])
    loss=ErrorDict(**reference["loss_config"])
    optimizer=Optimizer(name="sgd",learning_rate=1e-5,weight_decay=.1,
                        selective_weight_decay=False,lookahead_k=0,ema_decay=None)
    initial=initialize(pipeline,optimizer=optimizer)
    step=make_step(pipeline,loss,optimizer=optimizer)
    actual,_=step(initial,atoms)
    missing=[name for name,gradient in reference["gradients"].items() if gradient is None]
    assert missing, "Independent fixture must exercise structurally unused parameters"
    for name,gradient in reference["gradients"].items():
        previous=np.asarray(initial["params"][name])
        expected=previous if gradient is None else previous-1e-5*(gradient+.1*previous)
        np.testing.assert_allclose(actual["params"][name],expected,atol=1e-12,rtol=1e-10,err_msg=name)
    for field in ("m","v","vmax","slow"):
        assert actual["optimizer"][field].keys()==initial["optimizer"][field].keys()
        for name in missing:
            np.testing.assert_array_equal(actual["optimizer"][field][name],initial["optimizer"][field][name])
    continued,_=step(actual,atoms)
    for name in missing:
        np.testing.assert_array_equal(continued["params"][name],initial["params"][name])


def test_force_matches_directional_finite_difference(prepared):
    pipeline,atoms=prepared
    output=pipeline.forward(pipeline.state,atoms,training=True,properties=("energy",),forces=True)
    direction=jnp.sin(jnp.arange(atoms["positions"].size,dtype=jnp.float64)+1).reshape(atoms["positions"].shape)
    direction=direction/jnp.linalg.norm(direction)
    def energy(positions):
        return jnp.sum(pipeline.forward(pipeline.state,dict(atoms,positions=positions),
                                      training=True,properties=("energy",),forces=False)["energy"])
    h=1e-5
    finite=(energy(atoms["positions"]+h*direction)-energy(atoms["positions"]-h*direction))/(2*h)
    np.testing.assert_allclose(-jnp.sum(output["forces"]*direction),finite,atol=1e-8,rtol=1e-6)


@pytest.mark.parametrize("stage,weights",[
    ("density",{"density":1.,"density_grad":.1}),
    ("energy",{"energy":.2,"forces":.3}),
    ("joint",{"density":1.,"density_grad":.1,"energy":.2,"forces":.3}),
])
def test_complete_update_freeze_rules_and_checkpoint_resume(tmp_path,prepared,stage,weights):
    pipeline,atoms=prepared
    loss=ErrorDict(weights,loss_comp={key:["mse"] for key in weights})
    optimizer=Optimizer(name="sgd",learning_rate=1e-4)
    initial=initialize(pipeline,stage=stage,optimizer=optimizer)
    update=make_step(pipeline,loss,stage=stage,optimizer=optimizer)
    trained,metrics=update(initial,atoms)
    assert np.isfinite(float(metrics["loss"]))
    active=set(initial["optimizer"]["m"])
    changed=set()
    for name,param in initial["params"].items():
        if not np.array_equal(param,trained["params"][name]):
            changed.add(name)
        if name not in active:
            np.testing.assert_array_equal(param,trained["params"][name])
    assert changed and changed<=active
    path=tmp_path/"checkpoint"
    save_checkpoint(path,pipeline,trained,stage=stage,optimizer=optimizer,loss=loss)
    restored,restored_state,restored_stage,restored_optimizer,restored_loss=load_checkpoint(path)
    assert restored_stage==stage
    restored.prepare(atoms)
    resumed=make_step(restored,restored_loss,stage=stage,optimizer=restored_optimizer)
    expected,_=update(trained,atoms)
    actual,_=resumed(restored_state,atoms)
    assert_tree(actual,expected,atol=1e-12,rtol=1e-12)


def test_successive_geometry_predictions_reuse_compiled_function(prepared):
    pipeline,atoms=prepared
    first=pipeline.predict(atoms,properties=("energy",),forces=True)
    cached=tuple(pipeline._compiled.values())
    moved=dict(atoms,positions=atoms["positions"]+jnp.arange(atoms["positions"].size).reshape(atoms["positions"].shape)*.001)
    second=pipeline.predict(moved,properties=("energy",),forces=True)
    assert tuple(pipeline._compiled.values())==cached
    assert not np.array_equal(first["energy"],second["energy"])
    assert np.isfinite(np.asarray(second["forces"])).all()


def test_host_basis_metadata_is_kept_outside_compiled_inputs(prepared):
    pipeline,atoms=prepared
    raw=dict(atoms,atom_df_coeffs_basis={"H":"sto-3g"},host_note="QC preparation")
    numerical=pipeline.prepare(raw)
    assert "atom_df_coeffs_basis" not in numerical
    result=pipeline.predict(raw,properties=("energy",),forces=True)
    assert np.isfinite(np.asarray(result["forces"])).all()
    assert pipeline.host_metadata["atom_df_coeffs_basis"]=={"H":"sto-3g"}


def test_grid_dipole_property_and_loss_resolve_density_prerequisite(prepared):
    from copy import deepcopy
    from equiv_dens_jax.training import required_properties
    base,_=prepared
    node=deepcopy(base.node)
    node["children"]["property_models"]["children"]["dipole_moment"]={
        "type":"DipoleMomentCalc","path":"property_models.dipole_moment",
        "attrs":{},"children":{},"params":{},"buffers":{}}
    node["attrs"]["no_force_props"].append("dipole_moment")
    pipeline=Pipeline(node,base.state)
    assert pipeline.resolve_properties(("dipole_moment",))==("density","dipole_moment")
    properties,forces=required_properties(ErrorDict({"dipole_moment":1.}),pipeline)
    assert properties==("density","dipole_moment") and not forces


def test_actual_native_network_calculator_and_md_step(prepared):
    from equiv_dens_jax.md import Calculator,ase_calculator
    from equiv_dens_jax.data import unpack_atoms
    from ase import Atoms,units
    from ase.md.verlet import VelocityVerlet
    pipeline,data=prepared
    reference=pipeline.predict(data,properties=("energy",),forces=True)
    numbers=np.asarray(data["batch_atom_numbers"])
    positions=np.asarray(data["batch_positions"])
    calculated=Calculator(pipeline).calculate(numbers,positions)
    np.testing.assert_allclose(calculated["energy"],reference["energy"],atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(calculated["forces"],unpack_atoms(reference["forces"],data),atol=1e-9,rtol=1e-8)
    active=numbers[0]>0
    atoms=Atoms(numbers=numbers[0,active],positions=positions[0,active])
    atoms.calc=ase_calculator(pipeline,energy_to_ev=1.)
    atoms.set_velocities(np.zeros((int(active.sum()),3)))
    VelocityVerlet(atoms,timestep=.01*units.fs).run(2)
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()


def test_invalid_species_is_rejected_before_jax_index_clamping(prepared):
    pipeline,data=prepared
    invalid=dict(data,batch_atom_numbers=np.full_like(data["batch_atom_numbers"],100))
    with pytest.raises(ValueError,match="embedding Zmax"):
        pipeline.prepare(invalid)
