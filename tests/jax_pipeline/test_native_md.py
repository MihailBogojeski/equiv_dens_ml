"""Actual current network with live host preparation and ASE trajectory I/O."""
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from equiv_dens_jax.pipeline import Pipeline
from equiv_dens_jax.serialization import load_tree
from equiv_dens_jax.md import Calculator,run_md
from equiv_dens_jax.md_preparation import make_md_preparation

jax.config.update("jax_enable_x64",True)


def test_actual_network_live_grid_and_two_step_md(tmp_path):
    from ase import Atoms
    from ase.io import write
    from ase.io.trajectory import Trajectory
    reference=Path(os.environ.get("DENSNET_JAX_PIPELINE_REFERENCE","scratch/jax_pipeline/pipeline-tiny"))
    if not (reference/"manifest.json").exists():
        pytest.fail("Generate independent current-model architecture before MD qualification")
    tree=load_tree(reference)
    pipeline=Pipeline(tree["model"],tree["state"])
    basis=tmp_path/"basis.npy"
    atoms_data=tmp_path/"atomic.npy"
    np.save(basis,{"H":[(1,1,0)],"O":[(8,1,0)]},allow_pickle=True)
    np.save(atoms_data,{1:{"mo_basis":"sto-3g","mo_coeff":np.ones((1,1)),"mo_occ":np.ones(1)},
                        8:{"mo_basis":"sto-3g","mo_coeff":np.eye(5),"mo_occ":np.array([2.,2.,2.,1.,1.])}},allow_pickle=True)
    prepare=make_md_preparation({"basis_path":str(basis),"atom_density_path":str(atoms_data),
         "trusted_legacy":True,"atom_density_type":"mo_coeffs","remove_atom_density":True,
         "density_grad":True,"include_atom_dipole":True,"grid_level":0,"grid_count":16,"seed":3})
    positions=np.asarray(tree["inputs"]["batch_positions"])[0]
    numbers=np.asarray(tree["inputs"]["batch_atom_numbers"])[0]
    calculator=Calculator(pipeline,properties=("energy","density"),prepare_properties=prepare)
    first=calculator.calculate(numbers,positions)
    assert first["density"].shape==(1,16)
    assert first["density_grad"].shape==(1,16,3)
    assert first["forces"].shape==(1,3,3)
    changed=positions.copy()
    changed[1,0]+=.01
    second=calculator.calculate(numbers,changed)
    for row in (first,second):
        assert all(np.isfinite(value).all() for value in row.values())
    assert not np.array_equal(first["density"],second["density"])
    geometry=tmp_path/"water.xyz"
    write(geometry,Atoms(numbers=numbers,positions=positions))
    output=tmp_path/"native.traj"
    result=run_md(pipeline,geometry,output,steps=2,timestep_fs=.001,temperature_k=20.,seed=7,
                  energy_to_ev=1.,properties=("energy","density"),prepare_properties=prepare)
    assert np.isfinite(result.positions).all()
    with Trajectory(output) as trajectory:
        assert len(trajectory)==3
        assert all(np.isfinite(frame.get_forces()).all() for frame in trajectory)
