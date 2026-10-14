"""Data, portable state and calculator boundary contracts."""
import json
import sys
import numpy as np
import pytest

from equiv_dens_jax.data import prepare_atoms,unpack_atoms,sample_grid,load_batches
from equiv_dens_jax.serialization import save_tree,load_tree
from equiv_dens_jax.md import Calculator,ase_calculator


def test_ragged_batch_graph_force_packing_and_grid_alignment(tmp_path):
    numbers=np.array([[8,1,1],[8,1,0]])
    positions=np.arange(18,dtype=float).reshape(2,3,3)/10
    forces=positions+1
    coords=np.arange(42).reshape(2,7,3)
    atoms=prepare_atoms(dict(atom_numbers=numbers,positions=positions,forces=forces,
                            coords=coords,density=coords[...,0],coord_weights=np.ones((2,7))))
    np.testing.assert_array_equal(unpack_atoms(atoms["forces"],atoms)[numbers>0],forces[numbers>0])
    batches=atoms["atom_batch_idx"][0]
    assert np.all(batches[atoms["idx_i"]]==batches[atoms["idx_j"]])
    assert np.all(atoms["idx_i"]!=atoms["idx_j"])
    assert len(atoms["idx_i"])==8
    sampled=sample_grid(atoms,4,seed=13)
    np.testing.assert_array_equal(sampled["coords"][...,0],sampled["density"])
    save_tree(tmp_path/"data",{"batches":[sampled]})
    loaded=load_batches(tmp_path/"data")[0]
    np.testing.assert_array_equal(loaded["positions"],atoms["positions"])


def test_bundle_roundtrip_tuple_keys_scalars_and_corruption(tmp_path):
    path=tmp_path/"state"
    tree={(8,2):np.arange(15,dtype=np.float64).reshape(3,5),
          "optimizer":{"count":np.int64(3),"limit":float("inf")},"empty":(),"none":None}
    save_tree(path,tree)
    result=load_tree(path)
    np.testing.assert_array_equal(result[(8,2)],tree[(8,2)])
    assert result["optimizer"]==tree["optimizer"]
    archive=path/json.loads((path/"manifest.json").read_text())["arrays"]
    archive.write_bytes(archive.read_bytes()+b"corruption")
    with pytest.raises(ValueError,match="checksum"):
        load_tree(path)


class HarmonicPotential:
    """Known internal Å/eV potential with independently declared I/O units."""
    def __init__(self,distance_in=1.,distance_out=1.,energy_out=1.):
        def conversion(distance):
            name={1.:"angstrom_to_angstrom",.52917721092:"bohr_to_angstrom",
                  1/.52917721092:"angstrom_to_bohr"}[distance]
            return {"type":"UnitsConversion","path":"","children":{},
                    "attrs":{"dist_conversion_func":{"function":name}}}
        self.node={"children":{"conversions_in":conversion(distance_in),
                               "conversions_out":conversion(distance_out)}}
        self.distance_in=distance_in
        self.distance_out=distance_out
        self.energy_out=energy_out

    def predict(self,atoms,**kwargs):
        p=atoms["positions"]*self.distance_in
        return {"energy":np.sum(p*p,axis=(1,2))[:,None]*.5*self.energy_out,
                "forces":-p*self.energy_out/self.distance_out}


@pytest.mark.parametrize("distance_in,distance_out",[(1.,1.),(.52917721092,1/.52917721092)])
def test_ase_units_centering_and_two_step_md(tmp_path,distance_in,distance_out):
    from ase import Atoms,units
    from ase.md.verlet import VelocityVerlet
    from ase.io.trajectory import Trajectory
    positions=np.array([[0.,0.,0.],[.8,.2,.1]])
    atoms=Atoms("HH",positions=positions)
    # 2 output energy units per eV, converted back at the adapter boundary.
    atoms.calc=ase_calculator(HarmonicPotential(distance_in,distance_out,2.),energy_to_ev=.5,center=False)
    np.testing.assert_allclose(atoms.get_potential_energy(),.5*np.sum(positions**2),atol=1e-14)
    np.testing.assert_allclose(atoms.get_forces(),-positions,atol=1e-14)
    atoms.set_velocities(np.zeros((2,3)))
    with Trajectory(tmp_path/"md.traj","w",atoms) as trajectory:
        integrator=VelocityVerlet(atoms,timestep=.1*units.fs)
        integrator.attach(trajectory.write,interval=1)
        integrator.run(2)
    assert np.isfinite(atoms.positions).all()
    assert not np.array_equal(atoms.positions,positions)
    with Trajectory(tmp_path/"md.traj") as trajectory:
        assert len(trajectory)==3
    assert "torch" not in sys.modules


def test_charge_center_calculator_and_batch_mapping():
    numbers=np.array([[8,1,1],[8,1,0]])
    positions=np.arange(18,dtype=float).reshape(2,3,3)/10
    values=Calculator(HarmonicPotential(),center=True).calculate(numbers,positions)
    centers=np.sum(positions*numbers[...,None],axis=1)/numbers.sum(axis=1)[:,None]
    np.testing.assert_allclose(values["forces"][numbers>0],-(positions-centers[:,None])[numbers>0])
