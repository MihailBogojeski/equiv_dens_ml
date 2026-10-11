"""Torch-free energy/force calculator and ASE molecular-dynamics integration."""
from __future__ import annotations

import numpy as np
from .data import prepare_atoms,unpack_atoms


class Calculator:
    """Host calculator returning completed native JAX energies/forces.

    Positions supplied here are in the model's input distance units. Returned
    energies/forces use its output units; this boundary performs no additional
    unit conversion. Centering follows the current MD adapter's charge centroid.
    Optional grid/SAD preparation is an explicit host function of current atoms.
    """
    def __init__(self,pipeline,*,properties=("energy",),cutoff=None,jit=True,
                 center=True,prepare_properties=None):
        self.pipeline=pipeline
        self.properties=tuple(properties)
        self.cutoff=cutoff
        self.jit=jit
        self.center=center
        self.prepare_properties=prepare_properties

    def calculate(self,numbers,positions):
        numbers=np.asarray(numbers,dtype=np.int64)
        positions=np.asarray(positions)
        if numbers.ndim==1:
            numbers=numbers[None]
        if positions.ndim==2:
            positions=positions[None]
        if self.center:
            center=np.sum(positions*numbers[...,None],axis=1)/np.sum(numbers,axis=1)[:,None]
            positions=positions-center[:,None]
        atoms=prepare_atoms({"atom_numbers":numbers,"positions":positions},cutoff=self.cutoff)
        if self.prepare_properties is not None:
            atoms=self.prepare_properties(atoms)
        output=self.pipeline.predict(atoms,properties=self.properties,forces=True,jit=self.jit,convert_inputs=True)
        result={"energy":np.asarray(output["energy"]),"forces":unpack_atoms(output["forces"],atoms)}
        for name in ("dipole_moment","df_coeffs","density","density_grad"):
            if name in output:
                result[name]=np.asarray(output[name])
        return result


def ase_calculator(pipeline,*,energy_to_ev,**kwargs):
    """Adapt to ASE's eV/Angstrom contract with an explicit energy conversion.

    Internal positions use Angstrom, as in DFTNetwork. Input coordinates are
    converted to the model's declared input units, and output forces to eV/Å.
    ``energy_to_ev`` describes the declared output energy unit.
    """
    from ase.calculators.calculator import Calculator as ASECalculator,all_changes
    from .pipeline import _factor,_DISTANCE
    from .common import child
    incoming=child(pipeline.node,"conversions_in")
    outgoing=child(pipeline.node,"conversions_out")
    input_distance=_factor(incoming,"dist_conversion_func",_DISTANCE)
    output_distance=_factor(outgoing,"dist_conversion_func",_DISTANCE)
    if not np.isfinite(energy_to_ev) or energy_to_ev<=0:
        raise ValueError("energy_to_ev must be positive and finite")
    evaluator=Calculator(pipeline,**kwargs)
    class NativeCalculator(ASECalculator):
        implemented_properties=["energy","forces","dipole"]
        def calculate(self,atoms=None,properties=("energy","forces"),system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            values=evaluator.calculate(self.atoms.numbers,self.atoms.positions/input_distance)
            self.results={"energy":float(values["energy"].reshape(-1)[0])*energy_to_ev,
                          "forces":values["forces"][0]*energy_to_ev*output_distance}
            if "dipole_moment" in values:
                self.results["dipole"]=values["dipole_moment"][0]
    return NativeCalculator()


def run_md(pipeline,geometry,output,*,steps,timestep_fs=0.5,temperature_k=300.,seed=0,
           energy_to_ev,properties=("energy",),cutoff=None,jit=True,prepare_properties=None):
    """ASE velocity-Verlet trajectory using the native potential; no new trainer."""
    from ase import units
    from ase.io import read
    from ase.io.trajectory import Trajectory
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary
    if steps<0 or timestep_fs<=0 or temperature_k<0:
        raise ValueError("Invalid MD steps, time step or temperature")
    atoms=read(geometry)
    atoms.calc=ase_calculator(pipeline,energy_to_ev=energy_to_ev,properties=properties,cutoff=cutoff,jit=jit,
                             prepare_properties=prepare_properties)
    MaxwellBoltzmannDistribution(atoms,temperature_K=temperature_k,rng=np.random.default_rng(seed))
    Stationary(atoms)
    integrator=VelocityVerlet(atoms,timestep=timestep_fs*units.fs)
    with Trajectory(output,"w",atoms) as trajectory:
        # ASE invokes attached observers at step zero as well as after steps.
        integrator.attach(trajectory.write,interval=1)
        integrator.run(steps)
    return atoms
