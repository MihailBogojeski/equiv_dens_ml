# !/usr/bin/env python3
import equiv_dens.compat  # noqa: F401 - apply T_co patch before schnetpack import

import os
import sys
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling
import equiv_dens.utils.base as utils
from equiv_dens.training.model_loader import load_model
from equiv_dens.md.dft_network_calculator import DFTNetworkCalculator

import numpy as np
from functools import partial

# from respa_md import RESPAVerlet, RESPALangevin
from schnetpack.md import System
from schnetpack.md import MaxwellBoltzmannInit
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md import Simulator
from schnetpack.md.simulation_hooks import thermostats
from schnetpack.md.simulation_hooks import logging_hooks
from schnetpack.md.calculators import MDCalculator, MDCalculatorError
from schnetpack.md.data import HDF5Loader


# MD calculator class
class DummyCalculator(MDCalculator):
    def __init__(self,
                 props,
                 required_properties,
                 force_handle,
                 position_conversion=1.0,
                 force_conversion=1.0,
                 property_conversion={},
                 stress_handle=None,
                 stress_conversion=1.0,
                 detach=True,
                 time_step=0.5,
                 ):
        # energy prediction model
        super().__init__(
            required_properties,
            force_handle,
            detach=detach,
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            property_conversion=property_conversion,
        )
        self.props = props
        self.time_step = time_step / 1000
        self.step = 0

    def calculate(self, system):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        # set model to evaluation mode to disable graph creation
        i = self.step
        mols = utils.npy_to_ase(props['_positions'][i].squeeze(0) * 10,
                                props['_atomic_numbers'])
        system.load_molecules(mols)
        # print('masses', system.masses)
        # print('masses shape', system.masses.shape)
        # print('system nmols', system.n_molecules)
        system.momenta = torch.tensor(props['velocities'][i]).to(system.masses) * system.masses

        forces = props['forces'][i]
        system.forces = (
            torch.tensor(forces).to(system.masses).view(system.n_replicas, system.n_molecules, system.max_n_atoms, 3)
        )
        system.momenta = system.momenta - 0.5 * system.forces * self.time_step
        # print('md system positions', system.positions)
        # print('md system velocities', system.velocities)
        # print('props velocities', props['velocities'][i])
        # print('md system cells', system.cells)
        system.properties = {}
        for prop in target_properties:
            system.properties[prop] = torch.tensor(props[prop][i]).to(system.masses)
        self.step += 1


load_file = sys.argv[1] 
data = HDF5Loader(load_file, load_properties=True)
props = data.properties 
save_file = load_file[:-5] + '_compressed.hdf5' 
if os.path.exists(save_file):
    os.remove(save_file)
n_replicas = 1
md_system = System(n_replicas, device='cpu')
time_step = 0.5  # fs

target_properties = ['energy', 'forces', 'dipole_moment']
# Setup the integrator
md_integrator = VelocityVerlet(time_step)

md_calculator = DummyCalculator(props,
                                required_properties=target_properties,
                                force_handle='forces',
                                position_conversion="nm",
                                force_conversion="kcal/mol/nm",
                                property_conversion={},
                                time_step=time_step,
                                )
pos = props['_positions']
print('properties', props.keys())
print('pos shape', pos.shape)
print('forces shape', props['forces'].shape)


# load the structure
data_streams = [
    logging_hooks.MoleculeStream(),
    logging_hooks.PropertyStream(target_properties=target_properties),
]

buffer_size = 100
# create the file logger
file_logger = logging_hooks.FileLogger(
    save_file,
    buffer_size,
    data_streams=data_streams
)
print('save file', save_file)

md_simulator = Simulator(md_system, md_integrator, md_calculator,
                         simulator_hooks=[file_logger],
                         restart=False)

md_simulator.simulate(pos.shape[0] - 1)
