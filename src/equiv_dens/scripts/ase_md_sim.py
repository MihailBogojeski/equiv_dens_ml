#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from equiv_dens.nn.dft_network import DFTNetwork
from equiv_dens.nn.representation.spherical_harmonic import EquivariantSphericalHarmonics
from equiv_dens.nn.property_output.energy import ComplexEnergyNetwork, SimpleEnergyNetwork,\
    SphericalHarmonicsEnergyNetwork, SimpleEnergyNetworkv2
from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DensityExpansion
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.errors import ErrorDict
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    dftpy_grid, CubicalGrid, spherical_grid, rot_spherical_sampling
from equiv_dens.density_functionals.LDA import LDAFunctional
import equiv_dens.utils.base as utils
import copy

import numpy as np
from functools import partial

from dftpy.pseudo import LocalPseudo
import numpy as np
import sys
import time
from ase import Atoms
import ase.io
from ase import units
from ase.calculators.interface import Calculator
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
import subprocess
# from respa_md import RESPAVerlet, RESPALangevin

run_results = []


def predict_energy_worker(X):
    global density_kernel, energy_kernel
    return energy_kernel.predict(density_kernel.predict(X))


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split(b'\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


# MD calculator class
class MLCalculator(Calculator):
    def __init__(self, model, data_atoms=data_atoms,
                 atoms=None, verbose=0,
                 gpu=False, n_jobs=10,
                 density_expansion=False):
        # energy prediction model
        self.model = model
        # density prediction model
        self.data_atoms = data_atoms
        self.atom_types = self.data_atoms['atom_numbers']
        self.atoms = atoms
        self.calc = None
        self.mylog = []
        self.verbose = verbose
        self.gpu = gpu
        self.n_jobs = n_jobs
        self.device = None
        self.potential_loaded = False
        self.energy_loaded = False
        self.density_expansion = density_expansion

    def get_energy_via_ML(self, atoms):
        """ Computes energy

        Input is atom_pos  and atom_types of a single molecule geometry.

        """
        assert np.all(atoms['atom_numbers'] == self.atom_types)

        start_total = time.time()

        # pass inputs through model
        start = time.time()
        atoms = self.model(atoms)

        end = time.time()
        if self.verbose > 0:
            print('Predict time', end - start)
        end_total = time.time()
        if self.verbose > 0:
            print('Total predict time', end_total - start_total)

        print('Predicted energy', energy.reshape(1, -1)[0, 0])
        return atoms['energy'] / 23.061, atoms['forces'] / 23.061  # from kcal/mol to eV

    def calculation_required(self, atoms, quantities=None):
        return (self.calc is None) or (self.calc['atoms'] != atoms)

    def _make_calc(self, atoms):
        # Calculate both energy and forces via one call to the ML models
        print('Calculating forces using ML')
        if not np.all(atoms.get_atomic_numbers() == self.atom_types):
            raise RuntimeError('ASE switched atom types around.')

        in_atoms = {key: self.data_atoms[key] for key in self.data_atoms.keys()}
        in_atoms['positions'] = atoms.get_positions()
        energy, forces = self.get_energy_via_ML(in_atoms)

        if self.verbose > 0:
            print('forces', forces)

        self.calc = {'forces': forces.reshape(-1, 3),
                     'energy': energy,
                     'atoms': atoms.copy()}

        return energy, forces

    def get_potential_energy(self, atoms=None, force_consistent=False):
        # if force_consistent:
        #    raise NotImplementedError('Asking for force conistent energy?')
        if atoms is None:
            atoms = self.atoms
        if self.calculation_required(atoms):
            self._make_calc(atoms)

        return self.calc['energy']

    def get_forces(self, atoms):
        if self.calculation_required(atoms):
            self._make_calc(atoms)
        return self.calc['forces']

    # def get_slow_forces(self, atoms):
    #     print('Calculate slow forces')
    #     self.respa = True
    #     self._make_calc(atoms)
    #     forces = self.calc['forces']
    #     self.respa = False
    #     self.calc = None
    #     return forces


class MDLogger:
    def __init__(self, atoms, results_dir=None, work_dir=None, reset_log=True,
                 log_suffix='', run_type=''):
        self.atoms = atoms
        if results_dir is not None and work_dir is not None:
            self.log_dir = os.path.join(results_dir, 'md_logs', work_dir)
        else:
            self.log_dir = 'md_logs'

        self.reset_log = reset_log

        self.log_suffix = ''
        if run_type != '':
            self.log_suffix += '_' + run_type
        if log_suffix != '':
            self.log_suffix += '_' + log_suffix

        if not reset_log:
            last_position = list(ase.io.iread(os.path.join(self.log_dir, 'positions' + self.log_suffix + '.xyz')))[-1]
            last_velocities = list(ase.io.iread(os.path.join(self.log_dir, 'velocities' + self.log_suffix + '.xyz')))[-1]
            atoms.set_positions(last_position.get_positions())
            atoms.set_velocities(last_velocities.get_positions())

    def __call__(self):  # store a reference to atoms in the definition.
        print('Calling logger to ', os.path.join(self.log_dir, 'potential_energy' + self.log_suffix))
        if self.reset_log:
            open_mode = 'w'
            self.reset_log = False
        else:
            open_mode = 'a'

        with open(os.path.join(self.log_dir, 'potential_energy' + self.log_suffix), open_mode) as f:
            if self.atoms._calc.fast_energy is None:
                curr_energy = self.atoms.get_potential_energy()
            else:
                curr_energy = self.atoms._calc.fast_energy[0]
            f.write('%.18e\n' % curr_energy)
        with open(os.path.join(self.log_dir, 'kinetic_energy' + self.log_suffix), open_mode) as f:
            f.write('%.18e\n' % self.atoms.get_kinetic_energy())
        with open(os.path.join(self.log_dir, 'temperature' + self.log_suffix), open_mode) as f:
            f.write('%.18e\n' % self.atoms.get_temperature())

        # if self.atoms._calc.fast_energy is None:
        forces = self.atoms.get_forces()
        # else:
        #     forces = self.atoms._calc.fast_forces
        with open(os.path.join(self.log_dir, 'forces' + self.log_suffix + '.xyz'), open_mode) as f:
            f.write('%d\n' % len(forces))
            f.write(' generated by ML\n')
            for j in range(len(forces)):
                f.write(' %s %.18e %.18e %.18e\n' % (self.atoms.get_calculator().data_atoms['atom_types'][j], forces[j][0],
                                                     forces[j][1], forces[j][2]))
        velocities = self.atoms.get_velocities()
        if velocities is not None:
            with open(os.path.join(self.log_dir, 'velocities' + self.log_suffix + '.xyz'), open_mode) as f:
                f.write('%d\n' % len(velocities))
                f.write(' generated by ML\n')
                for j in range(len(velocities)):
                    f.write(' %s %.18e %.18e %.18e\n' % (self.atoms.get_calculator().data_atoms['atom_types'][j], velocities[j][0],
                                                         velocities[j][1], velocities[j][2]))

        pos = self.atoms.get_positions()
        with open(os.path.join(self.log_dir, 'positions' + self.log_suffix + '.xyz'), open_mode) as f:
            f.write('%d\n' % len(pos))
            f.write(' generated by ML\n')
            for j in range(len(pos)):
                f.write(' %s %.18e %.18e %.18e\n' % (self.atoms.get_calculator().data_atoms['atom_types'][j], pos[j][0],
                                                     pos[j][1], pos[j][2]))

    def get_prev_position(self):
        pos_list = list(ase.io.iread(os.path.join(self.log_dir, 'positions' + self.log_suffix + '.xyz')))
        if len(pos_list) > self.rollback_steps:
            last_position = pos_list[-self.rollback_steps]
        else:
            last_position = pos_list[0]
        return last_position

    def get_prev_velocities(self):
        vel_list = list(ase.io.iread(os.path.join(self.log_dir, 'velocities' + self.log_suffix + '.xyz')))
        if len(vel_list > self.rollback_steps):
            last_velocities = vel_list[-self.rollback_steps]
        else:
            last_velocities = vel_list[0]
        return last_velocities


# class MDLogger:
#     def __init__(self, atoms, results_dir=None, work_dir=None, reset_log=True,
#                  log_suffix='', run_type='', energy_type='', temp_threshold=1000,
#                  rollback=True, rollback_steps=50, rollback_criterion='temperature',
#                  max_pos=None, min_pos=None):
#         self.atoms = atoms
#         if results_dir is not None and work_dir is not None:
#             self.log_dir = os.path.join(results_dir, work_dir, 'md_logs')
#         else:
#             self.log_dir = 'md_logs'
#
#         self.reset_log = reset_log
#
#         self.log_suffix = ''
#         if run_type != '':
#             self.log_suffix += '_' + run_type
#         if energy_type != '':
#             self.log_suffix += '_' + energy_type
#         if log_suffix != '':
#             self.log_suffix += '_' + log_suffix
#
#         self.temp_threshold = temp_threshold
#         self.rollback = rollback
#         self.rollback_steps = rollback_steps
#         self.rollback_criterion = rollback_criterion
#         self.max_pos = max_pos
#         self.min_pos = min_pos
#
#         if not reset_log:
#             last_position = list(ase.io.iread(os.path.join(self.log_dir, 'positions' + self.log_suffix + '.xyz')))[-1]
#             last_velocities = list(ase.io.iread(os.path.join(self.log_dir, 'velocities' + self.log_suffix + '.xyz')))[-1]
#             atoms.set_positions(last_position.get_positions())
#             atoms.set_velocities(last_velocities.get_positions())
#
#     def __call__(self):  # store a reference to atoms in the definition.
#         print('Calling logger to ', os.path.join(self.log_dir, 'potential_energy' + self.log_suffix))
#         charge_to_str = {8: 'O', 1: 'H', 6: 'C'}
#
#         if self.reset_log:
#             open_mode = 'w'
#             self.reset_log = False
#         else:
#             open_mode = 'a'
#
#         with open(os.path.join(self.log_dir, 'potential_energy' + self.log_suffix), open_mode) as f:
#             if self.atoms._calc.fast_energy is None:
#                 curr_energy = self.atoms.get_potential_energy()
#             else:
#                 curr_energy = self.atoms._calc.fast_energy[0]
#             f.write('%.18e\n' % curr_energy)
#         with open(os.path.join(self.log_dir, 'kinetic_energy' + self.log_suffix), open_mode) as f:
#             f.write('%.18e\n' % self.atoms.get_kinetic_energy())
#         with open(os.path.join(self.log_dir, 'temperature' + self.log_suffix), open_mode) as f:
#             f.write('%.18e\n' % self.atoms.get_temperature())
#
#         if self.atoms._calc.fast_energy is None:
#             forces = self.atoms.get_forces()
#         else:
#             forces = self.atoms._calc.fast_forces
#             print('Multistep forces shape', forces.shape)
#         print('Regular forces shape', forces.shape)
#         with open(os.path.join(self.log_dir, 'forces' + self.log_suffix + '.xyz'), open_mode) as f:
#             f.write('%d\n' % len(forces))
#             f.write(' generated by ML\n')
#             for j in range(len(forces)):
#                 f.write(' %s %.18e %.18e %.18e\n' % (charge_to_str[self.atoms.get_calculator().atom_types[j]], forces[j][0],
#                                                      forces[j][1], forces[j][2]))
#         velocities = self.atoms.get_velocities()
#         if velocities is not None:
#             with open(os.path.join(self.log_dir, 'velocities' + self.log_suffix + '.xyz'), open_mode) as f:
#                 f.write('%d\n' % len(velocities))
#                 f.write(' generated by ML\n')
#                 for j in range(len(velocities)):
#                     f.write(' %s %.18e %.18e %.18e\n' % (charge_to_str[self.atoms.get_calculator().atom_types[j]], velocities[j][0],
#                                                          velocities[j][1], velocities[j][2]))
#
#         pos = self.atoms.get_positions()
#         with open(os.path.join(self.log_dir, 'positions' + self.log_suffix + '.xyz'), open_mode) as f:
#             f.write('%d\n' % len(pos))
#             f.write(' generated by ML\n')
#             for j in range(len(pos)):
#                 f.write(' %s %.18e %.18e %.18e\n' % (charge_to_str[self.atoms.get_calculator().atom_types[j]], pos[j][0],
#                                                      pos[j][1], pos[j][2]))
#         heavy = self.atoms.get_calculator().atom_types > 1
#         if len(self.atoms.get_calculator().atom_types) == 3:
#             heavy = [True, True, True]
#         if len(self.atoms.get_calculator().atom_types) > 10:
#             heavy = self.atoms.get_calculator().atom_types == 6
#
#         pos_aligned = transform_molecule(pos * dft_utils.to_bohr, self.atoms.get_calculator().base_pos,
#                                          heavy) / dft_utils.to_bohr
#         with open(os.path.join(self.log_dir, 'positions_aligned' + self.log_suffix + '.xyz'), open_mode) as f:
#             f.write('%d\n' % len(pos_aligned))
#             f.write(' generated by ML\n')
#             for j in range(len(pos_aligned)):
#                 f.write(' %s %.18e %.18e %.18e\n' % (charge_to_str[self.atoms.get_calculator().atom_types[j]], pos_aligned[j][0],
#                                                      pos_aligned[j][1], pos_aligned[j][2]))
#         print(self.atoms.get_potential_energy(), self.atoms.get_temperature())
#
#         if self.atoms.get_potential_energy == 1:
#             rollback_pos = self.get_prev_position()
#             self.atoms.set_positions(rollback_pos.get_positions())
#             randint = np.random.randint(10000)
#             random_state = np.random.RandomState(randint)
#             myMaxwellBoltzmannDistribution(self.atoms, random_state=random_state)
#             self.log_suffix = self.log_suffix + '_' + str(randint)
#
#         if self.rollback_criterion == 'temperature' and self.atoms.get_temperature() > self.temp_threshold:
#             if not self.rollback:
#                 print('Temperature exceeded threshold, not rolling back!')
#                 sys.exit()
#             print("Temperature threshold passed, rolling back {} steps!".format(self.rollback_steps))
#
#             rollback_pos = self.get_prev_position()
#
#             self.atoms.set_positions(rollback_pos.get_positions())
#             randint = np.random.randint(10000)
#             random_state = np.random.RandomState(randint)
#             myMaxwellBoltzmannDistribution(self.atoms, random_state=random_state)
#             self.log_suffix = self.log_suffix + '_' + str(randint)
#
#         if self.rollback_criterion == 'variance' and\
#                 (np.any(self.max_pos - self.atoms.get_positions() < 0) or
#                  np.any(self.min_pos - self.atoms.get_positions() > 0)):
#             if not self.rollback:
#                 print('Atom positions outside of deviation bounds, not rolling back!')
#                 sys.exit()
#             print("Atom position outside of deviation bounds, rolling back {} steps!".format(self.rollback_steps))
#             print(self.max_pos - self.atoms.get_positions())
#             print(self.min_pos - self.atoms.get_positions())
#             print(self.min_pos)
#             print(self.max_pos)
#
#             rollback_pos = self.get_prev_position()
#
#             self.atoms.set_positions(rollback_pos.get_positions())
#             randint = np.random.randint(10000)
#             random_state = np.random.RandomState(randint)
#             myMaxwellBoltzmannDistribution(self.atoms, random_state=random_state)
#             self.log_suffix = self.log_suffix + '_' + str(randint)
#
#     def get_prev_position(self):
#         pos_list = list(ase.io.iread(os.path.join(self.log_dir, 'positions' + self.log_suffix + '.xyz')))
#         if len(pos_list) > self.rollback_steps:
#             last_position = pos_list[-self.rollback_steps]
#         else:
#             last_position = pos_list[0]
#         return last_position
#
#     def get_prev_velocities(self):
#         vel_list = list(ase.io.iread(os.path.join(self.log_dir, 'velocities' + self.log_suffix + '.xyz')))
#         if len(vel_list > self.rollback_steps):
#             last_velocities = vel_list[-self.rollback_steps]
#         else:
#             last_velocities = vel_list[0]
#         return last_velocities



@ex.config
def config():
    md_steps = 200
    results_dir = '/home/ml-dft/md/'
    train_dir = 'water'
    test_dir = 'water'
    base_dir = 'water'
    train_inds = list(range(0, 50))
    test_inds = list(range(50, 102))
    run_id = 1
    output_file = None
    n_jobs = 10
    energy_type = 'ccsdt'
    alt_energy_type = 'diff'
    grid_spacing = 0.15
    gaussian_width = 0.38
    grid_file = None
    density_kernel = 'rbf'
    energy_kernel = 'rbf'
    density_kernel_params = {}
    energy_kernel_params = {}
    # rbf kernel
    density_alpha_params = list(np.logspace(-20, 0, 10))
    density_gamma_params = list(1 / np.sqrt(2 * np.logspace(-10, 0, 10)))
    energy_alpha_params = list(np.logspace(-20, 0, 30))
    energy_gamma_params = list(1 / np.sqrt(2 * np.logspace(-15, 0, 30)))
    use_true_densities = False
    verbose = 0
    new_run = True
    log_suffix = ''
    temperature = 300
    gpu = False
    temp_threshold = 10000
    std_threshold = 5
    rollback_steps = 50
    rollback_criterion = 'temperature'
    analytic_forces = False
    rollback = True
    seed = 1234
    start_ind = 0
    pbe_calculation = False
    variance_check = False
    variance_threshold = 1
    langevin = True
    respa_timestep = 0

    if verbose > 0:
        print('Parameters:')
        print('run_id', run_id)
        print('new_run', new_run)
        print('md_steps', md_steps)
        print('results_dir:', results_dir)
        print('train_dir:', train_dir)
        print('test_dir:', test_dir)
        print('base_dir:', base_dir)
        print('output_file', output_file)
        print('energy_type', energy_type)
        print('grid_spacing:', grid_spacing)
        print('gaussian_width:', gaussian_width)
        print('grid_file:', grid_file)
        print('density_kernel:', density_kernel)
        print('energy_kernel:', energy_kernel)
        print('density_kernel_params:', density_kernel_params)
        print('energy_kernel_params:', energy_kernel_params)
        print('density_alpha_params:', density_alpha_params)
        print('density_gamma_params:', density_gamma_params)
        print('energy_alpha_params:', energy_alpha_params)
        print('energy_gamma_params:', energy_gamma_params)
        print('use_true_densities:', use_true_densities)
        print('verbose:', verbose)
        print('log_suffix:', log_suffix)
        print('temperature:', temperature)
        print('gpu:', gpu)
        print('n_jobs:', n_jobs)
        print('temp_threshold', temp_threshold)
        print('std_threshold', std_threshold)
        print('rollback', rollback)
        print('rollback_steps', rollback_steps)
        print('rollback_criterion', rollback_criterion)
        print('analytic_forces', analytic_forces)
        print('seed', seed)
        print('start_ind', start_ind)
        print('pbe_calculation', pbe_calculation)
        print('variance_check', variance_check)
        print('variance_threshold', variance_threshold)
        print('langevin', langevin)
        print('respa_timestep', respa_timestep)
        print('alt_energy_type', alt_energy_type)
    if verbose > 1:
        print('train_inds:', train_inds)
        print('test_inds:', test_inds)


def myMaxwellBoltzmannDistribution(atoms, temp=300 * units.kB, random_state=np.random):
    xi = random_state.standard_normal((len(atoms.get_masses()), 3))
    momenta = xi * np.sqrt(atoms.get_masses() * temp)[:, np.newaxis]
    atoms.set_momenta(atoms.get_momenta() + momenta)


def load_model(args, dataset):
    z_vals = dataset.atoms['atom_numbers']
    # define model
    clebsch_gordan = ClebschGordanMatrix()
    repr_model = EquivariantSphericalHarmonics(
        orbitals=dataset.orbitals,
        order=args.order,
        num_features=args.num_features,
        num_basis_functions=args.num_basis_functions,
        num_modules=args.num_modules,
        num_residual_pre_x=args.num_residual_pre_x,
        num_residual_post_x=args.num_residual_post_x,
        num_residual_pre_vi=args.num_residual_pre_vi,
        num_residual_pre_vj=args.num_residual_pre_vj,
        num_residual_post_v=args.num_residual_post_v,
        num_residual_output=args.num_residual_output,
        num_radial_components=args.num_radial_components,
        basis_functions=args.basis_functions,
        cutoff=args.cutoff,
        activation=args.activation,
        clebsch_gordan=clebsch_gordan,
        verbose=args.verbose,
        timing=args.timing,
    )
    dens_model = DensityCoeffsNetwork(
        orbitals=dataset.orbitals,
        order=args.order[-1],
        num_features=args.num_features,
        positive_coeffs=args.positive_coeffs,
        clebsch_gordan=clebsch_gordan,
        verbose=args.verbose,
        timing=args.timing,
    )

    expansion_model = DensityExpansion(dataset.orbitals, radial_coeffs=dataset.radial_coeffs,
                                       expansion_constraint=args.expansion_constraint,
                                       integral_constraint=args.integral_constraint,
                                       integral_scale=args.integral_scale,
                                       softmax_norm=args.softmax_norm, n_electrons=sum(z_vals),
                                       verbose=args.verbose,
                                       timing=args.timing,
                                       )

    calculate_forces = True

    if args.num_energy_features is None:
        args.num_energy_features = args.num_features

    if args.energy_model == 'spherical':
        print('building spherical harmonic energy model')
        en_model = SphericalHarmonicsEnergyNetwork(
            orbitals=dataset.orbitals,
            order=args.order_en,
            mixing_order=args.mixing_order_en,
            num_features=args.num_energy_features,
            num_basis_functions=args.num_basis_functions,
            num_modules=args.num_modules,
            num_residual_pre_x=args.num_residual_pre_x,
            num_residual_post_x=args.num_residual_post_x,
            num_residual_pre_vi=args.num_residual_pre_vi,
            num_residual_pre_vj=args.num_residual_pre_vj,
            num_residual_post_v=args.num_residual_post_v,
            num_residual_output=args.num_residual_output,
            num_radial_components=args.num_radial_components,
            basis_functions=args.basis_functions,
            cutoff=args.cutoff,
            activation=args.activation,
            clebsch_gordan=clebsch_gordan,
            calculate_forces=calculate_forces,
            verbose=args.verbose,
            timing=args.timing,
        )
    elif args.energy_model == 'complex':
        print('building complex energy model')
        en_model = ComplexEnergyNetwork(
            orbitals=dataset.orbitals,
            num_features=args.num_energy_features,
            num_basis_functions=args.num_basis_functions,
            num_modules=args.num_modules,
            num_residual_pre_x=args.num_residual_pre_x,
            num_residual_post_x=args.num_residual_post_x,
            num_residual_pre_vi=args.num_residual_pre_vi,
            num_residual_pre_vj=args.num_residual_pre_vj,
            num_residual_post_v=args.num_residual_post_v,
            num_residual_output=args.num_residual_output,
            num_radial_components=args.num_radial_components,
            basis_functions=args.basis_functions,
            cutoff=args.cutoff,
            activation=args.activation,
            calculate_forces=calculate_forces,
            verbose=args.verbose,
            timing=args.timing,
        )
    elif args.energy_model == 'simple':
        print('building simple energy model')
        en_model = SimpleEnergyNetwork(
            orbitals=dataset.orbitals,
            num_features=args.num_energy_features,
            num_layers=args.num_energy_output,
            activation=args.activation,
            calculate_forces=calculate_forces,
            verbose=args.verbose,
            timing=args.timing,
        )
    elif args.energy_model == 'simple2':
        print('building simple energy model')
        en_model = SimpleEnergyNetworkv2(
            order=args.order[-1],
            orbitals=dataset.orbitals,
            num_features=args.num_energy_features,
            activation=args.activation,
            calculate_forces=calculate_forces,
            verbose=args.verbose,
            clebsch_gordan=clebsch_gordan,
            timing=args.timing,
        )
    else:
        args.energy_model = None

    # if loss_weights['energy_min'] > 0:
    #     functional = LDAFunctional(z_vals, verbose=args.verbose,
    #                                energy_offset=args.energy_offset,
    #                                store_energy=(args.energy_model is None))
    #     functional_en_model = nn.Sequential(expansion_model, functional)
    density_model = nn.Sequential(repr_model, dens_model)

    property_models = {}
    calculate_forces_dict = {}
    if args.density_loss > 0:
        property_models['density'] = expansion_model
        calculate_forces_dict['density'] = False
    # if loss_weights['energy_min'] > 0:
    #     property_models['energy_min'] = functional_en_model
    #     calculate_forces_dict['energy_min'] = False
    if args.energy_model is not None:
        property_models['energy'] = en_model
        calculate_forces_dict['energy'] = calculate_forces

    print('property models', property_models)
    model = DFTNetwork(density_model, property_models, calculate_forces_dict=calculate_forces_dict, verbose=args.verbose)
    # print('dft network', model)

    print('args restart', args.restart)
    print('best_model_path', best_model_path)
    state_dict_path = os.path.join(args.restart, best_model_path)
    print('state_dict_path', state_dict_path)
    state_dict = torch.load(state_dict_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(args.dtype)
    if use_gpu:
        model.cuda()
    # if there are multiple GPUs, wrap the model in DataParallel
    # "module" is used whenever direct access is needed, e.g. for parameters,
    # whereas "model" may be DataParallel and is used for inference only

    if use_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model
    

def run_optimization(args):
    random_state = np.random.RandomState(seed=args.split_seed)
    start_ind = 10
    # start_ind = np.randint(len(dataset))
    data_atoms = dataset.get_properties(start_ind)

    calculator = MLCalculator(model=model, data_atoms=data_atoms
                              verbose=verbose, n_jobs=n_jobs,
                              gpu=args.use_gpu and torch.cuda.is_available()
                              density_expansion=args.density_weight > 0)

    atoms = Atoms(positions=data_atoms['positions'],  # from Bohr to Angstrom
                  numbers=data_atoms['atom_numbers'],
                  calculator=calculator)

    dyn = BFGS(atoms, maxstep=0.01)

    dyn.attach(MDLogger(atoms, results_dir=args.log_dir, work_dir=,args.restart, reset_log=args.new_run,
                        log_suffix=log_suffix, run_type='opt', interval=1)
    dyn.run(fmax=0.01)


# The actual Simulation
def run_molecular_dynamics(args, dataset, model):
    random_state = np.random.RandomState(seed=args.split_seed)
    start_ind = 10
    # start_ind = np.randint(len(dataset))
    data_atoms = dataset.get_properties(start_ind)

    calculator = MLCalculator(model=model, data_atoms=data_atoms
                              verbose=verbose, n_jobs=n_jobs,
                              gpu=args.use_gpu and torch.cuda.is_available()
                              density_expansion=args.density_weight > 0)

    atoms = Atoms(positions=data_atoms['positions'],  # from Bohr to Angstrom
                  numbers=data_atoms['atom_numbers'],
                  calculator=calculator)

    ts = 0.5
    # if respa_timestep > 0:
    #     myMaxwellBoltzmannDistribution(atoms, temp=temperature * units.kB, random_state=random_state)
    #     if langevin:
    #         dyn = RESPALangevin(atoms, ts * units.fs, respa_timestep, temperature * units.kB, 0.1, rng=random_state)
    #     else:
    #         dyn = RESPAVerlet(atoms, ts * units.fs, respa_timestep)
    # else:
    myMaxwellBoltzmannDistribution(atoms, temp=args.temperature * units.kB, random_state=random_state)
    if args.langevin:
        dyn = Langevin(atoms, ts * units.fs, args.temperature * units.kB, 0.1, rng=random_state)
    else:
        dyn = VelocityVerlet(atoms, ts * units.fs)

    # results = {'energy': [], 'forces': [], 'positions': [], 'time': [], 'kinetic': [], 'temperature': []}
    # pos_mean = np.mean(train_pos, axis=0) * dft_utils.to_angstrom
    # pos_std = np.std(train_pos, axis=0) * dft_utils.to_angstrom

    dyn.attach(MDLogger(atoms, results_dir=args.log_dir, work_dir=,args.restart, reset_log=args.new_run,
                        log_suffix=log_suffix, run_type='md', interval=1)
    dyn.run(args.md_steps)

if __name__ == "__main__":
    # read arguments
    args = parse_command_line_arguments()

    old_args = copy.copy(args)
    # no restart directory specified
    directory = args.restart  # load directory name
    # load latest checkpoint
    checkpoint_path = os.path.join(directory, 'checkpoints')  # checkpoint directory
    checkpoint = torch.load(os.path.join(
        checkpoint_path, 'latest_checkpoint.pth'), map_location='cpu')
    latest_checkpoint = checkpoint['step']
    model_code = checkpoint['ID']  # load ID
    for arg in vars(checkpoint['args']):
        setattr(args, arg, getattr(checkpoint['args'], arg))
    step = checkpoint['step']
    args.restart = old_args.restart
    args.np_dataset = old_args.np_dataset
    args.dens_dataset = old_args.dens_dataset
    args.orbitals_file = old_args.orbitals_file
    args.radial_coeffs_file = old_args.radial_coeffs_file
    args.np_dataset_test = old_args.np_dataset_test
    args.dens_dataset_test = old_args.dens_dataset_test
    args.pseudo_pot_path = old_args.pseudo_pot_path
    args.num_test = old_args.num_test
    args.test_batch_size = old_args.test_batch_size
    args.spherical_grid_level = old_args.spherical_grid_level
    args.cube_size = old_args.cube_size
    args.verbose = old_args.verbose
    restore = True

    best_model_path = 'best_' + model_code + '.pth'
    print('best_model_path', best_model_path)

    print('model code:', model_code)
    # determine whether GPU is used for training
    print('args use gpu', args.use_gpu)
    use_gpu = args.use_gpu and torch.cuda.is_available()

    # load dataset(s)
    print("loading density from" + str(args.dens_dataset) + "...")
    print("loading atoms from" + args.np_dataset + "...")

    if args.cube_grid:
        grid_origin = args.cube_origin
        grid_extent = np.array([args.cube_extent] * 3)
        grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                          extent=grid_extent,
                          origin=np.array([grid_origin] * 3))
        sampling_fn = cubical_sampling
    else:
        grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
        sampling_fn = spherical_sampling
        grid_origin = 0
        grid_extent = None

    dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                               orbitals_path=args.orbitals_file,
                               density_n_samp=10000000000,
                               required_properties=['coords'],
                               center_positions=False,
                               radial_coeffs_file=args.radial_coeffs_file,
                               dtype=args.dtype,
                               grid_fn=grid_fn,
                               sampling_fn=sampling_fn,
                               grid_extent=grid_extent,
                               grid_origin=grid_origin,
                               verbose=args.verbose)

    model = load_model(args, dataset)

    if args.simulation = 'md':
        run_molecular_dynamics(args, dataset, model)
    elif args.simulation = 'opt':
        run_optimization(args, dataset, model)
    else:
        raise('Simulation type "' + args.simulation + '" is not supported!')
