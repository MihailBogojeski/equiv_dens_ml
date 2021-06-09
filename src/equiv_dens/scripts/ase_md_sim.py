# !/usr/bin/env python3
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
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_sampling
import equiv_dens.utils.base as utils

import numpy as np
from functools import partial

from ase.optimize import BFGS
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
    def __init__(self, model, data_atoms=None,
                 atoms=None, verbose=0,
                 gpu=False, n_jobs=10,
                 density_expansion=False,
                 grid_sampling_fn=None,
                 grid_spec=None):
        # energy prediction model
        self.model = model
        # density prediction model
        self.data_atoms = data_atoms
        self.atom_numbers = self.data_atoms['atom_numbers'].squeeze()
        self.atom_symbols = utils.numbers_to_symbols(self.atom_numbers)
        self.atoms = atoms
        self.grid_spec = grid_spec
        self.grid_sampling_fn = grid_sampling_fn
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

        Input is atom_pos  and atom_numbers of a single molecule geometry.

        """
        # print('atoms', atoms['atom_numbers'])
        # print('data atoms', self.data_atoms['atom_numbers'])
        assert torch.all(atoms['atom_numbers'] == self.data_atoms['atom_numbers'])

        start_total = time.time()
        if self.gpu:
            for key in atoms.keys():
                if isinstance(atoms[key], torch.Tensor):
                    atoms[key] = atoms[key].cuda()

        # pass inputs through model
        start = time.time()
        atoms = self.model(atoms)

        end = time.time()
        if self.verbose > 0:
            print('Predict time', end - start)
        end_total = time.time()
        if self.verbose > 0:
            print('Total predict time', end_total - start_total)

        # print('Predicted energy', atoms['energy'].reshape(1, -1))
        return atoms['energy'].detach().cpu().numpy() / 23.061, atoms['forces'].detach().cpu().numpy() / 23.061  # from kcal/mol to eV

    def calculation_required(self, atoms, quantities=None):
        return (self.calc is None) or (self.calc['atoms'] != atoms)

    def _make_calc(self, atoms):
        # Calculate both energy and forces via one call to the ML models
        # print('Calculating forces using ML')
        if not np.all(atoms.get_atomic_numbers() == self.atom_numbers.detach().cpu().numpy()):
            raise RuntimeError('ASE switched atom types around.')

        in_atoms = {key: self.data_atoms[key] for key in self.data_atoms.keys()}
        in_atoms['positions'] = torch.Tensor(atoms.get_positions()).unsqueeze(0)
        # print('in atoms positions shape', in_atoms['positions'].shape)
        in_atoms['positions'] = in_atoms['positions'].to(self.data_atoms['positions'])
        if self.density_expansion:
            sample_coords, _ = self.grid_sampling_fn(self.grid_spec, 10000000000,
                                                     self.atom_symbols,
                                                     in_atoms['positions'])
            in_atoms['coords'] = sample_coords
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
        # print('Calling logger to ', os.path.join(self.log_dir, 'potential_energy' + self.log_suffix))
        if self.reset_log:
            open_mode = 'w'
            self.reset_log = False
        else:
            open_mode = 'a'

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        with open(os.path.join(self.log_dir, 'potential_energy' + self.log_suffix), open_mode) as f:
            curr_energy = self.atoms.get_potential_energy()
            f.write('%.18e\n' % curr_energy)
        with open(os.path.join(self.log_dir, 'kinetic_energy' + self.log_suffix), open_mode) as f:
            f.write('%.18e\n' % self.atoms.get_kinetic_energy())
        with open(os.path.join(self.log_dir, 'temperature' + self.log_suffix), open_mode) as f:
            f.write('%.18e\n' % self.atoms.get_temperature())

        forces = self.atoms.get_forces()
        # else:
        #     forces = self.atoms._calc.fast_forces
        with open(os.path.join(self.log_dir, 'forces' + self.log_suffix + '.xyz'), open_mode) as f:
            f.write('%d\n' % len(forces))
            f.write(' generated by ML\n')
            for j in range(len(forces)):
                f.write(' %s %.18e %.18e %.18e\n' % (self.atoms.get_calculator().atom_symbols[j], forces[j][0],
                                                     forces[j][1], forces[j][2]))
        velocities = self.atoms.get_velocities()
        if velocities is not None:
            with open(os.path.join(self.log_dir, 'velocities' + self.log_suffix + '.xyz'), open_mode) as f:
                f.write('%d\n' % len(velocities))
                f.write(' generated by ML\n')
                for j in range(len(velocities)):
                    f.write(' %s %.18e %.18e %.18e\n' % (self.atoms.get_calculator().atom_symbols[j], velocities[j][0],
                                                         velocities[j][1], velocities[j][2]))

        pos = self.atoms.get_positions()
        with open(os.path.join(self.log_dir, 'positions' + self.log_suffix + '.xyz'), open_mode) as f:
            f.write('%d\n' % len(pos))
            f.write(' generated by ML\n')
            for j in range(len(pos)):
                f.write(' %s %.18e %.18e %.18e\n' % (self.atoms.get_calculator().atom_symbols[j], pos[j][0],
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
    if args.density_weight > 0:
        property_models['density'] = expansion_model
        calculate_forces_dict['density'] = False
    # if loss_weights['energy_min'] > 0:
    #     property_models['energy_min'] = functional_en_model
    #     calculate_forces_dict['energy_min'] = False
    if args.energy_model is not None:
        property_models['energy'] = en_model
        calculate_forces_dict['energy'] = calculate_forces

    model = DFTNetwork(density_model, property_models, calculate_forces_dict=calculate_forces_dict, verbose=args.verbose)
    # print('dft network', model)

    print('args restart', args.restart)
    print('best_model_path', best_model_path)
    state_dict_path = os.path.join(args.restart, best_model_path)
    print('state_dict_path', state_dict_path)
    state_dict = torch.load(state_dict_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(args.dtype)
    if args.use_gpu:
        print('using GPU')
        model.cuda()
    # if there are multiple GPUs, wrap the model in DataParallel
    # "module" is used whenever direct access is needed, e.g. for parameters,
    # whereas "model" may be DataParallel and is used for inference only

    if args.use_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model


def run_optimization(args, dataset, model):
    start_ind = 10
    # start_ind = np.randint(len(dataset))
    data_atoms = dataset.get_properties(start_ind)

    calculator = MLCalculator(model=model, data_atoms=data_atoms,
                              verbose=args.verbose, n_jobs=args.num_workers,
                              gpu=args.use_gpu and torch.cuda.is_available(),
                              density_expansion=args.density_weight > 0,
                              grid_sampling_fn=dataset.sampling_fn,
                              grid_spec=dataset.grid_spec)

    atoms = Atoms(positions=data_atoms['positions'].squeeze(0).detach().cpu().numpy(),  # from Bohr to Angstrom
                  numbers=data_atoms['atom_numbers'].squeeze().detach().cpu().numpy(),
                  calculator=calculator)

    dyn = BFGS(atoms, maxstep=0.01)

    dyn.attach(MDLogger(atoms, results_dir=args.log_dir, work_dir=args.restart, reset_log=args.new_run,
                        log_suffix=args.log_suffix, run_type='opt'))
    dyn.run(fmax=0.01)


# The actual Simulation
def run_molecular_dynamics(args, dataset, model):
    random_state = np.random.RandomState(seed=args.split_seed)
    # start_ind = 10

    start_idx = np.random.randint(len(dataset), size=(args.test_batch_size,))
    data_atoms = dataset.get_properties(start_idx)
    print('positions shape', data_atoms['positions'].shape)

    calculator = MLCalculator(model=model, data_atoms=data_atoms,
                              verbose=args.verbose, n_jobs=args.num_workers,
                              gpu=args.use_gpu and torch.cuda.is_available(),
                              density_expansion=args.density_weight > 0,
                              grid_sampling_fn=dataset.sampling_fn,
                              grid_spec=dataset.grid_spec)

    atoms = Atoms(positions=data_atoms['positions'].squeeze(0).detach().cpu().numpy(),  # from Bohr to Angstrom
                  numbers=data_atoms['atom_numbers'].squeeze().detach().cpu().numpy(),
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

    dyn.attach(MDLogger(atoms, results_dir=args.log_dir, work_dir=args.restart, reset_log=args.new_run,
                        log_suffix=args.log_suffix, run_type='md'))
    dyn.run(args.md_steps)


if __name__ == "__main__":
    # read arguments
    args, hyperparam_args = parse_command_line_arguments()

    print('args np dir', args.np_dataset)
    # no restart directory specified
    directory = args.restart  # load directory name
    # load latest checkpoint
    checkpoint_path = os.path.join(directory, 'checkpoints')  # checkpoint directory
    checkpoint = torch.load(os.path.join(
        checkpoint_path, 'latest_checkpoint.pth'), map_location='cpu')
    latest_checkpoint = checkpoint['step']
    model_code = checkpoint['ID']  # load ID
    step = checkpoint['step']
    for arg in vars(checkpoint['args']):
        if args.fix_arguments:
            if arg in hyperparam_args:
                print('loading hyperparam arg', arg)
                setattr(args, arg, getattr(checkpoint['args'], arg))
        else:
            print('loading all arg', arg)
            setattr(args, arg, getattr(checkpoint['args'], arg))
    restore = True

    best_model_path = 'best_' + model_code + '.pth'
    print('best_model_path', best_model_path)

    print('model code:', model_code)
    # determine whether GPU is used for training
    print('args use gpu', args.use_gpu)
    args.use_gpu = args.use_gpu and torch.cuda.is_available()

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

    if args.simulation_type == 'md':
        run_molecular_dynamics(args, dataset, model)
    elif args.simulation_type == 'opt':
        run_optimization(args, dataset, model)
    else:
        raise('Simulation type "' + args.simulation + '" is not supported!')
