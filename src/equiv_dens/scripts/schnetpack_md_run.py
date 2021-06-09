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

import time
from schnetpack.md import System
# from respa_md import RESPAVerlet, RESPALangevin
from schnetpack.md import MaxwellBoltzmannInit
from schnetpack.md.integrators import VelocityVerlet
from schnetpack import Properties
from schnetpack.md.calculators import MDCalculator
from schnetpack.md import Simulator
from schnetpack.md.simulation_hooks import thermostats
from schnetpack.md.simulation_hooks import logging_hooks


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


# MD calculator class
class DFTNetworkCalculator(MDCalculator):
    def __init__(self,
                 model,
                 required_properties,
                 force_handle,
                 atoms_data=None,
                 verbose=0,
                 n_jobs=10,
                 density_expansion=False,
                 position_conversion="A",
                 force_conversion="kcal/mol/A",
                 grid_spec=None,
                 grid_sampling_fn=None,
                 property_conversion={},
                 device='cpu',
                 detach=True):
        # energy prediction model
        super().__init__(
            required_properties,
            force_handle,
            detach=detach,
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            property_conversion=property_conversion,
        )
        print('position conversion', self.position_conversion)
        print('force conversion', self.force_conversion)
        self.model = model
        # density prediction model
        self.grid_sampling_fn = grid_sampling_fn
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.device = device
        self.grid_spec = {}
        for key in grid_spec.keys():
            self.grid_spec[key] = (torch.Tensor(grid_spec[key][0]).to(device),
                                   torch.Tensor(grid_spec[key][1]).to(device))  # convert Bohr grid to Angstrom
        self.density_expansion = density_expansion

    def calculate(self, system):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        # set model to evaluation mode to disable graph creation
        start = time.time()
        self.model.eval()

        inputs = self._generate_input(system)
        self.results = self.model(inputs)
        self._update_system(system)
        print('Step time:', time.time() - start)

    def _generate_input(self, system):
        """
        Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
        formatted input for the schnetpack model.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            dict(torch.Tensor): Schnetpack inputs in dictionary format.
        """
        positions, atom_types, atom_masks, cells, pbc = self._get_system_molecules(
            system
        )
        inputs = {'positions': positions,
                  'atom_numbers': atom_types
                  }
        if self.density_expansion:
            sample_coords, _ = self.grid_sampling_fn(self.grid_spec, 10000000000,
                                                     utils.numbers_to_symbols(atom_types[0].squeeze().detach().cpu().numpy()),
                                                     inputs['positions'])
            inputs['coords'] = sample_coords

        return inputs

    # def calculation_required(self, atoms, quantities=None):
    #     return (self.calc is None) or (self.calc['atoms'] != atoms)
    #
    # def _make_calc(self, atoms):
    #     # Calculate both energy and forces via one call to the ML models
    #     # print('Calculating forces using ML')
    #     if not np.all(atoms.get_atomic_numbers() == self.atom_numbers.detach().cpu().numpy()):
    #         raise RuntimeError('ASE switched atom types around.')
    #
    #     in_atoms = {key: self.data_atoms[key] for key in self.data_atoms.keys()}
    #     in_atoms['positions'] = torch.Tensor(atoms.get_positions()).unsqueeze(0)
    #     # print('in atoms positions shape', in_atoms['positions'].shape)
    #     in_atoms['positions'] = in_atoms['positions'].to(self.data_atoms['positions'])
    #     energy, forces = self.get_energy_via_ML(in_atoms)
    #
    #     if self.verbose > 0:
    #         print('forces', forces)
    #
    #     self.calc = {'forces': forces.reshape(-1, 3),
    #                  'energy': energy,
    #                  'atoms': atoms.copy()}
    #
    #     return energy, forces
    #
    # def get_potential_energy(self, atoms=None, force_consistent=False):
    #     # if force_consistent:
    #     #    raise NotImplementedError('Asking for force conistent energy?')
    #     if atoms is None:
    #         atoms = self.atoms
    #     if self.calculation_required(atoms):
    #         self._make_calc(atoms)
    #
    #     return self.calc['energy']
    #
    # def get_forces(self, atoms):
    #     if self.calculation_required(atoms):
    #         self._make_calc(atoms)
    #     return self.calc['forces']


def run_molecular_dynamics(args, dataset, model):
    np.random.seed(args.split_seed)
    # start_ind = 10

    start_idx = np.random.randint(len(dataset), size=(args.test_batch_size,))
    atoms_data = dataset.get_properties(start_idx)
    mols = utils.npy_to_ase(atoms_data['positions'].detach().cpu().numpy(), utils.numbers_to_symbols(atoms_data['atom_numbers'][0].squeeze()))
    print('positions shape', atoms_data['positions'].shape)
    # Check if a GPU is available and use a CPU otherwise
    if args.use_gpu:
        md_device = "cuda"
    else:
        md_device = "cpu"

    # Number of molecular replicas
    n_replicas = 1

    # Initialize the system
    md_system = System(n_replicas, device=md_device)

    # Load the structure
    md_system.load_molecules(mols)
    system_temperature = 300  # Kelvin

    # Set up the initializer
    md_initializer = MaxwellBoltzmannInit(
        system_temperature,
        remove_translation=True,
        remove_rotation=True)

    # Initialize momenta of the system
    md_initializer.initialize_system(md_system)

    time_step = 0.5  # fs

    # Setup the integrator
    md_integrator = VelocityVerlet(time_step)

    # Generate the calculator
    md_calculator = DFTNetworkCalculator(
        model,
        required_properties=[Properties.energy, Properties.forces],
        force_handle='forces',
        atoms_data=atoms_data,
        position_conversion="A",
        force_conversion="kcal/mol/A",
        property_conversion={},
        density_expansion=args.density_weight > 0,
        grid_spec=dataset.grid_spec,
        grid_sampling_fn=dataset.sampling_fn,
        device=md_device,
    )

    # Set temperature and thermostat constant
    bath_temperature = 300  # K
    time_constant = 100  # fs

    # Initialize the thermostat
    langevin = thermostats.LangevinThermostat(bath_temperature, time_constant)

    simulation_hooks = [langevin]
    if args.log_suffix != '':
        args.log_suffix = '_' + args.log_suffix
    log_file = os.path.join(args.md_log_dir, 'simulation' + args.log_suffix + '.hdf5')

    # Size of the buffer
    buffer_size = 100

    # Set up data streams to store positions, momenta and all properties
    data_streams = [
        logging_hooks.MoleculeStream(),
        logging_hooks.PropertyStream(target_properties=['energy', 'forces', 'density']),
    ]

    # Create the file logger
    file_logger = logging_hooks.FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams
    )

    # Update the simulation hooks
    simulation_hooks.append(file_logger)

    # Set the path to the checkpoint file
    chk_file = os.path.join(args.md_log_dir, 'simulation.chk')

    # Create the checkpoint logger
    checkpoint = logging_hooks.Checkpoint(chk_file, every_n_steps=100)

    # Update the simulation hooks
    simulation_hooks.append(checkpoint)

    md_simulator = Simulator(md_system, md_integrator, md_calculator,
                             simulator_hooks=simulation_hooks,
                             restart=True)

    md_simulator.simulate(args.md_steps)


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

    args.md_log_dir = os.path.join(args.log_dir, 'md_logs', args.restart.split('/')[-1])
    if not os.path.exists(args.md_log_dir):
        os.mkdirs(args.md_log_dir)

    if args.simulation_type == 'md':
        run_molecular_dynamics(args, dataset, model)
    # elif args.simulation_type == 'opt':
    #     run_optimization(args, dataset, model)
    else:
        raise('Simulation type "' + args.simulation + '" is not supported!')
