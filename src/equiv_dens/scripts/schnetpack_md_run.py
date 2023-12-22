# !/usr/bin/env python3
import os
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
import wandb

# from respa_md import RESPAVerlet, RESPALangevin
from schnetpack.md import System
from schnetpack.md import MaxwellBoltzmannInit
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md import Simulator
from schnetpack.md.simulation_hooks import thermostats
from schnetpack.md.simulation_hooks import callback_hooks


def wandb_summary(wandb, md_system):
    wandb.log({'energy': torch.mean(md_system.energy)})
    wandb.log({'forces': torch.mean(torch.norm(md_system.forces, dim=-1))})
    pos = md_system.positions
    n_mols = md_system.n_molecules
    pos = torch.reshape(pos, (n_mols, -1, 3))
    distances, _ = utils.calculate_distances_and_directions(pos)
    wandb.log({'distances': torch.mean(distances)})


def run_molecular_dynamics(args, dataset, model):
    wandb.login()
    args_dict = vars(args)
    if args.args_file_name is not None:
        wandb_id = args.args_file_name + '_'
    else:
        wandb_id = ''
    wandb_date = directory.split('/')[-1].split('_')[0]
    wandb_name = wandb_id + wandb_date
    wandb_id = wandb_name + '_' + model_code
    wandb_run = wandb.init(project='equiv_dens', config=args_dict,
                           name=wandb_name, id=wandb_id, resume='allow')
    np.random.seed(args.split_seed)
    # start_ind = 10
    if args.start_idx is None:
        start_idx = np.random.randint(len(dataset), size=(args.test_batch_size,))
    else:
        start_idx = args.start_idx
    print('dataset atoms positions type', dataset.atoms['positions'].dtype)
    atoms_data = dataset.get_properties(start_idx)
    # print('data energy', atoms_data['energy'])
    # print('avg_force', torch.mean(torch.norm(atoms_data['forces'], dim=-1)))
    print('data positions type', type(atoms_data['positions']))
    mols = utils.npy_to_ase(atoms_data['batch_positions'].detach().cpu().numpy(),
                            atoms_data['batch_atom_numbers'].detach().cpu().numpy())
    print('positions shape', atoms_data['positions'].shape)
    # Check if a GPU is available and use a CPU otherwise
    # Number of molecular replicas
    n_replicas = 1

    # Initialize the system
    md_system = System()

    # Load the structure
    md_system.load_molecules(mols, n_replicas)
    system_temperature = args.temperature  # Kelvin

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

    required_properties = ['energy', 'forces']
    if args.dipole_moment_weight > 0:
        required_properties.append('dipole_moment')
    if args.density_weight > 0:
        required_properties.append('spherical_coeffs')
        required_properties.append('radial_width')
        required_properties.append('radial_scale')
        # required_properties.append('density')
    # Generate the calculator
    md_calculator = DFTNetworkCalculator(
        model,
        required_properties=required_properties,
        force_key='forces',
        energy_key='energy',
        position_unit=args.position_conversion,
        energy_unit=args.energy_conversion,
        density_expansion=args.density_weight > 0,
        grid_spec=dataset.grid_spec,
        grid_sampling_fn=dataset.sampling_fn,
        use_gpu=args.use_gpu,
        cutoff=args.cutoff,
        pyscf_grid=args.pyscf_grid,
    )

    simulation_hooks = []

    if args.log_suffix != '':
        args.log_suffix = '_' + args.log_suffix
    log_file = os.path.join(args.md_log_dir, 'simulation' + args.log_suffix + '.hdf5')

    # Size of the buffer
    buffer_size = 100

    # Set up data streams to store positions, momenta and all properties
    target_properties = [p for p in required_properties]
    data_streams = [
        callback_hooks.MoleculeStream(store_velocities=True),
        callback_hooks.PropertyStream(target_properties=target_properties),
    ]

    # Create the file logger
    file_logger = callback_hooks.FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams
    )

    # Update the simulation hooks
    simulation_hooks.append(file_logger)

    # Set the path to the checkpoint file
    chk_file = os.path.join(args.md_log_dir, 'simulation' + args.log_suffix + '.chk')

    # Create the checkpoint logger
    checkpoint = callback_hooks.Checkpoint(chk_file, every_n_steps=1000)

    # Update the simulation hooks
    simulation_hooks.append(checkpoint)

    if args.langevin or args.warm_up:
        # Set temperature and thermostat constant
        bath_temperature = args.temperature  # K
        time_constant = 100  # fs

        # Initialize the thermostat
        langevin = thermostats.LangevinThermostat(bath_temperature, time_constant)
        print('initialized thermostat')

        if args.langevin:
            simulation_hooks.append(langevin)
        elif args.new_run:
            warmup_hooks = simulation_hooks + [langevin]
            warmup_simulator = Simulator(md_system, md_integrator, md_calculator,
                                         simulator_hooks=warmup_hooks)
            if args.use_gpu:
                warmup_simulator = warmup_simulator.to('cuda')
            warmup_simulator = warmup_simulator.to(args.dtype)
            steps = 0
            while steps < args.md_steps//20:
                steps += 100
                warmup_simulator.simulate(100)
                wandb_summary(wandb_run, md_system)
            print('finishing warm up')

    md_simulator = Simulator(md_system, md_integrator, md_calculator,
                             simulator_hooks=simulation_hooks)

    if os.path.exists(chk_file):
        print('restarting past model')
        state_dict = torch.load(chk_file)
        md_simulator.restart_simulation(state_dict)
    if args.use_gpu:
        md_simulator = md_simulator.to('cuda')
    md_simulator = md_simulator.to(args.dtype)

    steps = 0
    while steps < args.md_steps:
        steps += 100
        md_simulator.simulate(100)
        wandb_summary(wandb_run, md_system)


if __name__ == "__main__":
    # read arguments
    args, hyperparam_args = parse_command_line_arguments()

    print('type dtype', type(args.dtype))
    print('args np md_initializer', args.np_dataset)
    print('dtype', args.dtype)
    # no restart directory specified
    directory = args.restart  # load directory name
    # load latest checkpoint
    checkpoint_path = os.path.join(directory, 'checkpoints')  # checkpoint directory
    checkpoint = torch.load(os.path.join(
        checkpoint_path, 'latest_checkpoint.pth'), map_location='cpu')
    latest_checkpoint = checkpoint['step']
    model_code = checkpoint['ID']  # load ID
    for arg in vars(checkpoint['args']):
        if args.fix_arguments:
            if arg in hyperparam_args:
                print('loading hyperparam arg', arg)
                setattr(args, arg, getattr(checkpoint['args'], arg))
    step = checkpoint['step']
    restore = True
    print('dtype', args.dtype)

    args.best_model_path = 'best_' + model_code + '.pth'
    print('best_model_path', args.best_model_path)

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
        sampling_fn = partial(spherical_radial_sampling, rotate=False)
        grid_origin = 0
        grid_extent = None

    dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                               orbitals_path=args.orbitals_file,
                               density_n_samp=10000000000,
                               required_properties=['energy', 'forces'],
                               center_positions=False,
                               radial_coeffs_file=args.radial_coeffs_file,
                               dtype=args.dtype,
                               grid_fn=grid_fn,
                               pyscf_grid=args.pyscf_grid,
                               sampling_fn=sampling_fn,
                               grid_extent=grid_extent,
                               grid_origin=grid_origin,
                               cutoff=args.cutoff,
                               verbose=args.verbose,
                               df_loss_weights=args.df_loss_weights)

    test_dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                                    orbitals_path=args.orbitals_file,
                                    density_n_samp=10000000000,
                                    required_properties=['energy', 'forces'],
                                    center_positions=False,
                                    radial_coeffs_file=args.radial_coeffs_file,
                                    dtype=args.dtype,
                                    grid_fn=grid_fn,
                                    pyscf_grid=args.pyscf_grid,
                                    sampling_fn=sampling_fn,
                                    grid_extent=grid_extent,
                                    grid_origin=grid_origin,
                                    cutoff=args.cutoff,
                                    verbose=args.verbose,
                                    df_loss_weights=args.df_loss_weights)
    # if args.center_energy:
    #     if args.atomic_energies is None:
    #         energy_mean = dataset.atoms['energy'].mean()
    #         dataset.center_energy(energy_mean)
    #     else:
    #         atomic_energies = np.load(args.atomic_energies, allow_pickle=True).item()
    #         dataset.normalize_energy(atomic_energies)

    print('dataset grid_spec type', dataset.grid_spec['H'][0].type())
    model = load_model(args, dataset)

    print('model is loaded')
    args.md_log_dir = os.path.join(args.log_dir, 'md_logs', args.restart.split('/')[-1])
    if not os.path.exists(args.md_log_dir):
        os.makedirs(args.md_log_dir)

    if args.simulation_type == 'md':
        run_molecular_dynamics(args, test_dataset, model)
    # elif args.simulation_type == 'opt':
    #     run_optimization(args, dataset, model)
    else:
        raise('Simulation type "' + args.simulation + '" is not supported!')
