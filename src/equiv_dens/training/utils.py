"""
This module contains various helper functions that are used prepare a model for training.

Author: Mihail Bogojeski
"""
import torch
from equiv_dens.utils.misc import generate_id
from datetime import datetime
from functools import partial
import os
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.training.lookahead import Lookahead
from equiv_dens.training.errors import ErrorDict
import numpy as np


def init_training_vars(args, hyperparam_args):
    """
    Initialize training variables based on command line args.

    Args:
        args: Namespace object containing command line arguments.
        hyperparam_args: Namespace object containing hyperparameter arguments.

    Returns:
        args: Updated args Namespace.
        hyperparam_args: Updated hyperparam_args Namespace.
        train_vars: Dictionary containing training variables.
    """
    if args.restart is None:
        # generate "unique" id for the run (very unlikely that two runs will have the same ID)
        model_code = generate_id()
        directory = os.path.join(args.save_dir, datetime.utcnow().strftime("%Y-%m-%d_") +
                                 model_code)  # generate directory name
        # create directories
        if not os.path.exists(directory):
            os.makedirs(directory)
        # write command line arguments to file (useful for reproducibility)
        with open(os.path.join(directory, 'args.txt'), 'w') as f:
            for key in args.__dict__.keys():
                # special case for list input
                if isinstance(args.__dict__[key], list):
                    for entry in args.__dict__[key]:
                        f.write('--' + key + '=' + str(entry) + "\n")
                else:
                    f.write('--' + key + '=' + str(args.__dict__[key]) + "\n")
        checkpoint = None
        step = 0
        restore = False
        data_split_indices = None
        # restarts run from latest checkpoint
    else:
        directory = args.restart  # load directory name
        # load latest checkpoint
        checkpoint_path = os.path.join(directory, 'checkpoints')  # checkpoint directory
        checkpoint = torch.load(
            os.path.join(checkpoint_path, 'latest_checkpoint.pth'),
            map_location='cpu',
            weights_only=False,
        )
        model_code = checkpoint['ID']  # load ID
        if args.fix_hyperparams:
            pass
        elif args.fix_arguments:
            for arg in vars(checkpoint['args']):
                if arg in hyperparam_args:
                    print('loading hyperparam arg', arg)
                    setattr(args, arg, getattr(checkpoint['args'], arg))
        step = checkpoint['step']
        restore = True
        data_split_indices = checkpoint['data_split_indices']
    if args.density_grad_weight > 0:
        args.density_grad = True
    train_vars = {'model_code': model_code, 'directory': directory, 'checkpoint': checkpoint,
                  'step': step, 'restore': restore, 'data_split_indices': data_split_indices}

    return args, hyperparam_args, train_vars


def init_grid_vars(args, test=False):
    """
    Initialize data variables.

    Args:
        args: Namespace object containing command line arguments.
        test: Boolean indicating whether to initialize test data variables.

    Returns:
        grid_vars: Dictionary containing data variables.
    """
    if test or not args.rotate_grid:
        rotate = False
    else:
        rotate = True
    if args.pyscf_grid:
        grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
        sampling_fn = None
        grid_origin = 0
        grid_extent = None
    elif args.cube_grid:
        grid_origin = args.cube_origin
        grid_extent = np.array([args.cube_extent] * 3)
        grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                          extent=grid_extent,
                          origin=np.array([grid_origin] * 3))
        sampling_fn = cubical_sampling
    else:
        grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
        sampling_fn = partial(spherical_radial_sampling, rotate=rotate)
        grid_origin = 0
        grid_extent = None
    grid_vars = {'rotate': rotate, 'grid_fn': grid_fn, 'sampling_fn': sampling_fn,
                 'grid_origin': grid_origin, 'grid_extent': grid_extent}
    return grid_vars


def modify_args_by_phase(args, orig_args, phase):
    """
    Modify arguments based on the current training phase.

    Args:
        args: Namespace object containing command line arguments.
        orig_args: Namespace object containing original command line arguments.
        phase: String indicating the current training phase.

    Returns:
        args: Updated args Namespace.
    """
    args.df_weight = 0.0
    args.density_weight = 0.0
    args.dipole_moment_weight = 0.0
    args.energy_weight = 0.0
    args.forces_weight = 0.0
    args.learning_rate = orig_args.learning_rate
    args.stop_at_learning_rate = orig_args.stop_at_learning_rate
    args.validation_interval = orig_args.validation_interval
    args.decay_patience = orig_args.decay_patience
    args.max_steps = orig_args.max_steps

    if phase == 'df':
        args.df_weight = orig_args.df_weight
        if orig_args.density_weight > 0:
            if args.fast_df:
                args.max_steps = args.max_steps / 10
                args.validation_interval = args.validation_interval / 10
                args.decay_patience = args.decay_patience * 2
    elif phase == 'density':
        args.density_weight = orig_args.density_weight
        if orig_args.df_weight > 0:
            args.learning_rate = args.learning_rate / 10
        if orig_args.core_density_basis > 0:
            args.core_density_basis = 0
        args.density_loss_comp = orig_args.density_loss_comp
        args.density_loss_comp_weights = orig_args.density_loss_comp_weights
    elif phase == 'density_fine_tuning':
        args.density_weight = orig_args.density_weight
        args.learning_rate = orig_args.learning_rate * orig_args.fine_tuning_lr_factor
        args.stop_at_learning_rate = orig_args.stop_at_learning_rate\
            * orig_args.fine_tuning_stop_lr_factor
        args.density_loss_comp = orig_args.density_loss_comp_ft
        args.density_loss_comp_weights = orig_args.density_loss_comp_ft_weights
    elif phase == 'dipole_moment':
        args.dipole_moment_weight = orig_args.dipole_moment_weight
        if orig_args.density_weight > 0:
            args.learning_rate = args.learning_rate / 1000
            args.stop_at_learning_rate = args.stop_at_learning_rate/100
        elif orig_args.df_weight > 0:
            args.learning_rate = args.learning_rate / 10
    elif phase == 'core_density':
        args.density_weight = orig_args.density_weight
        args.core_density_basis = orig_args.core_density_basis
        args.learning_rate - args.learning_rate / 10
        args.density_loss_comp = ['mse']
        args.density_loss_comp_weights = [1.0]
    elif phase == 'energy':
        args.energy_weight = orig_args.energy_weight
        args.forces_weight = orig_args.forces_weight

    return args


def get_required_properties_from_args(args):
    """
    Get required properties from command line arguments.

    Args:
        args: Namespace object containing command line arguments.

    Returns:
        required_properties: List of required properties.
    """
    required_properties = []
    if (args.density_weight + args.dipole_moment_weight > 0) \
        or args.density_from_df or args.density_from_free_atoms:
        required_properties.append('density')
    if args.df_weight > 0 or args.density_from_df:
        required_properties.append('df_coeffs')
    if args.dipole_moment_weight > 0:
        required_properties.append('dipole_moment')
    if args.energy_weight > 0:
        required_properties.append('energy')
    if args.forces_weight > 0:
        required_properties.append('forces')

    return required_properties


def prepare_cubic_datasets(args, required_properties, train_indices, valid_indices):
    """
    Prepare cubic grid validation datasets.

    Args:
        args: Namespace object containing command line arguments.
        required_properties: List of required properties.
        train_indices: List of training indices.
        valid_indices: List of validation indices.

    Returns:
        valid_cube_dataset: AtomsDensityData validation dataset with cubic grid.
    """
    if args.cube_grid_valid:
        grid_origin = args.cube_origin
        grid_extent = np.array([args.cube_extent] * 3)
        cube_grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                               extent=grid_extent,
                               origin=np.array([grid_origin] * 3))
        cube_sampling_fn = cubical_sampling

        cube_dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                                        orbitals_path=args.orbitals_file,
                                        density_n_samp=args.density_subsamples,
                                        required_properties=required_properties,
                                        center_positions=False,
                                        radial_coeffs_file=args.radial_coeffs_file,
                                        L0_coeffs_file=args.L0_coeffs_file,
                                        dtype=args.dtype,
                                        grid_fn=cube_grid_fn,
                                        sampling_fn=cube_sampling_fn,
                                        verbose=args.verbose,
                                        cutoff=args.cutoff,
                                        df_loss_weights=args.df_loss_weights,
                                        atom_dens_path=args.atom_dens_path,
                                        atom_dens_type=args.atom_dens_type,
                                        density_grad=args.density_grad,
                                        calc_basis_path=args.calc_basis_file,
                                        dpm_intor=args.dpm_intor,
                                        )

        valid_cube_dataset = torch.utils.data.Subset(cube_dataset, valid_indices)

        if args.center_energy and 'energy' in required_properties:
            if args.atomic_energies is None:
                energy_mean = cube_dataset.atoms['energy'][train_indices].mean()
                cube_dataset.center_energy(energy_mean)
            else:
                atomic_energies = np.load(args.atomic_energies, allow_pickle=True).item()
                cube_dataset.normalize_energy(atomic_energies)
        return valid_cube_dataset
    else:
        return None


def prepare_datasets(args, required_properties, grid_vars, data_split_indices, density_n_samp=None):
    """
    Prepare training, validation and test datasets.

    Args:
        args: Namespace object containing command line arguments.
        required_properties: List of required properties.
        grid_vars: Dictionary containing grid variables.
        data_split_indices: List of indices for splitting training, validation and test data.

    Returns:
        train_dataset: AtomsDensityData object for training dataset.
        valid_dataset: AtomsDensityData validation dataset.
        test_dataset: AtomsDensityData test dataset.
        valid_cube_dataset: AtomsDensityData validation dataset with cubic grid.
    """
    if density_n_samp is None:
        density_n_samp = args.density_subsamples
    train_dataset, valid_dataset, test_dataset = None, None, None

    dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                               orbitals_path=args.orbitals_file,
                               density_n_samp=density_n_samp,
                               required_properties=required_properties,
                               center_positions=False,
                               radial_coeffs_file=args.radial_coeffs_file,
                               L0_coeffs_file=args.L0_coeffs_file,
                               dtype=args.dtype,
                               grid_fn=grid_vars['grid_fn'],
                               pyscf_grid=args.pyscf_grid,
                               pyscf_rotate=grid_vars['rotate'],
                               sampling_fn=grid_vars['sampling_fn'],
                               grid_extent=grid_vars['grid_extent'],
                               grid_origin=grid_vars['grid_origin'],
                               verbose=args.verbose,
                               timing=args.timing,
                               cutoff=args.cutoff,
                               df_loss_weights=args.df_loss_weights,
                               atom_dens_path=args.atom_dens_path,
                               atom_dens_type=args.atom_dens_type,
                               density_grad=args.density_grad,
                               calc_basis_path=args.calc_basis_file,
                               dpm_intor=args.dpm_intor,
                               )

# split into train / valid / test
    if data_split_indices is None and args.np_dataset_valid is None:
        train_dataset, valid_dataset, test_dataset = seeded_random_split(
            lengths=[args.num_train, args.num_valid, len(dataset) - (args.num_train + args.num_valid)],
            dataset=dataset, seed=args.split_seed
        )

        data_split_indices = {'train': train_dataset.indices,
                              'valid': valid_dataset.indices,
                              'test': test_dataset.indices}
    elif args.np_dataset_valid is not None:
        valid_dataset = AtomsDensityData(np_path=args.np_dataset_valid, density_path=args.dens_dataset_valid,
                                         orbitals_path=args.orbitals_file,
                                         density_n_samp=density_n_samp,
                                         required_properties=required_properties,
                                         center_positions=False,
                                         radial_coeffs_file=args.radial_coeffs_file,
                                         L0_coeffs_file=args.L0_coeffs_file,
                                         dtype=args.dtype,
                                         grid_fn=grid_vars['grid_fn'],
                                         pyscf_grid=args.pyscf_grid,
                                         pyscf_rotate=False,
                                         sampling_fn=grid_vars['sampling_fn'],
                                         grid_extent=grid_vars['grid_extent'],
                                         grid_origin=grid_vars['grid_origin'],
                                         verbose=args.verbose,
                                         cutoff=args.cutoff,
                                         df_loss_weights=args.df_loss_weights,
                                         atom_dens_path=args.atom_dens_path,
                                         atom_dens_type=args.atom_dens_type,
                                         density_grad=args.density_grad,
                                         calc_basis_path=args.calc_basis_file,
                                         dpm_intor=args.dpm_intor,
                                         )

        if data_split_indices is None or args.ignore_split_indices:
            valid_inds = np.random.choice(np.arange(len(valid_dataset)), args.num_valid, replace=False)
            valid_dataset = torch.utils.data.Subset(valid_dataset, valid_inds)
            train_dataset, _, test_dataset = seeded_random_split(
                lengths=[args.num_train, 0, len(dataset) - args.num_train],
                dataset=dataset, seed=args.split_seed
            )
            data_split_indices = {'train': train_dataset.indices,
                                  'valid': valid_dataset.indices,
                                  'test': test_dataset.indices}
        else:
            train_dataset = torch.utils.data.Subset(dataset, data_split_indices['train'][:args.num_train])
            valid_dataset = torch.utils.data.Subset(valid_dataset, data_split_indices['valid'][:args.num_valid])
            test_dataset = torch.utils.data.Subset(dataset, data_split_indices['test'])
    else:
        train_dataset = torch.utils.data.Subset(dataset, data_split_indices['train'][:args.num_train])
        valid_dataset = torch.utils.data.Subset(dataset, data_split_indices['valid'][:args.num_valid])
        test_dataset = torch.utils.data.Subset(dataset, data_split_indices['test'])

    if args.num_test is not None:
        test_dataset.indices = test_dataset.indices[:args.num_test]

    if args.np_dataset_test is not None:
        test_dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                                        orbitals_path=args.orbitals_file,
                                        density_n_samp=density_n_samp,
                                        required_properties=required_properties,
                                        center_positions=False,
                                        radial_coeffs_file=args.radial_coeffs_file,
                                        dtype=args.dtype,
                                        grid_fn=grid_vars['grid_fn'],
                                        pyscf_grid=args.pyscf_grid,
                                        pyscf_rotate=False,
                                        sampling_fn=grid_vars['sampling_fn'],
                                        grid_extent=grid_vars['grid_extent'],
                                        grid_origin=grid_vars['grid_origin'],
                                        cutoff=args.cutoff,
                                        df_loss_weights=args.df_loss_weights,
                                        atom_dens_path=args.atom_dens_path,
                                        atom_dens_type=args.atom_dens_type,
                                        density_grad=args.density_grad,
                                        calc_basis_path=args.calc_basis_file,
                                        dpm_intor=args.dpm_intor,
                                        )

        if args.num_test is not None:
            test_size = args.num_test
        else:
            test_size = len(test_dataset)

        test_dataset = torch.utils.data.Subset(test_dataset, np.arange(test_size))

    # print('valid dataset size', len(valid_dataset))
    valid_cube_dataset = prepare_cubic_datasets(args, required_properties,
                                                train_dataset.indices,
                                                valid_dataset.indices)

    if args.center_energy and 'energy' in required_properties:
        if args.atomic_energies is None:
            train_ind = train_dataset.indices
            energy_mean = dataset.atoms['energy'][train_ind].mean()
            dataset.center_energy(energy_mean)
            if isinstance(test_dataset, torch.utils.data.Subset):
                test_dataset.dataset.center_energy(energy_mean)
            else:
                test_dataset.center_energy(energy_mean)
            if isinstance(valid_dataset, torch.utils.data.Subset):
                valid_dataset.dataset.center_energy(energy_mean)
            else:
                valid_dataset.center_energy(energy_mean)
        else:
            atomic_energies = np.load(args.atomic_energies, allow_pickle=True).item()
            dataset.normalize_energy(atomic_energies)
            if isinstance(test_dataset, torch.utils.data.Subset):
                test_dataset.dataset.normalize_energy(atomic_energies)
            else:
                test_dataset.normalize_energy(atomic_energies)
            if isinstance(valid_dataset, torch.utils.data.Subset):
                valid_dataset.dataset.normalize_energy(atomic_energies)
            else:
                valid_dataset.normalize_energy(atomic_energies)

    return dataset, train_dataset, valid_dataset, test_dataset, valid_cube_dataset, data_split_indices


def init_error_dict(args, test=False):
    """
    Initialize error dictionary for training.

    Args:
        args: Command line arguments.
        test: Whenther error dict will be used for testing.
    Returns:
        error_dict: Object for specifying the error functions used for the different properties.
    """
    loss_weights = {}
    loss_weights['density'] = args.density_weight
    loss_weights['density_grad'] = args.density_grad_weight
    loss_weights['dipole_moment'] = args.dipole_moment_weight
    loss_weights['df_coeffs'] = args.df_weight
    loss_weights['energy'] = args.energy_weight
    loss_weights['forces'] = args.forces_weight
    loss_weights['energy_min'] = args.energy_min_weight
    if not test:
        weights_decay = {}
        weights_decay['density'] = args.density_weight_decay
        weights_decay['density_grad'] = args.density_grad_weight_decay
        weights_decay['dipole_moment'] = args.dipole_moment_weight_decay
        weights_decay['df_coeffs'] = args.df_weight_decay
        weights_decay['energy'] = args.energy_weight_decay
        weights_decay['forces'] = args.forces_weight_decay
        weights_decay['energy_min'] = args.energy_min_weight_decay
        weights_min = {}
        weights_min['density'] = args.density_weight_min
        weights_min['density_grad'] = args.density_grad_weight_min
        weights_min['dipole_moment'] = args.dipole_moment_weight_min
        weights_min['df_coeffs'] = args.df_weight_min
        weights_min['energy'] = args.energy_weight_min
        weights_min['forces'] = args.forces_weight_min
        weights_min['energy_min'] = args.energy_min_weight_min

    loss_comp = {}
    loss_comp['dipole_moment'] = args.dipole_moment_loss_comp
    loss_comp['df_coeffs'] = args.df_loss_comp
    loss_comp['energy'] = args.energy_loss_comp
    loss_comp['forces'] = args.forces_loss_comp
    loss_comp['density_grad'] = args.density_grad_loss_comp

    loss_comp_weights = {}
    loss_comp_weights['df_coeffs'] = {loss_comp: loss_weight
                                      for loss_comp, loss_weight
                                      in zip(args.df_loss_comp, args.df_loss_comp_weights)}
    loss_comp_weights['dipole_moment'] = {loss_comp: loss_weight
                                          for loss_comp, loss_weight
                                          in zip(args.dipole_moment_loss_comp,
                                                 args.dipole_moment_loss_comp_weights)}
    loss_comp_weights['energy'] = {loss_comp: loss_weight
                                   for loss_comp, loss_weight
                                   in zip(args.energy_loss_comp, args.energy_loss_comp_weights)}
    loss_comp_weights['forces'] = {loss_comp: loss_weight
                                   for loss_comp, loss_weight
                                   in zip(args.forces_loss_comp, args.forces_loss_comp_weights)}
    loss_comp_weights['density_grad'] = {loss_comp: loss_weight
                                         for loss_comp, loss_weight
                                         in zip(args.density_grad_loss_comp,
                                                args.density_grad_loss_comp_weights)}
    if not test:
        loss_comp['density'] = args.density_loss_comp
        loss_comp_weights['density'] = {loss_comp: loss_weight
                                        for loss_comp, loss_weight
                                        in zip(args.density_loss_comp,
                                               args.density_loss_comp_weights)}
        error_dict = ErrorDict(loss_weights, weights_balance=args.weights_balance,
                               percentage_error=args.percentage_error,
                               weights_decay=weights_decay, weights_min=weights_min,
                               loss_comp=loss_comp, loss_comp_weights=loss_comp_weights,
                               df_loss_weights=args.df_loss_weights,
                               )
    else:
        loss_comp['density'] = ['perc_mae', 'perc_rmse']
        loss_comp_weights['density'] = {'perc_mae': 1, 'perc_rmse': 1}
        error_dict = ErrorDict(loss_weights, weights_balance=args.weights_balance,
                               percentage_error=args.percentage_error,
                               loss_comp=loss_comp, loss_comp_weights=loss_comp_weights,
                               df_loss_weights=args.df_loss_weights,
                               )
    return error_dict


def prepare_optimizers(args, model, phase=None):
    """
    Prepare optimizers and schedulers.

    Args:
        args: Command line arguments.
        model: GNN model to train.
        phase: Current phase of the training process.

    Returns:
        optimizers: List of initialized optimizers.
        schedulers: List of initialized schedulers.
        ema_params: Parameters used for parameter averaging.
    """
    if args.use_parameter_averaging:
        ema_params = {'decay': args.ema_decay, 'start_epoch': args.ema_start_epoch}
    else:
        ema_params = None

    # build list of parameters to optimize (with or without weight decay)
    parameters = []
    weight_decay_parameters = []
    en_weight_decay_parameters = []
    offset_param = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'radial_fn' not in name and 'embedding' not in name:
            if 'energy' in name and args.en_weight_decay != 0:
                en_weight_decay_parameters.append(param)
            else:
                weight_decay_parameters.append(param)
        elif name == 'en_offset':
            offset_param.append(param)
        else:
            parameters.append(param)
    if phase == 'energy' or args.core_density_basis > 0:
        for param_group in model.density_repr_model.parameters():
            param_group.requires_grad = False
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

    parameter_list = [
        {'params': parameters},
        {'params': weight_decay_parameters, 'weight_decay': float(args.weight_decay)},
        {'params': en_weight_decay_parameters, 'weight_decay': float(args.en_weight_decay)}]

    # choose optimizer
    optimizers = init_optimizers(args, parameter_list, offset_param)
    # initialize Lookahead
    if args.lookahead_k > 0:
        optimizers = [Lookahead(optimizer, k=args.lookahead_k) for optimizer in optimizers]

    # learning rate scheduler (decays learning rate if validation loss plateaus)
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizers[0], mode='min', factor=args.decay_factor, patience=args.decay_patience, verbose=args.verbose)]
    if args.energy_offset:
        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[1], mode='min', factor=args.decay_factor, patience=args.decay_patience, verbose=args.verbose))

    return optimizers, schedulers, ema_params


def init_optimizers(args, parameter_list, offset_param):
    """
    Initialize optimizers.

    Args:
        args: Namespace object containing command line arguments.
        parameter_list: List of parameters to optimize.
        offset_param: List of parameters to use when optimizing energy offset.
    """
    optimizers = []
    if args.optimizer == 'adam':  # Adam
        print("using Adam optimizer")
        optimizers.append(torch.optim.Adam(parameter_list, lr=args.learning_rate, eps=args.epsilon,
                                           betas=(args.beta1, args.beta2), weight_decay=0.0))
        if args.energy_offset:
            optimizers.append(torch.optim.Adam(offset_param, lr=100 * args.learning_rate, eps=args.epsilon,
                                               betas=(args.beta1, args.beta2), weight_decay=0.0))
    elif args.optimizer == 'amsgrad':  # AMSGrad
        print("using AMSGrad optimizer")
        optimizers.append(torch.optim.Adam(parameter_list, lr=args.learning_rate, eps=args.epsilon,
                                           betas=(args.beta1, args.beta2), weight_decay=0.0, amsgrad=True))
        if args.energy_offset:
            optimizers.append(torch.optim.Adam(offset_param, lr=100 * args.learning_rate, eps=args.epsilon,
                                               betas=(args.beta1, args.beta2), weight_decay=0.0, amsgrad=True))
    elif args.optimizer == 'sgd':  # Stochastic Gradient Descent
        print("using Stochastic Gradient Descent optimizer")
        optimizers.append(torch.optim.SGD(
            parameter_list, lr=args.learning_rate, momentum=args.momentum, weight_decay=0.0))
        if args.energy_offset:
            optimizers.append(torch.optim.SGD(
                offset_param, lr=100 * args.learning_rate, momentum=args.momentum, weight_decay=0.0))

    return optimizers
