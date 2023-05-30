#!/usr/bin/env python3
import os
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.errors import ErrorDict
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.training.model_loader import load_model
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling

import numpy as np
from functools import partial
import time
from equiv_dens.data.custom_samplers import set_up_data_loader

# from torch import autograd

"""
################################################
################ INITIALIZATION ################
################################################
"""
# read arguments
args, hyperparam_args = parse_command_line_arguments()

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
if 'data_split_indices' in checkpoint.keys():
    data_split_indices = checkpoint['data_split_indices']
else:
    data_split_indices = None

print('model code:', model_code)
# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
use_gpu = args.use_gpu and torch.cuda.is_available()

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")
print('args num test', args.num_test)

if args.cube_grid:
    grid_origin = args.cube_origin
    grid_extent = np.array([args.cube_extent] * 3)
    grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                      extent=grid_extent,
                      origin=np.array([grid_origin] * 3))
    sampling_fn = cubical_sampling
else:
    grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
    sampling_fn = spherical_radial_sampling
    grid_origin = 0
    grid_extent = None

required_properties = []
if args.density_weight > 0:
    required_properties.append('density')
if args.df_weight > 0:
    required_properties.append('df_coeffs')
if args.dipole_moment_weight > 0:
    required_properties.append('dipole_moment')
if args.energy_weight > 0:
    required_properties.append('energy')
if args.forces_weight > 0:
    required_properties.append('forces')

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=args.density_subsamples,
                           required_properties=required_properties,
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=args.verbose,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
                           )

if data_split_indices is None or args.ignore_split_indices:
    num_test = 0 if args.np_dataset_test is not None else args.num_test
    print('num_test', num_test)
    train_dataset, _, test_dataset = seeded_random_split(
        dataset, [args.num_train, 0, num_test], seed=args.split_seed)

    data_split_indices = {'train': train_dataset.indices,
                          'test': test_dataset.indices,
                          }
else:
    train_dataset = torch.utils.data.Subset(dataset, data_split_indices['train'])
    test_dataset = torch.utils.data.Subset(dataset, data_split_indices['test'])

if args.num_test is not None:
    test_dataset.indices = test_dataset.indices[:args.num_test]

print('args dataset test', args.np_dataset_test)
if args.np_dataset_test is not None:
    test_dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                                    orbitals_path=args.orbitals_file,
                                    density_n_samp=args.density_subsamples,
                                    required_properties=required_properties,
                                    center_positions=False,
                                    radial_coeffs_file=args.radial_coeffs_file,
                                    dtype=args.dtype,
                                    grid_fn=grid_fn,
                                    sampling_fn=sampling_fn,
                                    grid_extent=grid_extent,
                                    grid_origin=grid_origin,
                                    cutoff=args.cutoff,
                                    df_loss_weights=args.df_loss_weights,
                                    projected_density=args.projected_density,
                                    )

    if args.num_test is not None:
        test_size = args.num_test
    else:
        test_size = len(test_dataset)

    test_dataset = torch.utils.data.Subset(test_dataset, np.arange(test_size))

print('test dataset len', len(test_dataset))
print('args center energy')
if args.center_energy:
    if args.atomic_energies is None:
        train_ind = train_dataset.indices
        energy_mean = dataset.atoms['energy'][train_ind].mean()
        dataset.center_energy(energy_mean)
        print('centering training energy')
        if args.np_dataset_test is not None:
            print('centering test energy')
            test_dataset.dataset.center_energy(energy_mean)
    else:
        atomic_energies = np.load(args.atomic_energies, allow_pickle=True).item()
        dataset.normalize_energy(atomic_energies)
        if args.np_dataset_test is not None:
            test_dataset.dataset.normalize_energy(atomic_energies)

# determine weights of different quantities for scaling loss
loss_weights = {}
loss_weights['df_coeffs'] = args.df_weight
loss_weights['density'] = args.density_weight
loss_weights['dipole_moment'] = args.dipole_moment_weight
loss_weights['energy'] = args.energy_weight
loss_weights['forces'] = args.forces_weight
loss_weights['energy_min'] = args.energy_min_weight

error_dict = ErrorDict(loss_weights, weights_balance=args.weights_balance,
                       percentage_error=args.percentage_error,
                       # relative_en=True,
                      )

z_vals = dataset.atoms['atom_numbers']
# determine weights of different quantities for scaling loss
# loss_weights['full_hamiltonian'] = args.full_hamiltonian_weight
# loss_weights['core_hamiltonian'] = args.core_hamiltonian_weight
# loss_weights['overlap_matrix'] = args.overlap_matrix_weight
# loss_weights['energy'] = args.energy_weight
# loss_weights['forces'] = args.forces_weight

# if energies / forces are used for training, the extreme errors
# at the beginning of training usually lead to NaNs. For this
# reason gradients are only allowed to flow through loss terms
# if the MAE is smaller than a certain threshold.

print('train + valid mean', np.mean(train_dataset))
print('train + valid std')
print('test mean')
print('test std')
# prepare data loaders
if not hasattr(args, 'test_batch_size'):
    args.test_batch_size = args.valid_batch_size

if isinstance(test_dataset, torch.utils.data.Subset):
    def collate_fn(batch):
        return test_dataset.dataset.get_properties(batch)
else:
    def collate_fn(batch):
        return test_dataset.get_properties(batch)

print('test dataset size', len(test_dataset))
print('args.test batch_size', args.test_batch_size)
test_data_loader = set_up_data_loader(test_dataset, args.test_batch_size,
                                      args.electron_num_batching,
                                      args.batch_efficiency, use_gpu, False)

# define model
print('args.df_weight', args.df_weight)
model = load_model(args, dataset)

if use_gpu:
    print("Testing on " + str(torch.cuda.device_count()) + " GPUs:")
else:
    print("Testing on the CPU:")

test_errors = error_dict.empty()
model.eval()
for test_batch_num, data in enumerate(test_data_loader):
    start = time.time()
    # send data to GPU

    if use_gpu:
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()

    if args.timing:
        print('test load time', time.time() - start)
    # forward step
    print('step', test_batch_num)
    # print('positions shape', data['positions'].shape)
    data = model.conversions_in(data)
    data = model.scaling(data)
    predictions = model(data)
    data = model.scaling.transform_back(data)
    data = model.conversions_out(data)
    # print(lkajsdlkjasfd)
    # print('energy pred', predictions['energy'])
    if args.verbose > 0:
        if 'density' in predictions.keys():
            print('test density intergal', torch.sum(predictions['density'] * predictions['coord_weights'], dim=1))
        if 'energy' in predictions.keys():
            print('pred energy', predictions['energy'].view((-1, )))
            print('true energy', data['energy'].view((-1, )))

    # print('spherical density integral', torch.sum(predictions['density'] * data['coord_weights'], dim=-1))
    # compute error metrics
    errors = error_dict.compute(predictions, data)

    # update test_errors (running average)
    for key in errors.keys():
        test_errors[key] += (errors[key].item() -
                             test_errors[key]) / (test_batch_num + 1)
    predictions = None
    data = None
    errors = None
    if args.timing:
        print('test step time', time.time() - start)

print(test_errors)
