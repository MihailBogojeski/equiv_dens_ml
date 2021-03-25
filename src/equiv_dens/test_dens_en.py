#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from equiv_dens.nn.dft_network import DFTNetwork
from equiv_dens.nn.representation.spherical_harmonic import EquivariantSphericalHarmonics
from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DensityExpansion
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.property_output.energy import ComplexEnergyNetwork, SimpleEnergyNetwork
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.errors import ErrorDict
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.data.batch_loader import BatchLoader
from equiv_dens.utils.grids import cubical_grid, cubical_sampling, spherical_grid
import copy

import numpy as np
from functools import partial

# from torch import autograd

"""
################################################
################ INITIALIZATION ################
################################################
"""
# read arguments
args = parse_command_line_arguments()
print('args use gpu', args.use_gpu)

old_args = copy.copy(args)
print('old args', old_args)
directory = args.restart  # load directory name
# load latest checkpoint
checkpoint_path = os.path.join(directory, 'checkpoints')  # checkpoint directory
checkpoint = torch.load(os.path.join(
    checkpoint_path, 'latest_checkpoint.pth'), map_location='cpu')
latest_checkpoint = checkpoint['step']
model_code = checkpoint['ID']  # load ID
for arg in vars(checkpoint['args']):
    setattr(args, arg, getattr(checkpoint['args'], arg))
args.restart = old_args.restart
args.np_dataset_test = old_args.np_dataset_test
args.dens_dataset_test = old_args.dens_dataset_test
args.test_batch_size = old_args.test_batch_size
args.verbose = old_args.verbose
step = checkpoint['step']
restore = True
best_model_path = 'best_' + model_code + '.pth'

print('model code:', model_code)
# no restart directory specified

# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
use_gpu = args.use_gpu and torch.cuda.is_available()

if use_gpu:
    device = 'cuda'
else:
    device = 'cpu'

# load dataset(s)
print("loading density from" + args.dens_dataset + "...")
print("loading atoms from" + args.np_dataset + "...")

# density_file = '/home/mihail/data/water_rot/full_densities.hdf5'
# np_file = 'h2o_overlap_static.npy'

grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=args.density_subsamples,
                           required_properties=['density', 'energy', 'forces'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           verbose=args.verbose)
# split into train / valid / test
train_dataset, valid_dataset, test_dataset = seeded_random_split(
    dataset, [args.num_train, args.num_valid, len(dataset) - (args.num_train + args.num_valid)], seed=args.split_seed)

print('args np dataset', args.np_dataset_test)
print('args dens dataset', args.dens_dataset_test)
if args.np_dataset_test is not None:
    print('loading test dataset')
    test_dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                                    orbitals_path=args.orbitals_file,
                                    density_n_samp=10000000000,
                                    required_properties=['density', 'energy', 'forces'],
                                    center_positions=False,
                                    radial_coeffs_file=args.radial_coeffs_file,
                                    dtype=args.dtype,
                                    grid_fn=grid_fn,
                                    verbose=args.verbose,
                                    )

if args.center_energy:
    train_ind = train_dataset.indices
    energy_mean = dataset.atoms['energy'][train_ind].mean()
    dataset.center_energy(energy_mean)
    if args.np_dataset_test is not None:
        test_dataset.center_energy(energy_mean)

print('train dataset len', len(train_dataset))
print('valid dataset len', len(valid_dataset))
print('test dataset len', len(test_dataset))

# cube_grid_fn = partial(cubical_grid, nx=50, ny=50, nz=50,
#                        extent=np.array([4.1483, 4.1483, 4.1483]),
#                        origin=np.array([-2.0318, -2.0318, -2.0318]))
# cube_sampling_fn = cubical_sampling
#
# test_cube_dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
#                                      orbitals_path=args.orbitals_file,
#                                      density_n_samp=10000000000,
#                                      required_properties=['density'],
#                                      center_positions=False,
#                                      radial_coeffs_file=args.radial_coeffs_file,
#                                      dtype=args.dtype,
#                                      grid_fn=cube_grid_fn,
#                                      sampling_fn=cube_sampling_fn)
#
# test_cube_dataset = torch.utils.data.Subset(test_cube_dataset, test_dataset.indices)
#
loss_weights = {}
loss_weights['density'] = args.density_weight
loss_weights['energy'] = args.energy_weight
loss_weights['forces'] = args.forces_weight
# determine weights of different quantities for scaling loss
# if energies / forces are used for training, the extreme errors
# at the beginning of training usually lead to NaNs. For this
# reason gradients are only allowed to flow through loss terms
# if the MAE is smaller than a certain threshold.
error_dict = ErrorDict(loss_weights, weights_balance=args.weights_balance,
                       percentage_error=args.percentage_error,)

# prepare data loaders
if not hasattr(args, 'test_batch_size'):
    args.test_batch_size = args.valid_batch_size
test_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(test_dataset),
                                             batch_size=args.test_batch_size, drop_last=False)
# test_cube_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(test_cube_dataset),
#                                                   batch_size=args.test_batch_size, drop_last=False)

test_data_loader = BatchLoader(test_dataset, batch_sampler=test_sampler,
                               num_workers=args.num_workers, pin_memory=use_gpu)
# test_cube_loader = BatchLoader(test_cube_dataset, batch_sampler=test_cube_sampler,
#                                num_workers=args.num_workers, pin_memory=use_gpu)

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
    order=args.order,
    num_features=args.num_features,
    positive_coeffs=args.positive_coeffs,
    clebsch_gordan=clebsch_gordan,
    compressed_extraction=args.compressed_extraction,
    verbose=args.verbose,
    timing=args.timing,
)

expansion_model = DensityExpansion(dataset.orbitals, radial_coeffs=dataset.radial_coeffs,
                                   expansion_constraint=args.expansion_constraint,
                                   integral_constraint=args.integral_constraint,
                                   verbose=args.verbose,
                                   softmax_norm=args.softmax_norm,
                                   timing=args.timing,
                                   )


# determine what should be calculated based on loss weights
# tmp = (loss_weights['energy'] > 0) or (loss_weights['forces'] > 0)
# model.calculate_full_hamiltonian = (
#     loss_weights['full_hamiltonian'] > 0) or tmp
# model.calculate_core_hamiltonian = (
#     loss_weights['core_hamiltonian'] > 0) or tmp
# model.calculate_overlap_matrix = (
#     (loss_weights['overlap_matrix'] > 0) or tmp) and not args.orthonormal_basis
# model.calculate_energy = loss_weights['energy'] > 0
# model.calculate_forces = loss_weights['forces'] > 0

print('loss weights forces', loss_weights['forces'])
calculate_forces = loss_weights['forces'] > 0

if args.energy_model == 'complex':
    print('building complex energy model')
    en_model = ComplexEnergyNetwork(
        orbitals=dataset.orbitals,
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
        calculate_forces=calculate_forces,
        compressed_extraction=args.compressed_extraction,
    )
elif args.energy_model == 'simple':
    print('building simple energy model')
    en_model = SimpleEnergyNetwork(
        orbitals=dataset.orbitals,
        num_features=args.num_features,
        num_layers=args.num_energy_output,
        activation=args.activation,
        calculate_forces=calculate_forces,
        compressed_extraction=args.compressed_extraction,
        verbose=args.verbose,
        timing=args.timing,
    )
else:
    args.energy_model = None

# send model to GPU (if use_gpu is True)
density_model = nn.Sequential(repr_model, dens_model)
property_models = {'density': expansion_model}
calculate_forces_dict = {'density': False}

print('calculate forces', calculate_forces)
if args.energy_model is not None:
    property_models['energy'] = en_model
    calculate_forces_dict['energy'] = calculate_forces
print('calculate force dict', calculate_forces_dict)


model = DFTNetwork(density_model, property_models, verbose=args.verbose, calculate_forces_dict=calculate_forces_dict)
print(args.restart)
print(best_model_path)
state_dict_path = os.path.join(args.restart, best_model_path)
print(state_dict_path)
state_dict = torch.load(state_dict_path, map_location='cpu')
model.load_state_dict(state_dict)
model.to(device)
model.to(args.dtype)

test_errors = error_dict.empty()
for test_batch_num, data in enumerate(test_data_loader):
    # send data to GPU
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)

    # forward step
    predictions = model(data)
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

print(test_errors)
