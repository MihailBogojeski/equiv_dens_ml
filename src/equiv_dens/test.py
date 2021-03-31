#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from equiv_dens.nn.dft_network import DFTNetwork
from equiv_dens.nn.representation.spherical_harmonic import EquivariantSphericalHarmonics
from equiv_dens.nn.property_output.energy import ComplexEnergyNetwork, SimpleEnergyNetwork
from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DensityExpansion
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.errors import ErrorDict
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.data.batch_loader import BatchLoader
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    dftpy_grid, CubicalGrid, spherical_grid, rot_spherical_sampling
from equiv_dens.density_functionals.LDA import LDAFunctional
import equiv_dens.utils.base as utils
import copy

import numpy as np
from functools import partial

from dftpy.pseudo import LocalPseudo
# from torch import autograd

"""
################################################
################ INITIALIZATION ################
################################################
"""
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
args.np_dataset_test = old_args.np_dataset_test
args.dens_dataset_test = old_args.dens_dataset_test
args.num_test = old_args.num_test
args.test_batch_size = old_args.test_batch_size
restore = True
if 'data_split_indices' in checkpoint.keys():
    data_split_indices = checkpoint['data_split_indices']
else:
    data_split_indices = None
best_model_path = 'best_' + model_code + '.pth'
print('best_model_path', best_model_path)

print('model code:', model_code)
# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
use_gpu = args.use_gpu and torch.cuda.is_available()

if use_gpu:
    device = 'cuda'
else:
    device = 'cpu'

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
    sampling_fn = rot_spherical_sampling
    grid_origin = 0
    grid_extent = None


dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density', 'energy', 'forces'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=args.verbose)

if data_split_indices is None:
    train_dataset, valid_dataset, test_dataset = seeded_random_split(
        dataset, [args.num_train, args.num_valid, len(dataset) - (args.num_train + args.num_valid)], seed=args.split_seed)

    data_split_indices = {'train': train_dataset.indices,
                          'valid': valid_dataset.indices,
                          'test': test_dataset.indices,
                          }
else:
    train_dataset = torch.utils.data.Subset(dataset, data_split_indices['train'])
    valid_dataset = torch.utils.data.Subset(dataset, data_split_indices['valid'])
    test_dataset = torch.utils.data.Subset(dataset, data_split_indices['test'])

if args.num_test is not None:
    test_dataset.indices = test_dataset.indices[:args.num_test]

if args.np_dataset_test is not None:
    test_dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                                    orbitals_path=args.orbitals_file,
                                    density_n_samp=10000000000,
                                    required_properties=['density', 'energy', 'forces'],
                                    center_positions=False,
                                    radial_coeffs_file=args.radial_coeffs_file,
                                    dtype=args.dtype,
                                    grid_fn=grid_fn,
                                    sampling_fn=sampling_fn,
                                    grid_extent=grid_extent,
                                    grid_origin=grid_origin)

    if args.num_test is not None:
        test_size = args.num_test
    else:
        test_size = len(test_dataset)

    test_dataset = torch.utils.data.Subset(dataset, np.arange(test_size))

if args.center_energy:
    train_ind = train_dataset.indices
    energy_mean = dataset.atoms['energy'][train_ind].mean()
    dataset.center_energy(energy_mean)
    if args.np_dataset_test is not None:
        test_dataset.dataset.center_energy(energy_mean)

# determine weights of different quantities for scaling loss
loss_weights = {}
loss_weights['density'] = old_args.density_weight
loss_weights['energy'] = old_args.energy_weight
loss_weights['forces'] = old_args.forces_weight
loss_weights['energy_min'] = old_args.energy_min_weight

error_dict = ErrorDict(loss_weights, weights_balance=args.weights_balance,
                       percentage_error=args.percentage_error)

z_vals = dataset.atoms['atom_numbers']
if loss_weights['energy_min']:
    grid_origin = args.cube_origin
    grid_extent = np.array([args.cube_extent] * 3)
    grid_cl = CubicalGrid(dataset.atoms, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                          origin=[0, 0, 0], extent=utils.angstrom_to_bohr(grid_extent), device=device, dtype=args.dtype)

    cube_gap = utils.angstrom_to_bohr(args.cube_extent) / args.cube_size
    print('cube_extent', utils.angstrom_to_bohr(args.cube_extent))
    print('cube_size', args.cube_size)
    print('cube_gap', cube_gap)
    grid = dftpy_grid(np.diag(utils.angstrom_to_bohr(grid_extent)), cube_gap)
# print('grid.lattice', grid.lattice)
# print('grid size', grid.r.shape)
# print('ions lattice', dataset.ions[0].pos.cell.lattice)

    file_names = {'H': 'H.pbe-kjpaw_psl.0.1.UPF', 'C': 'C.pbe-kjpaw_psl.0.1.UPF',
                  'O': 'O.pbe-n-kjpaw_psl.0.1.UPF'}
    PP_list = {key: os.path.join(args.pseudo_pot_path, file_names[key]) for key in file_names.keys()}
# print('pseudo potentials', PP_list)
    pseudo_pot = LocalPseudo(grid=grid, ions=None, PP_list=PP_list, PME=True)
    pseudo_pot.restart(grid=grid, ions=dataset.ions[0])

    dataset.add_fixed_properties({'grid': grid_cl, 'dftpy_grid': grid, 'pseudo_pot': pseudo_pot})

    z_vals = []
    print('ions0', dataset.ions[0])
    for t in dataset.atoms['atom_types']:
        z_vals.append(dataset.ions[0].Zval[t])
    z_vals = np.array(z_vals)
    print('dataset atom numbers', dataset.atoms['atom_numbers'])
    print('z_vals', z_vals)
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

# prepare data loaders
if not hasattr(args, 'test_batch_size'):
    args.test_batch_size = args.valid_batch_size

print('test dataset size', len(test_dataset))
test_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(test_dataset),
                                             batch_size=args.test_batch_size, drop_last=False)

test_data_loader = BatchLoader(test_dataset, batch_sampler=test_sampler,
                               num_workers=args.num_workers, pin_memory=use_gpu)

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
        verbose=args.verbose,
        timing=args.timing,
    )
elif args.energy_model == 'simple':
    print('building simple energy model')
    en_model = SimpleEnergyNetwork(
        orbitals=dataset.orbitals,
        num_features=args.num_features,
        num_layers=args.num_energy_output,
        activation=args.activation,
        calculate_forces=calculate_forces,
        verbose=args.verbose,
        timing=args.timing,
    )
else:
    args.energy_model = None

if loss_weights['energy_min'] > 0:
    functional = LDAFunctional(z_vals, verbose=args.verbose, energy_offset=args.energy_offset, store_energy=(args.energy_model is None))
    functional_en_model = nn.Sequential(expansion_model, functional)

density_model = nn.Sequential(repr_model, dens_model)

property_models = {}
calculate_forces_dict = {}
if (loss_weights['density'] + loss_weights['energy_min']) > 0:
    property_models['density'] = expansion_model
    calculate_forces_dict['density'] = False
if loss_weights['energy_min'] > 0:
    property_models['energy_min'] = functional_en_model
    calculate_forces_dict['energy_min'] = False
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
model.to(device)
model.to(args.dtype)
# if there are multiple GPUs, wrap the model in DataParallel
# "module" is used whenever direct access is needed, e.g. for parameters,
# whereas "model" may be DataParallel and is used for inference only

test_errors = error_dict.empty()
for test_batch_num, data in enumerate(test_data_loader):
    # send data to GPU
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)

    # forward step
    print('step')
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
