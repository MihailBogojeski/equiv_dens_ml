#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
from datetime import datetime
from tensorboardX import SummaryWriter
from equiv_dens.nn.dft_network import DFTNetwork
from equiv_dens.nn.representation.spherical_harmonic import EquivariantSphericalHarmonics
from equiv_dens.nn.property_output.energy import ComplexEnergyNetwork, SimpleEnergyNetwork
from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DensityExpansion
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.utils.misc import generate_id
from equiv_dens.training.errors import ErrorDict
from equiv_dens.training.trainer import Trainer
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.training.lookahead import Lookahead
from equiv_dens.data.batch_loader import BatchLoader
from equiv_dens.utils.grids import cubical_grid, cubical_sampling, dftpy_grid, CubicalGrid
from equiv_dens.density_functionals.LDA import LDAFunctional
import equiv_dens.utils.base as utils

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

# no restart directory specified
if args.restart is None:
    # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    model_code = generate_id()
    directory = datetime.utcnow().strftime("%Y-%m-%d_") + \
        model_code  # generate directory name
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
    latest_checkpoint = 0
    step = 0
    restore = False
# restarts run from latest checkpoint
else:
    directory = args.restart  # load directory name
    # load latest checkpoint
    checkpoint_path = os.path.join(directory, 'checkpoints')  # checkpoint directory
    checkpoint = torch.load(os.path.join(
        checkpoint_path, 'latest_checkpoint.pth'), map_location='cpu')
    latest_checkpoint = checkpoint['step']
    model_code = checkpoint['ID']  # load ID
    args = checkpoint['args']  # overwrite args
    step = checkpoint['step']
    restore = True

max_steps = args.max_steps

print('model code:', model_code)
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
grid_origin = -2.0318
grid_extent = np.array([4.1483, 4.1483, 4.1483])
cube_grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                       extent=grid_extent,
                       origin=np.array([grid_origin] * 3))
cube_sampling_fn = cubical_sampling


dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density', 'energy', 'forces'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=cube_grid_fn,
                           sampling_fn=cube_sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin)
# split into train / valid / test
train_dataset, valid_dataset, test_dataset = seeded_random_split(
    dataset, [args.num_train, args.num_valid, len(dataset) - (args.num_train + args.num_valid)], seed=args.split_seed)

if args.center_energy:
    train_ind = train_dataset.indices
    energy_mean = dataset.atoms['energy'][train_ind].mean()
    dataset.center_energy(energy_mean)

grid_cl = CubicalGrid(None, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                      origin=[0, 0, 0], extent=utils.angstrom_to_bohr(grid_extent), device=device, dtype=args.dtype)

grid = dftpy_grid(np.diag(utils.angstrom_to_bohr(grid_extent)), args.cube_gap)
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
print(dataset.atoms['atom_numbers'])
print(z_vals)
# determine weights of different quantities for scaling loss
loss_weights = {}
loss_weights['density'] = args.density_weight
loss_weights['energy'] = args.energy_weight
loss_weights['forces'] = args.forces_weight
loss_weights['energy_min'] = args.energy_min_weight
weights_decay = {}
weights_decay['density'] = args.density_weight_decay
weights_decay['energy'] = args.energy_weight_decay
weights_decay['forces'] = args.forces_weight_decay
weights_decay['energy_min'] = args.energy_min_weight_decay
weights_min = {}
weights_min['density'] = args.density_weight_min
weights_min['energy'] = args.energy_weight_min
weights_min['forces'] = args.forces_weight_min
weights_min['energy_min'] = args.energy_min_weight_min

error_dict = ErrorDict(loss_weights, weights_balance=args.weights_balance,
                       percentage_error=args.percentage_error, weights_decay=weights_decay, weights_min=weights_min)
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
train_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset),
                                              batch_size=args.train_batch_size, drop_last=False)
valid_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(valid_dataset),
                                              batch_size=args.valid_batch_size, drop_last=False)

train_data_loader = BatchLoader(train_dataset, batch_sampler=train_sampler,
                                num_workers=args.num_workers, pin_memory=use_gpu)
valid_data_loader = BatchLoader(valid_dataset, batch_sampler=valid_sampler,
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
)
dens_model = DensityCoeffsNetwork(
    orbitals=dataset.orbitals,
    order=args.order,
    num_features=args.num_features,
    positive_coeffs=args.positive_coeffs,
    clebsch_gordan=clebsch_gordan,
    verbose=args.verbose)

expansion_model = DensityExpansion(dataset.orbitals, radial_coeffs=dataset.radial_coeffs,
                                   expansion_constraint=args.expansion_constraint,
                                   integral_constraint=args.integral_constraint,
                                   integral_scale=args.integral_scale,
                                   verbose=args.verbose,
                                   softmax_norm=args.softmax_norm, n_electrons=sum(z_vals),
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
        calculate_forces=calculate_forces)
elif args.energy_model == 'simple':
    print('building simple energy model')
    en_model = SimpleEnergyNetwork(
        orbitals=dataset.orbitals,
        num_features=args.num_features,
        num_layers=2,
        activation=args.activation,
        calculate_forces=calculate_forces)
else:
    args.energy_model = None

functional = LDAFunctional(z_vals, verbose=args.verbose, energy_offset=args.energy_offset, store_energy=(args.energy_model is None))

density_model = nn.Sequential(repr_model, dens_model)

functional_en_model = nn.Sequential(expansion_model, functional)

property_models = {}
calculate_forces_dict = {}
if error_dict.loss_weights['energy_min'] > 0:
    property_models['energy_min'] = functional_en_model
    calculate_forces_dict['energy_min'] = False
if args.energy_model is not None:
    property_models['energy'] = en_model
    calculate_forces_dict['energy'] = calculate_forces

print('property models', property_models)
model = DFTNetwork(density_model, property_models, calculate_forces_dict=calculate_forces_dict, verbose=args.verbose)
# print('dft network', model)

# if there are multiple GPUs, wrap the model in DataParallel
# "module" is used whenever direct access is needed, e.g. for parameters,
# whereas "model" may be DataParallel and is used for inference only
if args.use_parameter_averaging:
    ema_params = {'decay': args.ema_decay, 'start_epoch': args.ema_start_epoch}
else:
    ema_params = None

# build list of parameters to optimize (with or without weight decay)
parameters = []
weight_decay_parameters = []
offset_param = []
param_names = []
for name, param in model.named_parameters():
    if 'weight' in name and 'radial_fn' not in name and 'embedding' not in name:
        weight_decay_parameters.append(param)
    elif name == 'en_offset':
        offset_param.append(param)
    else:
        parameters.append(param)

parameter_list = [
    {'params': parameters},
    {'params': weight_decay_parameters, 'weight_decay': float(args.weight_decay)}]

# choose optimizer
optimizers = []
if args.optimizer == 'adam':  # Adam
    print("using Adam optimizer")
    optimizers.append(torch.optim.Adam(parameter_list, lr=args.learning_rate, eps=args.epsilon, betas=(
        args.beta1, args.beta2), weight_decay=0.0))
    if args.energy_offset:
        optimizers.append(torch.optim.Adam(offset_param, lr=100 * args.learning_rate, eps=args.epsilon, betas=(
            args.beta1, args.beta2), weight_decay=0.0))
elif args.optimizer == 'amsgrad':  # AMSGrad
    print("using AMSGrad optimizer")
    optimizers.append(torch.optim.Adam(parameter_list, lr=args.learning_rate, eps=args.epsilon, betas=(
        args.beta1, args.beta2), weight_decay=0.0, amsgrad=True))
    if args.energy_offset:
        optimizers.append(torch.optim.Adam(offset_param, lr=100 * args.learning_rate, eps=args.epsilon, betas=(
            args.beta1, args.beta2), weight_decay=0.0, amsgrad=True))
elif args.optimizer == 'sgd':  # Stochastic Gradient Descent
    print("using Stochastic Gradient Descent optimizer")
    optimizers.append(torch.optim.SGD(
        parameter_list, lr=args.learning_rate, momentum=args.momentum, weight_decay=0.0))
    if args.energy_offset:
        optimizers.append(torch.optim.SGD(
            offset_param, lr=100 * args.learning_rate, momentum=args.momentum, weight_decay=0.0))

# initialize Lookahead
if args.lookahead_k > 0:
    optimizer = Lookahead(optimizers[0], k=args.lookahead_k)

# learning rate scheduler (decays learning rate if validation loss plateaus)

schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizers[0], mode='min', factor=args.decay_factor, patience=args.decay_patience, verbose=args.verbose)]
if args.energy_offset:
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizers[1], mode='min', factor=args.decay_factor, patience=args.decay_patience, verbose=args.verbose))

# create summary writer for tensorboard
summary = SummaryWriter(logdir=os.path.join(
    directory, 'logs'), purge_step=step)


trainer = Trainer(model_path=directory, model=model, error_dict=error_dict,
                  optimizers=optimizers, schedulers=schedulers,
                  train_loader=train_data_loader,
                  validation_loaders=[valid_data_loader],
                  checkpoint_interval=args.checkpoint_interval,
                  validation_interval=args.validation_interval,
                  summary_interval=args.summary_interval,
                  ema_params=ema_params,
                  args=args,
                  restore=restore,
                  max_steps=args.max_steps,
                  clip_norm=args.clip_norm,
                  stop_at_learning_rate=args.stop_at_learning_rate,
                  valid_check_best=[True],
                  verbose=args.verbose,
                  )

trainer.run(args.max_steps, device=device, dtype=args.dtype)
