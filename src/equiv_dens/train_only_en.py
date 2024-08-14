#!/usr/bin/env python3
import os
import torch
from datetime import datetime
import wandb
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.utils.misc import generate_id
from equiv_dens.training.errors import ErrorDict
from equiv_dens.training.trainer import Trainer
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.training.lookahead import Lookahead
from equiv_dens.training.model_loader import load_model
from equiv_dens.data.custom_samplers import set_up_data_loader

import numpy as np
from functools import partial

# from torch import autograd

"""
################################################
################ INITIALIZATION ################
################################################
"""
# read arguments
args, hyperparam_args = parse_command_line_arguments()
wandb.login()

# no restart directory specified
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
    latest_checkpoint = 0
    step = 0
    restore = False
    data_split_indices = None
    # restarts run from latest checkpoint
else:
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
        else:
            print('loading all arg', arg)
            setattr(args, arg, getattr(checkpoint['args'], arg))

    step = checkpoint['step']
    restore = True
    data_split_indices = checkpoint['data_split_indices']

args_dict = vars(args)
if args.args_file_name is not None:
    wandb_id = args.args_file_name + '_'
else:
    wandb_id = ''
wandb_name = wandb_id + datetime.utcnow().strftime("%Y-%m-%d")
wandb_id = wandb_name + '_' + model_code
wandb_run = wandb.init(project='equiv_dens', config=args_dict,
                       name=wandb_name, id=wandb_id, resume='allow')

if args.no_restore:
    restore = False

print('model code:', model_code)
print('max steps:', args.max_steps)
print('num train:', args.num_train)
print('num valid:', args.num_valid)
# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
use_gpu = args.use_gpu and torch.cuda.is_available()

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

# density_file = '/home/mihail/data/water_rot/full_densities.hdf5'
# np_file = 'h2o_overlap_static.npy'

required_properties = []
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
                           L0_coeffs_file=args.L0_coeffs_file,
                           dtype=args.dtype,
                           verbose=args.verbose,
                           timing=args.timing,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
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
                                     density_n_samp=args.density_subsamples,
                                     required_properties=required_properties,
                                     center_positions=False,
                                     radial_coeffs_file=args.radial_coeffs_file,
                                     L0_coeffs_file=args.L0_coeffs_file,
                                     dtype=args.dtype,
                                     verbose=args.verbose,
                                     cutoff=args.cutoff,
                                     df_loss_weights=args.df_loss_weights,
                                     projected_density=args.projected_density,
                                     )
    if data_split_indices is None or args.ignore_split_indices:
        train_inds = np.random.choice(np.arange(len(dataset)), args.num_train, replace=False)
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
                                    density_n_samp=args.density_subsamples,
                                    required_properties=required_properties,
                                    center_positions=False,
                                    radial_coeffs_file=args.radial_coeffs_file,
                                    dtype=args.dtype,
                                    cutoff=args.cutoff,
                                    df_loss_weights=args.df_loss_weights,
                                    projected_density=args.projected_density,
                                    )

    if args.num_test is not None:
        test_size = args.num_test
    else:
        test_size = len(test_dataset)

    test_dataset = torch.utils.data.Subset(test_dataset, np.arange(test_size))

print('valid dataset size', len(valid_dataset))

if args.center_energy:
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
loss_comp = {}
loss_comp['density'] = args.density_loss_comp
loss_comp['energy'] = args.energy_loss_comp
loss_comp['forces'] = args.forces_loss_comp

error_dict = ErrorDict(loss_weights, weights_balance=args.weights_balance,
                       percentage_error=args.percentage_error,
                       weights_decay=weights_decay, weights_min=weights_min,
                       loss_comp=loss_comp,
                       )

# print('error dict relative en', error_dict.relative_en)
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

# prepare data loaders
train_data_loader = set_up_data_loader(train_dataset, args.train_batch_size,
                                       args.electron_num_batching, use_gpu, True)
valid_data_loader = set_up_data_loader(valid_dataset, args.valid_batch_size,
                                       args.electron_num_batching, use_gpu, False)
test_data_loader = set_up_data_loader(test_dataset, args.test_batch_size,
                                      args.electron_num_batching, use_gpu, False)
# define model
model = load_model(args, dataset, train=True)

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

for name, param in model.property_models['energy'].named_parameters():
    if 'weight' in name and 'radial_fn' not in name and 'embedding' not in name:
        weight_decay_parameters.append(param)
    elif name == 'en_offset':
        offset_param.append(param)
    else:
        parameters.append(param)

if args.load_from is not None:
    print('loading from', args.load_from)
    load_code = args.load_from.split('_')[-1]
    model_dict = torch.load(os.path.join(args.load_from, 'best_' + load_code + '.pth'), map_location='cpu')
    density_dict = {}

    for key in model_dict.keys():
        if 'repr_model' in key:
            split_key = key.split('.')[1:]
            new_key = '.'.join(split_key)
            density_dict[new_key] = model_dict[key]

    model.repr_model.load_state_dict(density_dict)
    for param_group in model.repr_model.parameters():
        param_group.requires_grad = False

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
    optimizers[0] = Lookahead(optimizers[0], k=args.lookahead_k)

# learning rate scheduler (decays learning rate if validation loss plateaus)

schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizers[0], mode='min', factor=args.decay_factor, patience=args.decay_patience, verbose=args.verbose)]
if args.energy_offset:
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizers[1], mode='min', factor=args.decay_factor, patience=args.decay_patience, verbose=args.verbose))

# create summary writer for tensorboard

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params is {}'.format(total_params))
validation_loaders = [valid_data_loader]
valid_check_best = [True]
if args.cube_grid_valid:
    validation_loaders.append(valid_cube_loader)
    valid_check_best.append(False)

trainer = Trainer(model_path=directory, model=model, error_dict=error_dict,
                  optimizers=optimizers, schedulers=schedulers,
                  train_loader=train_data_loader,
                  validation_loaders=validation_loaders,
                  keep_n_checkpoints=args.keep_checkpoints,
                  checkpoint_interval=args.checkpoint_interval,
                  validation_interval=args.validation_interval,
                  summary_interval=args.summary_interval,
                  wandb=wandb_run,
                  ema_params=ema_params,
                  args=args,
                  hyperparam_args=hyperparam_args,
                  restore=restore,
                  max_steps=args.max_steps,
                  clip_norm=args.clip_norm,
                  stop_at_learning_rate=args.stop_at_learning_rate,
                  valid_check_best=valid_check_best,
                  verbose=args.verbose,
                  timing=args.timing,
                  data_split_indices=data_split_indices,
                  )

# with torch.autograd.detect_anomaly():
trainer.run(args.max_steps, use_gpu=use_gpu, dtype=args.dtype)
print('Starting test evaluation!!!')
error_dict = ErrorDict(loss_weights, weights_balance=args.weights_balance,
                       percentage_error=args.percentage_error,
                       weights_decay=weights_decay, weights_min=weights_min,
                       loss_comp=loss_comp,
                       # relative_en=True,
                       )
test_errors = error_dict.empty()
for test_batch_num, data in enumerate(test_data_loader):
    model.eval()
    # send data to GPU
    if use_gpu:
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()

    # forward step
    data = model.conversions_in(data)
    predictions = model(data)
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

for key in test_errors.keys():
    wandb_run.summary[key + '_test'] = test_errors[key]
print('test errors', test_errors)
