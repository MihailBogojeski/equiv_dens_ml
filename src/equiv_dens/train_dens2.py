#!/usr/bin/env python3
import os
import math
import torch
from torch.nn.functional import softplus
from datetime import datetime
from tensorboardX import SummaryWriter
from nn.neural_network_dens2 import NeuralNetwork
from training.parse_command_line_arguments import parse_command_line_arguments
from training.util import generate_id, empty_error_dict, compute_error_dict
from training.density_dataset import AtomsDensityData
from training.hamiltonian_dataset import seeded_random_split
from training.exponential_moving_average import ExponentialMovingAverage
from training.lookahead import Lookahead
from training.batch_loader import BatchLoader
from nn.modules.spherical_harmonics_expansion import SphericalHarmonicsExpansion
from training.grids import cubical_grid, cubical_sampling
from functools import partial
import numpy as np
import time


"""
################################################
################ INITIALIZATION ################
################################################
"""
# read arguments
args = parse_command_line_arguments()
print('args use gpu', args.use_gpu)

# no restart directory specified
if args.restart is None:
    # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    ID = generate_id()
    directory = datetime.utcnow().strftime("%Y-%m-%d_") + \
        ID  # generate directory name
    checkpoint_dir = os.path.join(
        directory, 'checkpoints')  # checkpoint directory
    # create directories
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
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
# restarts run from latest checkpoint
else:
    directory = args.restart  # load directory name
    checkpoint_dir = os.path.join(
        directory, 'checkpoints')  # checkpoint directory
    # load latest checkpoint
    checkpoint = torch.load(os.path.join(
        checkpoint_dir, 'latest_checkpoint.pth'), map_location='cpu')
    latest_checkpoint = checkpoint['epoch']
    ID = checkpoint['ID']  # load ID
    max_steps = args.max_steps
    args = checkpoint['args']  # overwrite args
    args.max_steps = max_steps

# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
use_gpu = args.use_gpu and torch.cuda.is_available()

# load dataset(s)
print("loading density from" + args.dens_dataset + "...")
print("loading atoms from" + args.np_dataset + "...")

# density_file = '/home/mihail/data/water_rot/full_densities.hdf5'
# np_file = 'h2o_overlap_static.npy'

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=args.density_subsamples,
                           required_properties=['density'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype)
# split into train / valid / test
train_dataset, valid_dataset, test_dataset = seeded_random_split(
    dataset, [args.num_train, args.num_valid, len(dataset) - (args.num_train + args.num_valid)], seed=args.split_seed)
print('train dataset len', len(train_dataset))
print('valid dataset len', len(valid_dataset))
print('test dataset len', len(test_dataset))

cube_grid_fn = partial(cubical_grid, nx=50, ny=50, nz=50,
                       extent=np.array([4.1483, 4.1483, 4.1483]),
                       origin=np.array([-2.0318, -2.0318 , -2.0318]))
cube_sampling_fn = cubical_sampling

valid_cube_dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                                      orbitals_path=args.orbitals_file,
                                      density_n_samp=10000000000,
                                      required_properties=['density'],
                                      center_positions=False,
                                      radial_coeffs_file=args.radial_coeffs_file,
                                      dtype=args.dtype,
                                      grid_fn=cube_grid_fn,
                                      sampling_fn=cube_sampling_fn)

valid_cube_dataset = torch.utils.data.Subset(valid_cube_dataset, valid_dataset.indices)

# determine weights of different quantities for scaling loss
loss_weights = {}
loss_weights['density'] = args.density_weight
loss_weights['energy'] = args.energy_weight
# loss_weights['full_hamiltonian'] = args.full_hamiltonian_weight
# loss_weights['core_hamiltonian'] = args.core_hamiltonian_weight
# loss_weights['overlap_matrix'] = args.overlap_matrix_weight
# loss_weights['energy'] = args.energy_weight
# loss_weights['forces'] = args.forces_weight

# if energies / forces are used for training, the extreme errors
# at the beginning of training usually lead to NaNs. For this
# reason gradients are only allowed to flow through loss terms
# if the MAE is smaller than a certain threshold.
max_errors = {}
max_errors['density'] = np.inf

# prepare data loaders
train_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(train_dataset),
                                              batch_size=args.train_batch_size, drop_last=False)
valid_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(valid_dataset),
                                              batch_size=args.valid_batch_size, drop_last=False)
valid_cube_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(valid_cube_dataset),
                                                   batch_size=args.valid_batch_size, drop_last=False)

train_data_loader = BatchLoader(train_dataset, batch_sampler=train_sampler,
                                num_workers=args.num_workers, pin_memory=use_gpu)
valid_data_loader = BatchLoader(valid_dataset, batch_sampler=valid_sampler,
                                num_workers=args.num_workers, pin_memory=use_gpu)
valid_cube_loader = BatchLoader(valid_cube_dataset, batch_sampler=valid_cube_sampler,
                                num_workers=args.num_workers, pin_memory=use_gpu)

# define model
if args.load_from is None:
    equiv_model = NeuralNetwork(
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
        activation=args.activation)
    expansion_model = SphericalHarmonicsExpansion(dataset.orbitals, radial_coeffs=dataset.radial_coeffs,
                                                  expansion_constraint=args.expansion_constraint,
                                                  integral_constraint=args.integral_constraint)
else:
    equiv_model = NeuralNetwork(load_from=args.load_from)
    expansion_model = SphericalHarmonicsExpansion(dataset.orbitals, radial_coeffs=dataset.radial_coeffs,
                                                  expansion_constraint=args.expansion_constraint,
                                                  integral_constraint=args.integral_constraint)

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

# convert the model to the correct dtype
equiv_model.to(args.dtype)
expansion_model.to(args.dtype)

# send model to GPU (if use_gpu is True)
if use_gpu:
    equiv_model.cuda()
    expansion_model.cuda()

# if there are multiple GPUs, wrap the model in DataParallel
# "module" is used whenever direct access is needed, e.g. for parameters,
# whereas "model" may be DataParallel and is used for inference only
if use_gpu and torch.cuda.device_count() > 1:
    equiv_model = torch.nn.DataParallel(equiv_model)
    equiv_module = equiv_model.module
    expansion_model = torch.nn.DataParallel(expansion_model)
    expansion_module = expansion_model.module
else:
    equiv_module = equiv_model
    expansion_module = expansion_model

# for keeping an exponential moving average of the model parameters (usually leads to better models)
if args.use_parameter_averaging:
    exponential_moving_average = ExponentialMovingAverage(
        equiv_module, decay=args.ema_decay, start_epoch=args.ema_start_epoch)
else:
    exponential_moving_average = None

# build list of parameters to optimize (with or without weight decay)
parameters = []
weight_decay_parameters = []
for name, param in equiv_module.named_parameters():
    if 'weight' in name and 'radial_fn' not in name and 'embedding' not in name:
        weight_decay_parameters.append(param)
    else:
        parameters.append(param)

parameter_list = [
    {'params': parameters},
    {'params': weight_decay_parameters, 'weight_decay': float(args.weight_decay)}]

# choose optimizer
if args.optimizer == 'adam':  # Adam
    print("using Adam optimizer")
    optimizer = torch.optim.Adam(parameter_list, lr=args.learning_rate, eps=args.epsilon, betas=(
        args.beta1, args.beta2), weight_decay=0.0)
elif args.optimizer == 'amsgrad':  # AMSGrad
    print("using AMSGrad optimizer")
    optimizer = torch.optim.Adam(parameter_list, lr=args.learning_rate, eps=args.epsilon, betas=(
        args.beta1, args.beta2), weight_decay=0.0, amsgrad=True)
elif args.optimizer == 'sgd':  # Stochastic Gradient Descent
    print("using Stochastic Gradient Descent optimizer")
    optimizer = torch.optim.SGD(
        parameter_list, lr=args.learning_rate, momentum=args.momentum, weight_decay=0.0)

# initialize Lookahead
if args.lookahead_k > 0:
    optimizer = Lookahead(optimizer, k=args.lookahead_k)

# learning rate scheduler (decays learning rate if validation loss plateaus)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=args.decay_factor, patience=args.decay_patience, verbose=1)

# restore state from checkpoint
if checkpoint is not None:  # no checkpoint is specified
    step = checkpoint['step']
    epoch = checkpoint['epoch']
    best_errors = checkpoint['best_errors']
    valid_errors = checkpoint['valid_errors']
    equiv_module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if exponential_moving_average is not None:
        checkpoint_ema = checkpoint['exponential_moving_average']
        for key in exponential_moving_average.ema.keys():
            with torch.no_grad():
                exponential_moving_average.ema[key].data.copy_(
                    checkpoint_ema[key].data)
# or initialize step / epoch to 0 and errors to infinity
else:
    step = 0
    epoch = 0
    best_errors = empty_error_dict(loss_weights, fill_value=math.inf)
    valid_errors = empty_error_dict(loss_weights, fill_value=math.inf)

# create summary writer for tensorboard
summary = SummaryWriter(logdir=os.path.join(
    directory, 'logs'), purge_step=step)

"""
###############################################
################ TRAINING LOOP ################
###############################################
"""
if use_gpu:
    print("Training on " + str(torch.cuda.device_count()) + " GPUs:")
else:
    print("Training on the CPU:")

# initialize train metrics
if args.use_gradient_clipping:
    gradient_norm = 0
train_errors = empty_error_dict(loss_weights)  # reset train error metrics
train_batch_num = - 1
# initialize state
equiv_model.train()
expansion_model.eval()
train_iterator = iter(train_data_loader)
new_valid = False
new_best = False

start_time = time.time()
while step < args.max_steps + 1:
    # get the next batch
    start_load = time.time()
    try:
        data = next(train_iterator)
    except StopIteration:
        epoch += 1
        train_iterator = iter(train_data_loader)
        continue
    train_batch_num += 1
    # print('train loading time', time.time() - start_load)

    # send data to GPU
    if use_gpu:
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()

    # zero the parameter gradients
    optimizer.zero_grad()

    # with torch.autograd.set_detect_anomaly(True):  # TODO!!! TURN THIS OFF AGAIN

    # forward step
    coeffs = equiv_model(R=data['positions'])
    # print('coords space shape', data['coords'].shape)
    # print('coords space extremes', torch.min(data['coords'][0], dim=0)[0],
    #       torch.max(data['coords'][0], dim=0)[0])
    predictions = expansion_model(data['coords'], data['positions'],
                                  coeffs['spherical_coeffs'],
                                  coeffs['radial_width'],
                                  coeffs['radial_scale'])

    # print('density integral', torch.sum(predictions['density'] * data['coord_weights'], dim=-1))

    # compute error metrics
    if args.coord_weights:
        coord_weights = data['coord_weights']
    else:
        coord_weights = None

    errors = compute_error_dict(predictions, data, loss_weights, max_errors,
                                coord_weights=coord_weights, weights_balance=args.weights_balance)

    # backward step
    errors['loss'].backward()

    # apply gradient clipping
    if args.use_gradient_clipping:
        norm = torch.nn.utils.clip_grad_norm_(
            equiv_module.parameters(), args.clip_norm)
        gradient_norm += (norm - gradient_norm) / (train_batch_num + 1)

    # optimization step
    optimizer.step()

    # update parameter averages
    if args.use_parameter_averaging:
        exponential_moving_average(epoch)

    # update train_errors (running average)
    for key in train_errors.keys():
        train_errors[key] += (errors[key].item() -
                              train_errors[key]) / (train_batch_num + 1)

    # run validation each validation_interval
    if step % args.validation_interval == 0:
        with torch.no_grad():
            print('validation')
            # this is a signal to the summary writer
            new_valid = True

            # swap to exponentially averaged parameters for validation
            if args.use_parameter_averaging:
                exponential_moving_average.swap()

            # run once over the validation set
            valid_errors = empty_error_dict(
                loss_weights)  # reset valid error metrics
            valid_cube_errors = empty_error_dict(
                loss_weights)  # reset valid error metrics
            equiv_model.eval()  # sets model to evaluation mode
            start_load = time.time()
            for valid_batch_num, data in enumerate(valid_data_loader):
                # print('valid load time', time.time() - start_load)
                # send data to GPU
                if use_gpu:
                    for key in data.keys():
                        if isinstance(data[key], torch.Tensor):
                            data[key] = data[key].cuda()

                # forward step
                coeffs = equiv_model(R=data['positions'])
                # print('coords space shape', data['coords'].shape)
                # print('coords space extremes', torch.min(data['coords'][0], dim=0)[0],
                #       torch.max(data['coords'][0], dim=0)[0])
                predictions = expansion_model(data['coords'], data['positions'],
                                              coeffs['spherical_coeffs'],
                                              coeffs['radial_width'],
                                              coeffs['radial_scale'])

                # print('spherical density integral', torch.sum(predictions['density'] * data['coord_weights'], dim=-1))
                if args.coord_weights:
                    coord_weights = data['coord_weights']
                else:
                    coord_weights = None

                # compute error metrics
                errors = compute_error_dict(
                    predictions, data, loss_weights, max_errors,
                    coord_weights=coord_weights, weights_balance=args.weights_balance)

                # update valid_errors (running average)
                for key in valid_errors.keys():
                    valid_errors[key] += (errors[key].item() -
                                          valid_errors[key]) / (valid_batch_num + 1)

            # pass validation loss to learning rate scheduler
            scheduler.step(metrics=valid_errors['loss'])

            for valid_batch_num, data in enumerate(valid_cube_loader):
                # print('valid load time', time.time() - start_load)
                # send data to GPU
                if use_gpu:
                    for key in data.keys():
                        if isinstance(data[key], torch.Tensor):
                            data[key] = data[key].cuda()

                # forward step
                coeffs = equiv_model(R=data['positions'])
                # print('coords space shape', data['coords'].shape)
                # print('coords space extremes', data['coords'][0, 0], data['coords'][0, -1])
                predictions = expansion_model(data['coords'], data['positions'],
                                              coeffs['spherical_coeffs'],
                                              coeffs['radial_width'],
                                              coeffs['radial_scale'])

                print('cubical density integral', torch.sum(predictions['density'] * data['coord_weights'], dim=-1))
                if args.coord_weights:
                    coord_weights = data['coord_weights']
                else:
                    coord_weights = None

                # compute error metrics
                errors = compute_error_dict(
                    predictions, data, loss_weights, max_errors, coord_weights=coord_weights)

                # update valid_errors (running average)
                for key in valid_cube_errors.keys():
                    valid_cube_errors[key] += (errors[key].item() -
                                               valid_cube_errors[key]) / (valid_batch_num + 1)

            # pass validation loss to learning rate scheduler
            scheduler.step(metrics=valid_errors['loss'])

            # save if it outperforms previous best
            if valid_errors['loss'] < best_errors['loss']:
                new_best = True
                best_errors = valid_errors
                equiv_module.save(os.path.join(directory, 'best_' + str(ID) + '.pth'))
                # construct message for logging
                message = ''
                for key in best_errors.keys():
                    message += key + ': %.6f' % best_errors[key] + '\ n'
                summary.add_text('best models', message, step)

            # swap back to original parameters for training
            if args.use_parameter_averaging:
                exponential_moving_average.swap()

            # set model back to training mode
            equiv_model.train()

            start_load = time.time()
    # write summary to console
    if step % args.summary_interval == 0:
        # write error summaries
        for key in train_errors.keys():
            summary.add_scalar(key + '/train', train_errors[key], step)

        if new_valid:
            for key in valid_errors.keys():
                summary.add_scalar(key + '/valid', valid_errors[key], step)
            new_valid = False

        if new_best:
            for key in best_errors.keys():
                summary.add_scalar(key + '/best', best_errors[key], step)
            new_best = False

        if args.use_gradient_clipping:
            summary.add_scalar('gradient/norm', gradient_norm, step)

        # write summaries for scalar model parameters (always)
        summary.add_scalar(
            'rbf/alpha', softplus(equiv_module.radial_basis_functions._alpha), step)

        # write optional summaries for model parameters
        if args.write_parameter_summaries:
            for name, param in equiv_module.named_parameters():
                splitted_name = name.split('.', 1)
                if len(splitted_name) > 1:
                    first, last = splitted_name
                else:
                    first = 'nn'
                    last = splitted_name[0]
                if param.numel() > 1 and param.requires_grad:  # only tensors get written as histogram
                    summary.add_histogram(
                        first + '/' + last, param.clone().cpu().data.numpy(), step)

        # print progress to consoles
        progress_string = str(step).zfill(
            len(str(args.max_steps))) + "/" + str(args.max_steps)
        progress_string += " epoch: %6d" % epoch
        for key in loss_weights.keys():
            if loss_weights[key] > 0:
                progress_string += "\n  " + key + ":\n"
                progress_string += "    train mae: %10.6f" % train_errors[key + '_mae']
                progress_string += "    train loss: %10.6f" % train_errors['loss']
                progress_string += "    valid mae: %10.6f" % valid_errors[key + '_mae']
                progress_string += "    valid loss: %10.6f" % valid_errors['loss']
                progress_string += "    valid cube mae: %10.6f" % valid_cube_errors[key + '_mae']
                progress_string += "     best mae: %10.6f" % best_errors[key + '_mae']
                progress_string += "    best loss: %10.6f" % best_errors['loss']
        print(progress_string)
        end_time = time.time()
        print("time elapsed:", end_time - start_time)
        start_time = end_time

        # reset train metrics
        if args.use_gradient_clipping:
            gradient_norm = 0
        train_errors = empty_error_dict(
            loss_weights)  # reset train error metrics
        train_batch_num = - 1

    # increment step counter
    step += 1

    # save checkpoint (always the last step)
    if step % args.checkpoint_interval == 0:
        # move latest checkpoint (so it is not overwritten)
        if os.path.isfile(os.path.join(checkpoint_dir, 'latest_checkpoint.pth')):
            os.rename(os.path.join(checkpoint_dir, 'latest_checkpoint.pth'), os.path.join(
                checkpoint_dir, 'checkpoint_' + str(latest_checkpoint).zfill(10) + '.pth'))
        latest_checkpoint = step

        # overwrite latest checkpoint
        torch.save({
            'ID': ID,
            'args': args,
            'step': step,
            'epoch': epoch,
            'best_errors': best_errors,
            'valid_errors': valid_errors,
            'model_state_dict': equiv_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'exponential_moving_average': (exponential_moving_average.ema if args.use_parameter_averaging else None)
        }, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
        summary.add_text('checkpoints', 'saved checkpoint', step)

        # remove oldest checkpoints
        if args.keep_checkpoints >= 0:  # for negative arguments, all checkpoints are kept
            for file in os.listdir(checkpoint_dir):
                if file.startswith("checkpoint") and file.endswith('.pth'):
                    checkpoint_step = int(file.split('.pth')[0].split('_')[-1])
                    if checkpoint_step < step - args.checkpoint_interval * args.keep_checkpoints:
                        filename = os.path.join(checkpoint_dir, file)
                        if os.path.isfile(filename):
                            os.remove(filename)

    # decide whether to stop the run based on learning rate
    stop_training = True
    for param_group in optimizer.param_groups:
        stop_training = stop_training and (
            param_group['lr'] < args.stop_at_learning_rate)
    if stop_training:
        print("Learning rate is smaller than " +
              str(args.stop_at_learning_rate) + "! Training stopped.")
        break

# close summary writer
summary.close()
