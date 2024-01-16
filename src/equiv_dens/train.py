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
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling
from equiv_dens.training import utils as train_utils

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

args, hyperparam_args, train_vars = train_utils.init_training_vars(args, hyperparam_args)
checkpoint = train_vars['checkpoint']

args_dict = vars(args)
if args.args_file_name is not None:
    wandb_id = args.args_file_name + '_'
else:
    wandb_id = ''
wandb_name = wandb_id + datetime.utcnow().strftime("%Y-%m-%d")
wandb_id = wandb_name + '_' + train_vars['model_code']
wandb_run = wandb.init(project='equiv_dens', config=args_dict,
                       name=wandb_name, id=wandb_id, resume='allow')

if args.no_restore:
    restore = False

print('model code:', train_vars['model_code'])
print('max steps:', args.max_steps)
print('normalize dens', args.normalize)
print('normalize en', args.normalize_en)
# determine whether GPU is used for training
print('before use gpu')
use_gpu = args.use_gpu and torch.cuda.is_available()
print('after use gpu')

# load dataset(s)
print("loading atoms from" + args.np_dataset + "...")
print("loading density from" + str(args.dens_dataset) + "...")

# density_file = '/home/mihail/data/water_rot/full_densities.hdf5'
# np_file = 'h2o_overlap_static.npy'

grid_vars = train_utils.init_grid_vars(args)

required_properties = train_utils.get_required_properties_from_args(args)


dataset, train_dataset, valid_dataset, test_dataset, valid_cube_dataset, data_split_indices =\
    train_utils.prepare_datasets(args, required_properties,
                                 grid_vars, train_vars['data_split_indices'])


error_dict = train_utils.init_error_dict(args)

z_vals = dataset.atoms['atom_numbers']

# prepare data loaders
train_data_loader = set_up_data_loader(train_dataset, args.train_batch_size,
                                       args.electron_num_batching,
                                       args.batch_efficiency, use_gpu, True)
valid_data_loader = set_up_data_loader(valid_dataset, args.valid_batch_size,
                                       args.electron_num_batching,
                                       args.batch_efficiency, use_gpu, False)
test_data_loader = set_up_data_loader(test_dataset, args.test_batch_size,
                                      args.electron_num_batching,
                                      args.batch_efficiency, use_gpu, False)
if args.cube_grid_valid:
    valid_cube_loader = set_up_data_loader(valid_cube_dataset, args.valid_batch_size,
                                           args.electron_num_batching,
                                           args.batch_efficiency, use_gpu, False)

# define model
model = load_model(args, dataset, train=True)

# if there are multiple GPUs, wrap the model in DataParallel
# "module" is used whenever direct access is needed, e.g. for parameters,
# whereas "model" may be DataParallel and is used for inference only
optimizers, schedulers, ema_params = train_utils.prepare_optimizers(args, model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params is {}'.format(total_params))
validation_loaders = [valid_data_loader]
valid_check_best = [True]
if args.cube_grid_valid:
    validation_loaders.append(valid_cube_loader)
    valid_check_best.append(False)

# print('error dict before training', error_dict.relative_en)

if args.use_gpu:
    model.cuda()
model.to(args.dtype)

trainer = Trainer(model_path=train_vars['directory'],
                  model=model, error_dict=error_dict,
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
                  restore=train_vars['restore'],
                  max_steps=args.max_steps,
                  clip_norm=args.clip_norm,
                  stop_at_learning_rate=args.stop_at_learning_rate,
                  valid_check_best=valid_check_best,
                  verbose=args.verbose,
                  timing=args.timing,
                  memory=args.memory,
                  data_split_indices=data_split_indices,
                  grid_scaling_annealing=args.grid_scaling_annealing,
                  grid_scaling_start=args.grid_scaling_start,
                  )
# with torch.autograd.detect_anomaly():
trainer.run(args.max_steps, use_gpu=use_gpu, dtype=args.dtype)

print('Starting test evaluation!!!')
error_dict = train_utils.init_error_dict(args, test=True)
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
        if key not in test_errors.keys():
            test_errors[key] = errors[key].item()
        else:
            test_errors[key] += (errors[key].item() -
                                 test_errors[key]) / (test_batch_num + 1)
    predictions = None
    data = None
    errors = None

for key in test_errors.keys():
    wandb_run.summary[key + '_test'] = test_errors[key]
print('test errors', test_errors)
