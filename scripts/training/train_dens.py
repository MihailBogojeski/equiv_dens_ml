#!/usr/bin/env python3
import os
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training import utils as train_utils
from equiv_dens.training.model_loader import load_model
from equiv_dens.training.errors import ErrorDict
from equiv_dens.training.trainer import Trainer
from equiv_dens.training.lookahead import Lookahead
from equiv_dens.data.custom_samplers import set_up_data_loader

# read arguments
args, hyperparam_args = parse_command_line_arguments()
args, hyperparam_args, train_vars = train_utils.init_training_vars(args, hyperparam_args)
directory = train_vars['directory']
restore = train_vars['restore']
print('model code:', train_vars['model_code'])
print('args use gpu', args.use_gpu)
use_gpu = args.use_gpu and torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

# load dataset(s)
grid_vars = train_utils.init_grid_vars(args)
required_properties = ['density']
dataset, train_dataset, valid_dataset, test_dataset, valid_cube_dataset, _ = train_utils.prepare_datasets(
    args, required_properties, grid_vars, train_vars['data_split_indices'])
print('train dataset len', len(train_dataset))
print('valid dataset len', len(valid_dataset))
print('test dataset len', len(test_dataset))
# determine weights of different quantities for scaling loss
# Density-only training: ignore energy/forces even if config specifies them
loss_weights = {}
loss_weights['density'] = args.density_weight
loss_weights['energy'] = 0.0
loss_weights['forces'] = 0.0
# loss_weights['full_hamiltonian'] = args.full_hamiltonian_weight
# loss_weights['core_hamiltonian'] = args.core_hamiltonian_weight
# loss_weights['overlap_matrix'] = args.overlap_matrix_weight
# loss_weights['energy'] = args.energy_weight
# loss_weights['forces'] = args.forces_weight

# if energies / forces are used for training, the extreme errors
# at the beginning of training usually lead to NaNs. For this
# reason gradients are only allowed to flow through loss terms
# if the MAE is smaller than a certain threshold.
error_dict = ErrorDict(loss_weights, weights_balance=args.weights_balance,
                       percentage_error=args.percentage_error,)

# prepare data loaders
train_data_loader = set_up_data_loader(train_dataset, args.train_batch_size,
                                       args.electron_num_batching,
                                       args.batch_efficiency, use_gpu, True)
valid_data_loader = set_up_data_loader(valid_dataset, args.valid_batch_size,
                                       args.electron_num_batching,
                                       args.batch_efficiency, use_gpu, False) if len(valid_dataset) > 0 else None

valid_cube_loader = None
if args.cube_grid_valid and valid_cube_dataset is not None and len(valid_dataset) > 0:
    valid_cube_loader = set_up_data_loader(valid_cube_dataset, args.valid_batch_size,
                                          args.electron_num_batching,
                                          args.batch_efficiency, use_gpu, False)

# define model
model = load_model(args, dataset, train=True)
optimizers, schedulers, ema_params = train_utils.prepare_optimizers(args, model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params is {}'.format(total_params))

validation_loaders = [valid_data_loader] if valid_data_loader is not None else []
valid_check_best = [True] * len(validation_loaders)
if valid_cube_loader is not None:
    validation_loaders.append(valid_cube_loader)
    valid_check_best.append(False)

trainer = Trainer(model_path=directory, model=model, error_dict=error_dict,
                  optimizers=optimizers, schedulers=schedulers,
                  train_loader=train_data_loader,
                  validation_loaders=validation_loaders,
                  checkpoint_interval=args.checkpoint_interval,
                  validation_interval=args.validation_interval,
                  summary_interval=args.summary_interval,
                  ema_params=ema_params,
                  args=args,
                  restore=restore,
                  keep_n_checkpoints=args.keep_checkpoints,
                  max_steps=args.max_steps,
                  clip_norm=args.clip_norm,
                  stop_at_learning_rate=args.stop_at_learning_rate,
                  valid_check_best=valid_check_best,
                  verbose=args.verbose,
                  timing=args.timing,
                  )

trainer.run(args.max_steps, use_gpu=use_gpu, dtype=args.dtype)
