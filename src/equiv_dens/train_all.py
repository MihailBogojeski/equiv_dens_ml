#!/usr/bin/env python3
import torch
import wandb
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.trainer import Trainer
from equiv_dens.training.model_loader import load_model
from equiv_dens.data.custom_samplers import set_up_data_loader
from equiv_dens.training import utils as train_utils
import copy

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
wandb_date = train_vars['directory'].split('/')[-1].split('_')[0]
wandb_name = wandb_id + wandb_date
wandb_id = wandb_name + '_' + train_vars['model_code']
wandb_run = wandb.init(project='equiv_dens', config=args_dict,
                       name=wandb_name, id=wandb_id, resume='allow', mode=args.wandb_mode)
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

if checkpoint is None or 'training_phases' not in checkpoint:
    training_phases = []
    if args.df_weight > 0:
        training_phases.append('df')
    if args.density_weight > 0:
        training_phases.append('density')
    if args.dipole_moment_weight > 0:
        training_phases.append('dipole_moment')
    if args.density_fine_tuning:
        training_phases.append('density_fine_tuning')
    if args.energy_weight > 0:
        training_phases.append('energy')
else:
    training_phases = checkpoint['training_phases']

if args.core_density_basis > 0 and 'core_density' not in training_phases:
    training_phases.append('core_density')
ongoing_phases = [phase for phase in training_phases]

print('training_phases', training_phases)
orig_args = copy.deepcopy(args)

for phase in training_phases:
    args = train_utils.modify_args_by_phase(args, orig_args, phase)

    required_properties = train_utils.get_required_properties_from_args(args)

    dataset, train_dataset, valid_dataset, test_dataset, valid_cube_dataset, data_split_indices =\
        train_utils.prepare_datasets(args, required_properties,
                                     grid_vars, train_vars['data_split_indices'])

    error_dict = train_utils.init_error_dict(args)
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
    if args.restart is None:
        args.restart = train_vars['directory']

    optimizers, schedulers, ema_params = train_utils.prepare_optimizers(args, model, phase)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))
    validation_loaders = [valid_data_loader]
    valid_check_best = [True]
    if args.cube_grid_valid:
        validation_loaders.append(valid_cube_loader)
        valid_check_best.append(False)

    if args.use_gpu:
        model.cuda()
    model.to(args.dtype)

    print('restore before training', phase, train_vars['restore'])
    print('num neighbors', args.num_neighbours)
    print('normalize_en', args.normalize_en)
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
                      training_phases=ongoing_phases,
                      )
    # with torch.autograd.detect_anomaly():
    trainer.run(args.max_steps, use_gpu=use_gpu, dtype=args.dtype)
    print('finished trainer run of phase', phase)
    ongoing_phases.remove(phase)
    restore = False
print('Starting test evaluation!!!')

args.df_weight = 0.0
args.density_weight = 1.0
args.dipole_moment_weight = 1.0
if 'energy' in training_phases:
    args.energy_weight = 1.0
    args.forces_weight = 1.0
else:
    args.energy_weight = 0.0
    args.forces_weight = 0.0

required_properties = train_utils.get_required_properties_from_args(args)
grid_vars['rotate'] = False

_, _, _, test_dataset, _, _ = train_utils.prepare_datasets(args, required_properties,
                                                           grid_vars, data_split_indices)

test_data_loader = set_up_data_loader(test_dataset, args.test_batch_size,
                                      args.electron_num_batching,
                                      args.batch_efficiency, use_gpu, False)

error_dict = train_utils.init_error_dict(args, test=True)

if training_phases[-1] == 'dipole_moment':
    error_dict.loss_weights['energy'] = 0
    error_dict.loss_weights['forces'] = 0

model = load_model(args, dataset, train=False)
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
    # print('energy pred', predictions['energy'])
    if args.verbose > 1:
        if 'density' in predictions.keys():
            print('test density intergal', torch.sum(predictions['density'] * predictions['coord_weights'], dim=1))
            print('true density intergal', torch.sum(data['density'] * data['coord_weights'], dim=1))
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
