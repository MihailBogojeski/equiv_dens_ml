#!/usr/bin/env python3
import os
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.errors import ErrorDict
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.training.model_loader import load_model
from equiv_dens.utils.grids import cubical_grid, cubical_sampling, \
    spherical_grid, spherical_radial_sampling
from equiv_dens.utils import base as utils
import equiv_dens.training.utils as train_utils

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

args, hyperparam_args, test_vars = train_utils.init_training_vars(args, hyperparam_args)
checkpoint = test_vars['checkpoint']
args_dict = vars(args)

print('model code:', test_vars['model_code'])
# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
use_gpu = args.use_gpu and torch.cuda.is_available()

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")
print('args num test', args.num_test)

grid_vars = train_utils.init_grid_vars(args, test=True)

required_properties = train_utils.get_required_properties_from_args(args)

dataset, train_dataset, valid_dataset, test_dataset, _, data_split_indices =\
    train_utils.prepare_datasets(args, required_properties,
                                 grid_vars, test_vars['data_split_indices'],
                                 density_n_samp=1000000000000)


datasets = [test_dataset]

if args.test_eval_all:
    datasets = [train_dataset, valid_dataset, test_dataset]

for test_dat in datasets:
    error_dict = train_utils.init_error_dict(args)
    test_data_loader = set_up_data_loader(test_dat, args.test_batch_size,
                                          args.electron_num_batching,
                                          args.batch_efficiency, use_gpu, False)

    # define model
    model = load_model(args, dataset)

    if use_gpu:
        print("Testing on " + str(torch.cuda.device_count()) + " GPUs:")
    else:
        print("Testing on the CPU:")

    test_errors = error_dict.empty()
    model.eval()
    prop_stats = {}
    print('required_properties', required_properties)
    saved_properties = required_properties + ['positions', 'atom_numbers']
    saved_results = None

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
        # print('step', test_batch_num)
        # print('positions shape', data['positions'].shape)
        data = model.transform_input(data)
        predictions = model(data)
        data = model.transform_back_input(data)

        # print('energy pred', predictions['energy'])
        if args.verbose > 0:
            if 'density' in predictions.keys():
                print('test density intergal', torch.sum(predictions['density'] * predictions['coord_weights'], dim=1))
            if 'energy' in predictions.keys():
                print('pred energy', predictions['energy'].view((-1, )))
                print('true energy', data['energy'].view((-1, )))
                print('energy correlation', np.corrcoef(predictions['energy'].view((-1,)).numpy(force=True), data['energy'].view((-1,)).numpy(force=True)))

        # print('spherical density integral', torch.sum(predictions['density'] * data['coord_weights'], dim=-1))
        # compute error metrics
        errors = error_dict.compute(predictions, data)
        compress_props = ['positions', 'atom_numbers']
        if args.forces_weight > 0:
            compress_props.append('forces')
        data = utils.batch_compressed_atoms(data, compress_props)
        if args.test_save:
            predictions = utils.batch_compressed_atoms(predictions, compress_props)
            if saved_results is None:
                saved_results = {}
                for key in saved_properties:
                    saved_results[key] = predictions[key].detach().cpu()
                    # print('saved results', key, ' after', saved_results[key].shape)
            else:
                for key in saved_properties:
                    # print('saved results', key, ' extend before', saved_results[key].shape)
                    # print('predictions', key, ' extend before', predictions[key].shape)
                    if isinstance(predictions[key], torch.Tensor):
                        saved_results[key] = torch.cat((saved_results[key], predictions[key].detach().cpu()), dim=0) 
                    # print('saved results', key, ' extend after', saved_results[key].shape)
                    # if key == 'density':
                        # print('data density integral', torch.sum(data['density'] * data['coord_weights'], dim=1))
                        # print('res density integral', torch.sum(predictions['density'] * data['coord_weights'], dim=1))

        # update test_errors (running average)
        for key in errors.keys():
            if key not in test_errors.keys():
                test_errors[key] = errors[key].item()
            else:
                test_errors[key] += (errors[key].item() -
                                     test_errors[key]) / (test_batch_num + 1)
        for key in required_properties:
            if key not in prop_stats.keys():
                prop_stats[key] = [data[key].detach().cpu().numpy()]
            else:
                prop_stats[key].append(data[key].detach().cpu().numpy())
        predictions = None
        data = None
        errors = None
        if args.timing:
            print('test step time', time.time() - start)
    print(test_errors)

if args.test_save:
    print('saving test output in ', os.path.join(test_vars['directory'], args.test_save_name))
    torch.save(saved_results, os.path.join(test_vars['directory'], args.test_save_name))
for key in prop_stats.keys():
    # print(prop_stats[key])
    prop_stats[key] = np.concatenate(prop_stats[key], axis=0)
    print(key, 'mean magnitude:', np.mean(np.linalg.norm(prop_stats[key], axis=-1)))
    print(key, 'std:', np.std(prop_stats[key]))
