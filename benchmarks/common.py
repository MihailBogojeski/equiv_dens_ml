"""Shared helpers: build args / dataset / model / batch through the normal training pipeline."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
DATA = '/workspace'  # config paths are relative to this directory
CLUSTERS = os.environ.get('CLUSTERS', '/workspace/datasets/water_clusters')

from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.model_loader import load_model
from equiv_dens.training import utils as train_utils
from equiv_dens.data.custom_samplers import set_up_data_loader

DEFAULT_CONFIG = os.path.join(ROOT, 'config/training/h2o_small_all_001.txt')


def dataset_overrides(dataset, batch_size):
    """dataset: 'h2o_small' or a water-cluster size such as 'water8'."""
    ov = [f'--train_batch_size={batch_size}']
    if dataset != 'h2o_small':
        n_conf = len(np.load(f'{CLUSTERS}/{dataset}_augccpvdz.npy', allow_pickle=True).item()['energy'])
        ov += [f'--np_dataset={CLUSTERS}/{dataset}_augccpvdz.npy',
               f'--dens_dataset={CLUSTERS}/{dataset}_augccpvdz_df_augccpvqzjkfit.npy',
               '--np_dataset_valid=None', '--dens_dataset_valid=None',
               '--np_dataset_test=None', '--dens_dataset_test=None',
               f'--num_train={n_conf}', '--num_valid=0', '--num_test=0']
    return ov


def parse_args(config_files, overrides):
    os.chdir(DATA)
    sys.argv = ['x'] + [f'@{f}' for f in config_files] + list(overrides)
    args, hyperparam_args = parse_command_line_arguments()
    if args.density_grad_weight > 0:
        args.density_grad = True
    return args


def build(config_files, overrides, train=True):
    """Returns args, dataset (the full AtomsDensityData), train_dataset (Subset), model, error_dict."""
    args = parse_args(config_files, overrides)
    grid_vars = train_utils.init_grid_vars(args)
    required = train_utils.get_required_properties_from_args(args)
    dataset, train_dataset, _, _, _, _ = train_utils.prepare_datasets(args, required, grid_vars, None)
    model = load_model(args, dataset, train=train)
    error_dict = train_utils.init_error_dict(args)
    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    model.to(args.dtype)
    return args, dataset, train_dataset, model, error_dict


def to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
