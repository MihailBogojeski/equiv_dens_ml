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
import numpy as np
import time
from functools import partial
from training.grids import cubical_grid, cubical_sampling

directory = '2020-07-06_0MCs9AUf'  # load directory name
#directory = '2020-04-30_To0wdEze'
checkpoint_dir = os.path.join(
    directory, 'checkpoints')  # checkpoint directory
# load latest checkpoint
checkpoint = torch.load(os.path.join(
    checkpoint_dir, 'latest_checkpoint.pth'), map_location='cpu')
latest_checkpoint = checkpoint['epoch']
ID = checkpoint['ID']  # load ID
args = checkpoint['args']  # overwrite args
# args.use_gpu = False

use_gpu = args.use_gpu and torch.cuda.is_available()

# load dataset(s)
print("loading density from" + args.dens_dataset + "...")
print("loading atoms from" + args.np_dataset + "...")

# density_file = '/home/mihail/data/water_rot/full_densities.hdf5'
# np_file = 'h2o_overlap_static.npy'

args.num_workers = 0

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=args.density_subsamples,
                           required_properties=['density'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype)

cube_grid_fn = partial(cubical_grid, nx=50, ny=50, nz=50,
                       extent=2.5 * np.array([4.1483, 4.1483, 4.1483]),
                       origin=2.5 * np.array([-2.0318, -2.0318 , -2.0318]))
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

# determine weights of different quantities for scaling loss
sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(dataset),
                                              batch_size=6, drop_last=False)

data_loader = BatchLoader(dataset, batch_sampler=sampler,
                                num_workers=args.num_workers, pin_memory=use_gpu)
loss_weights = {}
loss_weights['density'] = args.density_weight
loss_weights['energy'] = args.energy_weight

valid_cube_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(valid_cube_dataset),
                                                   batch_size=args.valid_batch_size, drop_last=False)
valid_cube_loader = BatchLoader(valid_cube_dataset, batch_sampler=valid_cube_sampler,
                                num_workers=args.num_workers, pin_memory=use_gpu)

rad_iterator = iter(data_loader)
cube_iterator = iter(valid_cube_loader)

# define model
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
expansion_model = SphericalHarmonicsExpansion(dataset.orbitals,
                                              radial_coeffs=dataset.radial_coeffs, constraint_type=args.expansion_constraint)

# convert the model to the correct dtype
equiv_model.to(args.dtype)
expansion_model.to(args.dtype)

# send model to GPU (if use_gpu is True)
if use_gpu:
    equiv_model.cuda()
    expansion_model.cuda()


equiv_module = equiv_model
expansion_module = expansion_model

# for keeping an exponential moving average of the model parameters (usually leads to better models)
if checkpoint is not None:  # no checkpoint is specified
    step = checkpoint['step']
    epoch = checkpoint['epoch']
    best_errors = checkpoint['best_errors']
    valid_errors = checkpoint['valid_errors']
    equiv_module.load_state_dict(checkpoint['model_state_dict'])
# or initialize step / epoch to 0 and errors to infinity
else:
    step = 0
    epoch = 0
    best_errors = empty_error_dict(loss_weights, fill_value=math.inf)
    valid_errors = empty_error_dict(loss_weights, fill_value=math.inf)

print('best errors', best_errors)
print('valid_errors', valid_errors)
print('constraint', args.expansion_constraint)

sample = 5
pos = valid_cube_dataset[[sample]]['positions']
print('positions', pos)
coords = valid_cube_dataset[[sample]]['coords']
print('coords shape', coords.shape)
dens_true = valid_cube_dataset[[sample]]['density']
print('density.shape', dens_true.shape)
print('density_integral', torch.sum(dens_true))
dens_true_integral = torch.sum(dens_true)
coeffs = equiv_model(R=pos)
dens_pred = expansion_model(coords, pos,
                            coeffs['spherical_coeffs'],
                            coeffs['radial_width'],
                            coeffs['radial_scale'])
dens_pred = dens_pred['density']
dens_pred_integral = torch.sum(dens_pred)
dens_diff_integral = torch.sum(torch.abs(dens_pred - dens_true))
print('density pred integral', dens_pred_integral)
print('density_difference integral', dens_diff_integral)
print('density interal ratio', dens_diff_integral/dens_true_integral)
print('density MAE', torch.mean(torch.abs(dens_pred - dens_true)))
