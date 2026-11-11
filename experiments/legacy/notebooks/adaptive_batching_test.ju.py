# %%
import os
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling

import numpy as np
from functools import partial

import random
# %load_ext autoreload
# %autoreload 2

# %%
args, hyperparam_args = parse_command_line_arguments(arg_file='water_dyn_spherical_new.txt')

print('type dtype', type(args.dtype))
args.fix_arguments = True
print('args np dir', args.np_dataset)
# no restart directory specified
if args.restart is None: # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    # create directories
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
    step = checkpoint['step']
    for arg in vars(checkpoint['args']):
        if args.fix_arguments:
            if arg in hyperparam_args:
                print('loading hyperparam arg', arg)
                setattr(args, arg, getattr(checkpoint['args'], arg))
        else:
            print('loading all arg', arg)
            setattr(args, arg, getattr(checkpoint['args'], arg))
    restore = True

args.use_gpu = False
# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")
args.verbose = 0
args.use_gpu = False
args.radii_adjust = True 
if args.cube_grid:
    grid_origin = args.cube_origin
    grid_extent = np.array([args.cube_extent] * 3)
    grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                      extent=grid_extent,
                      origin=np.array([grid_origin] * 3))
    sampling_fn = cubical_sampling
else:
    grid_fn = partial(spherical_grid, level=2)
    sampling_fn = partial(spherical_radial_sampling, rotate=False)
    grid_origin = 0
    grid_extent = None

args.cutoff = 5
args.dens_dataset = 'datasets/water_combined_pyscf_def2svp_dft_f_augccpvqzjkfit.npy'
args.np_dataset = 'datasets/water_combined_grouped.npy'
dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,\
                           required_properties=['density', 'df_coeffs'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=torch.float32,
                           grid_fn=grid_fn,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=0,
                           radii_adjust=args.radii_adjust,
                           cutoff=args.cutoff)

dataset_df = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density', 'df_coeffs'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=torch.float32,
                           grid_fn=grid_fn,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=0,
                           radii_adjust=args.radii_adjust, projected_density=True,
                           cutoff=args.cutoff)
print('dataset basis', dataset.orbital_basis_num)
small_dataset = torch.utils.data.Subset(dataset, range(4900, 5100))
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %%
# args, hyperparam_args = parse_command_line_arguments(arg_file='thiophene_12_all_001.txt')
args, hyperparam_args = parse_command_line_arguments(arg_file='thiophene_poly_all_001.txt')

print('type dtype', type(args.dtype))
args.fix_arguments = True
print('args np dir', args.np_dataset)
# no restart directory specified
if args.restart is None:
    # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    # create directories
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
    step = checkpoint['step']
    for arg in vars(checkpoint['args']):
        if args.fix_arguments:
            if arg in hyperparam_args:
                print('loading hyperparam arg', arg)
                setattr(args, arg, getattr(checkpoint['args'], arg))
        else:
            print('loading all arg', arg)
            setattr(args, arg, getattr(checkpoint['args'], arg))
    restore = True

args.use_gpu = False
# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")
args.verbose = 0
args.use_gpu = False
args.radii_adjust = True
if args.cube_grid:
    grid_origin = args.cube_origin
    grid_extent = np.array([args.cube_extent] * 3)
    grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                      extent=grid_extent,
                      origin=np.array([grid_origin] * 3))
    sampling_fn = cubical_sampling
else:
    grid_fn = partial(spherical_grid, level=2)
    sampling_fn = partial(spherical_radial_sampling, rotate=False)
    grid_origin = 0
    grid_extent = None

args.cutoff = 5
# args.dens_dataset = 'datasets/water_combined_pyscf_def2svp_dft_f_augccpvqzjkfit.npy'
# args.np_dataset = 'datasets/water_combined_grouped.npy'
dataset = AtomsDensityData(np_path=args.np_dataset, density_path=None,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['energy', 'forces'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=torch.float32,
                           grid_fn=grid_fn,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=0,
                           radii_adjust=args.radii_adjust,
                           cutoff=args.cutoff)

# dataset_df = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
#                               orbitals_path=args.orbitals_file,
#                               density_n_samp=10000000000,
#                               required_properties=['density', 'df_coeffs'],
#                               center_positions=False,
#                               radial_coeffs_file=args.radial_coeffs_file,
#                               dtype=torch.float32,
#                               grid_fn=grid_fn,
#                               sampling_fn=sampling_fn,
#                               grid_extent=grid_extent,
#                               grid_origin=grid_origin,
#                               verbose=0,
#                               radii_adjust=args.radii_adjust, projected_density=True,
#                               cutoff=args.cutoff)
print('dataset basis', dataset.orbital_basis_num)
small_dataset = torch.utils.data.Subset(dataset, range(900, 1100))
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %%
import equiv_dens.data.custom_samplers as samplers
samp_dataset = dataset

if isinstance(samp_dataset, torch.utils.data.Subset):
    def collate_fn(batch):
        return samp_dataset.dataset.get_properties(batch)
else:
    def collate_fn(batch):
        return samp_dataset.get_properties(batch)
sampler = samplers.SimilarSizeSampler(samp_dataset, shuffle=False)
data_loader = torch.utils.data.DataLoader(samp_dataset, num_workers=args.num_workers,
                                          sampler=sampler,
                                          collate_fn=collate_fn)
count = 0
for dat in data_loader:
    print(count, dat['_idx'], torch.sum(dat['atom_numbers']))
    count += 1

# %%
import equiv_dens.data.custom_samplers as samplers
samp_dataset = dataset

if isinstance(samp_dataset, torch.utils.data.Subset):
    def collate_fn(batch):
        return samp_dataset.dataset.get_properties(batch)
else:                     
    def collate_fn(batch):
        return samp_dataset.get_properties(batch)
sampler = samplers.SimilarSizeSampler(samp_dataset, shuffle=True)
batch_sampler = samplers.AdaptiveBatchSampler(sampler, max_num_elec=args.train_batch_size,
                                              drop_last=False, batch_efficiency=0.7)
data_loader = torch.utils.data.DataLoader(samp_dataset, num_workers=args.num_workers,
                                          batch_sampler=batch_sampler,
                                          collate_fn=collate_fn)
count = 0
for dat in data_loader:
    max_enum = torch.max(torch.sum(dat['batch_atom_numbers'], dim=1))
    usage = torch.mean(torch.sum(torch.tensor(dat['batch_atom_numbers'], dtype=float), dim=1)) / max_enum
    total_elec_num = torch.sum(dat['batch_atom_numbers'])
    if usage < 0.6 or total_elec_num < args.train_batch_size / 2:
        print(count, 'max enum', float(max_enum), 'space usage', float(usage), 'total elec num', float(total_elec_num))
        print('elec nums', torch.sum(dat['batch_atom_numbers'], dim=1))
    count += 1

