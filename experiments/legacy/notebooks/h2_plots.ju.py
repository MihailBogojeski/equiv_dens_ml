# %%
import numpy as np
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from equiv_dens.utils import base as utils
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from functools import partial
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling
from equiv_dens.training.model_loader import load_model
import equiv_dens.utils.cubetools as cubetools
import os
plt.rc('text', usetex=True)

# %%
# water_dimer = np.load('datasets/water_dimer_test.npy', allow_pickle=True).item()
# print('len water_dimer', len(water_dimer['positions']))
# print(water_dimer['atom_numbers'].shape)
# min_dims = np.min(water_dimer['positions'], axis=(0,1)) - 1
# max_dims = np.max(water_dimer['positions'], axis=(0,1)) + 1
# span = max_dims - min_dims
# step_size = max(span)/50
# print('stepsize', step_size)
# grid_size = np.round(span/step_size).astype(np.int)
# dim_coords = []
# for i in range(len(min_dims)):
#     dim_coords.append(np.linspace(min_dims[i], max_dims[i], grid_size[i]))
# x, y, z = np.meshgrid(*dim_coords, indexing='ij')
# grid = np.stack([x, y, z], axis=-1)[np.newaxis, :]
# grid = torch.Tensor(grid).to(args.dtype)
# grid_weights = np.ones(grid.shape[:-1])
# grid_weights /= np.sum(grid_weights)
# grid_weights = torch.Tensor(grid_weights).to(args.dtype)


args, hyperparam_args = parse_command_line_arguments(arg_file='h2_partial_en_only.txt')
print('type dtype', type(args.dtype))
print('args np dir', args.np_dataset)
# no restart directory specified
args.fix_arguments = True
args.restart = '2022-06-09_Q1CyzdFf'
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

args.best_model_path = 'best_' + model_code + '.pth'
print('best_model_path', args.best_model_path)

print('model code:', model_code)
# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
args.use_gpu = args.use_gpu and torch.cuda.is_available()

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")
args.use_gpu = False
args.cube_grid = True
if args.cube_grid:
    args.grid_extent = 4
    args.cube_size = 100
    grid_origin = -args.grid_extent / 2
    grid_extent = np.array([args.cube_extent] * 3)
    grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                      extent=grid_extent,
                      origin=np.array([grid_origin] * 3))
    sampling_fn = cubical_sampling
    args.radii_adjust = False
else:
    grid_fn = partial(spherical_grid, level=2)
    sampling_fn = partial(spherical_radial_sampling, rotate=False)
    grid_origin = 0
    grid_extent = None
    args.radii_adjust = True

dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density', 'energy'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           radii_adjust=args.radii_adjust,
                           grid_fn=grid_fn,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=args.verbose)
torch.manual_seed(0)
np.random.seed(0)
#args.restart = None
args.density_weight = 0
data_split_indices = checkpoint['data_split_indices']
print(data_split_indices)
print('energy model', args.energy_model)
if args.center_energy:
    train_data = np.load(args.np_dataset, allow_pickle=True).item()
    energy_mean = np.mean(train_data['energy'])
    dataset.center_energy(energy_mean)

# %%
print(len(dataset))
sample = dataset.get_properties([20])
print(sample['positions'])
print(sample['coords'])

print('dens int', torch.sum(sample['density'] * sample['coord_weights']))
fname = 'datasets/h2_dens.cube'
#ase.io.cube.write_cube(f, mol[0], data=repr['density'].detach().cpu().numpy().squeeze())
#ase.io.cube.write_cube(f, mol[0], data=np.ones((50, 50, 50)))
#data = np.ones((50, 50, 50))
meta = {'atoms': []}
for i in range(sample['positions'].shape[1]): 
    pos = utils.angstrom_to_bohr(sample['positions'])
    meta['atoms'].append((sample['atom_numbers'].squeeze()[i], pos.squeeze()[i].tolist()))
print('atoms', meta['atoms'])
density_cube = sample['density'].squeeze().detach().cpu().numpy().reshape(args.cube_size, args.cube_size, args.cube_size)
print('density cube shape', density_cube.shape)
min_coords, _ = torch.min(sample['positions'].view(-1, 3), dim=0)
max_coords, _ = torch.max(sample['positions'].view(-1, 3), dim=0)
print('min coords', min_coords)
print('max coords', max_coords)
min_coords, _ = torch.min(sample['coords'].view(-1, 3), dim=0)
max_coords, _ = torch.max(sample['coords'].view(-1, 3), dim=0)
print('min coords', min_coords)
print('max coords', max_coords)
min_coords, _ = torch.min(utils.angstrom_to_bohr(sample['coords'].view(-1, 3)), dim=0)
max_coords, _ = torch.max(utils.angstrom_to_bohr(sample['coords'].view(-1, 3)), dim=0)
print('min coords', min_coords)
print('max coords', max_coords)
print('grid_weights', sample['coord_weights'])
lattice = (max_coords - min_coords).detach().cpu().numpy()
print('lattice', lattice)
step_size = lattice/density_cube.squeeze().shape
print('step_size', step_size)
meta['org'] = min_coords.tolist()
meta['xvec'] = [step_size[0], 0, 0]
meta['yvec'] = [0, step_size[1], 0]
meta['zvec'] = [0, 0, step_size[2]]
cubetools.write_cube(data=density_cube, meta=meta, fname=fname)
