# %%
import os
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling
import equiv_dens.utils.base as utils
from equiv_dens.training.model_loader import load_model
from equiv_dens.utils.misc import generate_id
from datetime import datetime

import numpy as np
from functools import partial, partialmethod
import time
from pyscf.dft import gen_grid, radi
from pyscf import gto
import torch.nn as nn
import ase.io
import equiv_dens.utils.cubetools as cubetools

# %load_ext autoreload
# %autoreload 2

# %%
# args, hyperparam_args = parse_command_line_arguments(arg_file='ethanol_all_012.txt')
# args, hyperparam_args = parse_command_line_arguments(arg_file='thiophene_6mer_all_001.txt')
args, hyperparam_args = parse_command_line_arguments(arg_file='thiophene_poly_all_001.txt')
print('type dtype', type(args.dtype))
args.fix_arguments = True
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
    args.best_model_path = 'best_' + model_code + '.pth'
    restore = True

args.use_gpu = args.use_gpu and torch.cuda.is_available()

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")
args.use_gpu = False
args.cube_grid = False
args.pyscf_grid = True
if args.cube_grid:
    args.cube_size = 75
    args.cube_extent = 9
    # grid_origin = args.cube_origin
    grid_origin = -args.cube_extent/2
    grid_extent = np.array([args.cube_extent] * 3)
    grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                      extent=grid_extent,
                      origin=np.array([grid_origin] * 3))
    sampling_fn = cubical_sampling
    args.radii_adjust = False
else:
    grid_fn = partial(spherical_grid, level=1)
    sampling_fn = partial(spherical_radial_sampling, rotate=True)
    grid_origin = 0
    # grid_origin = -grid_extent[0]/2
    grid_extent = None
    args.radii_adjust = False

args.timing = True

# dataset_1 = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
#                              orbitals_path=args.orbitals_file,
#                              density_n_samp=10000000000,
#                              required_properties=['density'],
#                              center_positions=False,
#                              radial_coeffs_file=args.radial_coeffs_file,
#                              dtype=args.dtype,
#                              grid_fn=grid_fn,
#                              sampling_fn=sampling_fn,
#                              grid_extent=grid_extent,
#                              grid_origin=grid_origin,
#                              verbose=args.verbose,
#                              timing=args.timing,
#                              radii_adjust=args.radii_adjust,)

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           pyscf_grid=True,
                           pyscf_rotate=True,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=args.verbose,
                           timing=args.timing,
                           radii_adjust=args.radii_adjust,)
torch.manual_seed(0)
np.random.seed(0)
# args.restart = None
args.verbose = 0
# model = load_model(args, dataset_1)
print('sampling_fn args', sampling_fn.args)
print('sampling_fn args', sampling_fn.func)
print('sampling_fn args', sampling_fn.keywords)

# %%
data1 = dataset_1.get_properties([0])
data2 = dataset_2.get_properties([0])
# print('data1 grid spec', dataset_1.grid_spec)
# print('data2 grid spec', dataset_2.grid_spec)
# print('data1 pos', data1['positions'])
# print('data2 pos', data2['positions'])
# print('data1 pos', data1['atom_numbers'])
# print('data2 pos', data2['atom_numbers'])
# print('data1 coords', torch.max(data1['coords']))
# print('data2 coords', torch.max(data2['coords']))
# print('data1 coords', torch.min(data1['coords']))
# print('data2 coords', torch.min(data2['coords']))
# print('data1 weights', data1['coord_weights'])
# print('data2 weights', data2['coord_weights'])
# print(data['coords'][:, :10])
# print(data['coord_weights'][:, :10])
print('density integral 1', torch.sum(data1['density'] * data1['coord_weights'], dim=1))
print('density integral 2', torch.sum(data2['density'] * data2['coord_weights'], dim=1))

# %%
data1 = dataset_1.get_properties([0])
data2 = dataset_2.get_properties([0])
print('density integral 1', torch.sum(data1['density'] * data1['coord_weights'], dim=1))
print('density integral 2', torch.sum(data2['density'] * data2['coord_weights'], dim=1))

# %%
data = dataset.get_properties([3001, 4001])
print(data['coords'].shape)
print('density integral', torch.sum(data['density'] * data['coord_weights'], dim=1))

# %%
data = dataset.get_properties([3001, 4001])
print(data['coords'].shape)
print('density integral', torch.sum(data['density'] * data['coord_weights'], dim=1))

# %%
# mol = utils.npy_to_ase(dataset.atoms['positions'][0:1], dataset.atoms['atom_numbers'][0:1])[0]
# utils.npy_to_ase(dataset.atoms['positions'][0:1], dataset.atoms['atom_numbers'][0:1])[0]
total_time = 0
for i in range(30, 40):
    numbers = dataset.atoms['atom_numbers'][i].astype(int)
    positions = dataset.atoms['positions'][i]
    mol_dict = list(zip(numbers, positions))
    if (np.sum(numbers) % 2 == 1):
        mol = gto.M(atom=mol_dict, spin=1)
    else:
        mol = gto.M(atom=mol_dict)

    grid_spec = gen_grid.gen_atomic_grids(mol, level=1, radi=radi.treutler)
    start = time.time()
    coords, weights = gen_grid.gen_partition(mol, grid_spec)
    coords = torch.tensor(coords).to(data['positions'])
    grid_time = time.time() - start
    total_time += grid_time
    print('grid time', time.time() - start)

print('total time', total_time)

print(coords.shape)

# %%
# mol = utils.npy_to_ase(dataset.atoms['positions'][0:1], dataset.atoms['atom_numbers'][0:1])[0]
# utils.npy_to_ase(dataset.atoms['positions'][0:1], dataset.atoms['atom_numbers'][0:1])[0]
all_numbers = []
all_positions = []
for i in range(30, 40):
    numbers = dataset.atoms['atom_numbers'][i].astype(int)
    positions = dataset.atoms['positions'][i] + 100 * i
    all_numbers.append(numbers)
    all_positions.append(positions)

all_numbers = np.concatenate(all_numbers)
all_positions = np.concatenate(all_positions)
mol_dict = list(zip(all_numbers, all_positions))
if (np.sum(numbers) % 2 == 1):
    mol = gto.M(atom=mol_dict, spin=1)
else:
    mol = gto.M(atom=mol_dict)

grid_spec = gen_grid.gen_atomic_grids(mol, level=1, radi=radi.treutler)
start_grid = time.time()
coords, weights = gen_grid.gen_partition(mol, grid_spec)
coords = torch.tensor(coords).to(data['positions'])
print('grid time', time.time() - start_grid)


print(coords.shape)

# %%
def get_pyscf_coords(grid_spec, density_n_samp, mols, idx):
    # mol = utils.npy_to_ase(dataset.atoms['positions'][0:1], dataset.atoms['atom_numbers'][0:1])[0]
    # utils.npy_to_ase(dataset.atoms['positions'][0:1], dataset.atoms['atom_numbers'][0:1])[0]
    start = time.time()
    max_len = 0
    all_coords = []
    for i in idx:
        loop_start = time.time()
        mol = mols[i]
        if not mol._built:
            build_start = time.time()
            # if self.verbose > 3:
            print('building mol', i)
            mol.build()
            # if self.timing:
            #     print('build time', time.time() - build_start)
        rot_spec = {key: (grid_spec[key][0] @ utils.torch_random_rotation_matrix().to(grid_spec[key][0]), grid_spec[key][1])
                    for key in grid_spec.keys()}
        print('rot_coords time', time.time() - start)
        coords, weights = gen_grid.gen_partition(mol, rot_spec)
        coords = torch.tensor(coords).to(data['positions'])
        all_coords.append(coords)
        max_len = max(coords.shape[1], max_len)
    pad_coords = nn.utils.rnn.pad_sequence(all_coords, batch_first=True, padding_value=0)
    print('grid time', time.time() - start)
    print(pad_coords.shape)
    return pad_coords

# %%
fast_coords = get_pyscf_coords(dataset.grid_spec, 10000000000, dataset.mols, list(range(20,30)))

