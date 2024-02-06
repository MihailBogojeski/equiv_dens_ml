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
from functools import partial
import ase.io
import equiv_dens.utils.cubetools as cubetools

# %load_ext autoreload
# %autoreload 2

# %%
args, hyperparam_args = parse_command_line_arguments(arg_file='ethanol_all_012.txt')
print('type dtype', type(args.dtype))
args.fix_arguments = True
print('args np dir', args.np_dataset)
# no restart directory specified
directory = args.restart  # load directory name
# load latest checkpoint
print('directory', directory)
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
args.cube_grid = False
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
    grid_fn = partial(spherical_grid, level=2)
    sampling_fn = partial(spherical_radial_sampling, rotate=False)
    grid_origin = 0
    # grid_origin = -grid_extent[0]/2
    grid_extent = None
    args.radii_adjust = True

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=args.verbose,
                           radii_adjust=args.radii_adjust,)
torch.manual_seed(0)
np.random.seed(0)
# args.restart = None
args.verbose = 0

model = load_model(args, dataset)

# %%
from equiv_dens.utils import orbitals
sample = dataset.get_properties([3])

dpm = orbitals.calc_dipole_moment(sample)['dipole_moment']

print('dpm', dpm)
print('dpm', torch.sum(dpm**2))
print('atom types', sample['atom_numbers'][0])
print('pos', sample['positions'])


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

# %%
# try out mixed dimer and water architectures
idx = 3
sample = dataset.get_properties([idx])
print('sample grid shape', sample['coords'].shape)
repr = model(sample)
print('charge', torch.sum(repr['atom_numbers'], dim=1))
print('density shape', repr['density'].shape)
print('density integral', torch.sum(repr['density'] * repr['coord_weights']))
print('sample density integral', torch.sum(sample['density'] * sample['coord_weights']))
print('positions', sample['positions'])

fname = 'datasets/ethanol_dens.cube'
#ase.io.cube.write_cube(f, mol[0], data=repr['density'].detach().cpu().numpy().squeeze())
#ase.io.cube.write_cube(f, mol[0], data=np.ones((50, 50, 50)))
#data = np.ones((50, 50, 50))
meta = {'atoms': []}
for i in range(sample['positions'].shape[1]): 
    pos = utils.angstrom_to_bohr(sample['positions'])
    meta['atoms'].append((sample['atom_numbers'].squeeze()[i], pos.squeeze()[i].tolist()))
print('atoms', meta['atoms'])
density_cube = repr['density'].squeeze().detach().cpu().numpy().reshape(args.cube_size, args.cube_size, args.cube_size)
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
print('grid_weights', repr['coord_weights'])
lattice = (max_coords - min_coords).detach().cpu().numpy()
print('lattice', lattice)
step_size = lattice/density_cube.squeeze().shape
print('step_size', step_size)
meta['org'] = min_coords.tolist()
meta['xvec'] = [step_size[0], 0, 0]
meta['yvec'] = [0, step_size[1], 0]
meta['zvec'] = [0, 0, step_size[2]]
cubetools.write_cube(data=density_cube, meta=meta, fname=fname)

# %%
from pyscf.dft import numint

scaled_sample_coords = sample['coords']
dens = torch.zeros((sample['coords'].shape[0], sample['coords'].shape[1]), dtype=dataset.dtype)
# mol_start = time.time()
# print('c, i', c, i)
mol = dataset.mols[idx]
if not mol._built:
    # build_start = time.time()
    mol.build()
    # print('build time', time.time() - build_start)
coeff_dict = dataset.coeffs[i]
# ao_start = time.time()
ao = numint.eval_ao(mol, scaled_sample_coords.squeeze())
print('ao shape', ao.shape)
orb = np.einsum('ki, ij -> kj', ao, coeff_dict['mo_coeff'])

orb_fname = 'datasets/ethanol_homo.cube'

data = orb[:, 12].reshape(args.cube_size, args.cube_size, args.cube_size)

cubetools.write_cube(data=data, meta=meta, fname=orb_fname)

# %%
args, hyperparam_args = parse_command_line_arguments(arg_file='resorcinol_all_012.txt')
print('type dtype', type(args.dtype))
args.fix_arguments = True
print('args np dir', args.np_dataset)
# no restart directory specified
directory = args.restart  # load directory name
# load latest checkpoint
print('directory', directory)
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
args.cube_grid = False
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
    grid_fn = partial(spherical_grid, level=2)
    sampling_fn = partial(spherical_radial_sampling, rotate=False)
    grid_origin = 0
    # grid_origin = -grid_extent[0]/2
    grid_extent = None
    args.radii_adjust = True

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=args.verbose,
                           radii_adjust=args.radii_adjust,)

dataset_df = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                              orbitals_path=args.orbitals_file,
                              density_n_samp=10000000000,
                              required_properties=['density'],
                              center_positions=False,
                              radial_coeffs_file=args.radial_coeffs_file,
                              dtype=args.dtype,
                              grid_fn=grid_fn,
                              projected_density=True,
                              sampling_fn=sampling_fn,
                              grid_extent=grid_extent,
                              grid_origin=grid_origin,
                              verbose=args.verbose,
                              radii_adjust=args.radii_adjust,)
torch.manual_seed(0)
np.random.seed(0)
# args.restart = None
args.verbose = 0

model = load_model(args, dataset)

# %%
from equiv_dens.utils import orbitals
sample = dataset.get_properties([8])

dpm = orbitals.calc_dipole_moment(sample)['dipole_moment']

print('dpm', dpm)
print('dpm', torch.sum(dpm**2))
print('pos', sample['positions'])


# %%
# try out mixed dimer and water architectures
torch.no_grad()
dens_model = torch.nn.Sequential(model.density_repr_model, model.property_models['density'])
idx = 8
sample = dataset.get_properties([idx])
# print('sample grid shape', sample['coords'].shape)
# repr = dens_model(sample)
# print('charge', torch.sum(repr['atom_numbers'], dim=1))
# print('density shape', repr['density'].shape)
# print('density integral', torch.sum(repr['density'] * repr['coord_weights']))
# print('sample density integral', torch.sum(sample['density'] * sample['coord_weights']))
# print('positions', sample['positions'])

fname = 'datasets/resorcinol_pred_dens.cube'
#ase.io.cube.write_cube(f, mol[0], data=repr['density'].detach().cpu().numpy().squeeze())
#ase.io.cube.write_cube(f, mol[0], data=np.ones((50, 50, 50)))
#data = np.ones((50, 50, 50))
meta = {'atoms': []}
for i in range(sample['positions'].shape[1]): 
    pos = utils.angstrom_to_bohr(sample['positions'])
    meta['atoms'].append((sample['atom_numbers'].squeeze()[i], pos.squeeze()[i].tolist()))
print('atoms', meta['atoms'])
# density_cube = repr['density'].squeeze().detach().cpu().numpy().reshape(args.cube_size, args.cube_size, args.cube_size)
# print('density cube shape', density_cube.shape)
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
# print('grid_weights', repr['coord_weights'])
lattice = (max_coords - min_coords).detach().cpu().numpy()
print('lattice', lattice)
step_size = lattice/torch.tensor([args.cube_size, args.cube_size, args.cube_size])
print('step_size', step_size)
meta['org'] = min_coords.tolist()
meta['xvec'] = [step_size[0], 0, 0]
meta['yvec'] = [0, step_size[1], 0]
meta['zvec'] = [0, 0, step_size[2]]
# cubetools.write_cube(data=density_cube, meta=meta, fname=fname)

# %%
fname = 'datasets/resorcinol_dens_2.cube'
data = sample['density'].squeeze().detach().cpu().numpy().reshape(args.cube_size, args.cube_size, args.cube_size)
cubetools.write_cube(data=data, meta=meta, fname=fname)
# sample_df = dataset_df.get_properties([idx])
# fname = 'datasets/resorcinol_df_dens.cube'
# data_df = sample_df['density'].squeeze().detach().cpu().numpy().reshape(args.cube_size, args.cube_size, args.cube_size)
# cubetools.write_cube(data=data_df, meta=meta, fname=fname)
# data_ml = density_cube

# %%
cubetools.write_cube(data=np.abs(data_df-data), meta=meta, fname='datasets/resorcinol_df_dens_diff.cube')
cubetools.write_cube(data=np.abs(data_ml-data), meta=meta, fname='datasets/resorcinol_ml_dens_diff.cube')

# %%
for i in range(3):
    fname = 'datasets/resorcinol_dens_ml_' + str(i+1) + '.cube'
    rot_mat = torch.tensor(utils.random_rotation_matrix()).to(sample['positions'])
    print('rot mat shape', rot_mat.shape)
    sample['positions'] = torch.einsum('bai, ij -> baj', sample['positions'], rot_mat)
    sample['coords'] = torch.einsum('bai, ij -> baj', sample['coords'], rot_mat)
    repr = dens_model(sample)
    density_cube = repr['density'].squeeze().detach().cpu().numpy().reshape(args.cube_size, args.cube_size, args.cube_size)
    cubetools.write_cube(data=density_cube, meta=meta, fname=fname)

# %%
args, hyperparam_args = parse_command_line_arguments(arg_file='water_dyn_spherical.txt')
print('type dtype', type(args.dtype))
args.fix_arguments = True
print('args np dir', args.np_dataset)
# no restart directory specified
if args.restart is None:
    # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    model_code = generate_id()
    directory = os.path.join(args.save_dir, datetime.utcnow().strftime("%Y-%m-%d_") +
                             model_code)  # generate directory name
    # create directories
    if not os.path.exists(directory):
        os.makedirs(directory)
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
    if args.fix_arguments:
        for arg in vars(checkpoint['args']):
            if arg in hyperparam_args:
                print('loading hyperparam arg', arg)
                setattr(args, arg, getattr(checkpoint['args'], arg))
    step = checkpoint['step']
    restore = True
    data_split_indices = checkpoint['data_split_indices']

if args.no_restore:
    restore = False

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
    args.cube_size = 50
    args.cube_extent = 5
    # grid_origin = args.cube_origin
    grid_origin = -args.cube_extent/2
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
    # grid_origin = -grid_extent[0]/2
    grid_extent = None
    args.radii_adjust = True

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=args.verbose,
                           radii_adjust=args.radii_adjust,)

torch.manual_seed(0)
np.random.seed(0)
# args.restart = None
args.verbose = 0

# %%
pos = dataset.atoms['positions']
atom_numbers = dataset.atoms['atom_numbers']
print(pos)
print(atom_numbers)
hydrogen_dist = np.sum((pos[:, 1] - pos[:, 2])**2, axis=1)
print('hydrogen_dist', hydrogen_dist[:20])
# %%
idx = 7
sample = dataset.get_properties([idx])
print('sample grid shape', sample['coords'].shape)
print('sample density integral', torch.sum(sample['density'] * sample['coord_weights']))
print('positions', sample['positions'])

fname = 'datasets/water_dens.cube'
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
lattice = (max_coords - min_coords).detach().cpu().numpy()
print('lattice', lattice)
step_size = lattice/density_cube.squeeze().shape
print('step_size', step_size)
meta['org'] = min_coords.tolist()
meta['xvec'] = [step_size[0], 0, 0]
meta['yvec'] = [0, step_size[1], 0]
meta['zvec'] = [0, 0, step_size[2]]
cubetools.write_cube(data=density_cube, meta=meta, fname=fname)

# %%
args, hyperparam_args = parse_command_line_arguments(arg_file='ethanethiol_all_011.txt')
print('type dtype', type(args.dtype))
args.fix_arguments = True
print('args np dir', args.np_dataset)
# no restart directory specified
directory = args.restart  # load directory name
# load latest checkpoint
print('directory', directory)
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
args.cube_grid = False
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
    grid_fn = partial(spherical_grid, level=2)
    sampling_fn = partial(spherical_radial_sampling, rotate=False)
    grid_origin = 0
    # grid_origin = -grid_extent[0]/2
    grid_extent = None
    args.radii_adjust = True

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['density'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           verbose=args.verbose,
                           radii_adjust=args.radii_adjust,)
torch.manual_seed(0)
np.random.seed(0)
# args.restart = None
args.verbose = 0

model = load_model(args, dataset)

# %%
from equiv_dens.utils import orbitals
sample = dataset.get_properties([17])

dpm = orbitals.calc_dipole_moment(sample)['dipole_moment']

center_of_mass = torch.sum(sample['batch_positions'] * sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
                 / torch.sum(sample['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)

print(torch.sum(sample['batch_atom_numbers'].unsqueeze(-1) > 0, dim=1, keepdim=True))  
print('center', center_of_mass)
print('dpm + center', dpm + center_of_mass)
print('dpm', torch.sum(dpm**2))
print('atom types', sample['atom_numbers'][0])
print('pos', sample['positions'])

# %%
torch.no_grad()
dens_model = torch.nn.Sequential(model.density_repr_model, model.property_models['density'])
idx = 17
sample = dataset.get_properties([idx])
# print('sample grid shape', sample['coords'].shape)
# repr = dens_model(sample)
# print('charge', torch.sum(repr['atom_numbers'], dim=1))
# print('density shape', repr['density'].shape)
# print('density integral', torch.sum(repr['density'] * repr['coord_weights']))
# print('sample density integral', torch.sum(sample['density'] * sample['coord_weights']))
# print('positions', sample['positions'])

fname = 'datasets/resorcinol_pred_dens.cube'
#ase.io.cube.write_cube(f, mol[0], data=repr['density'].detach().cpu().numpy().squeeze())
#ase.io.cube.write_cube(f, mol[0], data=np.ones((50, 50, 50)))
#data = np.ones((50, 50, 50))
meta = {'atoms': []}
for i in range(sample['positions'].shape[1]): 
    pos = utils.angstrom_to_bohr(sample['positions'])
    meta['atoms'].append((sample['atom_numbers'].squeeze()[i], pos.squeeze()[i].tolist()))
print('atoms', meta['atoms'])
# density_cube = repr['density'].squeeze().detach().cpu().numpy().reshape(args.cube_size, args.cube_size, args.cube_size)
# print('density cube shape', density_cube.shape)
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
# print('grid_weights', repr['coord_weights'])
lattice = (max_coords - min_coords).detach().cpu().numpy()
print('lattice', lattice)
step_size = lattice/torch.tensor([args.cube_size, args.cube_size, args.cube_size])
print('step_size', step_size)
meta['org'] = min_coords.tolist()
meta['xvec'] = [step_size[0], 0, 0]
meta['yvec'] = [0, step_size[1], 0]
meta['zvec'] = [0, 0, step_size[2]]
# cubetools.write_cube(data=density_cube, meta=meta, fname=fname)

# %%
fname = 'datasets/ethanethiol_dens.cube'
data = sample['density'].squeeze().detach().cpu().numpy().reshape(args.cube_size, args.cube_size, args.cube_size)
cubetools.write_cube(data=data, meta=meta, fname=fname)
# sample_df = dataset_df.get_properties([idx])
# fname = 'datasets/resorcinol_df_dens.cube'
# data_df = sample_df['density'].squeeze().detach().cpu().numpy().reshape(args.cube_size, args.cube_size, args.cube_size)
# cubetools.write_cube(data=data_df, meta=meta, fname=fname)
# data_ml = density_cube
