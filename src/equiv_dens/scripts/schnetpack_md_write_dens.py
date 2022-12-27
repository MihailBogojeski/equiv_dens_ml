from schnetpack.md.utils import HDF5Loader
import sys
import os
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling
import equiv_dens.utils.base as utils
from equiv_dens.training.model_loader import load_model
from equiv_dens.utils import orbitals
from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DensityExpansion
import numpy as np
from functools import partial
import argparse
import copy
import ase.io
import equiv_dens.utils.cubetools as cubetools

file = sys.argv[1]
batch = int(sys.argv[2])
every = int(sys.argv[3])

args, hyperparam_args = parse_command_line_arguments(arg_file='ethanol_all_001_md.txt')

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
args.cube_extent = 10
args.cube_size = 50
grid_extent = np.array([args.cube_extent] * 3)
grid_origin = -grid_extent[0]/2
grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                  extent=grid_extent,
                  origin=np.array([grid_origin] * 3))
sampling_fn = cubical_sampling


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
                           cutoff=args.cutoff,
                           projected_density=False,
                           df_loss_weights=args.df_loss_weights)


expansion_model = DensityExpansion(dataset.orbital_basis_num,
                                    radial_coeffs=dataset.radial_coeffs,
                                    expansion_constraint=args.expansion_constraint,
                                    integral_constraint=args.integral_constraint,
                                    integral_scale=args.integral_scale,
                                    softmax_norm=args.softmax_norm,
                                    verbose=args.verbose,
                                    timing=args.timing,
                                    grid_scaling_factor=args.grid_scaling_factor,
                                    )

print(dataset.dtype)
args.md_log_dir = os.path.join(args.log_dir, 'md_logs', args.restart.split('/')[-1])
if args.log_suffix != '':
    args.log_suffix = '_' + args.log_suffix

dens_write_dir = os.path.join(args.md_log_dir, 'densities')
if not os.path.exists(dens_write_dir):
    os.makedirs(dens_write_dir)

log_file = os.path.join(args.md_log_dir, 'simulation' + args.log_suffix + '.hdf5')

data = HDF5Loader(log_file, load_properties=True)
print(data.properties.keys())

print(dataset.dtype)
atoms = {}
for key in data.properties.keys():
    if data.properties[key] is None:
        continue
    print('key', key)
    if key == '_positions':
        atoms_key = 'positions'
        atoms[atoms_key] = data.properties['_positions'][::every] * 10
        print(atoms['positions'].shape)
        print(data.properties['_positions'].shape)
    elif key == '_atomic_numbers':
        atoms_key = 'atom_numbers'
        atoms[atoms_key] = data.properties['_atomic_numbers'][None, None, :].repeat(data.properties['_positions'][::every].shape[0], axis=0)
        print(atoms['atom_numbers'].shape)
        print(data.properties['_atomic_numbers'].shape)
    else:
        atoms_key = key
        atoms[atoms_key] = data.properties[key][::every]
print(atoms['radial_width'].shape)
print(atoms['radial_scale'].shape)
print(atoms['spherical_coeffs'].shape)
        
for j in range(atoms['positions'].shape[0]):
    dens_write_file = os.path.join(dens_write_dir,  'simulation' + args.log_suffix +
                                   '_dens_' + str(j * every) + '.cube')
    mol = {}
    for key in atoms.keys():
        mol[key] = torch.tensor(atoms[key]).to(dataset.dtype).squeeze(1)[[j],batch]
        print('key', key)
        print(mol[key].shape)

    print('mol positions shape', mol['positions'].shape)
    mol['batch_atom_numbers'] = mol['atom_numbers']
    mol['batch_atom_mask'] = mol['atom_numbers'] > 0
    mol['batch_positions'] = mol['positions']
    mol['coords'], mol['coord_weights'] = dataset.sampling_fn(dataset.grid_spec, dataset.density_n_samp,
                                                                mol['atom_numbers'],
                                                                mol['positions'])
    coeffs = orbitals.vector_to_coeffs_dict(mol, dataset.orbital_basis_num, mol['atom_numbers'])
    for key in coeffs:
        mol[key] = coeffs[key]
    print('coords shape', mol['coords'].shape)
    print('weights shape', mol['coord_weights'].shape)

    mol = expansion_model(mol)

    print(mol.keys())
    mol['density'] = mol['density'].view(args.cube_size, args.cube_size, args.cube_size)
    print('density shape', mol['density'].detach().cpu().numpy().squeeze().shape)
    #ase.io.cube.write_cube(f, mol[0], data=np.ones((50, 50, 50)))
    #data = np.ones((50, 50, 50))
    print('positions', mol['positions'])
    meta = {'atoms': []}
    for i in range(mol['positions'].shape[1]): 
        pos = utils.angstrom_to_bohr(mol['positions'])
        meta['atoms'].append((mol['atom_numbers'].squeeze()[i].to(torch.int), pos.squeeze()[i].tolist()))
    print('atoms', meta['atoms'])
    min_coords, _ = torch.min(utils.angstrom_to_bohr(mol['coords'].view(-1, 3)), dim=0)
    max_coords, _ = torch.max(utils.angstrom_to_bohr(mol['coords'].view(-1, 3)), dim=0)
    print('min coords', min_coords)
    print('max coords', max_coords)
    lattice = (max_coords - min_coords).detach().cpu().numpy()
    print('lattice', lattice)
    step_size = lattice/mol['density'].squeeze().shape
    print('step_size', step_size)
    meta['org'] = min_coords.tolist()
    meta['xvec'] = [step_size[0], 0, 0]
    meta['yvec'] = [0, step_size[1], 0]
    meta['zvec'] = [0, 0, step_size[2]]
    cubetools.write_cube(data=mol['density'].detach().cpu().numpy().squeeze(), meta=meta, fname=dens_write_file)
