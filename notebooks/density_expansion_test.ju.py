# %% import numpy as np
import os
from pyscf import gto
from pyscf.dft import numint
from pyscf.lib import param
from datetime import datetime

import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling, \
     CubicalGrid, spherical_grid, spherical_radial_sampling
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from equiv_dens.utils.misc import generate_id

from functools import partial
from argparse import Namespace
from equiv_dens.training import density_errors
import matplotlib.pyplot as plt
import numpy as np
from equiv_dens.training import model_loader

# %load_ext autoreload
# %autoreload 2
# %cd /home/mihail/Documents/workspace/equiv_dens/
# %%
main_args = Namespace()

# main_args.args_file = "args/resorcinol_all_001.txt"
# main_args.args_file = "args/ethanethiol_all_006_test.txt"
main_args.args_file = "args/ethanethiol_df_coeffs_001_test.txt"
# main_args.args_file = "args/water_all_001_test.txt"
# main_args.ref_np_load_file = "datasets/h2o_dynamic_centered.npy"
# main_args.ref_dens_load_file ="datasets/h2o_dynamic_pyscf_def2svp_dft_f_augccpvqzjkfit.npy"
main_args.ref_np_load_file = "datasets/ethanethiol_rand-1000.npy"
main_args.ref_dens_load_file = "datasets/ethanethiol_rand-1000_pyscf_augccpvdz.npy"
# main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
# main_args.save_file = 'ethanethiol_all_006'
# main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
# main_args.save_file = 'resorcinol_all_005'
main_args.res_load_file = 'datasets/ethanethiol_df_coeffs_001_test.pt'
main_args.save_file = 'ethanethiol_df_coeffs_001'
# main_args.res_load_file = 'datasets/water_all_001_test.pt'
# main_args.save_file = 'water_all_001'
main_args.df_error = True
main_args.use_gpu = True
main_args.num_samples = 100
main_args.make_plots = True

df_losses = None
# %%
args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)

# print('type dtype', type(args.dtype))
args.fix_arguments = True
# print('args np dir', args.np_dataset)
args.restart = None
args.pred_radial_coeffs = False

if args.restart is None:
    # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    model_code = generate_id()
    directory = os.path.join(args.save_dir, datetime.utcnow().strftime("%Y-%m-%d_") +
                             model_code)  # generate directory name
    # create directories
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # # write command line arguments to file (useful for reproducibility)
    # with open(os.path.join(directory, 'args.txt'), 'w') as f:
    #     for key in args.__dict__.keys():
    #         # special case for list input
    #         if isinstance(args.__dict__[key], list):
    #             for entry in args.__dict__[key]:
    #                 f.write('--' + key + '=' + str(entry) + "\n")
    #         else:
    #             f.write('--' + key + '=' + str(args.__dict__[key]) + "\n")
    checkpoint = None
    latest_checkpoint = 0
    step = 0
    restore = False
    data_split_indices = None
    # restarts run from latest checkpoint
else:
    # no restart directory specified
    directory = args.restart
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
                # print('loading hyperparam arg', arg)
                setattr(args, arg, getattr(checkpoint['args'], arg))
        else:
            # print('loading all arg', arg)
            setattr(args, arg, getattr(checkpoint['args'], arg))
    restore = True

args.df_weight = 1.0

print('model code:', model_code)

# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
args.use_gpu = False

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

args.verbose = 0
args.use_gpu = False
args.cube_grid = False
args.radii_adjust = True
args.expansion_constraint = None
args.cube_grid = False
if args.cube_grid:
    args.cube_origin = -1.0
    args.cube_extent = 2.0
    args.cube_size = 50
    args.radii_adjust = False
    grid_origin = args.cube_origin
    grid_extent = np.array([args.cube_extent] * 3)
    grid_fn = partial(cubical_grid, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                      extent=grid_extent,
                      origin=np.array([grid_origin] * 3))
    sampling_fn = cubical_sampling
else:
    args.spherical_grid_level = 1
    grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
    sampling_fn = partial(spherical_radial_sampling, rotate=False)
    grid_origin = 0
    grid_extent = None
    rotate = False

required_properties = ['energy', 'forces', 'df_coeffs', 'density', 'dipole_moment']

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file

print('np file', main_args.ref_np_load_file)
print('dens file', main_args.ref_dens_load_file)
print('dens file', args.dens_dataset_test)
print('np file', args.np_dataset_test)
print('dens file', args.dens_dataset_test)
dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000000000000,
                           required_properties=required_properties,
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           # pyscf_grid=args.pyscf_grid,
                           pyscf_rotate=rotate,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
                           radii_adjust=args.radii_adjust,
                           )
print('dataset length', len(dataset))
print('sample pos shape', dataset.get_properties([0])['positions'].shape)
print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
print('num samples', main_args.num_samples)

# %%
df_losses = None
if main_args.df_error:
    # df_losses = {'dens_mae': [], 'dens_rmse': [], 'dpm_mae': [],
    #              'dpm_rmse': [], 'kl_loss': [],
    #              'dpm_mag': [], 'dpm_ang': [], 'lda_mae': [],
    #              'coulomb': [], 'coulomb_int': [], 'mae_23': [], 'mae_43': [],
    #              'lda_23_mae': [], 'dpm_coord_rmse': [],
    #              'dpm_pos_coord_rmse': [], 'dpm_neg_coord_rmse': [],
    #              'dpm_int_rmse': []}

    dataset_df = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                                  orbitals_path=args.orbitals_file,
                                  density_n_samp=10000000000000000000000,
                                  required_properties=required_properties,
                                  center_positions=False,
                                  radial_coeffs_file=args.radial_coeffs_file,
                                  dtype=args.dtype,
                                  grid_fn=grid_fn,
                                  pyscf_grid=args.pyscf_grid,
                                  pyscf_rotate=rotate,
                                  sampling_fn=sampling_fn,
                                  grid_extent=grid_extent,
                                  grid_origin=grid_origin,
                                  cutoff=args.cutoff,
                                  df_loss_weights=args.df_loss_weights,
                                  projected_density=True,
                                  radii_adjust=args.radii_adjust,
                                  )
# %%
res_dataset = torch.load(main_args.res_load_file, map_location='cpu')
print(res_dataset.keys())
# %%
# Hirshfeld analysis
# %%
model = model_loader.load_model(args, dataset)
samp = dataset.get_properties([2])
samp_df = dataset_df.get_properties([2])

res = model(samp)

print('samp df_coeffs', samp_df['df_coeffs'].shape)
print('res df_coeffs', res['df_coeffs'].shape)

# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
df_coeffs_dict = {key: res[key] for key in res.keys()}

print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))

df_sph_coeffs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']},
                                               dataset.orbital_basis_num, samp_df['atom_numbers'], radial_coeffs=False,
                                               convert_to_equiv_dens=True)

df_coeffs_dict['spherical_coeffs'] = df_sph_coeffs['spherical_coeffs']
print('df sph coeffs L 1 H 1 converted', df_coeffs_dict['spherical_coeffs'][0][1, 1])

# df_sph_coeffs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']},
#                                                dataset.orbital_basis_num, samp_df['atom_numbers'], radial_coeffs=False,
#                                                convert_to_equiv_dens=False)
#
# df_coeffs_dict['spherical_coeffs'] = df_sph_coeffs['spherical_coeffs']
# print('df sph coeffs L 1 H 1', df_coeffs_dict['spherical_coeffs'][0][1, 1])


df_dens = model.property_models['density'](df_coeffs_dict)
print('df density integral', torch.sum(df_dens['density'] * df_dens['coord_weights'], dim=1))
print('df integral', torch.sum(samp_df['density'] * samp_df['coord_weights'], dim=1))
print('dens integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
print('df pred vs df diff', torch.sum(torch.abs(df_dens['density'] - samp_df['density']) * samp_df['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
print('df pred vs dens diff', torch.sum(torch.abs(df_dens['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
print('df vs dens diff', torch.sum(torch.abs(samp_df['density'] - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp['atom_numbers']))
# %%
def sample_projected_density(atoms, df_coeffs, dataset):
    df_coeffs = df_coeffs.detach().cpu().numpy()
    sample_coords = atoms['coords']
    scaled_sample_coords = atoms['coords'].detach().cpu().numpy() / param.BOHR  # convert Angstrom grid to Bohr
    dens = torch.zeros((sample_coords.shape[0], sample_coords.shape[1]))
    # mol_start = time.time()
    # print('c, i', c, i)

    atom = [(int(atoms['batch_atom_numbers'][0, i].detach().cpu().numpy()),
            atoms['batch_positions'][0, i].detach().cpu().numpy()) for i in range(atoms['batch_positions'].shape[1])] 
    #print(atom)
    mol = gto.M(atom=atom, basis=dataset.density_fitting['auxbasis'])
    mol_dict = mol.pack()
    df_coeffs_split = orbitals.split_df_coeffs(mol_dict['atom'], df_coeffs, dataset.orbital_basis_size)
    #print('df_coeffs split', df_coeffs_split)

    # ao_start = time.time()
    ao = numint.eval_ao(mol, scaled_sample_coords[0])
    # print('ao time', time.time() - ao_start)
    # rho_start = time.time()
    # print('df coeff', df_coeff.shape)
    # print('ao shape', ao.shape)
    rho = 0
    for j in range(df_coeffs.shape[0]):
        rho += np.einsum('ij,j->i', ao, df_coeffs[j])
    # print('rho time', time.time() - rho_start)
    dens = torch.from_numpy(rho)
    # print('mol_time', time.time() - mol_start)
    return dens
# %%
orbital_dict = orbitals.combine_orbital_basis(dataset.orbital_basis_num, 5)[0]
print('orbital_dict', orbital_dict)
samp_df = dataset_df.get_properties([0])
samp = dataset.get_properties([0])
#print(dataset_df.atoms['atom_numbers'][0])
#print('dataset_df orig coeffs', dataset_df.density_fitting['df_coeffs'][0])
df_coeffs = samp_df['df_coeffs'].detach().cpu().numpy()
print(dataset.orbital_basis_size)
atom = [(int(samp_df['batch_atom_numbers'][0, i].detach().cpu().numpy()),
        samp_df['batch_positions'][0, i].detach().cpu().numpy())
        for i in range(samp_df['batch_positions'].shape[1])] 
#print(atom)
df_coeffs_split = orbitals.split_df_coeffs(atom, df_coeffs.squeeze(), dataset_df.orbital_basis_size)
max_coeffs_len = max([len(coeffs[1]) for coeffs in df_coeffs_split])
#print(dataset.orbital_basis_num)
orbital_dict = orbitals.combine_orbital_basis(dataset.orbital_basis_num, 5)[0]
print('orbital_dict', orbital_dict)
dens_errors = []
coeff_errors = []
dpm_errors = []

masks_pos = {key: 0 for key in dataset.orbital_basis_size} 
max_orbitals = orbital_dict[8]
print('max orbitals', max_orbitals)
for L in range(2):
    nc = 2 * L + 1
    max_coeffs = max_orbitals[L][1]
    for i in range(1, max_coeffs + 1):
        for z in masks_pos.keys():
            if L >= len(orbital_dict[z]) or i >= orbital_dict[z][L][1] + 1:
                continue
            masks_pos[z] += nc
            print('L', L, 'max coeffs', max_coeffs, 'i', i, 'z', z, 'masks_pos', masks_pos[z])
        mask_coeffs = []
        for z, atom_coeffs in df_coeffs_split:
            atom_coeffs = torch.Tensor(atom_coeffs)
            mask = torch.zeros_like(atom_coeffs)
            mask[:masks_pos[z]] = 1
            # print('z', z, 'mask pos', masks_pos[z], 'mask', mask)
            mask_coeffs.append(atom_coeffs * mask)
    coeffs = torch.cat(mask_coeffs).unsqueeze(0)
    dens_coeff = sample_projected_density(samp_df, coeffs, dataset_df)
    dens_err = torch.sum(torch.abs(dens_coeff - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp_df['atom_numbers'])
    coeff_err = torch.mean(torch.abs(coeffs - samp_df['df_coeffs']))
    new_samp = {key: samp_df[key] for key in samp_df.keys()}
    new_samp['density'] = dens_coeff
    dpm = orbitals.calc_dipole_moment(new_samp)['dipole_moment']
    dpm_err = 4.8 * torch.norm(dpm - samp['dipole_moment'])
    dens_errors.append(float(dens_err))
    coeff_errors.append(float(coeff_err))
    dpm_errors.append(float(dpm_err))
    print('len dpm errors', len(dpm_errors))
print('df dens errors', dens_errors)
print('df coeff errors', coeff_errors)
print('df dpm errors', dpm_errors)
# %%
samp_df = dataset_df.get_properties([2])
samp = dataset.get_properties([2])
#print(dataset_df.atoms['atom_numbers'][0])
#print('dataset_df orig coeffs', dataset_df.density_fitting['df_coeffs'][0])
df_coeffs = samp_df['df_coeffs'].detach().cpu().numpy()
print(dataset.orbital_basis_size)
atom = [(int(samp_df['batch_atom_numbers'][0, i].detach().cpu().numpy()),
        samp_df['batch_positions'][0, i].detach().cpu().numpy())
        for i in range(samp_df['batch_positions'].shape[1])] 
#print(atom)
df_coeffs_split = orbitals.split_df_coeffs(atom, df_coeffs.squeeze(), dataset_df.orbital_basis_size)
max_coeffs_len = max([len(coeffs[1]) for coeffs in df_coeffs_split])
#print(dataset.orbital_basis_num)
orbital_dict = orbitals.combine_orbital_basis(dataset.orbital_basis_num, 5)[0]
print('orbital_dict', orbital_dict)
dens_errors = []
coeff_errors = []
dpm_errors = []
print('density shape', samp['density'].shape)
print('coord weights shape', samp['coord_weights'].shape)

masks_pos = {key: 0 for key in dataset.orbital_basis_size}
max_orbitals = orbital_dict[16]
masks = {}
for z, atom_coeffs in df_coeffs_split:
    if z not in masks:
        masks[z] = torch.zeros_like(torch.tensor(atom_coeffs))
print('max orbitals', max_orbitals)
eval_L = [0, 1, 2, 3, 4, 5]
for L in range(max(eval_L) + 1):
    nc = 2 * L + 1
    max_coeffs = max_orbitals[L][1]
    for i in range(1, max_coeffs + 1):
        for z in masks_pos.keys():
            if L >= len(orbital_dict[z]) or i >= orbital_dict[z][L][1] + 1:
                continue
            if L in eval_L:
                masks[z][masks_pos[z]:masks_pos[z] + nc] = 1
            masks_pos[z] += nc
            print('L', L, 'max coeffs', max_coeffs, 'i', i, 'z', z, 'masks_pos', masks_pos[z])

mask_coeffs = []
for z, atom_coeffs in df_coeffs_split:
    atom_coeffs = torch.Tensor(atom_coeffs)
    mask_coeffs.append(atom_coeffs * masks[z])

coeffs = torch.cat(mask_coeffs).unsqueeze(0)
dens_coeff = sample_projected_density(samp_df, coeffs, dataset_df)
dens_err = torch.sum(torch.abs(dens_coeff - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp_df['atom_numbers'])
coeff_err = torch.mean(torch.abs(coeffs - samp_df['df_coeffs']))
new_samp = {key: samp_df[key] for key in samp_df.keys()}
new_samp['density'] = dens_coeff
dpm = orbitals.calc_dipole_moment(new_samp)['dipole_moment']
dpm_err = 4.8 * torch.norm(dpm - samp['dipole_moment'])
dens_errors.append(float(dens_err))
coeff_errors.append(float(coeff_err))
dpm_errors.append(float(dpm_err))
print('len dpm errors', len(dpm_errors))
print('df dens errors', dens_errors)
print('df coeff errors', coeff_errors)
print('df dpm errors', dpm_errors)
# %%
# difference between dens_coeff and df_dens
new_df_coeffs_dict = {key: df_coeffs_dict[key] for key in df_coeffs_dict.keys()}
new_df_coeffs_dict['spherical_coeffs'] = []
for i in range(len(df_coeffs_dict['spherical_coeffs'])):
    new_df_coeffs_dict['spherical_coeffs'].append({})
    for key in df_coeffs_dict['spherical_coeffs'][i].keys():
        if key[1] == 1:
            mask = torch.tensor([1, 1, 1]).view(1, 1, -1, 1)
            # print('df coefs before', df_coeffs_dict['spherical_coeffs'][i][key])
            new_df_coeffs_dict['spherical_coeffs'][i][key] = mask * df_coeffs_dict['spherical_coeffs'][i][key]
            new_df_coeffs_dict['spherical_coeffs'][i][key] = new_df_coeffs_dict['spherical_coeffs'][i][key][:, :, [1, 2, 0], :]
        else:
            mask = 1
            new_df_coeffs_dict['spherical_coeffs'][i][key] = mask * df_coeffs_dict['spherical_coeffs'][i][key]
        if key[1] % 2 == 1:
            new_df_coeffs_dict['spherical_coeffs'][i][key] = -1 * new_df_coeffs_dict['spherical_coeffs'][i][key]

print('new df sph coeffs L 1 H 1', new_df_coeffs_dict['spherical_coeffs'][0][1, 1])

df_dens = model.property_models['density'](new_df_coeffs_dict)

print('dens diff', torch.sum(torch.abs(dens_coeff - df_dens['density']) * df_dens['coord_weights'], dim=1))
print('min max coeff_dens', torch.min(dens_coeff), torch.max(dens_coeff))
print('max exp dens, coeff', torch.min(df_dens['density']), torch.max(df_dens['density']))
print('dens diff vs samp_df', torch.sum(torch.abs(samp_df['density'] - df_dens['density']) * df_dens['coord_weights'], dim=1))
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib notebook
sample = 2
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
dens = df_dens['density'].detach().cpu().numpy().flatten()
x, y = np.meshgrid(np.arange(50), np.arange(50))
x_f = x.flatten()
y_f = y.flatten()
z_f = np.zeros_like(y_f) + 19
idx = np.ravel_multi_index((x_f, y_f, z_f), (50, 50, 50))

#X, Y, Z = np.meshgrid(np.arange(50) + 38, np.arange(50) + 38, np.arange(50) + 38)
#X = X.flatten()
#Y = Y.flatten()
#Z = Z.flatten()
#ndices = np.ravel_multi_index((X, Y, Z), (125, 125, 125))

print('idx', idx)
dens2d = dens[idx]
dens2d = dens2d.reshape(50, 50)

vmin = np.min(dens2d)
vmax = np.max(dens2d)
c1 = ax1.contourf(x, y, dens2d, vmin=vmin, vmax=vmax, levels=100)
ax1.set_aspect('equal')


p_dens = (dens_coeff.detach().cpu().numpy().flatten()[idx]).reshape(50, 50)

ax2 = fig.add_subplot(122)
c2 = ax2.contourf(x, y, p_dens, vmin=vmin, vmax=vmax, levels=100)
ax2.set_aspect('equal')

fig.colorbar(c2, ax=[ax1, ax2])
plt.show()
