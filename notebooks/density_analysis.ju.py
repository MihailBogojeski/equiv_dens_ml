# %%
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
main_args.args_file = "args/ethanethiol_all_006_test.txt"
# main_args.args_file = "args/ethanethiol_df_coeffs_001_test.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
main_args.save_file = 'ethanethiol_all_006'
# main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
# main_args.save_file = 'resorcinol_all_005'
# main_args.res_load_file = 'datasets/ethanethiol_df_coeffs_001_test.pt'
# main_args.save_file = 'ethanethiol_df_coeffs_001'
main_args.df_error = True
main_args.use_gpu = False 
main_args.num_samples = 100
main_args.make_plots = True

df_losses = None

# %%
args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)

# print('type dtype', type(args.dtype))
args.fix_arguments = True
# print('args np dir', args.np_dataset)
# args.restart = None
# args.pred_radial_coeffs = False

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
    # no restart directory specifie
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
print('args use gpu', args.use_gpu)
args.cube_grid = False
args.radii_adjust = True
args.expansion_constraint = None
if args.cube_grid:
    args.cube_origin = -0.25
    args.cube_extent = 0.5
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

dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
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
                           projected_density=args.projected_density,
                           )
print('dataset length', len(dataset))
print('sample pos shape', dataset.get_properties([0])['positions'].shape)
print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
print('num samples', main_args.num_samples)
print('args use gpu', args.use_gpu)

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
                                  )

# %%
res_dataset = torch.load(main_args.res_load_file, map_location='cpu')
print(res_dataset.keys())

# %%
# Hirshfeld analysis

# %%
# Hartree potential
num_pot = 5000
for idx in range(min(len(dataset), main_args.num_samples)):
    sample = dataset.get_properties([idx])
    sample_df = dataset_df.get_properties([idx])
    grid_idxs = torch.randint(res_dataset['density'].shape[1], (num_pot,))
    subsamp_grid = sample['coords']
    subsamp_weights = sample['coord_weights']
    coord_dists = torch.norm(sample['coords'].unsqueeze(1) - subsamp_grid.unsqueeze(2), dim=-1) + 1e-10
    # print('avg coord dists', torch.mean(coord_dists, dim=2))
    # print('subsamp weights', subsamp_weights)
    hartree_pot_ml = torch.sum((res_dataset['density'][[idx]] * sample['coord_weights']).unsqueeze(1)
                               / coord_dists, dim=2)
    hartree_pot_df = torch.sum((sample_df['density'] * sample['coord_weights']).unsqueeze(1)
                               / coord_dists, dim=2)
    hartree_pot = torch.sum((sample['density'] * sample['coord_weights']).unsqueeze(1)
                            / coord_dists, dim=2)
    # print('hartree_pot', hartree_pot[:, :10])
    # print('hartree_pot_ml', hartree_pot_ml[:, :10])
    # print('hartree_pot_df', hartree_pot_df[:, :10])
    hartree_err_ml = torch.abs(torch.sum(hartree_pot_ml * subsamp_weights) - torch.sum(hartree_pot * subsamp_weights))
    hartree_err_df = torch.abs(torch.sum(hartree_pot_df * subsamp_weights) - torch.sum(hartree_pot * subsamp_weights))
    print('hatree_pot errors ML:', hartree_err_ml, 
          'DF:', hartree_err_df,
          'ML/DF:', hartree_err_ml / hartree_err_df)

# %%
# Hartree potential
num_pot = 30000
hartree_pot_errors = []
for idx in range(min(len(dataset), main_args.num_samples)):
    sample = dataset.get_properties([idx])
    sample_df = dataset_df.get_properties([idx])
    grid_idxs = torch.LongTensor(np.random.choice(np.arange(res_dataset['density'].shape[1]), size=(num_pot,), replace=False))
    subsamp_grid = sample['coords'][:, grid_idxs]
    subsamp_weights = sample['coord_weights'][:, grid_idxs]
    coord_dists = torch.norm(sample['coords'].unsqueeze(1) - subsamp_grid.unsqueeze(2), dim=-1) + 1e-10
    # print('avg coord dists', torch.mean(coord_dists, dim=2))
    # print('subsamp weights', subsamp_weights)
    hartree_pot_ml = torch.sum((res_dataset['density'][[idx]] * sample['coord_weights']).unsqueeze(1)
                               / coord_dists, dim=2)
    hartree_pot_df = torch.sum((sample_df['density'] * sample['coord_weights']).unsqueeze(1)
                               / coord_dists, dim=2)
    hartree_pot = torch.sum((sample['density'] * sample['coord_weights']).unsqueeze(1)
                            / coord_dists, dim=2)
    # print('hartree_pot', hartree_pot[:, :10])
    # print('hartree_pot_ml', hartree_pot_ml[:, :10])
    # print('hartree_pot_df', hartree_pot_df[:, :10])
    hartree_err_ml = torch.abs(torch.sum(hartree_pot_ml * subsamp_weights) - torch.sum(hartree_pot * subsamp_weights))
    hartree_err_df = torch.abs(torch.sum(hartree_pot_df * subsamp_weights) - torch.sum(hartree_pot * subsamp_weights))
    hartree_pot_errors.append({"ML": hartree_err_ml, "DF": hartree_err_df})
    print('hatree_pot errors ML:', hartree_err_ml,
          'DF:', hartree_err_df,
          'ML/DF:', hartree_err_ml / hartree_err_df)

# %%
model = model_loader.load_model(args, dataset)
samp = dataset.get_properties([0])
samp_df = dataset_df.get_properties([0])

res = model(samp)

print('samp df_coeffs', samp_df['df_coeffs'].shape)
print('res df_coeffs', res['df_coeffs'].shape)

# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
df_coeffs_dict = {key: res[key] for key in res.keys()}

print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))

df_sph_coeffs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']}, dataset.orbital_basis_num, samp_df['atom_numbers'], radial_coeffs=False)

df_coeffs_dict['spherical_coeffs'] = df_sph_coeffs['spherical_coeffs']


df_dens = model.property_models['density'](df_coeffs_dict)
print('df density integral', torch.sum(df_dens['density'] * df_dens['coord_weights'], dim=1))

# %%
# Error correlations

hartree_pot_errs = {'df': [hartree_pot_errors[i]['DF'] for i in range(len(hartree_pot_errors))],
                    'ml': [hartree_pot_errors[i]['ML'] for i in range(len(hartree_pot_errors))]}
df_losses = np.load('datasets/' + main_args.save_file + '_df_losses.npy', allow_pickle=True).item()
res_losses = np.load('datasets/' + main_args.save_file + '.npy', allow_pickle=True).item()
print('df_losses', df_losses.keys())

fig, axs = plt.subplots(2, 3, figsize=(8, 5))


for i, loss in enumerate(['dens_mae', 'lda_mae']):
    axs[0, i].scatter(res_losses[loss], res_losses['dpm_rmse'])
    axs[1, i].scatter(df_losses[loss], df_losses['dpm_rmse'])

axs[0, 2].scatter(hartree_pot_errs['ml'], res_losses['dpm_rmse'])
axs[1, 2].scatter(hartree_pot_errs['df'], df_losses['dpm_rmse'])

axs[0, 0].set_ylabel('ML dipole moment RMSE')
axs[0, 0].set_xlabel('ML density MAE')
axs[0, 1].set_xlabel('ML LDA MAE')
axs[0, 2].set_xlabel('ML hartree pot MAE')

axs[0, 0].set_title('Corr = {:.4f}'.format(np.corrcoef(res_losses['dpm_rmse'], res_losses['dens_mae'])[0, 1]))
axs[0, 1].set_title('Corr = {:.4f}'.format(np.corrcoef(res_losses['dpm_rmse'], res_losses['lda_mae'])[0, 1]))
axs[0, 2].set_title('Corr = {:.4f}'.format(np.corrcoef(res_losses['dpm_rmse'], hartree_pot_errs['ml'])[0, 1]))

axs[1, 0].set_ylabel('DF dipole moment RMSE')
axs[1, 0].set_xlabel('DF density MAE')
axs[1, 1].set_xlabel('DF LDA MAE')
axs[1, 2].set_xlabel('DF hartree pot MAE')

axs[1, 0].set_title('Corr = {:.4f}'.format(np.corrcoef(df_losses['dpm_rmse'], df_losses['dens_mae'])[0, 1])) axs[1, 1].set_title('Corr = {:.4f}'.format(np.corrcoef(df_losses['dpm_rmse'], df_losses['lda_mae'])[0, 1])) axs[1, 2].set_title('Corr = {:.4f}'.format(np.corrcoef(df_losses['dpm_rmse'], hartree_pot_errs['df'])[0, 1]))
for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig('figures/ethanethiol_all_006_losses_vs_dpm_loss_correlation.png', dpi=300)
print('mean ml hartree loss', np.mean(hartree_pot_errs['ml']))
print('mean df hartree loss', np.mean(hartree_pot_errs['df']))

# %%
model = model_loader.load_model(args, dataset)
samp = dataset.get_properties([0])
samp_df = dataset_df.get_properties([0])

res = model(samp)

print('samp df_coeffs', samp_df['df_coeffs'].shape)
print('res df_coeffs', res['df_coeffs'].shape)

# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
df_coeffs_dict = {key: res[key] for key in res.keys()}

print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))

df_sph_coeffs = orbitals.vector_to_coeffs_dict({'spherical_coeffs': samp_df['df_coeffs']}, dataset.orbital_basis_num, samp_df['atom_numbers'], radial_coeffs=False)

df_coeffs_dict['spherical_coeffs'] = df_sph_coeffs['spherical_coeffs']


df_dens = model.property_models['density'](df_coeffs_dict)
print('df density integral', torch.sum(df_dens['density'] * df_dens['coord_weights'], dim=1))
print('df integral', torch.sum(samp_df['density'] * samp_df['coord_weights'], dim=1))
print('dend integral', torch.sum(samp['density'] * samp['coord_weights'], dim=1))
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
print('df_pred', res_dataset['df_coeffs'])
print('df dens', samp_df['df_coeffs'])
print('error', torch.mean(torch.abs(res_dataset['df_coeffs'][[0]] - samp_df['df_coeffs'])))
print('res positions', res_dataset['positions'][[0]])
print('df positions', samp_df['positions'])

# %%
samp_df = dataset_df.get_properties([0])
samp = dataset.get_properties([0])
#print(dataset_df.atoms['atom_numbers'][0])
#print('dataset_df orig coeffs', dataset_df.density_fitting['df_coeffs'][0])
dens_errors = []
coeff_errors = []
dpm_errors = []
pred_errors = []
a = torch.linspace(0, 1, 100)
coeffs = [samp_df['df_coeffs'] * a[i] + res_dataset['df_coeffs'][[0]] * (1 - a[i]) for i in range(len(a))]
for coeff in coeffs:
    dens_coeff = sample_projected_density(samp_df, coeff, dataset_df)
    dens_err = torch.sum(torch.abs(dens_coeff - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp_df['atom_numbers'])
    coeff_err = torch.mean(torch.abs(coeff - samp_df['df_coeffs']))
    new_samp = {key: samp_df[key] for key in samp_df.keys()}
    new_samp['density'] = dens_coeff
    dpm = orbitals.calc_dipole_moment(new_samp)['dipole_moment']
    dpm_err = 4.8 * torch.norm(dpm - samp['dipole_moment'])
    dens_errors.append(float(dens_err))
    coeff_errors.append(float(coeff_err))
    dpm_errors.append(float(dpm_err))
print('df dens errors', dens_errors)
print('df coeff errors', coeff_errors)
print('df dpm errors', dpm_errors)

# %%
# normalize dens_errors from 0 to 1
dens_errors_norm = (dens_errors - np.min(dens_errors)) / (np.max(dens_errors) - np.min(dens_errors))
coeff_errors_norm = (coeff_errors - np.min(coeff_errors)) / (np.max(coeff_errors) - np.min(coeff_errors))
dpm_errors_norm = (dpm_errors - np.min(dpm_errors)) / (np.max(dpm_errors) - np.min(dpm_errors))

fig = plt.figure()
plt.plot(a, dens_errors_norm, label='density error')
plt.plot(a, coeff_errors_norm, label='coeffs error')
plt.plot(a, dpm_errors_norm, label='dpm error')
plt.xlabel('Interpolation from predidicted to DF coeffs')
plt.ylabel('Normalized error')
plt.legend()
plt.show()

# %%
# Interpolation between prediction and target
samp_df = dataset_df.get_properties([0])
samp = dataset.get_properties([0])
#print(dataset_df.atoms['atom_numbers'][0])
#print('dataset_df orig coeffs', dataset_df.density_fitting['df_coeffs'][0])
dens_errors = []
coeff_errors = []
dpm_errors = []
pred_errors = []
a = torch.logspace(-10, 0, 20)
print(a)
a = torch.flip(1 - a, [0])
print(a)

# %%
coeffs = [samp_df['df_coeffs'] * a[i] + res_dataset['df_coeffs'][[0]] * (1 - a[i]) for i in range(len(a))]
for coeff in coeffs:
    dens_coeff = sample_projected_density(samp_df, coeff, dataset_df)
    dens_err = torch.sum(torch.abs(dens_coeff - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp_df['atom_numbers'])
    coeff_err = torch.mean(torch.abs(coeff - samp_df['df_coeffs']))
    new_samp = {key: samp_df[key] for key in samp_df.keys()}
    new_samp['density'] = dens_coeff
    dpm = orbitals.calc_dipole_moment(new_samp)['dipole_moment']
    dpm_err = 4.8 * torch.norm(dpm - samp['dipole_moment'])
    dens_errors.append(float(dens_err))
    coeff_errors.append(float(coeff_err))
    dpm_errors.append(float(dpm_err))
print('df dens errors', dens_errors)
print('df coeff errors', coeff_errors)
print('df dpm errors', dpm_errors)

# %%
# normalize dens_errors from 0 to 1
dens_errors_norm = (dens_errors - np.min(dens_errors)) / (np.max(dens_errors) - np.min(dens_errors))
coeff_errors_norm = (coeff_errors - np.min(coeff_errors)) / (np.max(coeff_errors) - np.min(coeff_errors))
dpm_errors_norm = (dpm_errors - np.min(dpm_errors)) / (np.max(dpm_errors) - np.min(dpm_errors))

fig = plt.figure()
plt.plot(torch.log(a), dens_errors_norm, label='density error')
plt.plot(torch.log(a), coeff_errors_norm, label='coeffs error')
plt.plot(torch.log(a), dpm_errors_norm, label='dpm error')
plt.xlabel('Interpolation from predidicted to DF coeffs')
plt.ylabel('Normalized error')
plt.legend()
plt.show()

# %%
# Check which coefficients are important for the dipole moment 
samp_df = dataset_df.get_properties([0])
samp = dataset.get_properties([0])
df_coeffs = samp_df['df_coeffs'].detach().cpu().numpy()

atom = [(int(samp_df['batch_atom_numbers'][0, i].detach().cpu().numpy()),
        samp_df['batch_positions'][0, i].detach().cpu().numpy())
        for i in range(samp_df['batch_positions'].shape[1])] 
#print(atom)
df_coeffs_split = orbitals.split_df_coeffs(atom, df_coeffs.squeeze(), dataset_df.orbital_basis_size)
max_coeffs_len = max([len(coeffs[1]) for coeffs in df_coeffs_split])

# %%
dens_errors = []
coeff_errors = []
dpm_errors = []

for i in range(1, max_coeffs_len):
    mask_coeffs = []
    for _, atom_coeffs in df_coeffs_split:
        atom_coeffs = torch.Tensor(atom_coeffs)
        mask = torch.zeros_like(atom_coeffs)
        mask[:i] = 1
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
print('df dens errors', dens_errors)
print('df coeff errors', coeff_errors)
print('df dpm errors', dpm_errors)

# %%
# normalize dens_errors from 0 to 1
dens_errors_norm = (dens_errors - np.min(dens_errors)) / (np.max(dens_errors) - np.min(dens_errors))
coeff_errors_norm = (coeff_errors - np.min(coeff_errors)) / (np.max(coeff_errors) - np.min(coeff_errors))
dpm_errors_norm = (dpm_errors - np.min(dpm_errors)) / (np.max(dpm_errors) - np.min(dpm_errors))

fig = plt.figure()
plt.plot(dens_errors_norm, label='density error')
plt.plot(coeff_errors_norm, label='coeffs error')
plt.plot(dpm_errors_norm, label='dpm error')
plt.xlabel('Number of coefficients included per atom')
plt.ylabel('Normalized error')
plt.legend()
plt.show()

# %%
print(dataset.orbital_basis_size)
#print(dataset.orbital_basis_num)
orbital_dict = orbitals.combine_orbital_basis(dataset.orbital_basis_num, 5)[0]
print('orbital_dict', orbital_dict)
dens_errors = []
coeff_errors = []
dpm_errors = []

masks_pos = {key: 0 for key in dataset.orbital_basis_size} 
max_orbitals = orbital_dict[16]
print('max orbitals', max_orbitals)
for L in range(6):
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
# normalize dens_errors from 0 to 1
dens_errors_norm = (dens_errors - np.min(dens_errors)) / (np.max(dens_errors) - np.min(dens_errors))
coeff_errors_norm = (coeff_errors - np.min(coeff_errors)) / (np.max(coeff_errors) - np.min(coeff_errors))
dpm_errors_norm = (dpm_errors - np.min(dpm_errors)) / (np.max(dpm_errors) - np.min(dpm_errors))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(dens_errors_norm, label='density error')
axs[0].plot(coeff_errors_norm, label='coeffs error')
axs[0].plot(dpm_errors_norm, label='dpm error')
axs[0].set_xlabel('Number of coefficients included')
axs[0].set_ylabel('Normalized error')
axs[0].set_xticks(ticks=[14, 26, 36, 41, 44, 46], labels = ['L=0', '1', '2', '3', '4', '5'])
axs[1].plot(dens_errors_norm, label='density error')
axs[1].plot(coeff_errors_norm, label='coeffs error')
axs[1].plot(dpm_errors_norm, label='dpm error')
axs[1].set_yscale('log')
axs[1].set_xlabel('Number of coefficients included')
axs[1].set_ylabel('Normalized error (log)')
axs[1].set_xticks(ticks=[14, 26, 36, 41, 44, 46], labels = ['L=0', '1', '2', '3', '4', '5'])
fig.text(0.5, -0.15,
         'These figures show how the errors of the density coefficients, density and dipole\n' + 
         'moment (normalized to a range of 0 to 1) change as more basis functions are included,\n' +
         'starting from the ones with zero angular degree to the basis functions of highest \n' +
         'angular degree. We see that the dipole moment errors converge faster than the density errors,\n' +
         'already achieving minimal dipole moment errors, while the absolute density error is still\n'
         + 'larger than 1%.', ha='center', va='center', fontsize=12)
plt.tight_layout()
plt.legend()
plt.savefig('figures/ethanethiol_errors_by_order.pdf', dpi=300)
plt.show()

# %%
#print(dataset.orbital_basis_num)
orbital_dict = orbitals.combine_orbital_basis(dataset.orbital_basis_num, 5)[0]
print('orbital_dicts', orbital_dict)
dens_errors = []
coeff_errors = []
dpm_errors = []
min_exp = -1
max_exp = -1
for key in dataset.radial_coeffs:
    for exp_cf, _ in dataset.radial_coeffs[key]:
        if min_exp < 0:
            min_exp = float(exp_cf)
        min_exp = min(min_exp, exp_cf)
        max_exp = max(max_exp, exp_cf)

print('min_exp', float(min_exp), 'max_exp', float(max_exp))
# exp_space = torch.linspace(float(min_exp), float(max_exp), 100)
# exp_space = exp_space.flip(0) - 0.0001
rad_width = []
for z in samp_df['atom_numbers'].flatten():
    count = 0
    for z, orb, L in orbital_dict[float(z)]:
        for _ in range(orb):
            rad_width.extend([float(dataset.radial_coeffs[float(z)][count][0])] * ((2 * L) + 1))
            # print('count', count, 'len radial width', len(rad_width))
            count += 1
print('radial width', rad_width)
print('len radial width', len(rad_width))
rad_width = torch.Tensor(rad_width)
exp_space = torch.sort(torch.unique(rad_width))[0]
exp_space = exp_space.flip(0) - 0.0001
print('exp_space', exp_space)
for i in range(exp_space.shape[0]):
    exp_cf = exp_space[i]
    mask_coeffs = (rad_width >= exp_cf).unsqueeze(0)
    coeffs = samp_df['df_coeffs'] * mask_coeffs 
    dens_coeff = orbitals.sample_projected_density(samp_df, coeffs, dataset_df)
    dens_err = torch.sum(torch.abs(dens_coeff - samp['density']) * samp['coord_weights'], dim=1) / torch.sum(samp_df['atom_numbers'])
    coeff_err = torch.mean(torch.abs(coeffs - samp_df['df_coeffs']))
    new_samp = {key: samp_df[key] for key in samp_df.keys()}
    new_samp['density'] = dens_coeff
    dpm = orbitals.calc_dipole_moment(new_samp)['dipole_moment']
    dpm_err = 4.8 * torch.norm(dpm - samp['dipole_moment'])
    dens_errors.append(float(dens_err))
    coeff_errors.append(float(coeff_err))
    dpm_errors.append(float(dpm_err))

print('df dens errors', dens_errors)
print('df coeff errors', coeff_errors)
print('df dpm errors', dpm_errors)

# %%
# normalize dens_errors from 0 to 1
dens_errors_norm = (dens_errors - np.min(dens_errors)) / (np.max(dens_errors) - np.min(dens_errors))
coeff_errors_norm = (coeff_errors - np.min(coeff_errors)) / (np.max(coeff_errors) - np.min(coeff_errors))
dpm_errors_norm = (dpm_errors - np.min(dpm_errors)) / (np.max(dpm_errors) - np.min(dpm_errors))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(dens_errors_norm, label='density error')
axs[0].plot(coeff_errors_norm, label='coeffs error')
axs[0].plot(dpm_errors_norm, label='dpm error')
axs[0].set_xlabel('Width treshold for included coefficients')
axs[0].set_ylabel('Normalized error')
tick_coords = np.round(np.linspace(0, len(exp_space)-1, 6)).astype(int)
print('tick_coords', tick_coords)
axs[0].set_xticks(ticks=tick_coords, labels = ['{:.2f}'.format(float(exp_space[i])) for i in tick_coords])
axs[1].plot(dens_errors_norm, label='density error')
axs[1].plot(coeff_errors_norm, label='coeffs error')
axs[1].plot(dpm_errors_norm, label='dpm error')
axs[1].set_yscale('log')
axs[1].set_xlabel('width treshold for included coefficients')
axs[1].set_ylabel('Normalized error (log)')
axs[1].set_xticks(ticks=tick_coords, labels = ['{:.2f}'.format(float(exp_space[i])) for i in tick_coords])
fig.text(0.5, -0.15,
         'These figures show how the errors of the density coefficients,density and dipole\n' + 
         'moment (normalized to a range of 0 to 1) change as more basis functions are included,\n' +
         'starting from the ones with the lowest radial width to ones with the highest. We see\n' +
         'that the density and dipole moment errors change at a similar rate, and that even \n' +
         'the basis functions with the smalles width are important for achieving high accuracy for\n' +
         'the dipole moment.', ha='center', va='center', fontsize=12)
plt.tight_layout()
plt.legend()
plt.savefig('figures/ethanethiol_errors_by_width.pdf', dpi=300)
plt.show()

# %%
print('main args use gpu', main_args.use_gpu)
print('args use gpu', args.use_gpu)
model = model_loader.load_model(args, dataset)
samp = dataset.get_properties([0])
samp_df = dataset_df.get_properties([0])

res = model(samp)

print('samp df_coeffs', samp_df['df_coeffs'].shape)
print('res df_coeffs', res['df_coeffs'].shape)

# print('res_radial width', res['radial_width'])
# print('dataset radial coeffs', dataset.radial_coeffs)
print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))

sph_coeffs_vec = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                                radial_coeffs=False, convert_to_pyscf=False)

# %%
print(dataset.orbital_basis_size)
#print(dataset.orbital_basis_num)
orbital_dict = orbitals.combine_orbital_basis(dataset.orbital_basis_num, 5)[0]
print('orbital_dict', orbital_dict)
dens_errors = []
coeff_errors = []
dpm_errors = []

masks_pos = {key: 0 for key in dataset.orbital_basis_size} 
max_orbitals = orbital_dict[16]

atom = [(int(samp_df['batch_atom_numbers'][0, i].detach().cpu().numpy()),
        samp_df['batch_positions'][0, i].detach().cpu().numpy())
        for i in range(samp_df['batch_positions'].shape[1])] 
#print(atom)
df_coeffs_split = orbitals.split_df_coeffs(atom, sph_coeffs_vec['spherical_coeffs'].squeeze(), dataset.orbital_basis_size)
max_coeffs_len = max([len(coeffs[1]) for coeffs in df_coeffs_split])
print('max orbitals', max_orbitals)
for L in range(6):
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
        df_coeffs_dict = {key: res[key] for key in res.keys()}



        df_dens = model.property_models['density'](df_coeffs_dict)
        coeffs_dict = orbitals.vector_to_coeffs_dict({'spherical_coeffs': coeffs}, dataset.orbital_basis_num,
                                                     res['batch_atom_numbers'], radial_coeffs=False,
                                                     convert_to_equiv_dens=False)
        df_coeffs_dict['spherical_coeffs'] = coeffs_dict['spherical_coeffs']
        dens_coeff = model.property_models['density'](df_coeffs_dict)['density']
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
# normalize dens_errors from 0 to 1
dens_errors_norm = (dens_errors - np.min(dens_errors)) / (np.max(dens_errors) - np.min(dens_errors))
dpm_errors_norm = (dpm_errors - np.min(dpm_errors)) / (np.max(dpm_errors) - np.min(dpm_errors))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(dens_errors_norm, label='density error')
axs[0].plot(dpm_errors_norm, label='dpm error')
axs[0].set_xlabel('Number of coefficients included')
axs[0].set_ylabel('Normalized error')
axs[0].set_xticks(ticks=[14, 26, 36, 41, 44, 46], labels = ['L=0', '1', '2', '3', '4', '5'])
axs[1].plot(dens_errors_norm, label='density error')
axs[1].plot(dpm_errors_norm, label='dpm error')
axs[1].set_yscale('log')
axs[1].set_xlabel('Number of coefficients included')
axs[1].set_ylabel('Normalized error (log)')
axs[1].set_xticks(ticks=[14, 26, 36, 41, 44, 46], labels = ['L=0', '1', '2', '3', '4', '5'])
fig.text(0.5, -0.15,
         'These figures show how the errors of the machine learned density and dipole\n' + 
         'moment (normalized to a range of 0 to 1) change as more ML-basis functions are included,\n' +
         'starting from the ones with zero angular degree to the basis functions of highest \n' +
         'angular degree. We see that the dipole moment errors converge faster than the density errors,\n' +
         'already achieving minimal dipole moment errors, while the absolute density error is still\n'
         + 'larger than 4%. Here we see again that the higher spherical harmonic orders do not\n' +
         'further improve the dipole moment, however they are crucial for a good absolute density error.',
         ha='center', va='center', fontsize=12)
plt.tight_layout()
plt.legend()
plt.savefig('figures/ethanethiol_errors_by_order.pdf', dpi=300)
plt.show()

