# %%
from pyscf import gto, dft, df, lib
from pyscf.scf import hf
from pyscf.lib import param
import scipy

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
from equiv_dens.utils.hirshfeld_analysis import get_atm_nrks, free_atom_spline, eval_spline_density
from equiv_dens.utils.grids import spherical_grid
from pyscf.dft import gen_grid, radi
import os

# %load_ext autoreload
# %autoreload 2
# %cd /home/mihail/Documents/workspace/equiv_dens/
# %%
mol = gto.M(atom='O  0  0  1; H  0,  0, 2; N 0,  0, 3; C 0, 0, 4; S 0, 0, 5', basis='augccpvdz')

# dict of spherical atomic RKS density 
# TODO use becke scheme and see if difference is there -> no difference
mf_elems = get_atm_nrks(mol, xc='PBE')
result = {}
for el in mf_elems:
    mf_elem = mf_elems[el]
    anum = utils.symbols_to_numbers([el[0]])[0]
    result[anum] = {}
    result[anum]["mo_coeff"] = mf_elem.mo_coeff
    result[anum]["mo_occ"] = mf_elem.mo_occ
    result[anum]["spline_interp"] = free_atom_spline(mf_elem)
# np.save('datasets/free_atom_densities.npy', result, allow_pickle=True)
# %%
for anum in result.keys():
    atom_str = str(anum) + ' 0  0  0'
    if (anum % 2 == 1):
        mol = gto.M(atom=atom_str, spin=1, basis='augccpvdz')
    else:
        mol = gto.M(atom=atom_str, basis='augccpvdz')
    mol.build()
    if (anum % 2 == 1):
        auxmol = gto.M(atom=atom_str, spin=1, basis='augccpvqzjkfit')
    else:
        auxmol = gto.M(atom=atom_str, basis='augccpvqzjkfit')
    auxmol.build()
    atoms = {'atom_numbers': torch.tensor(anum).view(1, 1),
             'positions': torch.tensor([0, 0, 0]).view(1, 1, 3)}

    dm1 = hf.make_rdm1(result[anum]['mo_coeff'], result[anum]['mo_occ'])

    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
    ints_2c2e = auxmol.intor('int2c2e')

    nao = mol.nao
    naux = auxmol.nao

    df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
    df_coef = df_coef.reshape(naux, nao, nao)
    df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)
    df_basis = torch.tensor(df_basis).view(1, -1)

    print('df basis shape', df_basis.shape)
    rot_spec = spherical_grid(atoms, level=1)
    coords, weights = gen_grid.get_partition(mol, rot_spec)
    coords = torch.tensor(coords).unsqueeze(0)
    weights = torch.tensor(weights).unsqueeze(0)
    # print('df basis', df_basis)

    atoms['batch_atom_numbers'] = torch.tensor(atoms['atom_numbers'])
    atoms['batch_positions'] = torch.tensor(atoms['positions'])
    atoms['coords'] = coords
    atoms['coord_weights'] = weights
    dens_full = orbitals.sample_density(atoms, mo_coeff=result[anum]['mo_coeff'],
                                        mo_occ=result[anum]['mo_occ'])
    dens_full = torch.unsqueeze(dens_full, dim=0)
    print(dens_full.shape)
    print('dens full integral', torch.sum(dens_full * atoms['coord_weights'], dim=-1))

    dens_spline = eval_spline_density(result[anum]['spline_interp'], atoms['coords'])
    dens_spline = torch.tensor(dens_spline)
    print(dens_spline.shape)
    print('dens spline integral', torch.sum(dens_spline * atoms['coord_weights'], dim=-1))

    dens_df = orbitals.sample_projected_density(atoms, df_basis, 'augccpvqzjkfit')
    dens_df = dens_df.unsqueeze(0)
    print(dens_df.shape)
    print('dens df integral', torch.sum(dens_df * atoms['coord_weights'], dim=-1))

    dens_diff = torch.sum(torch.abs(dens_spline - dens_full) * atoms['coord_weights']) / torch.sum(atoms['atom_numbers'])
    print('dens_diff spline', dens_diff)

    dens_diff = torch.sum(torch.abs(dens_df - dens_full) * atoms['coord_weights']) / torch.sum(atoms['atom_numbers'])
    print('dens_diff df', dens_diff)
    result[anum]['df_basis'] = 'augccpvqzjkfit'
    result[anum]['df_coeffs'] = df_basis
# dens = orbitals.sample_density(atoms, mf_elems)
# splines = result.pop('interpolate_fn_rho_free_elem')
np.save('datasets/free_atom_densities_augccpvdz_augccpvqzjkfit.npy', result, allow_pickle=True)
# %%
main_args = Namespace()

main_args.args_file = "args/resorcinol_all_005.txt"
# main_args.args_file = "args/CO_dens_001.txt"
# main_args.args_file = "args/h2o_dens_002.txt"
# main_args.args_file = "args/ethanethiol_all_006_test.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
# main_args.save_file = 'CO_dens_001.txt'
# main_args.save_file = 'h2o_dens_002.txt'
main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
main_args.save_file = 'resorcinol_all_005'
# main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
# main_args.save_file = 'ethanethiol_all_006_test'
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
    data_split_indices = checkpoint['data_split_indices']

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
    args.cube_origin = -2
    args.cube_extent = 4
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
                           center_positions=True,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           pyscf_grid=args.pyscf_grid,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
                           radii_adjust=args.radii_adjust,
                           calc_data=True,
                           atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit.npy',
                           atom_dens_type='spline'
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
                                  center_positions=True,
                                  radial_coeffs_file=args.radial_coeffs_file,
                                  dtype=args.dtype,
                                  grid_fn=grid_fn,
                                  pyscf_grid=args.pyscf_grid,
                                  sampling_fn=sampling_fn,
                                  grid_extent=grid_extent,
                                  grid_origin=grid_origin,
                                  cutoff=args.cutoff,
                                  df_loss_weights=args.df_loss_weights,
                                  projected_density=True,
                                  radii_adjust=args.radii_adjust,
                                  atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit.npy',
                                  atom_dens_type='spline'
                                  )

# %%
# idx = 4

idx = [1, 2, 3, 4, 5]
denss = {}
datasets_ad = {}
for t in ['spline', 'df_coeffs', 'mo_coeffs']:
    datasets_ad[t] = AtomsDensityData(np_path=args.np_dataset_test,
                                      density_path=args.dens_dataset_test,
                                      orbitals_path=args.orbitals_file,
                                      density_n_samp=10000000000000000000000,
                                      required_properties=required_properties,
                                      center_positions=True,
                                      radial_coeffs_file=args.radial_coeffs_file,
                                      dtype=args.dtype,
                                      grid_fn=grid_fn,
                                      pyscf_grid=args.pyscf_grid,
                                      sampling_fn=sampling_fn,
                                      grid_extent=grid_extent,
                                      grid_origin=grid_origin,
                                      cutoff=args.cutoff,
                                      df_loss_weights=args.df_loss_weights,
                                      projected_density=args.projected_density,
                                      radii_adjust=args.radii_adjust,
                                      calc_data=True,
                                      atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit.npy',
                                      atom_dens_type=t,
                                      )
    samp = datasets_ad[t].get_properties(idx)
    print(t + ' error', torch.sum(torch.abs(samp['density'] - samp['atom_density']) * samp['coord_weights'], dim=-1) /
          torch.sum(samp['batch_atom_numbers'], dim=1))
    print(t + ' intergral', torch.sum(torch.abs(samp['atom_density']) * samp['coord_weights'], dim=-1))

    denss[t] = samp['atom_density']

print('spline df diff', torch.sum(torch.abs(denss['spline'] - denss['df_coeffs']) * samp['coord_weights'], dim=-1) /
      torch.sum(samp['batch_atom_numbers'], dim=1))
print('spline mo diff', torch.sum(torch.abs(denss['spline'] - denss['mo_coeffs']) * samp['coord_weights'], dim=-1) /
      torch.sum(samp['batch_atom_numbers'], dim=1))
print('mo df diff', torch.sum(torch.abs(denss['mo_coeffs'] - denss['df_coeffs']) * samp['coord_weights'], dim=-1) /
      torch.sum(samp['batch_atom_numbers'], dim=1))
# %%
idx = [1, 2, 3, 4, 5]
bases = {'spline': None, 'df_coeffs': dataset_df.density_fitting['auxbasis'], 'mo_coeffs': datasets_ad['mo_coeffs'].mols[0].basis}
for t in ['spline', 'df_coeffs', 'mo_coeffs']:
    samp = datasets_ad[t].get_properties(idx)
    dens, atom_dens = orbitals.sample_atom_density(samp['batch_positions'], samp['batch_atom_numbers'], samp['coords'],
                                                   bases[t], t, datasets_ad[t].atom_dens)
    print(t + ' error', torch.sum(torch.abs(dens - denss[t]) * samp['coord_weights'], dim=-1) /
          torch.sum(samp['batch_atom_numbers'], dim=1))
    dens_sum = torch.sum(atom_dens, dim=1)
    print(t + ' sum error', torch.sum(torch.abs(dens - dens_sum) * samp['coord_weights'], dim=-1) /
          torch.sum(samp['batch_atom_numbers'], dim=1))
    print(t + ' true dens error', torch.sum(torch.abs(dens - samp['density']) * samp['coord_weights'], dim=-1) /
          torch.sum(samp['batch_atom_numbers'], dim=1))
# %%
# idx = 4

idx = [1, 2, 3, 4, 5]
denss = {}
datasets_ad = {}
for t in ['spline', 'df_coeffs', 'mo_coeffs']:
    datasets_ad[t] = AtomsDensityData(np_path=args.np_dataset_test,
                                      density_path=args.dens_dataset_test,
                                      orbitals_path=args.orbitals_file,
                                      density_n_samp=10000000000000000000000,
                                      required_properties=required_properties,
                                      center_positions=True,
                                      radial_coeffs_file=args.radial_coeffs_file,
                                      dtype=args.dtype,
                                      grid_fn=grid_fn,
                                      pyscf_grid=args.pyscf_grid,
                                      sampling_fn=sampling_fn,
                                      grid_extent=grid_extent,
                                      grid_origin=grid_origin,
                                      cutoff=args.cutoff,
                                      df_loss_weights=args.df_loss_weights,
                                      projected_density=args.projected_density,
                                      radii_adjust=args.radii_adjust,
                                      calc_data=True,
                                      atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit.npy',
                                      atom_dens_type=t,
                                      )
    samp = datasets_ad[t].get_properties(idx)
    print(t + ' error', torch.sum(torch.abs(samp['density'] - samp['atom_density']) * samp['coord_weights'], dim=-1) /
          torch.sum(samp['batch_atom_numbers'], dim=1))
    print(t + ' intergral', torch.sum(torch.abs(samp['atom_density']) * samp['coord_weights'], dim=-1))

    denss[t] = samp['atom_density']

print('spline df diff', torch.sum(torch.abs(denss['spline'] - denss['df_coeffs']) * samp['coord_weights'], dim=-1) /
      torch.sum(samp['batch_atom_numbers'], dim=1))
print('spline mo diff', torch.sum(torch.abs(denss['spline'] - denss['mo_coeffs']) * samp['coord_weights'], dim=-1) /
      torch.sum(samp['batch_atom_numbers'], dim=1))
print('mo df diff', torch.sum(torch.abs(denss['mo_coeffs'] - denss['df_coeffs']) * samp['coord_weights'], dim=-1) /
      torch.sum(samp['batch_atom_numbers'], dim=1))
