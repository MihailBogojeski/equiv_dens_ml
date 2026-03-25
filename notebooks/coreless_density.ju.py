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
from equiv_dens.utils.hirshfeld_analysis import get_atm_nrks, free_atom_spline,\
    eval_spline_density, hirshfeld_partitioning
from equiv_dens.utils.grids import spherical_grid
from pyscf.dft import gen_grid, radi
import os

# %%
# %load_ext autoreload
# %autoreload 2
# %%
mol = gto.M(atom='O  0  0  1; H  0,  0, 2; N 0,  0, 3; C 0, 0, 4; S 0, 0, 5; F 0, 0 6; Cl 0, 0, 7', basis='augccpvdz')

# dict of spherical atomic RKS density
# TODO use becke scheme and see if difference is there -> no difference
mf_elems = get_atm_nrks(mol, xc='PBE')
result = {}
for el in mf_elems:
    mf_elem = mf_elems[el]
    print('el', el)
    anum = utils.symbols_to_numbers([el[:-1]])[0]
    print('anum', anum)
    result[anum] = {}
    result[anum]["mo_coeff"] = mf_elem.mo_coeff
    print('mo coeff shape', result[anum]["mo_coeff"].shape)
    result[anum]["mo_occ"] = mf_elem.mo_occ
    result[anum]["spline_interp"] = free_atom_spline(mf_elem)
# np.save('datasets/free_atom_densities.npy', result, allow_pickle=True)
# %%
for anum in result.keys():
    atom_str = str(anum) + ' 0  0  0'
    mol = gto.M(atom=atom_str, spin=anum, basis='augccpvdz')
    mol.build()
    auxmol = gto.M(atom=atom_str, spin=anum, basis='augccpvqzjkfit')
    auxmol.build()
    atoms = {'atom_numbers': torch.tensor(anum).view(1, 1),
             'positions': torch.tensor([0, 0, 0]).view(1, 1, 3)}

    dm1 = hf.make_rdm1(result[anum]['mo_coeff'], result[anum]['mo_occ'])
    print('dm shape', dm1.shape)

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

# main_args.args_file = "args/resorcinol_all_005.txt"
# main_args.args_file = "args/CO_dens_001.txt"
# main_args.args_file = "args/h2o_dens_002.txt"
main_args.args_file = "args/ethanethiol_all_006_test.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
# main_args.save_file = 'CO_dens_001.txt'
# main_args.save_file = 'h2o_dens_002.txt'
# main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
# main_args.save_file = 'resorcinol_all_005'
main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
main_args.save_file = 'ethanethiol_all_006_test'
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
                           atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy',
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
    dens, atom_dens = orbitals.sample_atom_density(samp['batch_positions'], 
                                                   samp['batch_atom_numbers'], samp['coords'],
                                                   bases[t], t, datasets_ad[t].atom_dens,
                                                   individual_dens=True)
    print('dens shape', dens.shape)
    print('atom dens shape', atom_dens.shape)
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
    sep_atom_dens = datasets_ad[t].sample_atom_density(samp['batch_positions'],
                                                       samp['batch_atom_numbers'],
                                                       samp['coords'],
                                                       individual_dens=True)
    print('sep atom dens shape', sep_atom_dens.shape)
    wA, elec_charges, dipoles = hirshfeld_partitioning(samp['density'], sep_atom_dens, samp['batch_positions'],
                                              samp['batch_atom_numbers'],
                                              samp['coords'], samp['coord_weights'])
    print('elec_charges', elec_charges)
    print('atom_numbers', samp['batch_atom_numbers'])

# %%
# idx = 4
import time

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
    
    start = time.time()
    samp = datasets_ad[t].get_properties(idx)
    print(t + ' sampling time', time.time() - start)
    print(t + ' error', torch.sum(torch.abs(samp['density'] - samp['atom_density']) * samp['coord_weights'], dim=-1) /
          torch.sum(samp['batch_atom_numbers'], dim=1))
    print(t + ' intergral', torch.sum(torch.abs(samp['atom_density']) * samp['coord_weights'], dim=-1))

    denss[t] = samp['atom_density']
    print('atom density shape', samp['atom_density'].shape)

print('spline df diff', torch.sum(torch.abs(denss['spline'] - denss['df_coeffs']) * samp['coord_weights'], dim=-1) /
      torch.sum(samp['batch_atom_numbers'], dim=1))
print('spline mo diff', torch.sum(torch.abs(denss['spline'] - denss['mo_coeffs']) * samp['coord_weights'], dim=-1) /
      torch.sum(samp['batch_atom_numbers'], dim=1))
print('mo df diff', torch.sum(torch.abs(denss['mo_coeffs'] - denss['df_coeffs']) * samp['coord_weights'], dim=-1) /
      torch.sum(samp['batch_atom_numbers'], dim=1))

# %%

model = model_loader.load_model(args, datasets_ad['mo_coeffs'])

res = model(samp)

auxmol = orbitals.ml_basis_to_auxmol(res)
# %%
print('auxmol basis', auxmol._basis)

mols = utils.npy_to_pyscf(samp['batch_positions'].numpy(force=True), samp['batch_atom_numbers'].numpy(force=True), basis=auxmol._basis)

print(mols)
# %%

atom_dens = datasets_ad['mo_coeffs'].atom_dens
print(atom_dens)
print('mo_basis')
basis = 'augccpvdz'

atom = [[1, [0,0,0]]]

mol = gto.M(atom=atom, basis=basis, spin=1)
mol.build()
print(mol._basis)
# %%
atom_dens_min = {z: {} for z in atom_dens.keys()}
basis = 'augccpvdz'
auxbasis = 'augccpvqzjkfit'
for z in atom_dens.keys():
    print('z', z)
    at_dens = atom_dens[z]
    symbol = utils.numbers_to_symbols([z])[0]
    keys = list(at_dens.keys())
    atom_dens[z]["mo_basis"] = {}
    atom_dens[z]["df_basis"] = {}
    atom_dens_min[z]["mo_basis"] = {}
    atom_dens_min[z]["df_basis"] = {}
    print(keys)
    for key in keys:
        print('key', key)
        if 'spline' in key:
            atom_dens_min[z][key] = at_dens[key]
        elif 'coeff' in key:
            if key == 'mo_coeff':
                bas = basis
            elif key == 'df_coeffs':
                bas = auxbasis
            print('basis', bas)
            atom = [[z, [0,0,0]]]
            mol = gto.M(atom=atom, basis=bas, spin=z%2)
            mol.build()
            prefix = key.split("_")[0]
            atom_dens[z][prefix + "_basis"][z] = mol._basis[symbol]
            s_orb_count = np.sum([len(b[1]) - 1 for b in atom_dens[z][prefix + "_basis"][z] if b[0] == 0])
            max_degree = np.max([b[0] for b in atom_dens[z][prefix + "_basis"][z]])
            print('max degree', max_degree)
            orbital_spec, _, _ = orbitals.combine_pyscf_basis(atom_dens[z][prefix + "_basis"], max_degree)
            n_s_orb = orbital_spec[z][0][1]
            if key == 'mo_coeff':
                mo_occ = atom_dens[z]["mo_occ"]
                max_occ = len(mo_occ)

                # iterate over mo_occ list in reverse
                for i in range(len(mo_occ) - 1, -1, -1):
                    if mo_occ[i] == 0:
                        max_occ = i
                    else:
                        break
                num_min_orb = max(max_occ, n_s_orb)
                curr_orb = 0
                num_min_coeff = 0

                for z, num_L_orb, L in orbital_spec[z]:
                    if curr_orb + num_L_orb > num_min_orb:
                        num_min_coeff += (num_min_orb - curr_orb) * ((2 * L) + 1)
                        break
                    else:
                        num_min_coeff += num_L_orb * ((2 * L) + 1)
                        curr_orb += num_L_orb

                print('atom_dens basis', atom_dens[z][prefix + "_basis"])
                atom_dens_min[z][key] = atom_dens[z][key][:num_min_coeff, :num_min_coeff]

                curr_orb = 0
                atom_dens_min[z][prefix + "_basis"][z] = []

                for orb in atom_dens[z][prefix + "_basis"][z]:
                    curr_num_orb = len(orb[1]) - 1
                    if curr_orb + curr_num_orb < num_min_orb:
                        atom_dens_min[z][prefix + "_basis"][z].append(orb)
                        curr_orb += curr_num_orb
                    else:
                        rest_orb = num_min_orb - curr_orb
                        atom_dens_min[z][prefix + "_basis"][z].append([orb[0]] + [orb[i][:(rest_orb + 1)] for i in range(1, len(orb))])
                        break
                print('orbital spec', orbital_spec[z])
                print('num_min_orb', num_min_orb)
                print('num_min_coeff', num_min_coeff)
                atom_dens_min[z]['mo_occ'] = mo_occ[:num_min_coeff]
                print('atom_dens_min basis', atom_dens_min[z][prefix + "_basis"])
            if key == 'df_coeffs':
                print('df coeffs shape', atom_dens[z][key].shape)
                atom_dens_min[z][key] = atom_dens[z][key][:, :n_s_orb]
                atom_dens_min[z][prefix + "_basis"][z] = atom_dens[z][prefix + "_basis"][z][:n_s_orb]
                print('atom_dens_min basis', atom_dens_min[z][prefix + "_basis"])

# %%
print('atom_dens', atom_dens)
print('atom_dens_min', atom_dens_min)
print('atom_dens', atom_dens[8]['mo_basis'])
print('atom_dens min', atom_dens_min[8]['mo_basis'])
# %%
np.save('datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy', atom_dens, allow_pickle=True)
np.save('datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy', atom_dens_min, allow_pickle=True)
# %%
# compare outcomes of old and new free atom density data
#idx = 4
import time

idx = [1, 2, 3, 4, 5]
denss = {}
datasets_ad = {}
for t in ['spline', 'df_coeffs', 'mo_coeffs']:
    dataset_old = AtomsDensityData(np_path=args.np_dataset_test,
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
    
    dataset_new = AtomsDensityData(np_path=args.np_dataset_test,
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
                                   atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy',
                                   atom_dens_type=t,
                                   )
    samp_old = dataset_old.get_properties(idx)
    samp_new = dataset_new.get_properties(idx)
    print(t + ' error', torch.sum(torch.abs(samp_new['density'] - samp_new['atom_density']) * samp_new['coord_weights'], dim=-1) /
          torch.sum(samp_new['batch_atom_numbers'], dim=1))
    print(t + ' intergral', torch.sum(torch.abs(samp_new['atom_density']) * samp_new['coord_weights'], dim=-1))
    print(t + ' error new old', torch.sum(torch.abs(samp_new['atom_density'] - samp_old['atom_density']) * samp_new['coord_weights'], dim=-1) /
          torch.sum(samp_new['batch_atom_numbers'], dim=1))

    print('atom density shape', samp_new['atom_density'].shape)

# print('spline df diff', torch.sum(torch.abs(denss['spline'] - denss['df_coeffs']) * samp['coord_weights'], dim=-1) /
#       torch.sum(samp['batch_atom_numbers'], dim=1))
# print('spline mo diff', torch.sum(torch.abs(denss['spline'] - denss['mo_coeffs']) * samp['coord_weights'], dim=-1) /
#       torch.sum(samp['batch_atom_numbers'], dim=1))
# print('mo df diff', torch.sum(torch.abs(denss['mo_coeffs'] - denss['df_coeffs']) * samp['coord_weights'], dim=-1) /
#       torch.sum(samp['batch_atom_numbers'], dim=1))



# %%
# compare outcomes of new and minimal free atom density data
#idx = 4
import time

idx = [1, 2, 3, 4, 5]
denss = {}
datasets_ad = {}
for t in ['spline', 'df_coeffs', 'mo_coeffs']:
# for t in ['mo_coeffs']:
    dataset_min = AtomsDensityData(np_path=args.np_dataset_test,
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
                                   atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy',
                                   atom_dens_type=t,
                                   split_atom_dens=True,
                                   timing=True
                                   )

    dataset_new = AtomsDensityData(np_path=args.np_dataset_test,
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
                                   atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy',
                                   atom_dens_type=t,
                                   split_atom_dens=True,
                                   timing=True
                                   )

    start = time.time()
    samp_min = dataset_min.get_properties(idx)
    print('samp min time', time.time() - start)
    start = time.time()
    samp_new = dataset_new.get_properties(idx)
    print('samp new time', time.time() - start)
    print(t + ' error', torch.sum(torch.abs(samp_min['density'] - samp_min['atom_density']) * samp_min['coord_weights'], dim=-1) /
          torch.sum(samp_min['batch_atom_numbers'], dim=1))
    print(t + ' intergral', torch.sum(torch.abs(samp_min['atom_density']) * samp_min['coord_weights'], dim=-1))
    print(t + ' error min old', torch.sum(torch.abs(samp_min['atom_density'] - samp_new['atom_density']) * samp_min['coord_weights'], dim=-1) /
          torch.sum(samp_min['batch_atom_numbers'], dim=1))
    print(t + 'split dens integral', torch.sum(samp_min['atom_density_split'] * samp_min['coord_weights'].unsqueeze(1), dim=-1))

    print('atom density shape', samp_new['atom_density'].shape)

# %%
atom_dens = dataset_new.atom_dens
print('atom_dens types', atom_dens[1].keys())
atom_dens_min = dataset_min.atom_dens

for i in range(6, samp_min['batch_atom_numbers'].shape[1]):
    z = samp_min['batch_atom_numbers'][0, i]
    print("full vs min free atom density for", z)

    pos = samp_min['batch_positions'][:, [i]]
    anum = samp_min['batch_atom_numbers'][:, [i]]

    coords = samp_min['coords']
    samp_dens, _ = orbitals.sample_atom_density(pos, anum, coords, 'mo_coeffs', atom_dens)
    samp_dens_min, _ = orbitals.sample_atom_density(pos, anum, coords, 'mo_coeffs', atom_dens_min)

    print('samp dens integral', torch.sum(samp_dens * samp_min['coord_weights'], dim=-1))
    print('min dens integral', torch.sum(samp_dens_min * samp_min['coord_weights'], dim=-1))
    print('samp dens diff', torch.sum(torch.abs(samp_dens - samp_dens_min) * samp_min['coord_weights'], dim=-1) /
        torch.sum(samp_min['batch_atom_numbers'], dim=1))
