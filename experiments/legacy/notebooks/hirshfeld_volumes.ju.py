# %%
from pyscf import gto, dft, df, lib, scf
from pyscf.scf import atom_ks
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
from pyscf.dft import numint
from equiv_dens.training import model_loader
from equiv_dens.utils.hirshfeld_analysis import get_atm_nrks, free_atom_spline,\
    eval_spline_density, hirshfeld_partitioning
from equiv_dens.utils.grids import spherical_grid
from pyscf.dft import gen_grid, radi
import os
from pyscf.data.elements import NRSRHFS_CONFIGURATION
from pyscf.scf import atom_hf, ADIIS
from pyscf.dft import rks
import matplotlib.pyplot as plt
from pyscf.scf.atom_hf import AtomSphericAverageRHF

# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..
# %%
class AtomSphAverageRKS(rks.RKS, atom_hf.AtomSphericAverageRHF):
    def __init__(self, mol, *args, **kwargs):
        atom_hf.AtomSphericAverageRHF.__init__(self, mol)
        rks.RKS.__init__(self, mol, *args, **kwargs)

        # SAP guess is perfect for atoms
        self.init_guess = 'vsap'

    eig = atom_hf.AtomSphericAverageRHF.eig
    get_occ = atom_hf.AtomSphericAverageRHF.get_occ
    get_grad = atom_hf.AtomSphericAverageRHF.get_grad


AtomSphericAverageRKS = AtomSphAverageRKS
# %%
spin_dict = {1: 1, 8: 2, 6: 2, 7: 3, 16: 2}
anum = 16
mol = gto.M(atom=[[anum, [0, 0, 0]]], basis='augccpvdz', spin=spin_dict[anum])
grid = (120, 770)
atomic_configuration = NRSRHFS_CONFIGURATION
xc = 'PBE'
basis = mol.basis

atm_scf_result = {}

elem_chrg = anum
atm = gto.Mole(atom=[[anum, [0, 0, 0]]], basis=basis, spin=spin_dict[anum]).build()

atm_ks = AtomSphericAverageRKS(atm)
atm_ks.atomic_configuration = atomic_configuration
atm_ks.xc = xc
print('ks grid coords', atm_ks.grids.coords)
print('ks grid atom_grid', atm_ks.grids.atom_grid)
my_diis_obj = ADIIS()
my_diis_obj.space = 12
atm_ks.diis = my_diis_obj
atm_ks.chkfile = False
atm_ks.run()
print('ks grid coords', atm_ks.grids.coords.shape)
print('ks grid atom_grid', atm_ks.grids.atom_grid)
print('mean coords', np.mean(atm_ks.grids.coords))
# dict of spherical atomic RKS density
# TODO use becke scheme and see if difference is there -> no difference
print('energy', atm_ks.e_tot)
print(atm_ks.mo_coeff)
# %%

ao = numint.eval_ao(mol, atm_ks.grids.coords)
rho_atm = numint.eval_rho2(mol, ao, mo_coeff=atm_ks.mo_coeff, mo_occ=atm_ks.mo_occ)
print('density integral', np.sum(rho_atm * atm_ks.grids.weights))
print('density r integral', np.sum(rho_atm * atm_ks.grids.weights * np.linalg.norm(atm_ks.grids.coords, axis=-1)))
print('density r2 integral', np.sum(rho_atm * atm_ks.grids.weights * np.linalg.norm(atm_ks.grids.coords, axis=-1)**2))
print('density r3 integral', np.sum(rho_atm * atm_ks.grids.weights * np.linalg.norm(atm_ks.grids.coords, axis=-1)**3))
# %%
dists = np.linalg.norm(atm_ks.grids.coords, axis=-1)
print('grid_coords', atm_ks.grids.coords)
print(dists.shape)
print('dists', dists)
print(np.unique(dists).shape)
u_atm_dists = []
rho_atm_means = []
rho_atm_stds = []
for dist in np.unique(dists):
    u_atm_dists.append(dist)
    print('dist', dist)
    rho_atm_mean = np.mean(rho_atm[dists == dist])
    rho_atm_std = np.std(rho_atm[dists == dist])
    print('rho_atm_mean', rho_atm_mean)
    print('rho_atm_std', rho_atm_std)
    rho_atm_means.append(rho_atm_mean)
    rho_atm_stds.append(rho_atm_std)

# %%
mol = gto.M(atom='H  0  0  0', basis='augccpvdz', spin=1, symmetry=True)
grid = (120, 770)
atomic_configuration = NRSRHFS_CONFIGURATION
xc = 'PBE'
basis = mol.basis

ks = dft.RKS(mol)
ks.xc = xc
ks.chkfile = False
ks.run()
print('ks grid coords', ks.grids.coords.shape)
print('ks grid atom_grid', ks.grids.atom_grid)
print('mean coords', np.mean(ks.grids.coords))

print('energy', ks.e_tot)
ao = numint.eval_ao(mol, ks.grids.coords)
dm = ks.make_rdm1()
rho_ks = numint.eval_rho(mol, ao, dm)
print('density integral', np.sum(rho_ks * ks.grids.weights))
print('density r integral', np.sum(rho_ks * ks.grids.weights * np.linalg.norm(ks.grids.coords, axis=-1)))
print('density r2 integral', np.sum(rho_ks * ks.grids.weights * np.linalg.norm(ks.grids.coords, axis=-1)**2))
print('density r3 integral', np.sum(rho_ks * ks.grids.weights * np.linalg.norm(ks.grids.coords, axis=-1)**3))
# %%
dists = np.linalg.norm(ks.grids.coords, axis=-1)
print('grid_coords', ks.grids.coords)
print(dists.shape)
print('dists', dists)
print(np.unique(dists).shape)
u_ks_dists = []
rho_ks_means = []
rho_ks_stds = []
for dist in np.unique(dists):
    u_ks_dists.append(dist)
    print('dist', dist)
    rho_ks_mean = np.mean(rho_ks[dists == dist])
    rho_ks_std = np.std(rho_ks[dists == dist])
    print('rho_ks_mean', rho_ks_mean)
    print('rho_ks_std', rho_ks_std)
    rho_ks_means.append(rho_ks_mean)
    rho_ks_stds.append(rho_ks_std)

# %%
plt.figure()
plt.plot(u_ks_dists, rho_ks_means, c='blue', label='KS', linestyle='-')
plt.plot(u_atm_dists, rho_atm_means, c='orange', label='radial', linestyle='-')
plt.xlim(5, 15)
plt.ylim(0,0.0001)
plt.legend()
plt.show()
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
