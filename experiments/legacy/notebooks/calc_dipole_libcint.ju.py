# %%
# %load_ext autoreload
# %autoreload 2
# %cd ..

# %%
import numpy as np
import os
import torch
import equiv_dens.utils.base as utils
from equiv_dens.utils import orbitals
from equiv_dens.data.density_dataset import AtomsDensityData
from argparse import Namespace
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.utils.misc import generate_id
from equiv_dens.training import model_loader
from functools import partial
from equiv_dens.utils.grids import cubical_grid, cubical_sampling, \
    spherical_grid, spherical_radial_sampling
from datetime import datetime, timezone
from pyscf.dft import numint
from pyscf.lib import param
from pyscf import gto, dft
from pyscf.scf import hf
import pyscf.gto.mole
import time
import scipy as sp

# %%
main_args = Namespace()

# main_args.args_file = "args/resorcinol_all_001.txt"
main_args.args_file = "args/ethanethiol_all_006_test.txt"
# main_args.args_file = "args/ethanethiol_all_106_test.txt"
# main_args.args_file = "args/h2o_small_all_001.txt"
# main_args.args_file = "args/ethanethiol_df_coeffs_001_test.txt"
# main_args.args_file = "args/ethanethiol_all_001_coreless_test.txt"
# main_args.args_file = "args/resorcinol_all_001_coreless.txt"
# main_args.args_file = "args/ethanol_all_002_coreless_test.txt"
# main_args.args_file = "args/ethanethiol_all_004_coreless.txt"
# main_args.args_file = "args/ethanethiol_all_001_SH_even.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
# main_args.res_load_file = 'datasets/ethanethiol_all_006_test.pt'
# main_args.res_load_file = 'datasets/ethanethiol_all_001_coreless_test_results.npy'
# main_args.res_load_file = None
# main_args.res_load_file = 'datasets/resorcinol_all_005_test.pt'
main_args.save_file = 'ethanethiol_all_006'
# main_args.save_file = 'ethanethiol_all_106'
# main_args.save_file = 'h2o_small_all_001'
# main_args.save_file = 'resorcinol_all_005'
# main_args.save_file = 'ethanethiol_all_001_coreless'
# main_args.save_file = 'resorcinol_all_001_coreless'
# main_args.save_file = 'ethanol_all_002_coreless'
# main_args.save_file = 'ethanethiol_all_004_coreless'
# main_args.save_file = 'ethanethiol_all_001_SH_even'
# main_args.save_file = 'ethanethiol_df_coeffs_001'
main_args.df_error = True
main_args.use_gpu = True
main_args.num_samples = 100
main_args.make_plots = True

df_losses = None

# %%
# initializing args and dataset
args, hyperparam_args = parse_command_line_arguments(arg_file=main_args.args_file)

# print('type dtype', type(args.dtype))
args.fix_arguments = True
# print('args np dir', args.np_dataset)
# args.restart = None
# args.pred_radial_coeffs = False

if args.restart is None:
    # generate "unique" id for the run (very unlikely that two runs will have the same ID)
    model_code = generate_id()
    directory = os.path.join(args.save_dir, datetime.now(timezone.UTC).strftime("%Y-%m-%d_") +
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


print('model code:', model_code)

# determine whether GPU is used for training
print('args use gpu', args.use_gpu)
args.use_gpu = main_args.use_gpu

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

args.verbose = 0
args.use_gpu = main_args.use_gpu
print('args use gpu', args.use_gpu)
args.cube_grid = False
args.radii_adjust = False
args.expansion_constraint = None
args.integral_constraint = False
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

required_properties = ['energy', 'forces', 'df_coeffs', 'density', 'dipole_moment', 'mo_coeff']

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file

# args.np_dataset_test = "datasets/ethanethiol_md_traj_every1000_dft_augccpvdz.npy"
# args.dens_dataset_test = "datasets/ethanethiol_md_traj_every1000_dft_augccpvdz_df_augccpvqzjkfit.npy"

args.density_grad = True
dataset = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000000000000,
                           required_properties=required_properties,
                           center_positions=True,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_fn,
                           pyscf_grid=True,
                           sampling_fn=sampling_fn,
                           grid_extent=grid_extent,
                           grid_origin=grid_origin,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           projected_density=args.projected_density,
                           radii_adjust=args.radii_adjust,
                           calc_data=True,
                           atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy',
                           atom_dens_type='spline',
                           split_atom_dens=True,
                           density_grad=args.density_grad,
                           dpm_intor=True,
                           )

print('dataset length', len(dataset))
print('sample pos shape', dataset.get_properties([0])['positions'].shape)
print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
print('num samples', main_args.num_samples)
print('args use gpu', args.use_gpu)

# %%
# testing model
idx = [6]
samp = dataset.get_properties(idx)
if args.use_gpu:
    for key in samp.keys():
        if isinstance(samp[key], torch.Tensor):
            samp[key] = samp[key].cuda()
model = model_loader.load_model(args, dataset)

res = model(samp)

auxmol_ml = orbitals.ml_basis_to_auxmol(res, 0)
df_coeffs_ml = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
                                              radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
print('samp mo_coeff', samp['mo_coeff'])

df_coeffs_opt, auxmol_opts = orbitals.ml_basis_to_df_coeffs(res, 'augccpvdz', mo_coeff=samp['mo_coeff'], mo_occ=samp['mo_occ'])

# %%
# calculating density and dipole by numeric integral
dpm_ml = orbitals.calc_dipole_moment(res)['dipole_moment']

coords = samp['coords'][0].numpy(force=True) / param.BOHR

ao_opt = numint.eval_ao(auxmol_opts[0], coords, deriv=0)
rho = np.einsum('ij,j->i', ao_opt, df_coeffs_opt[0])
rho = torch.tensor(rho).to(samp['coord_weights'])
ao_ml = numint.eval_ao(auxmol_ml, coords, deriv=0)
rho_ml = np.einsum('ij,j->i', ao_ml, df_coeffs_ml.numpy(force=True))
rho_ml = torch.tensor(rho_ml).to(samp['coord_weights'])
print('rho shape', rho.shape)

print('rho integral', torch.sum(rho * samp['coord_weights'][0]))
print('rho ml integral', torch.sum(rho_ml * samp['coord_weights'][0]))
print('res integral', torch.sum(res['density'][0] * samp['coord_weights'][0]))

print('rho error', torch.sum(torch.abs(rho - samp['density'][0]) * samp['coord_weights'][0]) / torch.sum(samp['atom_numbers'], dim=1))
print('rho ml error', torch.sum(torch.abs(rho_ml - samp['density'][0]) * samp['coord_weights'][0]) / torch.sum(samp['atom_numbers'], dim=1))
print('res error', torch.sum(torch.abs(res['density'][0] - samp['density'][0]) * samp['coord_weights'][0]) / torch.sum(samp['atom_numbers'], dim=1))

# %%
# calculating dipole of ML-DF projection
res_opt = {key: res[key] for key in res.keys()}
res_opt['density'] = rho.unsqueeze(0)

dpm_opt = orbitals.calc_dipole_moment(res_opt)['dipole_moment']
print('dpm opt', dpm_opt)
print('dpm ml', dpm_ml)
print('dpm samp', samp['dipole_moment'])
print('dpm ml err', torch.norm(dpm_ml - samp['dipole_moment']))
print('dpm opt err', torch.norm(dpm_opt - samp['dipole_moment']))

# %%
# calculating density matrix based on MO coeffs
a_nums = auxmol_ml.atom_charges()
a_coords = auxmol_ml.atom_coords('angstrom')

atom = [(a_nums[i], a_coords[i]) for i in range(len(a_nums))]

mol = gto.M(atom=atom, basis='augccpvdz')

dm1 = hf.make_rdm1(samp['mo_coeff'][0].numpy(force=True), samp['mo_occ'][0].numpy(force=True))
print('mo_coeff', samp['mo_coeff'][0])

# %%
# calculating helper molecule and testing analytic nuclear integral
helper_mol = orbitals.build_1c1e_helper_mol(auxmol_ml)
int1e_nuc = gto.mole.intor_cross('int1e_nuc', helper_mol, auxmol_ml)
intor_idx = [
    auxmol_ml.bas_atom(ibas)
    for ibas in range(auxmol_ml.nbas) for _ in range(auxmol_ml.bas_angular(ibas) * 2 + 1)
]
int1e_nuc = int1e_nuc[intor_idx, range(auxmol_ml.nao)]
print('int1e_nuc', int1e_nuc.shape)

# %%
print('aux nuc en', np.einsum('i,i', int1e_nuc, df_coeffs_ml.numpy(force=True)) / 2)
print('aux nuc en', np.einsum('i,i', int1e_nuc, df_coeffs_opt[0]) / 2)
print('mol nuc en', np.einsum('ij,ji', dm1, mol.intor('int1e_nuc')))

# %%
#  calculating dipole moment of true density analyticaly based on DM
ao_dip = mol.intor_symmetric('int1e_r', comp=3)
el_dip = np.einsum('xij,ji->x', ao_dip, dm1).real
charges = mol.atom_charges()
coords = mol.atom_coords()
nucl_dip = np.einsum('i,ix->x', charges, coords)
mol_dip = nucl_dip - el_dip
print('el_dpm', utils.bohr_to_angstrom(el_dip))
print('mol_dpm', utils.bohr_to_angstrom(mol_dip))
print('mol dpm diff', np.linalg.norm(utils.bohr_to_angstrom(mol_dip) - samp['dipole_moment'].numpy(force=True)))

# %%
# calculating analytical dipole moment of ML density coeffs
int1e_r = gto.mole.intor_cross('int1e_r', helper_mol, auxmol_ml)
print('int1e_r', int1e_r.shape)

int1e_r = int1e_r[:, intor_idx, range(auxmol_ml.nao)]
print('int1e_r', int1e_r.shape)
ml_dip = utils.bohr_to_angstrom(nucl_dip - np.einsum('ji,i->j', int1e_r, df_coeffs_ml.numpy(force=True)))
opt_dip = utils.bohr_to_angstrom(nucl_dip - np.einsum('ji,i->j', int1e_r, df_coeffs_opt[0]))
print('aux dpm ml', ml_dip)
print('aux dpm opt', opt_dip)
print('ml dpm diff', np.linalg.norm(ml_dip - samp['dipole_moment'].numpy(force=True)))
print('opt dpm diff', np.linalg.norm(opt_dip - samp['dipole_moment'].numpy(force=True)))

# %%
# calculating analytical dipole moment of ML-DF projection
int1e = gto.mole.intor_cross('int1e_ovlp', helper_mol, auxmol_ml)
print('int1e', int1e.shape)

int1e = int1e[intor_idx, range(auxmol_ml.nao)]

ml_int = np.einsum('i,i', int1e, df_coeffs_ml.numpy(force=True))
opt_int = np.einsum('i,i', int1e, df_coeffs_opt[0])
print('aux ml int', ml_int)
print('aux opt int', opt_int)

# %%
# timing of the different dipole moment calculation methods
print('dpm model timing')
start = time.time()
res_dens = model.property_models['density'](res)
dpm_ml = orbitals.calc_dipole_moment(res_dens)['dipole_moment']
print('ML dipole time', time.time() - start)
start = time.time()
# auxmol_ml = orbitals.ml_basis_to_auxmol(res, 0)
# df_coeffs_ml = orbitals.coeffs_dict_to_vector(res, dataset.orbital_basis_num, res['batch_atom_numbers'],
#                                               radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs'].squeeze().detach()
# helper_mol = orbitals.build_1c1e_helper_mol(auxmol_ml)
# intor_idx = [
#     auxmol_ml.bas_atom(ibas)
#     for ibas in range(auxmol_ml.nbas) for _ in range(auxmol_ml.bas_angular(ibas) * 2 + 1)
# ]
# int1e_r = gto.mole.intor_cross('int1e_r', helper_mol, auxmol_ml)
# # print('int1e_r', int1e_r.shape)
# int1e_r = int1e_r[:, intor_idx, range(auxmol_ml.nao)]
# print('int1e_r', int1e_r)
# # print('int1e_r', int1e_r.shape)
# charges = auxmol_ml.atom_charges()
# coords = auxmol_ml.atom_coords()
# nucl_dip = np.einsum('i,ix->x', charges, coords)
# ml_dip = utils.bohr_to_angstrom(nucl_dip - np.einsum('ji,i->j', int1e_r, df_coeffs_ml.numpy(force=True)))
# print('nucl_dip', nucl_dip)
# print('ml_dip', ml_dip)
dpm_ml_intor = orbitals.calc_dipole_moment_analytic(res, dataset.orbital_basis_num, 'ml_coeffs')['dipole_moment']
dpm_ml_intor = dpm_ml_intor.numpy(force=True)
print('ML analytic dipole time', time.time() - start)
print('dmp ml int', dpm_ml)
print('dmp ml intor', dpm_ml_intor)
print('dmp int', samp['dipole_moment'])
print('dpm intor', utils.bohr_to_angstrom(mol_dip))
print('ml dpm samp diff', np.linalg.norm(dpm_ml.numpy(force=True) - samp['dipole_moment'].numpy(force=True)))
print('ml dpm intor samp diff', np.linalg.norm(dpm_ml_intor - samp['dipole_moment'].numpy(force=True)))
print('ml dpm mol diff', np.linalg.norm(dpm_ml.numpy(force=True) - utils.bohr_to_angstrom(mol_dip)))
print('ml dpm intor mol diff', np.linalg.norm(dpm_ml_intor - utils.bohr_to_angstrom(mol_dip)))
print('samp mol diff', np.linalg.norm(samp['dipole_moment'].numpy(force=True) - utils.bohr_to_angstrom(mol_dip)))
# print('dmp ml pyscf', dpm_ml_intor)
# %%
dpm_intor_samp = orbitals.calc_dipole_moment_analytic(samp, 'augccpvdz', 'mo_coeffs')['dipole_moment']
print('dpm_intor_samp', dpm_intor_samp)
# %%

dataset_df = AtomsDensityData(np_path=args.np_dataset_test, density_path=args.dens_dataset_test,
                              orbitals_path=args.orbitals_file,
                              density_n_samp=10000000000000000000000,
                              required_properties=required_properties,
                              center_positions=True,
                              radial_coeffs_file=args.radial_coeffs_file,
                              dtype=args.dtype,
                              grid_fn=grid_fn,
                              pyscf_grid=True,
                              sampling_fn=sampling_fn,
                              grid_extent=grid_extent,
                              grid_origin=grid_origin,
                              cutoff=args.cutoff,
                              df_loss_weights=args.df_loss_weights,
                              projected_density=True,
                              radii_adjust=args.radii_adjust,
                              calc_data=True,
                              atom_dens_path='datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy',
                              atom_dens_type='spline',
                              split_atom_dens=True,
                              density_grad=args.density_grad,
                              dpm_intor=True,
                              )
samp_df = dataset_df.get_properties(idx)
print('samp_df dpm int', samp_df['dipole_moment'])
dpm_intor_samp_df = orbitals.calc_dipole_moment_analytic(samp_df, 'augccpvqzjkfit', 'df_coeffs')['dipole_moment']
print('dpm_intor_samp_df', dpm_intor_samp_df)
print('ml dpm samp diff', np.linalg.norm(samp_df['dipole_moment'].numpy(force=True) - samp['dipole_moment'].numpy(force=True)))
print('ml dpm intor samp diff', np.linalg.norm(dpm_intor_samp_df.numpy(force=True) - samp['dipole_moment'].numpy(force=True)))
print('ml dpm mol diff', np.linalg.norm(samp_df['dipole_moment'].numpy(force=True) - utils.bohr_to_angstrom(mol_dip)))
print('ml dpm intor mol diff', np.linalg.norm(dpm_intor_samp_df.numpy(force=True) - utils.bohr_to_angstrom(mol_dip)))

# %%
# calculating analytic integrals for other energy components via the DM
mf = dft.RKS(mol)
mf.chkfile = False
mf.xc = "pbe"
mf.kernel()
print(mf.energy_nuc())
print('mf mo_coeff', mf.mo_coeff)
dm = mf.make_rdm1()
h1e = mf.get_hcore()
m_nuc = mol.intor('int1e_nuc')
m_kin = mol.intor('int1e_kin')
print('energy e nuc', np.einsum('ij,ji', dm, m_nuc))
print('energy e kin', np.einsum('ij,ji', dm, m_kin))
print('energy h core', np.einsum('ij,ji', dm, h1e))
print('energy e nuc', np.einsum('ij,ji', dm, m_nuc))
print('energy e kin + nuc', np.einsum('ij,ji', dm, m_kin) + np.einsum('ij,ji', dm, m_nuc))

# %%
# Testing out function to cut off far away integration grid coordinates for density
from equiv_dens.utils import grids

cutoff_coords = grids.spherical_grid_atom_cutoff(samp['coords'], samp['batch_positions'],
                                                 samp['batch_atom_numbers'], dataset.grid_spec)

print('cutoff coords', cutoff_coords)
print('coords', samp['coords'])
print('cutoff coords sum', torch.sum(cutoff_coords, dim=-1))

# %%
# testing out conversion from one density fitting basis to anohter
print('samp_atom numbers', samp['batch_atom_numbers'])
print(auxmol_ml.basis['H0'][0])
print(dataset.atom_dens[1]['df_basis'])
# ml_basis_min = {'H': auxmol_ml.basis['H0']{}
pos_0 = samp['batch_positions'][0, 0].numpy(force=True)
h_base = gto.M(atom=[[1, pos_0]], basis=dataset.atom_dens[1]['df_basis'], spin=1)
old_normalize = pyscf.gto.mole.NORMALIZE_GTO
pyscf.gto.mole.NORMALIZE_GTO = False
h_ml = gto.M(atom=[[1, pos_0]], basis={1: auxmol_ml.basis['H0'][0]}, spin=1)
h_ml.build()
pyscf.gto.mole.NORMALIZE_GTO = old_normalize

coords = samp['coords'][0] / param.BOHR

ao_base = numint.eval_ao(h_base, samp['coords'][0] / param.BOHR)
ao_h_ml = numint.eval_ao(h_ml, samp['coords'][0] / param.BOHR)
print('ao base integrals', np.sum(ao_base * samp['coord_weights'][0].unsqueeze(1).numpy(force=True), axis=0))
print('ao h ml integrals', np.sum(ao_h_ml * samp['coord_weights'][0].unsqueeze(1).numpy(force=True), axis=0))


rho_base = np.einsum('ij,j->i', ao_base, dataset.atom_dens[1]['df_coeffs'][0])
rho_base = torch.from_numpy(rho_base).to(samp['coords'])
print('rho_base integral', torch.sum(rho_base * samp['coord_weights'][0]))
ovlp = gto.mole.intor_cross('int1e_ovlp', h_base, h_ml)
ovlp_ml = h_ml.intor('int1e_ovlp')

ovlp_transform = sp.linalg.solve(ovlp_ml, ovlp.T)

coeffs_ml = np.einsum('ji,i->j', ovlp_transform, dataset.atom_dens[1]['df_coeffs'][0]) 
rho_ml = np.einsum('ij,j->i', ao_h_ml, coeffs_ml)
rho_ml = torch.from_numpy(rho_ml).to(samp['coords'])
print('rho_ml_base integral', torch.sum(rho_ml * samp['coord_weights'][0]))
print('df coeffs', dataset.atom_dens[1]['df_coeffs'])
print('df coeffs ml', coeffs_ml)
print('df coeffs ml orig', df_coeffs_ml[:5])
print('transformed density diff', torch.sum(torch.abs(rho_base - rho_ml) * samp['coord_weights'][0]))
# print('ao ml', ao_h_ml)
# print('ao base', ao_base)
# %%
pos_0 = samp['batch_positions'][0, 0].numpy(force=True)
h_ml = gto.M(atom=[[1, pos_0]], basis={1: auxmol_ml.basis['H0'][0]}, spin=1)

print("h_ml basis", h_ml.basis)
print('coords', coords.shape)
ao_ml = numint.eval_ao(auxmol_ml, coords, deriv=0)
print('ao ml', ao_ml[:20, :5])
# ao_h_ml = numint.eval_ao(h_ml, coords, deriv=0)
# print('ao h ml', ao_h_ml[:20, :5])
# print('ao_h_ml / ao_ml', ao_h_ml[:20, :5] / ao_ml[:20, :5])
# print('ao_ml shape', ao_ml.shape)
# rho_ml = np.einsum('ij,j->i', ao_ml[:, :5], df_coeffs_ml[:5])
# rho_ml = torch.tensor(rho_ml)
#
# print('rho ml integral', torch.sum(rho_ml * samp['coord_weights'][0]))
# print('ao ml', ao_ml[:, :5])

# %%
# print("auxmol ml basis", auxmol_ml.basis)

