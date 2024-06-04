# %%
import torch
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
import equiv_dens.utils.spherical_harmonics_deriv as sph_deriv
import numpy as np
from torchviz import make_dot

import os
from pyscf.dft import numint
from pyscf.lib import param
from datetime import datetime, timezone
from pyscf import gto, dft
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.utils.grids import cubical_grid, cubical_sampling, \
    spherical_grid, spherical_radial_sampling
from equiv_dens.utils.misc import generate_id

from functools import partial
from argparse import Namespace
import numpy as np
from equiv_dens.training import model_loader
import time
# %load_ext autoreload
# %autoreload 2
# %cd /home/mihail/Documents/workspace/equiv_dens/

# %%
# Setting up distances, directions and coeffs as variables
coords = torch.randn(1, 3, 3)
coords.requires_grad = True
print('coords', coords)
max_L = 5
center = torch.ones(1, 1, 3) / 10
coords = torch.cat([coords, center], dim=1)
d, u = utils.calculate_distances_and_directions(coords, center=center)
s = spherical_harmonics(max_L, u)
print('s requires grad', [ss.requires_grad for ss in s])
for L in range(len(s)):
    zeros = torch.zeros_like(s[L])
    s[L] = torch.where(torch.isnan(s[L]), zeros, s[L])  # making sure there are no nans to avoid NaNs
print('dist', d)
print('unit', u)
scales = [torch.randn(1, 1, 1, 5) for _ in range(max_L + 1)]
widths = [torch.randn(1, 1, 1, 5)**2 for _ in range(max_L + 1)]
coeffs = [torch.randn(1, 1, 2 * L + 1, 5) for L in range(max_L + 1)]
print([ss.shape for ss in s])
print('scales', scales)
print('widths', widths)
print('coeffs', coeffs)
# %%
# Init distances directions and coeffs from model to check calculation
main_args = Namespace()

# main_args.args_file = "args/resorcinol_all_001_coreless.txt"
# main_args.args_file = "args/ethanol_all_002_coreless_test.txt"
main_args.args_file = "args/ethanethiol_all_004_coreless.txt"
main_args.ref_np_load_file = None
main_args.ref_dens_load_file = None
# main_args.save_file = 'resorcinol_all_001_coreless'
# main_args.save_file = 'ethanol_all_002_coreless'
main_args.save_file = 'ethanethiol_all_004_coreless'
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

required_properties = ['energy', 'forces', 'df_coeffs', 'density', 'dipole_moment']

if main_args.ref_np_load_file is not None:
    args.np_dataset_test = main_args.ref_np_load_file
if main_args.ref_dens_load_file is not None:
    args.dens_dataset_test = main_args.ref_dens_load_file

# args.np_dataset_test = "datasets/ethanethiol_md_traj_every1000_dft_augccpvdz.npy"
# args.dens_dataset_test = "datasets/ethanethiol_md_traj_every1000_dft_augccpvdz_df_augccpvqzjkfit.npy"

args.spherical_grid_level = 1
args.density_grad = True
grid_fn = partial(spherical_grid, level=args.spherical_grid_level)
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
                           atom_dens_type='spline',
                           split_atom_dens=True,
                           density_grad=True,
                           )

model = model_loader.load_model(args, dataset)
model.eval()
idx = [666]
samp = dataset.get_properties(idx)
# samp_df = dataset_df.get_properties(idx)

res = model(samp)

print('dataset length', len(dataset))
print('sample pos shape', dataset.get_properties([0])['positions'].shape)
print('sample dens shape', dataset.get_properties([0])['density'].shape)
if main_args.num_samples < 1:
    main_args_num_samples = len(dataset)
print('num samples', main_args.num_samples)
print('args use gpu', args.use_gpu)

# %%
# Assinging directions, distances and coeffs values from model
# Setting up distances, directions and coeffs as variables
coords = res['coords'][:, :3]
print('coords', coords.shape)
coords.requires_grad = True
print('coords', coords)
# max_L = model.property_models['density'].orbitals_max_order
max_L = 4
center = res['batch_positions'][:, [0]]
print('center', center)
d, u = utils.calculate_distances_and_directions(coords, center=center)
s = spherical_harmonics(max_L, u)
print('s requires grad', [ss.requires_grad for ss in s])
for L in range(len(s)):
    zeros = torch.zeros_like(s[L])
    s[L] = torch.where(torch.isnan(s[L]), zeros, s[L])  # making sure there are no nans to avoid NaNs
print('dist', d)
print('unit', u)
scales = [res['radial_scale'][0][(1, L)] for L in range(max_L + 1)]
widths = [res['radial_width'][0][(1, L)] for L in range(max_L + 1)]
coeffs = [res['spherical_coeffs'][0][(1, L)] for L in range(max_L + 1)]
print([ss.shape for ss in s])
print('scales', scales)
print('widths', widths)
print('coeffs', coeffs)

# %%
# basic gto calculation
gto = 0

for L in range(max_L + 1):
    scale = scales[L]
    width = widths[L]
    coeff = coeffs[L]
    sph = s[L].unsqueeze(-1) * coeff
    print('sph shape', sph.shape)
    rbf = orbitals.gaussian_rbf(d.unsqueeze(-1), width, scale, L)
    gto_part = torch.sum(rbf * sph, dim=(-2, -1))
    print(gto_part)
    gto += gto_part
print(gto)

# %%
# testing autograd calculation for gto
grad = 0
for i in range(3):
    grad_part = torch.autograd.grad(gto.squeeze()[i], coords, retain_graph=True)[0]
    # print('grad part', grad_part)
    grad += grad_part
print(grad)

# %%
# main cell for calculating GTO gradient, comparing autodiff to explicit gradient
print('coords requires grad', coords.requires_grad)
print('u requires grad', u.requires_grad)
print('s requres grad', s[0].requires_grad)
print('coords', coords)
print('u', u)
gto = 0
gto_grad = 0
s_deriv = sph_deriv.spherical_harmonics_deriv(max_L, u)
# for L in range(len(s_deriv)):
#     zeros = torch.zeros_like(s_deriv[L])
#     s_deriv[L] = torch.where(torch.isnan(s_deriv[L]), zeros, s_deriv[L])  # making sure there are no nans to avoid NaNs
dcoords = torch.eye(3).unsqueeze(0).unsqueeze(0)
print(dcoords.shape)
# TODO: check robustness for zero coords
for L in range(max_L + 1):
    scale = scales[L]
    width = widths[L]
    coeff = coeffs[L]
    sph = s[L].unsqueeze(-1) * coeff
    print('L', L)
    print('s[L] shape', s[L].shape)
    print('s[l] deriv shape', s_deriv[L].shape)
    print('d shape', d.shape)
    print('coords shape', coords.shape)
    print('sph coeff shape', coeff.shape)
    print('width shape', width.shape)
    print('scale shape', scale.shape)
    print('sph shape', sph.shape)
    sph_autograd = 0
    du = 0
    if L != 0:
        for i in range(sph.squeeze(0).shape[0]):
            for j in range(sph.squeeze(0).shape[1]):
                for k in range(sph.squeeze(0).shape[2]):
                    sph_autograd += torch.autograd.grad(sph.squeeze(0)[i, j, k], coords, retain_graph=True)[0]
                    du += torch.autograd.grad(sph.squeeze(0)[i, j, k], u, retain_graph=True)[0]
                # print(sph_autograd)
    # print('sph du autograd', du)
    # print('s deriv cooeffs size', (s_deriv[L].unsqueeze(-1) * coeff.unsqueeze(-2)).shape)
    # print('coords deriv shape', (dcoords/d.unsqueeze(-1)
    #                         - ((coords - center).unsqueeze(-2)
    #                         * (coords - center).unsqueeze(-1)/d.unsqueeze(-1)**3)).shape)
    sph_grad = (s_deriv[L].unsqueeze(-1) * coeff.unsqueeze(-2))
    sph_grad_c = sph_grad.unsqueeze(2) * -(dcoords / d.unsqueeze(-1)
                                            - ((coords - center).unsqueeze(-2)
                                               * (coords - center).unsqueeze(-1)
                                               / d.unsqueeze(-1)**3)).unsqueeze(-2).unsqueeze(-1)
    sph_grad_c = sph_grad_c.sum(-2)
    zeros = torch.zeros_like(sph_grad_c)
    sph_grad_c = torch.where(torch.isnan(sph_grad_c), zeros, sph_grad_c)  # making sure there are no nans to avoid NaNs
    # print('sph_grad_c', sph_grad_c.sum((-1, -2)))
    # print('sph_autograd', sph_autograd)
    rbf = orbitals.gaussian_rbf(d.unsqueeze(-1), width, scale, L)
    print('rbf shape', rbf.shape)
    rbf_autograd = 0
    for i in range(rbf.squeeze(0, -2).shape[0]):
        for j in range(rbf.squeeze(0, -2).shape[1]):
            rbf_autograd += torch.autograd.grad(rbf.squeeze(0, -2)[i, j], coords, retain_graph=True)[0]
    # print('rbf autograd', rbf_autograd)
    print('gaussian rbd grad shape, ', orbitals.gaussian_rbf_deriv(d.unsqueeze(-1), width, scale, L).shape)
    print('coords/d shape', ((coords - center) / d).unsqueeze(-1).shape)
    rbf_grad = orbitals.gaussian_rbf_deriv(d.unsqueeze(-1), width, scale, L) * ((coords - center) / d).unsqueeze(-1)
    print('rbf_grad.shape', rbf_grad.shape)
    zeros = torch.zeros_like(rbf_grad)
    rbf_grad = torch.where(torch.isnan(rbf_grad), zeros, rbf_grad)  # making sure there are no nans to avoid NaNs
    # rbf_grad = rbf_grad.sum(-1)

    # print('rbf_deriv', rbf_grad)
    # print('coords.shape', coords.shape)
    gto_l = torch.sum(rbf * sph, dim=(-2, -1))
    gto_autograd = 0
    for i in range(gto_l.squeeze().shape[0]):
        gto_autograd += torch.autograd.grad(gto_l.squeeze()[i], coords, retain_graph=True)[0]
    # print('rbf shape', rbf.shape)
    # print('rbf grad shape', rbf_grad.shape)
    # print('rbf grad unsqueeze shape', rbf_grad.unsqueeze(-2).shape)
    # print('sph shape', sph.shape)
    # print('sph deriv unsqueeze coords', sph_grad_c.unsqueeze(-1).shape)
    # print('rbf_grad unsqueeze * sph', (rbf_grad.unsqueeze(-2) * sph).shape)
    # print('rbf* sph_grad unsqueeze ', (rbf * sph_grad_c.unsqueeze(-1)).shape)
    print('sph_grad_c', sph_grad_c.shape)
    print('rbf grad', rbf_grad.shape)
    print('rbf grad * sph', (rbf_grad.unsqueeze(-2) * sph.unsqueeze(-3)).shape)
    print('rbf * sph grad', (rbf.unsqueeze(-3) * sph_grad_c).shape)
    gto_deriv = (rbf_grad.unsqueeze(-2) * sph.unsqueeze(-3)).sum(-2)\
                + (rbf.unsqueeze(-3) * sph_grad_c).sum(-2)
    gto_deriv = gto_deriv.sum((-1))
    print('gto_deriv.shape', gto_deriv.shape)
    # print('sph_grad_c', sph_grad_c)
    # print('rbf', rbf)
    # print('sph', sph)
    print('gto autograd', gto_autograd)
    print('gto deriv', gto_deriv)
    gto += gto_l
    gto_grad += gto_deriv

# grad = torch.autograd.grad(gto[0,1], coords, retain_graph=True)
# print(grad)
grad = 0
for i in range(gto.squeeze().shape[0]):
    grad += torch.autograd.grad(gto.squeeze()[i], coords, retain_graph=True)[0]
print('gto', gto)
print('total gto coords grad', grad)
print('explicit gto coords grad', gto_grad)

# %%
# calculating gradient for u, the normalized interatomic direction
du = 0.0
for i in range(u.squeeze().shape[0]):
    for j in range(u.squeeze().shape[1]):
        du += torch.autograd.grad(u.squeeze()[i, j], coords, retain_graph=True)[0]

dcoords = torch.eye(3).unsqueeze(0).unsqueeze(-1)
dcoords = dcoords.expand((-1 ,-1 ,-1, 3))

u_deriv = -(1 / d - (coords * torch.sum(coords, dim=(-1), keepdim=True) / d**3))
print('du', du)
print('u_deriv', u_deriv)

# %%
# combining gradient of degree 1 gto with the derivative of u via chain rule
sph_grad = (s_deriv[1].unsqueeze(-1) * coeffs[1][..., [2, 0, 1], :]).sum(-1)
# sph_grad = (s_deriv[2].unsqueeze(-1) * coeffs[2]).sum((-1, -2))
print('sph grad shape', sph_grad.shape)
dcoords = torch.eye(3).unsqueeze(0).unsqueeze(0)
# dcoords = dcoords.expand((-1 , 3 ,-1, -1))
u_deriv = -(dcoords/d.unsqueeze(-1) - (coords.unsqueeze(-2) * coords.unsqueeze(-1)/d.unsqueeze(-1)**3))
print('u_deriv shape', u_deriv.shape)
print('u_deriv', (sph_grad.unsqueeze(-2) * u_deriv).sum(-1))

# %%
# further tests for calculating the derivative of the degree 1 GTO with plotting of gradient graph
coords2 = torch.tensor(coords)
coords2.requires_grad = True
coeffs1 = torch.tensor(coeffs[1])
class Direction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones((1, 3, 3)))

    def forward(self, Ri):
        Ri = Ri * self.dummy
        Rj = torch.zeros_like(Ri)
        rij = Rj - Ri  # displacement vectors
        dij = torch.norm(rij, dim=-1, keepdim=True)  # distances
        uij = rij / dij  # unit displacement vectors
        return uij

class L1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.ones((1, 3, 3)))

    def forward(self, u):
        u = u * self.dummy
        print('u', u)
        return np.sqrt(3) * u[..., [1, 2, 0]] 

class SphCoeffs(torch.nn.Module):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = torch.nn.Parameter(coeffs)

    def forward(self, s):
        return s.unsqueeze(-1) * self.coeffs

l1 = L1()
dir = Direction()
sphcoeffs = SphCoeffs(coeffs1)
model = torch.nn.Sequential(dir, l1, sphcoeffs)

print('coords', coords2)

sph_l1 = model(coords2)
# make_dot(sph_l1.mean(), params=dict(model.named_parameters()))
print("l1", sph_l1)
l = sph_l1.sum()
l.backward()
print('coords_grad', coords2.grad)
print('l1 dummy', l1.dummy.grad)
print('dir dummy', dir.dummy.grad)

# %%
# calculating gradient graph for degree 1 gto detailed steps version with initial gradien from spherical harmonics
coords2 = coords.clone().detach()
coords2.requires_grad = True
coeffs1 = coeffs[2].clone().detach()
c_pow2 = coords2**2
c_pow2.retain_grad()
c_pow2_sum = c_pow2.sum(-1, keepdim=True)
c_pow2_sum.retain_grad()
d = c_pow2_sum**(0.5)
d.retain_grad()
u2 = coords2 * d**(-1)  # unit displacement vectors
u2.retain_grad()
l = (u2 * sph_grad).sum()
# l = u2.sum()
l.backward(retain_graph=True)
print('u grad', u2.grad)
print('d grad', d.grad)
print('c_pow2_sum grad', c_pow2_sum.grad)
print('c_pow2 grad', c_pow2.grad)
print('one side?', 2 * coords2 * c_pow2.grad)
print('coords grad', coords2.grad)
print('coords', coords)
print('final grad?', 2 * coords2 * c_pow2.grad + sph_grad/d)
make_dot(l)

# %%
# explicitly calculating gradient for degree 1 GTO, ensuring correct dimensions and broadcasting
s_deriv = sph_deriv.spherical_harmonics_deriv(2, u)

sph_grad = (s_deriv[2].unsqueeze(-1) * coeffs[2]).sum((-1,-2))
print('sph_grad.shape', sph_grad.shape)

dudr = -(1 / d - (coords * torch.sum(coords, dim=(-1), keepdim=True) / d**3)) 
dudd = (coords / -d**2)
dddp = 1/(2 *  torch.sqrt(torch.sum(coords**2, -1, keepdim=True)))
dpds = 2*coords
sph_grad_c = sph_grad * dudr
print('s grad', s_deriv[1])
print('sph_grad', sph_grad)
print('u grad r', dudr)
print('u grad d', (sph_grad*dudd).sum(-1, keepdim=True))
print('d grad sqrt', (sph_grad*dudd*dddp).sum(-1, keepdim=True))
print('sqrt grad sum', dpds*(sph_grad*dddp*dudd).sum(-1, keepdim=True))
print('sqrt grad sum', dpds*(sph_grad*dddp*dudd).sum(-1, keepdim=True))
print('sum up', sph_grad*1/d + dpds*(sph_grad*dddp*dudd).sum(-1, keepdim=True))
# print('second part', (coords * torch.sum(coords, dim=(-1), keepdim=True) / d**3))
# print('sph_grad_c', sph_grad_c)

