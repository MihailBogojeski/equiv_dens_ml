# %%
import ase
import numpy as np
import pyscf
import time
import os
from pyscf.scf import hf
from pyscf import gto, dft, df, lib
from pyscf.gto import mole
import scipy

import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training.errors import ErrorDict
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
     CubicalGrid, spherical_grid, spherical_radial_sampling
from equiv_dens.training.model_loader import load_model
import equiv_dens.utils.base as utils

from functools import partial

import ase.io
import dftd4.pyscf as d4disp
from pyscf.dft import gen_grid, radi, numint
from pyscf.lib import param
from pymbd import mbd_energy_species
from pymbd.fortran import MBDGeom
from equiv_dens.utils import hirshfeld_analysis
from vdw import to_mbd

hf.MUTE_CHKFILE = True
# %%
# %load_ext autoreload
# %autoreload 2
# %%
geomdir = 'datasets/s66x8'
out_file = 'datasets/s66x8_d4.npy'
data = np.load('datasets/s66x8.npz', allow_pickle=True)
print(data[1][0])
free_atom_dict = np.load('datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy', allow_pickle=True).item()

for i in range(data.shape[0]):
    fname = data[i][0]['atom'].split('/')[-1]
    basis = 'augccpvdz'
    mol = gto.M(atom=os.path.join(geomdir, fname), basis=basis)
    pos = mol.atom_coords()
    nums = mol.atom_charges()
    if len(nums) > 10:
        continue
    if 'WaterWater' not in fname:
        continue
    print('fname', fname)
    atoms = [(nums[i], pos[i, :]) for i in range(len(nums))]
    disp = d4disp.DFTD4Dispersion(mol, xc='pbe')
    dkr = disp.kernel()
    print('D4 energy', utils.hartree_to_kcal(dkr[0]))
    # mf = to_mbd(dft.RKS(mol, xc="PBE"), variant="rsscs").run()
    # print(mf.e_vdw)  # -0.001245831
    # print('D4 gradient', dkr[1])
    grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=1)
    coords, weights = gen_grid.get_partition(mol, grid_spec)
    ao = numint.eval_ao(mol, coords, deriv=0)
    xctype = 'LDA'
    # calculate density using precomputed spherically  averaged free atom MO coefficients
    coeffs = {'mo_coeff': data[i][1]['mo_coeff'],
              'mo_occ': data[i][1]['mo_occ']}
    rho = numint.eval_rho2(mol, ao, xctype=xctype, **coeffs)
    single_atom_dens = []
    for atom in atoms:
        z = atom[0]
        free_atom_basis = free_atom_dict[z]['mo_basis']
        # create molecule object from single atom
        at = gto.M(atom=[atom], basis=free_atom_basis, spin=None, unit='Bohr')
        ao = numint.eval_ao(at, coords, deriv=0)
        xctype = 'LDA'
        # calculate density using precomputed spherically  averaged free atom MO coefficients
        coeffs = {'mo_coeff': free_atom_dict[z]['mo_coeff'],
                  'mo_occ': free_atom_dict[z]['mo_occ']}
        rho_at = numint.eval_rho2(mol, ao, xctype=xctype, **coeffs)
        single_atom_dens.append(rho_at)
    single_atom_dens = np.stack(single_atom_dens, axis=0)
    dens = torch.tensor(rho).unsqueeze(0)
    pos_t = torch.tensor(pos).unsqueeze(0)
    nums_t = torch.tensor(nums).unsqueeze(0)
    coords_t = torch.tensor(coords).unsqueeze(0)
    weights_t = torch.tensor(weights).unsqueeze(0)
    single_atom_dens = torch.tensor(single_atom_dens).unsqueeze(0)
    wA, atomic_charges, dipoles, volume_ratio = hirshfeld_analysis.hirshfeld_partitioning(dens,
                                                                                          single_atom_dens,
                                                                                          pos_t, nums_t,
                                                                                          coords_t, weights_t,
                                                                                          to_bohr=False)
    energy = MBDGeom(pos).mbd_energy_species(utils.numbers_to_symbols(nums), volume_ratio[0], 0.83)
    print('MBD energy', utils.hartree_to_kcal(energy))
# %%
i = 14
print('i', i)
fname = data[i][0]['atom'].split('/')[-1]
basis = 'augccpvdz'
mol = gto.M(atom=os.path.join(geomdir, fname), basis=basis)
pos = mol.atom_coords()
nums = mol.atom_charges()
print('len nums', len(nums))
atoms = [(nums[i], pos[i, :]) for i in range(len(nums))]
disp = d4disp.DFTD4Dispersion(mol, xc='pbe')
dkr = disp.kernel()
print('D4 energy', dkr[0])
mf = to_mbd(dft.RKS(mol, xc="PBE"), variant="rsscs").run()
print(mf.e_vdw)  # -0.001245831
# %%
# print('D4 gradient', dkr[1])
grid_spec = gen_grid.gen_atomic_grids(mol, radi_method=radi.treutler, level=1)
coords, weights = gen_grid.get_partition(mol, grid_spec)
ao = numint.eval_ao(mol, coords, deriv=0)
print('atoms', atoms)
xctype = 'LDA'
# calculate density using precomputed spherically  averaged free atom MO coefficients
coeffs = {'mo_coeff': data[i][1]['mo_coeff'],
          'mo_occ': data[i][1]['mo_occ']}
rho = numint.eval_rho2(mol, ao, xctype=xctype, **coeffs)
single_atom_dens = []
for atom in atoms:
    z = atom[0]
    free_atom_basis = free_atom_dict[z]['mo_basis']
    # create molecule object from single atom
    at = gto.M(atom=[atom], basis=free_atom_basis, spin=None, unit='Bohr')
    print('at charges', at.atom_charges())
    print('at pos', at.atom_coords())
    ao = numint.eval_ao(at, coords, deriv=0)
    xctype = 'LDA'
    # calculate density using precomputed spherically  averaged free atom MO coefficients
    coeffs = {'mo_coeff': free_atom_dict[z]['mo_coeff'],
              'mo_occ': free_atom_dict[z]['mo_occ']}
    rho_at = numint.eval_rho2(mol, ao, xctype=xctype, **coeffs)
    single_atom_dens.append(rho_at)
single_atom_dens = np.stack(single_atom_dens, axis=0)
dens = torch.tensor(rho).unsqueeze(0)
pos_t = torch.tensor(pos).unsqueeze(0)
nums_t = torch.tensor(nums).unsqueeze(0)
coords_t = torch.tensor(coords).unsqueeze(0)
weights_t = torch.tensor(weights).unsqueeze(0)
single_atom_dens = torch.tensor(single_atom_dens).unsqueeze(0)
wA, atomic_charges, dipoles, volume_ratio = hirshfeld_analysis.hirshfeld_partitioning(dens,
                                                                                      single_atom_dens,
                                                                                      pos_t, nums_t,
                                                                                      coords_t, weights_t,
                                                                                      to_bohr=False)
