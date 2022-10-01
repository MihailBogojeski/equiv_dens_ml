"""
This module contains all functionalities required to load atomistic data,
generate batches and compute statistics. It makes use of the ASE database
for atoms [#ase2]_.

References
----------
.. [#ase2] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
   Groves, Hammer, Hargus:
   The atomic simulation environment -- a Python library for working with atoms.
   Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import logging
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from schnetpack.data.partitioning import train_test_split
from pyscf import gto
from pyscf.dft import numint
from pyscf.lib import param
from equiv_dens.utils.grids import spherical_grid,\
    spherical_radial_sampling, treutler_atomic_radii_adjust 
import equiv_dens.utils.base as utils
from equiv_dens.utils import orbitals
from dftpy.formats import ase_io
from pyscf.dft import radi
# import time

logger = logging.getLogger(__name__)


class AtomsDataError(Exception):
    pass


class AtomsDensityData(Dataset):
    ENCODING = "utf-8"
    available_properties = None

    def __init__(
        self,
        np_path,
        density_path,
        orbitals_path,
        density_n_samp=10000,
        subset=None,
        required_properties=[],
        center_positions=True,
        radial_coeffs_file=None,
        L0_coeffs_file=None,
        grid_fn=spherical_grid,
        sampling_fn=spherical_radial_sampling,
        dtype=torch.float32,
        grid_extent=None,
        grid_origin=0,
        fixed_properties={},
        verbose=0,
        use_gpu=False,
        radii_adjust=True,
        projected_density=False,
        cutoff=7.937658158457616,
    ):
        self.density_path = density_path
        self.np_path = np_path
        self.orbitals_path = orbitals_path
        self.density_n_samp = density_n_samp
        self.subset = subset
        self.required_properties = required_properties
        self.radial_coeffs_file = radial_coeffs_file
        self.L0_coeffs_file = L0_coeffs_file
        self.grid_fn = grid_fn
        self.sampling_fn = sampling_fn
        self.dtype = dtype
        self.grid_extent = grid_extent
        self.grid_origin = grid_origin
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.radii_adjust = radii_adjust
        self.energy_centered = False
        self.projected_density = projected_density
        self.cutoff = cutoff
        if required_properties is None:
            self.required_properties = self.available_properties
        self.centered_positions = center_positions
        self.atoms = np.load(np_path, allow_pickle=True).item()
        if self.atoms['atom_numbers'].ndim == 1:
            self.atoms['atom_numbers'] = self.atoms['atom_numbers'][None, :]
        if self.atoms['atom_numbers'].shape[0] != self.atoms['positions'].shape[0]:
            self.atoms['atom_numbers'] = np.tile(self.atoms['atom_numbers'], (self.atoms['positions'].shape[0], 1))

        all_atom_numbers = np.unique(self.atoms['atom_numbers'].flatten())
        all_atom_numbers = all_atom_numbers[all_atom_numbers > 0]
        self.orbital_basis = np.load(orbitals_path, allow_pickle=True).item()
        self.orbital_basis_num = {}
        self.orbital_basis_size = {}
        for key in self.orbital_basis.keys():
            z = utils.symbols_to_numbers([key])[0]
            if z in all_atom_numbers:
                self.orbital_basis_num[z] = (self.orbital_basis[key])
                self.orbital_basis_size[z] = 0
                for orb in self.orbital_basis_num[z]:
                    self.orbital_basis_size[z] += orb[1] * ((2 * orb[2]) + 1)

        self.atoms['shifted_positions'] = self.atoms['positions'] - grid_origin
        if self.density_path is not None:
            calc_results = np.load(density_path, allow_pickle=True)
        self.mols = []
        self.coeffs = []
        self.density_fitting = {}
        self.ions = []
        ase_atoms = utils.npy_to_ase(self.atoms['shifted_positions'], self.atoms['atom_numbers'])
        # for i in range(10):
        for i in range(len(self.atoms['positions'])):
            if self.verbose > 3:
                print('loading sample', i)
            if self.density_path is not None:
                mol_dict, calc_dict = calc_results[i]
                coeff_dict = {'mo_coeff': calc_dict['mo_coeff'], 'mo_occ': calc_dict['mo_occ']}
                mol = gto.Mole(**mol_dict)
                self.mols.append(mol)
                self.coeffs.append(coeff_dict)
                if 'df_coeff' in calc_dict:
                    if 'df_coeffs' not in self.density_fitting.keys():
                        self.density_fitting['auxbasis'] = calc_dict['auxbasis']
                        df_coeffs_split = orbitals.split_df_coeffs(mol_dict['atom'], calc_dict['df_coeff'], self.orbital_basis_size) 
                        self.density_fitting['df_coeffs'] = [df_coeffs_split]
                    else:
                        df_coeffs_split = orbitals.split_df_coeffs(mol_dict['atom'], calc_dict['df_coeff'], self.orbital_basis_size) 
                        self.density_fitting['df_coeffs'].append(df_coeffs_split)
            a = ase_atoms[i]
            a.set_cell(grid_extent)
            self.ions.append(ase_io.ase2ions(a))

        if radial_coeffs_file is not None:
            self.radial_coeffs = {}
            radial_coeffs_atoms = np.load(radial_coeffs_file, allow_pickle=True).item()
            for key in radial_coeffs_atoms.keys():
                z = utils.symbols_to_numbers([key])[0]
                if z in all_atom_numbers:
                    self.radial_coeffs[z] = radial_coeffs_atoms[key]
        else:
            self.radial_coeffs = None

        if L0_coeffs_file is not None:
            self.L0_coeffs = {}
            L0_coeffs_types = np.load(L0_coeffs_file, allow_pickle=True).item()
            for coeff_type in L0_coeffs_types.keys():
                self.L0_coeffs[coeff_type] = []
                for key in L0_coeffs_types[coeff_type].keys():
                    z = utils.symbols_to_numbers([key])[0]
                    if z in all_atom_numbers:
                        L0_atom = L0_coeffs_types[coeff_type][key]
                        self.L0_coeffs[coeff_type][z] = L0_atom
        else:
            self.L0_coeffs = None

        self.grid_spec = grid_fn(self.atoms)
        if isinstance(self.grid_spec, dict):
            for key in self.grid_spec.keys():
                self.grid_spec[key] = (self.grid_spec[key][0].type(self.dtype),
                                       self.grid_spec[key][1].type(self.dtype))
        if self.use_gpu:
            if isinstance(self.grid_spec, dict):
                for key in self.grid_spec.keys():
                    self.grid_spec[key] = (self.grid_spec[key][0].cuda(),
                                           self.grid_spec[key][1].cuda())  # convert Bohr grid to Angstrom
        self.fixed_properties = fixed_properties
        if self.verbose > 1:
            print('dataset radial coeffs', self.radial_coeffs)
            print('dataset L0 coeffs', self.L0_coeffs)

        print('finished init')

    def create_splits(self, num_train=None, num_val=None, split_file=None):
        warnings.warn(
            "create_splits is deprecated, " +
            "use schnetpack.data.train_test_split instead",
            DeprecationWarning,
        )
        return train_test_split(self, num_train, num_val, split_file)

    def create_subset(self, idx):
        """
        Returns a new dataset that only consists of provided indices.
        Args:
            idx (numpy.ndarray): subset indices

        Returns:
            schnetpack.data.AtomsData: dataset with subset of original data
        """
        idx = np.array(idx)
        subidx = (
            idx if self.subset is None or len(idx) == 0 else np.array(self.subset)[idx]
        )
        return type(self)(
            np_path=self.np_path,
            desnity_path=self.density_path,
            orbitals_path=self.orbitals_path,
            density_n_samp=self.density_n_samp,
            subset=subidx,
            required_propertie=self.required_properties,
            center_positions=self.centered_positions,
            radial_coeffs_file=self.radial_coeffs_file,
            grid_fn=self.grid_fn,
            sampling_fn=self.sampling_fn,
            dtype=self.dtype,
            grid_extent=self.grid_extent,
            grid_origin=self.grid_origin,
            fixed_properties=self.fixed_properties,
        )

    def __len__(self):
        if self.subset is None:
            if len(self.mols) > 0:
                return len(self.mols)
            else:
                return self.atoms['positions'].shape[0]
        return len(self.subset)

    # property collection is done by the get_properties function when passed as collate_fn
    def __getitem__(self, idx):
        return idx

    def _subset_index(self, idx):
        # get row
        if self.subset is not None:
            idx = self.subset[idx]
        return idx

    def center_energy(self, energy_mean):
        if self.verbose > 0:
            print('energy before centering', np.mean(self.atoms['energy']))
            print('energy mean', energy_mean)
        if not self.energy_centered:
            self.atoms['energy'] -= energy_mean
            self.energy_centered = True
        if self.verbose > 0:
            print('energy after centering', np.mean(self.atoms['energy']))

    @property
    def energy(self):
        if self.subset is None:
            return self.atoms['energy']
        else:
            return self.atoms['energy'][self.subset]

    @property
    def forces(self):
        if self.subset is None:
            return self.atoms['forces']
        else:
            return self.atoms['forces'][self.subset]

    @property
    def atom_numbers(self):
        if self.subset is None:
            return self.atoms['atom_numbers']
        else:
            return self.atoms['atom_numbers'][self.subset]

    # collects the molecular properties for the batch, should be used as collate_fn
    def get_properties(self, idx):
        idx = self._subset_index(idx)
        if not hasattr(idx, '__len__'):
            idx = [idx]

        # extract properties
        atom_numbers = self.atoms['atom_numbers'][idx]
        atom_props = {'positions': self.atoms['positions'][idx]}
        mol_props = {}
        for pname in self.required_properties:
            if pname in self.atoms.keys():
                if self.atoms[pname][0].shape[0] > 1:
                    atom_props[pname] = self.atoms[pname][idx]
                else:
                    mol_props[pname] = self.atoms[pname][idx]
            elif pname == 'df_coeffs':
                atom_props[pname] = [self.density_fitting[pname][i] for i in idx]

        # print('atom numbers', atom_numbers)
        # print('props', atom_props)
        atom_numbers, props = utils.compress_batch_atoms(atom_numbers, atom_props, basis_size=self.orbital_basis_size)
        props.update(mol_props)
        # atom_numbers = torch.from_numpy(atom_numbers).type(self.dtype)
        positions = torch.from_numpy(props['positions']).type(self.dtype)
        properties = {}
        for pname in self.required_properties:
            # fallback for properties stored directly
            # in the row
            if pname == 'coords' or pname == 'density':
                properties['coords'], properties['coord_weights'] = self.get_coords(positions, atom_numbers)
                if pname == 'density':
                    if self.projected_density:
                        properties[pname] = self.sample_projected_density(idx, properties['coords'])
                    else:
                        properties[pname] = self.sample_density(idx, properties['coords'])
            else:
                properties[pname] = torch.from_numpy(props[pname]).type(self.dtype)

        # extract/calculate structure
        properties['atom_numbers_first_positions'] = utils.get_atom_num_first_positions(atom_numbers)
        properties['positions'] = positions
        properties['atom_numbers'] = torch.LongTensor(atom_numbers)
        properties['atom_mask'] = properties['atom_numbers'] > 0
        properties['idx'] = torch.LongTensor(idx).unsqueeze(-1)
        # properties['ions'] = [self.ions[i] for i in idx]
        # print('positions', positions)
        if self.centered_positions:
            # print('atom center', positions.mean(axis=0))
            positions -= torch.sum(positions * atom_numbers, 0)/torch.sum(atom_numbers, 1)
        properties["_idx"] = torch.LongTensor(np.array(idx, dtype=np.int))
        nl = utils.TorchNeighborList(self.cutoff)
        idx_is, idx_js, _ = nl.get_neighbors(properties)
        neighbor_batch_idx = []
        prev_max=0
        for i in range(len(idx_is)):
            idx_is[i] += prev_max
            idx_js[i] += prev_max 
            max_i = torch.max(idx_is[i])
            max_j = torch.max(idx_is[i])
            prev_max = max(max_i, max_j) + 1
            neighbor_batch_idx.append(torch.ones_like(idx_is[i]) * i)
        
        atom_batch_idx = np.zeros_like(atom_numbers)
        for i in range(len(atom_numbers)):
            atom_batch_idx[i, :] = i
        atom_batch_idx = torch.LongTensor(atom_batch_idx)


        idx_is = torch.cat(idx_is, dim=0)
        idx_js = torch.cat(idx_js, dim=0)
        neighbor_batch_idx = torch.cat(neighbor_batch_idx, dim=0)
        properties['idx_i'] = idx_is
        properties['idx_j'] = idx_js
        properties['neighbor_batch_idx'] = neighbor_batch_idx
        properties['batch_atom_numbers'] = properties['atom_numbers'] * 1
        properties['batch_atom_mask'] = (properties['atom_mask'] * 1).type(torch.bool)
        properties['batch_positions'] = properties['positions'] * 1
        properties['positions'] = positions.view(1, -1, *properties['positions'].shape[2:])
        properties['atom_numbers'] = properties['batch_atom_numbers'].flatten()
        properties['atom_mask'] = properties['batch_atom_mask'].flatten()
        properties['atom_numbers'] = properties['atom_numbers'][properties['atom_mask']].view(1, -1)
        properties['atom_batch_idx'] = atom_batch_idx.flatten()
        properties['atom_batch_idx'] = properties['atom_batch_idx'][properties['atom_mask']].view(1, -1)
        properties['positions'] = properties['positions'][:, properties['atom_mask']]
        if 'forces' in properties:
            properties['batch_forces'] = properties['forces'] * 1
            properties['forces'] = properties['forces'].view(1, -1, *properties['forces'].shape[2:])
            properties['forces'] = properties['forces'][:, properties['atom_mask']]

        for prop in self.fixed_properties.keys():
            properties[prop] = self.fixed_properties[prop]

        return properties

    def get_coords(self, positions, atom_numbers):
        if self.use_gpu:
            positions = positions.cuda()
        if self.radii_adjust:
            f_radii_adjust = treutler_atomic_radii_adjust(atom_numbers,
                                                          radi.BRAGG_RADII)
            sample_coords, coord_weights = self.sampling_fn(self.grid_spec, self.density_n_samp,
                                                            atom_numbers,
                                                            positions, radii_adjust=f_radii_adjust)
        else:
            sample_coords, coord_weights = self.sampling_fn(self.grid_spec, self.density_n_samp,
                                                            atom_numbers,
                                                            positions)
        # print('density nans', torch.sum(torch.isnan(properties[pname])))
        coords = sample_coords.type(self.dtype)
        coord_weights = coord_weights.type(self.dtype)

        return coords, coord_weights

    def sample_density(self, idx, sample_coords):
        scaled_sample_coords = sample_coords.detach().cpu().numpy() / param.BOHR  # convert Angstrom grid to Bohr
        dens = torch.zeros((sample_coords.shape[0], sample_coords.shape[1]), dtype=self.dtype)
        if len(self.mols) > 0:
            for c, i in enumerate(idx):
                # mol_start = time.time()
                # print('c, i', c, i)
                mol = self.mols[i]
                if not mol._built:
                    # build_start = time.time()
                    if self.verbose > 3:
                        print('building mol', i)
                    mol.build()
                    # print('build time', time.time() - build_start)
                coeff_dict = self.coeffs[i]
                # ao_start = time.time()
                ao = numint.eval_ao(mol, scaled_sample_coords[c])
                # print('ao time', time.time() - ao_start)
                # rho_start = time.time()
                if coeff_dict['mo_occ'].ndim > 1:
                    rho = 0
                    for j in range(coeff_dict['mo_occ'].shape[0]):
                        rho += numint.eval_rho2(mol, ao, mo_occ=coeff_dict['mo_occ'][j], 
                                                mo_coeff=coeff_dict['mo_coeff'][j])
                else:
                    rho = numint.eval_rho2(mol, ao, **coeff_dict)
                # print('rho time', time.time() - rho_start)
                dens[c, :] = torch.from_numpy(rho).type(self.dtype)
                # print('mol_time', time.time() - mol_start)

        return dens

    def sample_projected_density(self, idx, sample_coords):
        scaled_sample_coords = sample_coords.detach().cpu().numpy() / param.BOHR  # convert Angstrom grid to Bohr
        dens = torch.zeros((sample_coords.shape[0], sample_coords.shape[1]), dtype=self.dtype)
        if len(self.density_fitting) == 0:
            raise RuntimeError("Density fitting coefficients MUST be present in order to create projected density")
        if len(self.mols) > 0:
            for c, i in enumerate(idx):
                # mol_start = time.time()
                # print('c, i', c, i)
                mol = self.mols[i]
                if not mol._built and mol.basis != self.density_fitting['auxbasis']:
                    mol.basis = self.density_fitting['auxbasis']
                    # build_start = time.time()
                    if self.verbose > 3:
                        print('building mol', i)
                    mol.build()
                    # print('build time', time.time() - build_start)
                # df_coeff = np.concatenate(self.density_fitting['df_coeffs'][i])
                df_coeff = np.concatenate([coeff[1] for coeff in self.density_fitting['df_coeffs'][i]])
                # ao_start = time.time()
                ao = numint.eval_ao(mol, scaled_sample_coords[c])
                # print('ao time', time.time() - ao_start)
                # rho_start = time.time()
                if df_coeff.ndim > 1:
                    rho = 0
                    for j in range(df_coeff.shape[0]):
                        rho += np.einsum('ij,j->i', ao, df_coeff[j])
                else:
                    rho = np.einsum('ij,j->i', ao, df_coeff)
                # print('rho time', time.time() - rho_start)
                dens[c, :] = torch.from_numpy(rho).type(self.dtype)
                # print('mol_time', time.time() - mol_start)

        return dens

    def add_fixed_properties(self, property_dict):
        for prop in property_dict.keys():
            self.fixed_properties[prop] = property_dict[prop]
