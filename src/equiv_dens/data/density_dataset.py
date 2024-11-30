"""This module contains all functionalities required to load atomistic data,
generate batches and compute statistics. It makes use of the ASE database
for atoms [#ase2].

References
----------
.. [#ase2] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
   Groves, Hammer, Hargus:
   The atomic simulation environment -- a Python library for working with atoms.
   Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pyscf import gto
from pyscf.lib import param
from equiv_dens.utils.grids import (
    spherical_grid,
    spherical_radial_sampling,
    treutler_atomic_radii_adjust,
)
import equiv_dens.utils.base as utils
from equiv_dens.utils import orbitals
from pyscf.dft import gen_grid, radi
import time

logger = logging.getLogger(__name__)


class AtomsDataError(Exception):
    pass


class AtomsDensityData(Dataset):
    ENCODING = "utf-8"

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
        pyscf_grid=False,
        pyscf_rotate=False,
        sampling_fn=spherical_radial_sampling,
        dtype=torch.float32,
        grid_extent=None,
        grid_origin=0,
        fixed_properties={},
        verbose=0,
        timing=False,
        use_gpu=False,
        radii_adjust=True,
        projected_density=False,
        cutoff=7.937658158457616,  # cutoff distance for neighborhood selection
        ao_matrix_cutoff=7.937658158457616,  # cutoff distance for ao matrix construction
        df_loss_weights=False,
        calc_data=False,
        atom_dens_path=None,
        atom_dens_type=None,
        split_atom_dens=False,
        density_grad=False,
        calc_basis_path=None,
    ):
        self.density_path = density_path
        self.np_path = np_path
        self.orbitals_path = orbitals_path
        self.density_n_samp = density_n_samp
        self.subset = subset
        self.required_properties = [prop for prop in required_properties]
        self.radial_coeffs_file = radial_coeffs_file
        self.L0_coeffs_file = L0_coeffs_file
        self.grid_fn = grid_fn
        self.pyscf_grid = pyscf_grid
        self.pyscf_rotate = pyscf_rotate
        self.sampling_fn = sampling_fn
        self.dtype = dtype
        self.grid_extent = grid_extent
        self.grid_origin = grid_origin
        self.verbose = verbose
        self.timing = timing
        self.use_gpu = use_gpu
        self.radii_adjust = radii_adjust
        self.energy_centered = False
        self.timing = timing
        self.projected_density = projected_density
        self.cutoff = cutoff
        self.ao_matrix_cutoff = ao_matrix_cutoff
        self.calc_data = calc_data
        self.atom_dens_type = atom_dens_type
        self.split_atom_dens = split_atom_dens
        self.density_grad = density_grad
        if "dipole_moment" in self.required_properties:
            if "density" not in self.required_properties:
                self.required_properties.append("density")
            self.required_properties.remove("dipole_moment")
            self.calc_dpm = True
        else:
            self.calc_dpm = False
        self.centered_positions = center_positions
        self.atoms = np.load(np_path, allow_pickle=True).item()
        if self.atoms["atom_numbers"].ndim == 1:
            self.atoms["atom_numbers"] = self.atoms["atom_numbers"][None, :]
        if self.atoms["atom_numbers"].shape[0] != self.atoms["positions"].shape[0]:
            self.atoms["atom_numbers"] = np.tile(
                self.atoms["atom_numbers"], (self.atoms["positions"].shape[0], 1)
            )

        if atom_dens_path is not None:
            self.atom_dens = np.load(atom_dens_path, allow_pickle=True).item()
        else:
            self.atom_dens = None

        all_atom_numbers = np.unique(self.atoms["atom_numbers"].flatten())
        all_atom_numbers = all_atom_numbers[all_atom_numbers > 0]
        self.orbital_basis = np.load(orbitals_path, allow_pickle=True).item()
        self.orbital_basis_num = orbitals.get_num_basis(self.orbital_basis, all_atom_numbers)
        self.orbital_basis_size = orbitals.get_basis_size(self.orbital_basis_num)

        if calc_basis_path is not None:
            self.calc_basis = np.load(calc_basis_path, allow_pickle=True).item()
            self.calc_basis_num = orbitals.get_num_basis(self.calc_basis, all_atom_numbers)
            self.calc_basis_size = orbitals.get_basis_size(self.calc_basis_num)
        else:
            self.calc_basis = None
            self.calc_basis_num = None
            self.calc_basis_size = None

        self.atoms["shifted_positions"] = self.atoms["positions"] - grid_origin
        calc_results = []
        if self.density_path is not None:
            calc_results = np.load(density_path, allow_pickle=True)
        self.mols = []
        self.coeffs = []
        self.calc_dict = []
        self.density_fitting = {}
        calc_props = ['hamiltonian_matrix', 'density_matrix', 'mo_coeff', 'mo_energies']
        ase_atoms = utils.npy_to_ase(
            self.atoms["shifted_positions"], self.atoms["atom_numbers"]
        )
        # for i in range(10):
        for i in range(len(self.atoms["positions"])):
            if self.verbose > 3:
                print("loading sample", i)
            if self.density_path is not None:
                mol_dict, calc_dict = calc_results[i]
                coeff_dict = {
                    "mo_coeff": calc_dict["mo_coeff"],
                    "mo_occ": calc_dict["mo_occ"],
                }
                calc_prop_dict = {}
                mol = gto.Mole(**mol_dict)

                for calc_prop in calc_props:
                    if calc_prop in calc_dict and calc_prop in required_properties:
                        if 'coeff' in calc_prop:
                            print(calc_prop, 'shape', calc_dict[calc_prop].shape)
                            calc_prop_dict[calc_prop] = orbitals.split_ao_coeffs(
                                mol_dict['atom'],
                                calc_dict[calc_prop],
                                self.calc_basis_size,
                            )
                        elif 'matrix' in calc_prop:
                            calc_prop_dict[calc_prop] = orbitals.split_ao_matrix(
                                mol_dict['atom'],
                                calc_dict[calc_prop],
                                self.calc_basis_size,
                            )
                        else:
                            calc_prop_dict[calc_prop] = calc_dict[calc_prop]

                self.calc_dict.append(calc_prop_dict)
                self.mols.append(mol)
                self.coeffs.append(coeff_dict)
                if "df_coeff" in calc_dict:
                    if "df_coeffs" not in self.density_fitting.keys():
                        self.density_fitting["auxbasis"] = calc_dict["auxbasis"]
                        print('df_coeff shape', calc_dict['df_coeff'].shape)
                        df_coeffs_split = orbitals.split_ao_coeffs(
                            mol_dict["atom"],
                            calc_dict["df_coeff"],
                            self.orbital_basis_size,
                        )
                        self.density_fitting["df_coeffs"] = [df_coeffs_split]
                    else:
                        df_coeffs_split = orbitals.split_ao_coeffs(
                            mol_dict["atom"],
                            calc_dict["df_coeff"],
                            self.orbital_basis_size,
                        )
                        self.density_fitting["df_coeffs"].append(df_coeffs_split)
            a = ase_atoms[i]
            a.set_cell(grid_extent)

        if radial_coeffs_file is not None:
            self.radial_coeffs = {}
            radial_coeffs_atoms = np.load(radial_coeffs_file, allow_pickle=True).item()
            for key in radial_coeffs_atoms.keys():
                z = utils.symbols_to_numbers([key])[0]
                if z in all_atom_numbers:
                    self.radial_coeffs[z] = radial_coeffs_atoms[key]
            if df_loss_weights:
                self.coeff_weights = {}
                for z in self.orbital_basis_num.keys():
                    self.coeff_weights[z] = []
                    for j in range(len(self.orbital_basis_num[z])):
                        # print('init coeffs j', j)
                        orb = self.orbital_basis_num[z][j]
                        L = orb[2]
                        key = (z, L)
                        width = self.radial_coeffs[z][j][0]
                        scale = self.radial_coeffs[z][j][1] / orbitals.pyscf_gto_factor
                        integral = 1 / (orbitals.gto_norm(L, width))
                        self.coeff_weights[z].append(scale * integral / (2 * L + 1))
            else:
                self.coeff_weights = None
        else:
            self.radial_coeffs = None
            self.coeff_weights = None

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

        if self.pyscf_grid:
            self.grid_spec = grid_fn(self.atoms, bohr=True)
        else:
            self.grid_spec = grid_fn(self.atoms)
        if isinstance(self.grid_spec, dict):
            for key in self.grid_spec.keys():
                self.grid_spec[key] = (
                    self.grid_spec[key][0].type(self.dtype),
                    self.grid_spec[key][1].type(self.dtype),
                )
        if self.use_gpu:
            if isinstance(self.grid_spec, dict):
                for key in self.grid_spec.keys():
                    self.grid_spec[key] = (
                        self.grid_spec[key][0].cuda(),
                        self.grid_spec[key][1].cuda(),
                    )  # convert Bohr grid to Angstrom
        self.fixed_properties = fixed_properties
        if self.verbose > 1:
            print("dataset radial coeffs", self.radial_coeffs)
            print("dataset L0 coeffs", self.L0_coeffs)

        if self.verbose > 0:
            print("finished init")

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
            density_path=self.density_path,
            orbitals_path=self.orbitals_path,
            density_n_samp=self.density_n_samp,
            subset=subidx,
            required_properties=self.required_properties,
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
                return self.atoms["positions"].shape[0]
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
            print("energy before centering", np.mean(self.atoms["energy"]))
            print("energy mean", energy_mean)
        if not self.energy_centered:
            self.atoms["energy"] -= energy_mean
            self.energy_centered = True
        if self.verbose > 0:
            print("energy after centering", np.mean(self.atoms["energy"]))

    def normalize_energy(self, atom_energies):
        """Normalize energies by subtracting the sum of atom energies.

        Args:
        atom_energies (dict): dictionary of atom energies
        """
        if self.verbose > 0:
            print("energy before normalization", np.mean(self.atoms["energy"]))
        if not self.energy_centered:
            for i in range(self.atoms["atom_numbers"].shape[0]):
                at_en = 0
                for j in range(self.atoms["atom_numbers"].shape[1]):
                    at_en += atom_energies[self.atoms["atom_numbers"][i, j]]
                self.atoms["energy"][i] -= at_en
            self.energy_centered = True
        if self.verbose > 0:
            print("energy after normalization", np.mean(self.atoms["energy"]))

    @property
    def energy(self):
        if "energy" in self.required_properties:
            if self.subset is None:
                return self.atoms["energy"]
            else:
                return self.atoms["energy"][self.subset]
        else:
            raise KeyError("Energy not in required properties")

    @property
    def forces(self):
        if "forces" in self.required_properties:
            if self.subset is None:
                return self.atoms["forces"]
            else:
                return self.atoms["forces"][self.subset]
        else:
            raise KeyError("Forces not in required properties")

    @property
    def atom_numbers(self):
        if self.subset is None:
            return self.atoms["atom_numbers"]
        else:
            return self.atoms["atom_numbers"][self.subset]

    # collects the molecular properties for the batch, should be used as collate_fn
    def get_properties(self, idx):
        idx = self._subset_index(idx)
        if not hasattr(idx, "__len__"):
            idx = [idx]

        # extract properties
        props_start = time.time()
        atom_numbers = self.atoms["atom_numbers"][idx]
        atom_props = {"positions": self.atoms["positions"][idx]}
        mol_props = {}
        for pname in self.required_properties:
            if pname in self.atoms.keys():
                if self.atoms[pname][0].shape[0] > 1:
                    atom_props[pname] = self.atoms[pname][idx]
                else:
                    mol_props[pname] = self.atoms[pname][idx]
            elif pname == "df_coeffs":
                atom_props[pname] = [self.density_fitting[pname][i] for i in idx]
            elif pname in ['hamiltonian_matrix', 'density_matrix']:
                atom_props[pname] = [self.calc_dict[i][pname] for i in idx]
            elif pname == 'mo_coeff':
                atom_props[pname] = [self.calc_dict[i][pname] for i in idx]
                mol_props['mo_occ'] = [self.coeffs[i]['mo_occ'] for i in idx]
                mol_props['mo_occ'] = np.stack(mol_props['mo_occ'], axis=0)
            elif pname == 'mo_energies':
                mol_props[pname] = [self.calc_dict[i][pname] for i in idx]
                mol_props[pname] = np.stack(mol_props[pname], axis=0)
        # print('atom numbers', atom_numbers)
        # print('props', atom_props)
        atom_numbers, props = utils.compress_batch_atoms(
            atom_numbers, atom_props,
            df_basis_size=self.orbital_basis_size,
            ao_basis_size=self.calc_basis_size,
        )
        props.update(mol_props)
        # atom_numbers = torch.from_numpy(atom_numbers).type(self.dtype)
        if "positions" not in props.keys():
            print("idx", idx)
            print("atom_props", atom_props)
            print("props", props)
        positions = torch.from_numpy(props["positions"]).type(self.dtype)
        if self.centered_positions:
            # print('atom center', positions.mean(axis=0))
            pos_shift = -(torch.mean(positions, dim=1, keepdim=True))
        else:
            pos_shift = 0
        positions += pos_shift
        properties = {}
        if self.timing:
            print("props time", time.time() - props_start)
        dens_start = time.time()
        for pname in self.required_properties:
            # fallback for properties stored directly
            # in the row
            if pname == "coords" or ("density" in pname and "matrix" not in pname):
                coords_start = time.time()
                if self.pyscf_grid:
                    properties["coords"], properties["coord_weights"] = (
                        self.get_pyscf_coords(idx)
                    )
                else:
                    properties["coords"], properties["coord_weights"] = self.get_coords(
                        positions, atom_numbers
                    )
                if self.timing:
                    print("coords time:", time.time() - coords_start)
                if pname == "density":
                    density_start = time.time()
                    if self.projected_density:
                        properties[pname] = self.sample_projected_density(
                            idx, properties["coords"] - pos_shift,
                            density_grad=self.density_grad,
                        )
                    else:
                        properties[pname] = self.sample_density(
                            idx, properties["coords"] - pos_shift,
                            density_grad=self.density_grad,
                        )
                    if self.density_grad:
                        properties[pname + "_grad"] = properties[pname][..., 1:]
                        properties[pname] = properties[pname][..., 0]
                    if self.timing:
                        print("density time:", time.time() - density_start)
                    if self.atom_dens is not None:
                        if self.split_atom_dens:
                            properties["atom_density_split"] = self.sample_atom_density(
                                positions,
                                torch.LongTensor(atom_numbers),
                                properties["coords"],
                                individual_dens=True,
                                density_grad=self.density_grad,
                            )
                            properties["atom_density"] = torch.sum(
                                properties["atom_density_split"], dim=1
                            )
                            if self.density_grad:
                                properties["atom_density_split_grad"] = properties["atom_density_split"][..., 1:]
                                properties["atom_density_split"] = properties["atom_density_split"][..., 0]
                        else:
                            properties["atom_density"] = self.sample_atom_density(
                                positions,
                                torch.LongTensor(atom_numbers),
                                properties["coords"],
                                density_grad=self.density_grad,
                            )
                        if self.density_grad:
                            properties["atom_density_grad"] = properties["atom_density"][..., 1:]
                            properties["atom_density"] = properties["atom_density"][..., 0]
            else:
                properties[pname] = torch.from_numpy(props[pname]).type(self.dtype)
        if self.timing:
            print("dens props time", time.time() - dens_start)
        # extract/calculate structure
        properties['pos_shift'] = pos_shift
        properties["atom_numbers_first_positions"] = utils.get_atom_num_first_positions(
            atom_numbers
        )
        properties["positions"] = positions
        properties["atom_numbers"] = torch.LongTensor(atom_numbers)
        properties["atom_mask"] = properties["atom_numbers"] > 0
        properties["idx"] = torch.LongTensor(idx).unsqueeze(-1)
        # print('positions', positions)
        properties["_idx"] = torch.LongTensor(np.array(idx, dtype=int))
        neighbor_start = time.time()
        # print(f"cutoff: {self.cutoff}")
        # print(f"ao_matrix_cutoff: {self.ao_matrix_cutoff}")
        nl = utils.TorchNeighborList(self.cutoff)
        idx_is, idx_js, _ = nl.get_neighbors(properties)

        nl_ao_matrix = utils.TorchNeighborList(self.ao_matrix_cutoff)
        idx_is_ao_matrix, idx_js_ao_matrix, _ = nl_ao_matrix.get_neighbors(properties)

        neighbor_batch_idx = []
        prev_max = 0
        for i in range(len(idx_is)):
            n_atoms = torch.sum(properties["atom_mask"][i])
            idx_is[i] += prev_max
            idx_js[i] += prev_max
            prev_max += n_atoms
            neighbor_batch_idx.append(torch.ones_like(idx_is[i]) * i)

        neighbor_ao_matrix_batch_idx = []
        prev_max = 0
        for i in range(len(idx_is_ao_matrix)):
            n_atoms = torch.sum(properties["atom_mask"][i])
            idx_is_ao_matrix[i] += prev_max
            idx_js_ao_matrix[i] += prev_max
            prev_max += n_atoms
            neighbor_ao_matrix_batch_idx.append(torch.ones_like(idx_is_ao_matrix[i]) * i)

        atom_batch_idx = np.zeros_like(atom_numbers)
        for i in range(len(atom_numbers)):
            atom_batch_idx[i, :] = i
        atom_batch_idx = torch.LongTensor(atom_batch_idx)
        if self.timing:
            print("neighbor time", time.time() - neighbor_start)

        final_start = time.time()
        idx_is = torch.cat(idx_is, dim=0)
        idx_js = torch.cat(idx_js, dim=0)
        idx_is_ao_matrix = torch.cat(idx_is_ao_matrix, dim=0)
        idx_js_ao_matrix = torch.cat(idx_js_ao_matrix, dim=0)
        neighbor_batch_idx = torch.cat(neighbor_batch_idx, dim=0)
        neighbor_ao_matrix_batch_idx = torch.cat(neighbor_ao_matrix_batch_idx, dim=0)
        properties["idx_i"] = idx_is
        properties["idx_j"] = idx_js
        properties["idx_i_ao_matrix"] = idx_is_ao_matrix
        properties["idx_j_ao_matrix"] = idx_js_ao_matrix
        properties["neighbor_batch_idx"] = neighbor_batch_idx
        properties["neighbor_ao_matrix_batch_idx"] = neighbor_ao_matrix_batch_idx
        properties["batch_atom_numbers"] = properties["atom_numbers"] * 1
        properties["batch_atom_mask"] = (properties["atom_mask"] * 1).type(torch.bool)
        properties["batch_positions"] = properties["positions"] * 1
        properties["positions"] = positions.view(
            1, -1, *properties["positions"].shape[2:]
        )
        properties["atom_numbers"] = properties["batch_atom_numbers"].flatten()
        properties["atom_mask"] = properties["batch_atom_mask"].flatten()
        batch_nz = properties["atom_mask"].to(properties["positions"])
        batch_idx_pos = batch_nz * torch.arange(len(batch_nz)).to(batch_nz)
        properties["batch_idx_pos"] = batch_idx_pos[properties["atom_mask"]].to(
            torch.long
        )
        properties["atom_numbers"] = properties["atom_numbers"][
            properties["atom_mask"]
        ].view(1, -1)
        properties["atom_batch_idx"] = atom_batch_idx.flatten()
        properties["atom_batch_idx"] = properties["atom_batch_idx"][
            properties["atom_mask"]
        ].view(1, -1)
        properties["positions"] = properties["positions"][:, properties["atom_mask"]]
        if "forces" in properties:
            properties["batch_forces"] = properties["forces"] * 1
            properties["forces"] = properties["forces"].view(
                1, -1, *properties["forces"].shape[2:]
            )
            properties["forces"] = properties["forces"][:, properties["atom_mask"]]

        if self.calc_dpm:
            properties = orbitals.calc_dipole_moment(properties)

        for prop in self.fixed_properties.keys():
            properties[prop] = self.fixed_properties[prop]

        # if self.calc_data:
        #     for i in idx:
        #         mo_coeff = (
        #             torch.tensor(self.coeffs[i]["mo_coeff"])
        #             .unsqueeze(0)
        #             .to(properties["positions"])
        #         )
        #         mo_occ = (
        #             torch.tensor(self.coeffs[i]["mo_occ"])
        #             .unsqueeze(0)
        #             .to(properties["positions"])
        #         )
        #         if "mo_coeff" not in properties.keys():
        #             properties["mo_coeff"] = mo_coeff
        #             properties["mo_occ"] = mo_occ
        #         else:
        #             properties["mo_coeff"] = torch.cat(
        #                 [properties["mo_coeff"], mo_coeff], dim=0
        #             )
        #             properties["mo_occ"] = torch.cat(
        #                 [properties["mo_occ"], mo_occ], dim=0
        #             )
        if self.timing:
            print("final props time", time.time() - final_start)
            print("total time", time.time() - props_start)
        return properties

    def get_basic_properties(self, idx):
        idx = self._subset_index(idx)
        if not hasattr(idx, "__len__"):
            idx = [idx]

        # extract properties
        atom_numbers = self.atoms["atom_numbers"][idx]
        atom_props = {"positions": self.atoms["positions"][idx]}
        mol_props = {}
        for pname in self.required_properties:
            if pname in self.atoms.keys():
                if self.atoms[pname][0].shape[0] > 1:
                    atom_props[pname] = self.atoms[pname][idx]
                else:
                    mol_props[pname] = self.atoms[pname][idx]
            elif pname == "df_coeffs":
                atom_props[pname] = [self.density_fitting[pname][i] for i in idx]

        # print('atom numbers', atom_numbers)
        # print('props', atom_props)
        atom_numbers, props = utils.compress_batch_atoms(
            atom_numbers, atom_props, basis_size=self.orbital_basis_size
        )
        props.update(mol_props)
        # atom_numbers = torch.from_numpy(atom_numbers).type(self.dtype)
        positions = torch.from_numpy(props["positions"]).type(self.dtype)
        properties = {}
        # extract/calculate structure
        properties["positions"] = positions
        properties["atom_numbers"] = torch.LongTensor(atom_numbers)
        atom_numbers = torch.tensor(atom_numbers).to(positions)
        # print('positions', positions)
        if self.centered_positions:
            # print('atom center', positions.mean(axis=0))
            positions -= torch.sum(positions * atom_numbers, 0) / torch.sum(
                atom_numbers, 1
            )
        properties["_idx"] = torch.LongTensor(np.array(idx, dtype=int))

        for prop in self.fixed_properties.keys():
            properties[prop] = self.fixed_properties[prop]

        return properties

    def get_pyscf_coords(self, idx):
        """
        Get density grid coordinates using PySCF gen_grid.

        Args:
        idx (list of int): index of molecule(s) to get coordinates for
        Returns:
        coords (torch.Tensor): coordinates of grid points
        weights (torch.Tensor): integration weights of grid points
        """
        # mol = utils.npy_to_ase(dataset.atoms['positions'][0:1], dataset.atoms['atom_numbers'][0:1])[0]
        # utils.npy_to_ase(dataset.atoms['positions'][0:1], dataset.atoms['atom_numbers'][0:1])[0]
        all_coords = []
        all_weights = []
        for i in idx:
            mol = self.mols[i]
            if not mol._built:
                build_start = time.time()
                if self.verbose > 3:
                    print("building mol", i)
                mol.build()
                if self.timing:
                    print("build time", time.time() - build_start)
            if self.pyscf_rotate:
                rot_spec = {
                    key: (
                        self.grid_spec[key][0]
                        @ utils.torch_random_rotation_matrix().to(
                            self.grid_spec[key][0]
                        ),
                        self.grid_spec[key][1],
                    )
                    for key in self.grid_spec.keys()
                }
            else:
                rot_spec = self.grid_spec
            coords, weights = gen_grid.get_partition(mol, rot_spec)
            coords = torch.tensor(coords)
            weights = torch.tensor(weights)
            # print('coords shape', coords.shape)
            # print('weights shape', weights)
            # print('density n samp', self.density_n_samp)
            if self.density_n_samp > coords.shape[0]:
                coords = torch.tensor(coords).to(self.dtype)
                weights = torch.tensor(weights).to(self.dtype)
            else:
                rand_idx = np.random.choice(
                    np.arange(coords.shape[0]), size=self.density_n_samp, replace=False
                )
                coords = torch.tensor(coords[rand_idx]).to(self.dtype)
                weights = torch.tensor(weights[rand_idx]).to(self.dtype)
            all_coords.append(coords)
            all_weights.append(weights)
        pad_coords = (
            nn.utils.rnn.pad_sequence(all_coords, batch_first=True, padding_value=0)
            * utils.to_angstrom
        )
        pad_weights = nn.utils.rnn.pad_sequence(
            all_weights, batch_first=True, padding_value=0
        )
        return pad_coords, pad_weights

    def get_coords(self, positions, atom_numbers):
        if self.use_gpu:
            positions = positions.cuda()
        if self.radii_adjust:
            f_radii_adjust = treutler_atomic_radii_adjust(
                atom_numbers, radi.BRAGG_RADII
            )
            sample_coords, coord_weights = self.sampling_fn(
                self.grid_spec,
                self.density_n_samp,
                atom_numbers,
                positions,
                radii_adjust=f_radii_adjust,
            )
        else:
            start = time.time()
            sample_coords, coord_weights = self.sampling_fn(
                self.grid_spec, self.density_n_samp, atom_numbers, positions
            )
            print("sample coords time", time.time() - start)
        # print('density nans', torch.sum(torch.isnan(properties[pname])))
        coords = sample_coords.type(self.dtype)
        coord_weights = coord_weights.type(self.dtype)

        return coords, coord_weights

    def sample_density(self, idx, sample_coords, density_grad=False):
        scaled_sample_coords = (
            sample_coords.detach().cpu().numpy() / param.BOHR
        )  # convert Angstrom grid to Bohr
        mols = [self.mols[i] for i in idx]
        for i, mol in enumerate(mols):
            if not mol._built:
                build_start = time.time()
                if self.verbose > 2:
                    print("building mol", i)
                mol.build()
                if self.timing:
                    print("molecule build time", time.time() - build_start)
        coeffs = [self.coeffs[i] for i in idx]
        dens = orbitals.sample_density_base(
            mols, scaled_sample_coords, coeffs, projected=False, density_grad=density_grad,
        )

        return dens

    def sample_atom_density(
        self, positions, atom_numbers, coords, individual_dens=False, density_grad=False,
    ):
        dens, atom_dens = orbitals.sample_atom_density(
            positions,
            atom_numbers,
            coords,
            self.atom_dens_type,
            self.atom_dens,
            individual_dens,
            density_grad=density_grad,
        )
        if individual_dens:
            return atom_dens
        else:
            return dens

    def sample_projected_density(self, idx, sample_coords, density_grad=False):
        scaled_sample_coords = (
            sample_coords.detach().cpu().numpy() / param.BOHR
        )  # convert Angstrom grid to Bohr
        mols = [self.mols[i] for i in idx]
        for i, mol in enumerate(mols):
            if not mol._built or mol.basis != self.density_fitting["auxbasis"]:
                mol.basis = self.density_fitting["auxbasis"]
                # build_start = time.time()
                if self.verbose > 3:
                    print("building mol", i)
                mol.build()

        df_coeffs = [
            np.concatenate([coeff[1] for coeff in self.density_fitting["df_coeffs"][i]])
            for i in idx
        ]

        dens = orbitals.sample_density_base(
            mols, scaled_sample_coords, df_coeffs, projected=True, density_grad=density_grad,
        )
        return dens

    def add_fixed_properties(self, property_dict):
        """Add fixed properties to the dataset.

        Args:
        property_dict (dict): Dictionary of properties to add to the dataset.
        """
        for prop in property_dict.keys():
            self.fixed_properties[prop] = property_dict[prop]
