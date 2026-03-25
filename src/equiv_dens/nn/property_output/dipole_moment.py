import torch
import torch.nn as nn
from equiv_dens.utils import orbitals
from equiv_dens.utils import base as utils
from pyscf import gto


class DipoleMomentCalc(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, atoms, center_coordinates=True):
        density = atoms['density']
        positive_dens = (density >= 0).to(density)
        density = density * positive_dens
        n_electrons = orbitals.get_n_electrons(atoms['batch_atom_numbers'])
        scaling_factor = n_electrons / torch.sum(density * atoms['coord_weights'], dim=1, keepdim=True)
        density = density * scaling_factor
        if center_coordinates:
            center_of_mass = torch.sum(atoms['batch_positions'] * atoms['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
                             / torch.sum(atoms['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)
        else:
            center_of_mass = torch.zeros(atoms['batch_positions'].shape[0], 1, atoms['batch_positions'].shape[-1]).to(atoms['batch_positions'])
        positive_dipole_moment = torch.sum((atoms['batch_positions'] - center_of_mass) * atoms['batch_atom_numbers'].unsqueeze(-1), dim=1)
        # positive_dipole_moment = torch.sum(atoms['batch_positions'] * atoms['batch_atom_numbers'].unsqueeze(-1), dim=1)
        # print('positive_dipole_moment', positive_dipole_moment)
        weighted_dens = density * atoms['coord_weights']
        # # print('weighted dens shape', weighted_dens.shape)
        # # print('atom numbers shape', atoms['atom_numbers'].shape)
        # weighted_dens = weighted_dens / torch.sum(weighted_dens, dim=1, keepdim=True) * \
        #                 torch.sum(atoms['batch_atom_numbers'], dim=1, keepdim=True)

        negative_moment = weighted_dens.unsqueeze(-1) * (atoms['coords'] - center_of_mass)
        # negative_moment = weighted_dens.unsqueeze(-1) * atoms['coords']
        negative_dipole_moment = torch.sum(negative_moment, dim=1)
        # print('negative_dipole_moment', negative_dipole_moment)

        atoms['dipole_moment'] = positive_dipole_moment - negative_dipole_moment
        return atoms


class DipoleMomentIntorCalc(nn.Module):

    def __init__(self, orbital_basis, remove_atom_density=False):
        super().__init__()
        self.orbital_basis = orbital_basis
        self.remove_atom_density = remove_atom_density

    def forward(self, atoms):
        df_coeffs_ml = orbitals.coeffs_dict_to_vector(atoms, self.orbital_basis, atoms['batch_atom_numbers'],
                                                      radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs']
        charges = atoms['batch_atom_numbers']
        coords = atoms['batch_positions']

        nucl_dip = torch.sum(charges.unsqueeze(-1) * coords, dim=1)
        batch_dens_dip = []
        for i in range(atoms['batch_atom_numbers'].shape[0]):
            auxmol_ml = orbitals.ml_basis_to_auxmol(atoms, i, skip_zero=False)
            helper_mol = orbitals.build_1c1e_helper_mol(auxmol_ml)
            intor_idx = [
                auxmol_ml.bas_atom(ibas)
                for ibas in range(auxmol_ml.nbas) for _ in range(auxmol_ml.bas_angular(ibas) * 2 + 1)
            ]
            int1e_r = gto.mole.intor_cross('int1e_r', helper_mol, auxmol_ml)
            int1e_r = int1e_r[:, intor_idx, range(auxmol_ml.nao)]
            int1e_r = utils.bohr_to_angstrom(torch.from_numpy(int1e_r).to(nucl_dip))
            # ml_dip = utils.bohr_to_angstrom(nucl_dip - torch.einsum('ji,i->j', int1e_r, df_coeffs_ml))
            el_dip = torch.einsum('ji,i->j', int1e_r, df_coeffs_ml[i])
            batch_dens_dip.append(el_dip)

        dens_dip = torch.stack(batch_dens_dip, dim=0)

        if self.remove_atom_density:
            atoms = orbitals.intor_dipole_moment_free_atom(atoms)
            dens_dip += atoms['atom_dipole_moment']

        atoms['dipole_moment'] = nucl_dip - dens_dip

        return atoms
