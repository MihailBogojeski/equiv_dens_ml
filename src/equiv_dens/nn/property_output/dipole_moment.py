import torch
import torch.nn as nn
from equiv_dens.utils.orbitals import get_n_electrons 


class DipoleMomentCalc(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, atoms):
        density = atoms['density']
        positive_dens = (density >= 0).to(density)
        density = density * positive_dens
        n_electrons = get_n_electrons(atoms['batch_atom_numbers'])
        scaling_factor = n_electrons / torch.sum(density * atoms['coord_weights'], dim=1, keepdim=True)
        density = density * scaling_factor
        positive_dipole_moment = torch.sum(atoms['batch_positions'] * atoms['batch_atom_numbers'].unsqueeze(-1), dim=1)
        # print('positive_dipole_moment', positive_dipole_moment)
        weighted_dens = density * atoms['coord_weights']
        # # print('weighted dens shape', weighted_dens.shape)
        # # print('atom numbers shape', atoms['atom_numbers'].shape)
        # weighted_dens = weighted_dens / torch.sum(weighted_dens, dim=1, keepdim=True) * \
        #                 torch.sum(atoms['batch_atom_numbers'], dim=1, keepdim=True)

        negative_moment = weighted_dens.unsqueeze(-1) * atoms['coords']
        negative_dipole_moment = torch.sum(negative_moment, dim=1)
        # print('negative_dipole_moment', negative_dipole_moment)

        atoms['dipole_moment'] = positive_dipole_moment - negative_dipole_moment
        return atoms
