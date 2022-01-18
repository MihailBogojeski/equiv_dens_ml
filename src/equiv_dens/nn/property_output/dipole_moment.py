import torch
import torch.nn as nn
import numpy as np


class DipoleMomentCalc(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, atoms):
        positive_dipole_moment = torch.sum(atoms['positions'] * atoms['atom_numbers'].unsqueeze(-1), dim=1)
        # print('positive_dipole_moment', positive_dipole_moment)
        weighted_dens = atoms['density'].unsqueeze(-1) * atoms['coord_weights'].unsqueeze(-1)
        weighted_dens = weighted_dens / torch.sum(weighted_dens, dim=1) * torch.sum(atoms['atom_numbers'])

        negative_moment = weighted_dens * atoms['coords']
        negative_dipole_moment = torch.sum(negative_moment, dim=1)
        # print('negative_dipole_moment', negative_dipole_moment)

        atoms['dipole_moment'] = positive_dipole_moment - negative_dipole_moment
        return atoms
