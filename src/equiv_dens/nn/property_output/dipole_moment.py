import torch
import torch.nn as nn


class DipoleMomentCalc(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, atoms):
        # print('positions', atoms['positions'])
        # center = torch.mean(atoms['positions'] * atoms['atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)
        # print('center', center)
        # R_centered = atoms['positions'] - center
        # print('centered positions', R_centered)
        # _, max_dens_idx = torch.max(atoms['density'], dim=1)
        # max_dens_idx = max_dens_idx.view(-1, 1, 1)
        # max_dens_idx = max_dens_idx.expand(-1, -1, 3)
        # print('max_dens_idx', max_dens_idx)
        # print('max_dens_idx shape', max_dens_idx.shape)
        # print('max dens coords', torch.gather(atoms['coords'], 1, max_dens_idx))
        positive_dipole_moment = torch.sum(atoms['positions'] * atoms['atom_numbers'].unsqueeze(-1), dim=1)
        # print('positive_dipole_moment', positive_dipole_moment)
        weighted_dens = atoms['density'].unsqueeze(-1) * atoms['coord_weights'].unsqueeze(-1)
        negative_moment = weighted_dens * atoms['coords']
        negative_dipole_moment = torch.sum(negative_moment, dim=1)
        # print('negative_dipole_moment', negative_dipole_moment)

        atoms['dipole_moment'] = positive_dipole_moment - negative_dipole_moment
        return atoms
