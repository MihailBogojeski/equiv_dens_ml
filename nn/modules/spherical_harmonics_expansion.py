import torch
import torch.nn as nn
from nn.spherical_harmonics import spherical_harmonics


class SphericalHarmonicsExpansion(nn.Module):

    def __init__(self, orbitals, radial_coeffs=None, constraint_type=None):
        super().__init__()
        self.orbitals = orbitals
        self.order_max = 0
        self.constraint_type = constraint_type
        print('constraint type',  self.constraint_type)

        for i in range(len(self.orbitals)):
            for z, _, l in self.orbitals[i]:
                if l > self.order_max:
                    self.order_max = l

        self.radial_coeffs = radial_coeffs
        self.orbital_spec, self.radial_counts = self.combine_orbitals()
        self.init_radial_coeffs()

    def combine_orbitals(self):
        orbital_spec = [None] * len(self.orbitals)
        radial_counts = [None] * len(self.orbitals)
        for i in range(len(self.orbitals)):
            orbital_L_count = [0] * (self.order_max + 2)
            orbital_spec[i] = []
            radial_counts[i] = [[]] * (self.order_max + 2)
            # print('density L count len', len(density_L_count))
            z = self.orbitals[i][0][0]
            for j in range(len(self.orbitals[i])):
                orb = self.orbitals[i][j]
                L = orb[2]
                orbital_L_count[L] += 1
                radial_counts[i][L].append(orb[1])
            for L, c in enumerate(orbital_L_count):
                if c == 0:
                    continue
                orbital_spec[i].append((z, c, L))

        return orbital_spec, radial_counts

    def init_radial_coeffs(self):
        init_width = [None] * len(self.orbitals)
        init_scale = [None] * len(self.orbitals)
        for i in range(len(self.orbitals)):
            z = self.orbitals[i][0][0]
            init_width[i] = {}
            init_scale[i] = {}
            for j in range(len(self.orbitals[i])):
                orb = self.orbitals[i][j]
                L = orb[2]
                key = (z, L)
                width_coeff = torch.zeros((1, 1, max(self.radial_counts[i][L]), 1))
                scale_coeff = torch.zeros((1, 1, max(self.radial_counts[i][L]), 1))
                if self.radial_coeffs is not None:
                    n_coeff = len(self.radial_coeffs[i][j][0])
                    width_coeff[..., :n_coeff, 0] += torch.Tensor(self.radial_coeffs[i][j][0])
                    scale_coeff[..., :n_coeff, 0] += torch.Tensor(self.radial_coeffs[i][j][1])
                if key in init_width[i].keys():
                    init_width[i][key] = torch.cat((init_width[i][key], width_coeff), dim=-1)
                    init_scale[i][key] = torch.cat((init_scale[i][key], scale_coeff), dim=-1)
                else:
                    init_width[i][key] = width_coeff
                    init_scale[i][key] = scale_coeff

            for key in init_width[i].keys():
                self.register_buffer('init_width_{}_{}_{}'.format(i, key[0], key[1]), init_width[i][key])
                self.register_buffer('init_scale_{}_{}_{}'.format(i, key[0], key[1]), init_scale[i][key])

        return init_width, init_scale

    def init_width(self, i, key):
        return getattr(self, 'init_width_{}_{}_{}'.format(i, key[0], key[1]))

    def init_scale(self, i, key):
        return getattr(self, 'init_scale_{}_{}_{}'.format(i, key[0], key[1]))

    def forward(self, coords, atom_R, sph_coeffs, rad_width, rad_scale, eval_atoms=None, eval_L=None):
        # print('orbitals', self.orbitals)
        # print('coords', coords.shape)
        # print('atom_R shape', atom_R.shape)
        # print('rad_width keys', rad_width[0].keys())
        # print('rad_scale keys', rad_scale[0].keys())
        if eval_atoms is None:
            eval_atoms = list(range(len(self.orbitals)))
        if eval_L is None:
            eval_L = list(range(self.order_max + 1))
        result = {'density': 0}
        # print('eval atoms', eval_atoms)
        # print('eval L', eval_L)
        # print('order max', self.order_max)
        for i in eval_atoms:
            z = self.orbital_spec[i][0][0]
            # print('atom num', z)
            # print('orbitals i', self.orbital_spec[i])
            d, u = calculate_distances_and_directions(coords, center=atom_R[:, [i]])
            s = spherical_harmonics(self.order_max, u)
            # print('atom[i]', i)
            # print('dists', d)
            # print('min dist', torch.min(d))
            for l in range(len(s)):
                zeros = torch.zeros_like(s[l])
                s[l] = torch.where(torch.isnan(s[l]), zeros, s[l])  # making sure there are no nans to avoid NaNs

            for orb in self.orbital_spec[i]:
                # print('orbital', orb)
                L = orb[2]
                if L not in eval_L:
                    continue
                key = (z, L)
                # print('key', key)
                # width = rad_width[i][key]
                # width = self.init_width(i, key)
                width = rad_width[i][key] + self.init_width(i, key)
                # print('width', width)

                # scale = rad_scale[i][key]
                # scale = self.init_scale(i, key)
                scale = rad_scale[i][key] + self.init_scale(i, key)
                # print('scale', scale)

                # print('width nan', torch.sum(torch.isnan(width)))
                # print('scale nan', torch.sum(torch.isnan(scale)))
                # print('sph_coeffs nan', torch.sum(torch.isnan(sph_coeffs[i][key])))
                # print('harmonics nans', torch.sum(torch.isnan(s[L])))
                # print('s shape', s[L].shape)
                # print('key', key)
                # print('sph coeffs', sph_coeffs[i][key])
                # print('width shape', width.shape)
                # print('scale shape', scale.shape)
                sph = s[L].unsqueeze(-1) * sph_coeffs[i][key]
                # print('sph nan', torch.sum(torch.isnan(sph)))
                # print('sph prod shape', sph.shape)
                rbf = gaussian_rbf(d.unsqueeze(-1), width, scale)
                # print('rbf nan', torch.sum(torch.isnan(rbf)))
                # print('rbf shape', rbf.shape)
                result['density'] += torch.sum(rbf * sph, dim=(-2, -1))
        if self.constraint_type == 'sq':
            result['density'] = result['density']**2
        if self.constraint_type == 'abs':
            result['density'] = torch.sqrt(result['density']**2)

        return result


def gaussian_rbf(r, width, scale):
    # print('scale shape', scale.shape)
    # print('scale shape', scale.shape)
    # print('width shape', width.shape)
    # print('r shape', r.shape)
    rbf = scale * torch.exp(-width * (r)**2)
    return torch.sum(rbf, dim=-2, keepdim=True)


def calculate_distances_and_directions(r, center=None):
    if center is None:
        center = 0
    # print('r', r.type())
    # print('center', center.type())

    r = r - center
    d = torch.norm(r, dim=-1, keepdim=True)
    u = r / d
    return d, u
