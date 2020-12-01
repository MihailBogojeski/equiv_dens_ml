import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from equiv_dens.nn.spherical_harmonics import spherical_harmonics


class SphericalHarmonicsExpansion(nn.Module):

    def __init__(self, orbitals, radial_coeffs=None,
                 expansion_constraint=None,
                 integral_constraint=False,
                 softmax_norm=False, verbose=0):
        super().__init__()
        self.orbitals = orbitals
        self.order_max = 0
        self.expansion_constraint = expansion_constraint
        self.n_electrons = 0
        self.integral_constraint = integral_constraint
        self.softmax_norm = softmax_norm
        self.verbose = verbose
        if self.verbose:
            print('expansion constraint', self.expansion_constraint)
            print('integral constraint', self.integral_constraint)
        for i in range(len(self.orbitals)):
            self.n_electrons += self.orbitals[i][0][0]
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
        L0_coeffs = []
        L0_sph = []
        L0_d = []
        L0_width = []
        for i in range(len(self.orbital_spec)):
            z = self.orbital_spec[i][0][0]
            # print('atom num', z)
            # print('orbitals i', self.orbital_spec[i])
            d, u = calculate_distances_and_directions(coords, center=atom_R[:, [i]])
            s = spherical_harmonics(self.order_max, u)
            # print('atom[i]', i)
            # print('dists', d)
            # print('min dist', torch.min(d))
            for L in range(len(s)):
                zeros = torch.zeros_like(s[L])
                s[L] = torch.where(torch.isnan(s[L]), zeros, s[L])  # making sure there are no nans to avoid NaNs
            for orb in self.orbital_spec[i]:
                # print('orbital', orb)
                L = orb[2]
                key = (z, L)
                # print('key', key)
                # width = rad_width[i][key]
                # width = self.init_width(i, key)
                width = (rad_width[i][key] + 1) * self.init_width(i, key)
                # print('width', width)

                # scale = rad_scale[i][key]
                # scale = self.init_scale(i, key)
                scale = rad_scale[i][key] + self.init_scale(i, key)
                # print('scale', scale)
                sph_coeff = sph_coeffs[i][key]
                if L == 0:
                    L0_coeff = sph_coeff * scale
                    L0_coeffs.append(L0_coeff)
                    L0_sph.append(s[L])
                    L0_d.append(d)
                    L0_width.append(width)
                    continue
                # print('width nan', torch.sum(torch.isnan(width)))
                # print('scale nan', torch.sum(torch.isnan(scale)))
                # print('sph_coeffs nan', torch.sum(torch.isnan(sph_coeffs[i][key])))
                # print('harmonics nans', torch.sum(torch.isnan(s[L])))
                # print('s shape', s[L].shape)
                # print('key', key)
                # print('sph coeffs', sph_coeffs[i][key])
                # print('width shape', width.shape)
                # print('scale shape', scale.shape)
                sph = s[L].unsqueeze(-1) * sph_coeff
                # print('sph nan', torch.sum(torch.isnan(sph)))
                # print('sph prod shape', sph.shape)
                rbf = gaussian_rbf(d.unsqueeze(-1), width, scale,
                                   # normalize=(self.integral_constraint and L == 1))
                                   )
                # print('rbf nan', torch.sum(torch.isnan(rbf)))
                # print('rbf shape', rbf.shape)
                if i in eval_atoms and L in eval_L:
                    result['density'] += torch.sum(rbf * sph, dim=(-2, -1))
        L0_coeffs_comb = torch.cat([coeff.view((coeff.shape[0], -1)) for coeff in L0_coeffs], dim=1)
        if self.integral_constraint:
            if self.softmax_norm:
                L0_coeffs_comb = F.softmax(L0_coeffs_comb, dim=1)
                L0_coeffs_comb = L0_coeffs_comb * self.n_electrons
            else:
                coeffs_sum = torch.sum(L0_coeffs_comb, dim=1)
                scale_factor = self.n_electrons / coeffs_sum
                L0_coeffs_comb = L0_coeffs_comb * scale_factor
        coeffs_pointer = 0
        if 0 in eval_L:
            for i in range(len(L0_coeffs)):
                coeffs_size = np.prod(list(L0_coeffs[i].shape[1:]))
                curr_coeffs = L0_coeffs_comb[:, coeffs_pointer:(coeffs_size + coeffs_pointer)]
                if self.verbose > 1:
                    print('curr_coeffs', curr_coeffs)
                    print('curr_coeffs sum', torch.sum(curr_coeffs))
                    print('L0 width', L0_width[i])
                curr_coeffs = curr_coeffs.view(L0_coeffs[i].shape)
                coeffs_pointer += coeffs_size
                rbf = gaussian_rbf(L0_d[i].unsqueeze(-1), L0_width[i], curr_coeffs)
                sph = L0_sph[i].unsqueeze(-1)
                if i in eval_atoms:
                    result['density'] += torch.sum(rbf * sph, dim=(-2, -1))
        if self.expansion_constraint == 'sq':
            result['density'] = result['density']**2
        if self.expansion_constraint == 'abs':
            result['density'] = torch.sqrt(result['density']**2)
        if self.expansion_constraint == 'sp':
            dens = result['density']
            if self.verbose > 1:
                print('dens int pos before', torch.sum(result['density'][dens > 0] * 0.06021670784495335))
                print('dens int neg before', torch.sum(result['density'][dens < 0] * 0.06021670784495335))
            result['density'] = F.softplus(result['density'], beta=100000000) + 1e-30
            if self.verbose > 1:
                print('dens int pos before', torch.sum(result['density'][dens > 0] * 0.06021670784495335))
                print('dens int neg before', torch.sum(result['density'][dens < 0] * 0.06021670784495335))

        return result


def gaussian_rbf(r, width, scale, normalize=True):
    # print('scale shape', scale.shape)
    # print('scale shape', scale.shape)
    # print('width shape', width.shape)
    # print('r shape', r.shape)
    if normalize:
        scale_calc = scale * 8 * (width**(3 / 2)) / (np.pi**(3 / 2) * 53.9866)
    else:
        scale_calc = scale
    rbf = scale_calc * torch.exp(-width * (r)**2)
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
