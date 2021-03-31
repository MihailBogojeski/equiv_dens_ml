import torch
import torch.nn.functional as F
import torch.nn as nn
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.modules.spherical_harmonic_layers import SphericalLinear
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
from equiv_dens.utils.orbitals import combine_orbitals, gaussian_rbf, get_max_order, get_n_electrons
from equiv_dens.utils.base import calculate_distances_and_directions
import numpy as np
import time


class DensityCoeffsNetwork(nn.Module):
    """
    Neural network for computing density coefficients from spherical harmonic features in a rotationally equivariant way
    """

    def __init__(self,
                 orbitals=None,  # orbitals of atoms, defines layout and shape of output matrix
                 order=1,  # maximum order of spherical harmonics features
                 num_features=32,
                 positive_coeffs=False,
                 clebsch_gordan=None,
                 verbose=0,
                 compressed_extraction=False,
                 timing=False,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)
        self.create_graph = True  # can be set to False if the NN is only used for inference

        # store hyperparameter values
        self.orbitals = orbitals
        self.order = order
        self.num_features = num_features
        self.positive_coeffs = positive_coeffs
        self.verbose = verbose
        self.compressed_extraction = compressed_extraction
        self.timing = timing

        # extract nuclear charges from orbitals, determine maximum order, and
        # build the occupation mask (for extracting occupied orbitals in energy prediction)
        self.orbitals_max_order = get_max_order(orbitals)
        # for calculating nucleus - nucleus repulsion

        if clebsch_gordan is None:
            self.clebsch_gordan = ClebschGordanMatrix()
        else:
            self.clebsch_gordan = clebsch_gordan

        # error checking
        if self.order < self.orbitals_max_order:
            print("An orbital with L={} was found, but the neural network was initialized with L={}".format(
                self.orbitals_max_order, self.order))
            print("The neural network MUST have at least the same order as all orbitals!")
            quit()

        self.orbital_spec, self.radial_count = combine_orbitals(self.orbitals, self.orbitals_max_order)
        print('orbital_spec', self.orbital_spec)

        if self.compressed_extraction:
            self.L_counts, self.r_max, self.L_dict = self.compute_orbital_features_num_compressed()
        else:
            self.L_counts, self.r_max, self.L_dict = self.compute_orbital_features_num()
        print('L_counts', self.L_counts)
        print('L_dict', self.L_dict)
        print('max lcounts', max(self.L_counts))
        self.spherical_output = SphericalLinear(self.order, self.num_features,
                                                self.orbitals_max_order + 1,
                                                max(self.L_counts), self.clebsch_gordan)
        self.radial_width = nn.ModuleList([nn.Linear(self.num_features, self.L_counts[L] * self.r_max[L])
                                           for L in range(self.orbitals_max_order + 2)])
        self.radial_scale = nn.ModuleList([nn.Linear(self.num_features, self.L_counts[L] * self.r_max[L])
                                           for L in range(self.orbitals_max_order + 2)])

    """
    Collects spherical harmonics features into orbital coefficients of the appropriate size

    inputs:
        fs: List of spherical harmonics features of different orders, each element of shape [batch_size, 3, 2*order + 1, num_features]
    outputs:
        matrix: Array of orbital coefficients of shape [batch_size, num_orbitals]
    """

    def extract_coefficients(self, sph_fs, rad_width, rad_scale):
        spherical_coeffs = [None] * len(self.orbital_spec)
        radial_width = [None] * len(self.orbital_spec)
        radial_scale = [None] * len(self.orbital_spec)
        # print('len radial width', len(radial_width))
        # print('len rad width', len(rad_width))
        for i in range(len(self.orbital_spec)):
            z = self.orbital_spec[i][0][0]
            spherical_coeffs[i] = {}
            radial_width[i] = {}
            radial_scale[i] = {}
            for orb in self.orbital_spec[i]:
                L = orb[2]
                key = (z, L)
                inds = self.L_dict[key]
                sph_fs_i = sph_fs[L][:, [i], :, :]
                rad_w_i = rad_width[L][:, [i], :, :]
                rad_s_i = rad_scale[L][:, [i], :, :]
                # print('fs l=', L, 'shape:', sph_fs[L].shape)
                # print('inds', inds)
                spherical_coeffs[i][key] = sph_fs_i[..., inds]
                # print('spherical coeffs shape', spherical_coeffs[i][key].shape)
                # print('i', i)
                # print('L', L)
                radial_width[i][key] = rad_w_i[..., inds]
                radial_scale[i][key] = rad_s_i[..., inds]

        return spherical_coeffs, radial_width, radial_scale

    """
    Counts how many features of each order are needed to collect the orbital coefficients
    outputs:
        matrix: Number of features required for each orbital order
    """

    def compute_orbital_features_num(self):
        print('using expanded extraction')
        # counts the number of orbitals of each order across all atoms for the given basis
        L_counts = [0 for L in range(2 * self.order + 1)]
        # contains maximum number of radial components for each order across all atoms for the given basis
        r_max = [0 for L in range(2 * self.order + 1)]
        orbital_dict = {}
        # radial_dict = {}
        for i in range(len(self.orbital_spec)):
            z = self.orbital_spec[i][0][0]
            for j in range(len(self.orbital_spec[i])):
                orb = self.orbital_spec[i][j]
                rad_c = self.radial_count[i][j]
                L = orb[2]
                n = orb[1]
                # print('L', L)
                # print('n', n)
                key = (z, L)
                if key not in orbital_dict:
                    # print('lcounts range', L_counts[L], L_counts[L] + n)
                    orbital_dict[key] = range(L_counts[L], L_counts[L] + n)
                    L_counts[L] += n
                    # print('rmax L', r_max[L])
                    # print('rad_c L', rad_c[L])
                    r_max[L] = max(rad_c[L], r_max[L])
                    # return one radial function per orbital
                    # radial function consists of multiple gaussians each with width and factor
        return L_counts, r_max, orbital_dict

    """
    Counts how many features of each order are needed to collect the orbital coefficients
    outputs:
        matrix: Number of features required for each orbital order
    """

    def compute_orbital_features_num_compressed(self):
        print('using compressed extraction')
        # counts the number of orbitals of each order across all atoms for the given basis
        L_counts = [0 for L in range(2 * self.order + 1)]
        # contains maximum number of radial components for each order across all atoms for the given basis
        r_max = [0 for L in range(2 * self.order + 1)]
        orbital_dict = {}
        # radial_dict = {}
        for i in range(len(self.orbital_spec)):
            z = self.orbital_spec[i][0][0]
            for j in range(len(self.orbital_spec[i])):
                orb = self.orbital_spec[i][j]
                rad_c = self.radial_count[i][j]
                L = orb[2]
                n = orb[1]
                # print('L', L)
                # print('n', n)
                key = (z, L)
                if key not in orbital_dict:
                    # print('lcounts range', L_counts[L], L_counts[L] + n)
                    orbital_dict[key] = range(n)
                    if L_counts[L] < n:
                        L_counts[L] = n
                    # L_counts[L] += n
                    # print('rmax L', r_max[L])
                    # print('rad_c L', rad_c[L])
                    r_max[L] = max(rad_c[L], r_max[L])
                    # return one radial function per orbital
                    # radial function consists of multiple gaussians each with width and factor
        return L_counts, r_max, orbital_dict

    """
    Computes the spherical harmonics coefficients for the density

    inputs:
        R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
    outputs:
        C: Spherical harmonics coefficients
    """

    def forward(self, atoms):
        start = time.time()
        fs = atoms['sph_repr']
        if self.verbose > 3:
            print('distances', atoms['distances'])
            print('fs[0]:', fs[0][:, 0, :, :10])
            print('fs[1]:', fs[1][:, 0, :, :10])
        out_sph = self.spherical_output(fs)
        if self.positive_coeffs:
            out_sph[0] = F.softplus(out_sph[0])
        out_width = []
        out_scale = []
        for L in range(len(self.radial_width)):
            out_width.append(torch.tanh(self.radial_width[L](fs[0])))
            out_width[L] = out_width[L].view(*out_width[L].shape[:-2], self.r_max[L], self.L_counts[L])
            if self.positive_coeffs:
                out_scale.append(F.softplus(self.radial_scale[L](fs[0])))
            else:
                out_scale.append(self.radial_scale[L](fs[0]))
            out_scale[L] = out_scale[L].view(*out_scale[L].shape[:-2], self.r_max[L], self.L_counts[L])
        atoms['spherical_coeffs'], atoms['radial_width'], atoms['radial_scale'] =\
            self.extract_coefficients(out_sph, out_width, out_scale)
        # print('out sph[1][0]', out_sph[1][:, 0, :])
        # print('spherical_coeffs[1][0]', atoms['spherical_coeffs'][0][(8, 1)])
        # print('out sph[1][0]', out_sph[1][:, 1, :])
        # print('spherical_coeffs[1][0]', atoms['spherical_coeffs'][1][(1, 1)])
        atoms['L_dict'] = self.L_dict
        if self.timing:
            print('density coeffs time:', time.time() - start)

        return atoms


class DensityExpansion(nn.Module):
    """
    Module for expanding density coefficients into a density sampled on a given grid
    """

    def __init__(self, orbitals, radial_coeffs=None,
                 expansion_constraint=None,
                 integral_constraint=False,
                 softmax_norm=False,
                 n_electrons=None,
                 integral_scale=False,
                 verbose=0,
                 timing=False,
                 ):
        super().__init__()
        self.orbitals = orbitals
        self.expansion_constraint = expansion_constraint
        self.integral_constraint = integral_constraint
        self.softmax_norm = softmax_norm
        self.timing = timing
        self.verbose = verbose
        if integral_scale:
            self.register_parameter('integral_scale', nn.Parameter(torch.ones(size=(1,))))
        else:
            self.register_buffer('integral_scale', torch.ones(size=(1,)))
        if self.verbose:
            print('expansion constraint', self.expansion_constraint)
            print('integral constraint', self.integral_constraint)

        self.orbitals_max_order = get_max_order(self.orbitals)
        if n_electrons is None:
            self.n_electrons = get_n_electrons(self.orbitals)
        else:
            self.n_electrons = n_electrons

        self.radial_coeffs = radial_coeffs
        self.orbital_spec, self.radial_counts = combine_orbitals(self.orbitals, self.orbitals_max_order)
        self.init_radial_coeffs()

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

    def forward(self, atoms, eval_atoms=None, eval_L=None):
        start = time.time()
        if eval_atoms is None:
            eval_atoms = list(range(len(self.orbitals)))
        if eval_L is None:
            eval_L = list(range(self.orbitals_max_order + 1))
        atoms['density'] = 0
        L0_coeffs = []
        L0_sph = []
        L0_d = []
        L0_width = []
        for i in range(len(self.orbital_spec)):
            z = self.orbital_spec[i][0][0]
            # print('atom num', z)
            # print('orbitals i', self.orbital_spec[i])
            d, u = calculate_distances_and_directions(atoms['coords'], center=atoms['positions'][:, [i]])
            s = spherical_harmonics(self.orbitals_max_order, u)
            # print('atom[i]', i)
            # print('dists', d)
            # print('dists shape', d.shape)
            # print('min dist', torch.min(d))
            for L in range(len(s)):
                zeros = torch.zeros_like(s[L])
                s[L] = torch.where(torch.isnan(s[L]), zeros, s[L])  # making sure there are no nans to avoid NaNs
            for orb in self.orbital_spec[i]:
                # print('orbital', orb)
                L = orb[2]
                key = (z, L)
                width = (atoms['radial_width'][i][key] + 1) * self.init_width(i, key)
                zero_scale = (self.init_scale(i, key) != 0).to(atoms['radial_scale'][i][key])
                scale = (atoms['radial_scale'][i][key] + self.init_scale(i, key)) * zero_scale
                if self.verbose > 3:
                    print('init_width', self.init_width(i, key))
                    print('init_scale', self.init_scale(i, key))
                    print('width', width)
                    print('scale', scale)
                sph_coeff = atoms['spherical_coeffs'][i][key]
                if L == 0:
                    L0_coeff = sph_coeff * scale
                    L0_coeffs.append(L0_coeff)
                    L0_sph.append(s[L])
                    L0_d.append(d)
                    L0_width.append(width)
                    continue
                sph = s[L].unsqueeze(-1) * sph_coeff
                rbf = gaussian_rbf(d.unsqueeze(-1), width, scale)
                if i in eval_atoms and L in eval_L:
                    atoms['density'] += torch.sum(rbf * sph, dim=(-2, -1))
        L0_coeffs_comb = torch.cat([coeff.view((coeff.shape[0], -1)) for coeff in L0_coeffs], dim=1)
        atoms['L0_coeffs'] = L0_coeffs_comb
        if self.integral_constraint:
            if self.softmax_norm:
                L0_coeffs_comb = F.softmax(L0_coeffs_comb, dim=1)
                L0_coeffs_comb = L0_coeffs_comb * self.n_electrons
                L0_coeffs_comb = L0_coeffs_comb * torch.clamp(self.integral_scale, 0.5, 1.5)
                # print('coeffs_sum', torch.sum(L0_coeffs_comb, dim=1, keepdim=True))
            else:
                coeffs_sum = torch.sum(L0_coeffs_comb, dim=1, keepdim=True)
                scale_factor = self.n_electrons / coeffs_sum
                L0_coeffs_comb = L0_coeffs_comb * scale_factor
                L0_coeffs_comb = L0_coeffs_comb * torch.clamp(self.integral_scale, 0.5, 1.5)
                # print('coeffs_sum', torch.sum(L0_coeffs_comb, dim=1, keepdim=True))
        coeffs_pointer = 0
        if 0 in eval_L:
            for i in range(len(L0_coeffs)):
                coeffs_size = np.prod(list(L0_coeffs[i].shape[1:]))
                curr_coeffs = L0_coeffs_comb[:, coeffs_pointer:(coeffs_size + coeffs_pointer)]
                if self.verbose > 3:
                    print('curr_coeffs', curr_coeffs)
                    print('curr_coeffs sum', torch.sum(curr_coeffs))
                    print('L0 width', L0_width[i])
                curr_coeffs = curr_coeffs.view(L0_coeffs[i].shape)
                coeffs_pointer += coeffs_size
                rbf = gaussian_rbf(L0_d[i].unsqueeze(-1), L0_width[i], curr_coeffs)
                sph = L0_sph[i].unsqueeze(-1)
                if i in eval_atoms:
                    atoms['density'] += torch.sum(rbf * sph, dim=(-2, -1))
        if self.expansion_constraint == 'sq':
            atoms['density'] = atoms['density']**2
        if self.expansion_constraint == 'abs':
            atoms['density'] = torch.sqrt(atoms['density']**2)
        if self.expansion_constraint == 'sp':
            dens = atoms['density']
            if self.verbose > 3:
                print('dens int pos before', torch.sum(atoms['density'][dens > 0] * 0.06021670784495335))
                print('dens int neg before', torch.sum(atoms['density'][dens < 0] * 0.06021670784495335))
            atoms['density'] = F.softplus(atoms['density'], beta=100000000) + 1e-30
            if self.verbose > 3:
                print('dens int pos after', torch.sum(atoms['density'][dens > 0] * 0.06021670784495335))
                print('dens int neg after', torch.sum(atoms['density'][dens < 0] * 0.06021670784495335))

        if self.timing:
            print('density expansion time:', time.time() - start)
        return atoms
