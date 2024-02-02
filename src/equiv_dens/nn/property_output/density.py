import torch
import torch.nn.functional as F
import torch.nn as nn
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.modules.spherical_harmonic_layers import SphericalLinear
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
from equiv_dens.utils.orbitals import combine_orbital_basis, \
    gaussian_rbf, get_max_order, get_n_electrons, coeffs_dict_to_vector, \
    gto_norm, pyscf_gto_factor
from equiv_dens.utils.base import calculate_distances_and_directions, batch_compressed_atoms
import numpy as np
import time

class DensityCoeffsNetwork(nn.Module):
    """
    Neural network for computing density coefficients from spherical harmonic features in a rotationally equivariant way
    """

    def __init__(self,
                 orbital_basis=None,  # orbitals of atoms, defines layout and shape of output matrix
                 order=1,  # maximum order of spherical harmonics features
                 num_features=32,
                 positive_coeffs=False,
                 clebsch_gordan=None,
                 verbose=0,
                 compressed_extraction=False,
                 timing=False,
                 memory=False,
                 init_coeffs=None,
                 coeff_weights=None,
                 pred_radial_coeffs=True,
                 init_radial_coeffs=None,
                 ml_width_min=0,
                 ml_width_max=2,
                 scale_sph_order=True,
                 normalize=0,
                 parity=False,
                 core_basis_ratio=0,
                 linear_out=False,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)
        self.create_graph = True  # can be set to False if the NN is only used for inference

        # store hyperparameter values
        self.orbital_basis = orbital_basis
        self.order = order
        self.num_features = num_features
        self.positive_coeffs = positive_coeffs
        self.verbose = verbose
        self.compressed_extraction = compressed_extraction
        self.timing = timing
        self.memory = memory
        self.init_coeffs = init_coeffs
        self.pred_radial_coeffs = pred_radial_coeffs
        self.scale_sph_order = scale_sph_order
        self.normalize = normalize
        self.coeff_weights = coeff_weights
        self.core_basis_ratio = core_basis_ratio
        self.radial_coeffs = init_radial_coeffs
        self.ml_width_offset = ml_width_min
        self.ml_width_scale = (ml_width_max - ml_width_min) / 2
        self.linear_out = linear_out

        # if core basis ratio > 0, reduce orbital basis to only the highest width s orbitals
        if self.core_basis_ratio > 0:
            core_basis = {}
            for z in self.orbital_basis.keys():
                core_basis[z] = [orb for orb in self.orbital_basis[z] if orb[2] == 0]
                num_basis = round(len(core_basis[z]) * self.core_basis_ratio)
                core_basis[z] = core_basis[z][:num_basis]
            self.orbital_basis = core_basis
        # extract nuclear charges from orbitals, determine maximum order, and
        # build the occupation mask (for extracting occupied orbitals in energy prediction)
        self.orbitals_max_order = get_max_order(self.orbital_basis)
        # for calculating nucleus - nucleus repulsion

        if clebsch_gordan is None:
            self.clebsch_gordan = ClebschGordanMatrix()
        else:
            self.clebsch_gordan = clebsch_gordan

        # error checking
        # if (2 * self.order) < self.orbitals_max_order:
        #     print("An orbital with L={} was found, but the neural network was initialized with L={}".format(
        #         self.orbitals_max_order, self.order))
        #     print("The neural network MUST have an order of at least 1/2 of the maximum order of all orbitals!")
        #     quit()

        self.spherical_spec, self.radial_spec, self.radial_count = combine_orbital_basis(self.orbital_basis, self.orbitals_max_order)
        if self.verbose > 0:
            print('spherical_spec', self.spherical_spec)
            print('radial_spec', self.radial_spec)
            print('radial count', self.radial_count)

        if self.compressed_extraction:
            self.sph_counts, self.rad_counts, self.r_max, self.sph_dict, self.rad_dict = self.compute_orbital_features_num_compressed()
        else:
            self.sph_counts, self.rad_counts, self.r_max, self.sph_dict, self.rad_dict = self.compute_orbital_features_num()
        if self.verbose > 0:
            print('rad_counts', self.rad_counts)
            print('sph counts', self.sph_counts)
            print('rad_dict', self.rad_dict)
            print('sph_dict', self.sph_dict)
            print('max rad_counts', max(self.rad_counts))
        if self.init_coeffs is not None or self.core_basis_ratio > 0:
            self.output_bias = False
            self.output_zero_init = True
        else:
            self.output_bias = True
            self.output_zero_init = False

        self.spherical_output = SphericalLinear(self.order, self.num_features,
                                                self.orbitals_max_order,
                                                max(self.sph_counts), self.clebsch_gordan, bias=self.output_bias,
                                                zero_init=self.output_zero_init, normalize=self.normalize,
                                                parity=parity)
        if self.linear_out:
            self.linear_output = SphericalLinear(self.orbitals_max_order, max(self.sph_counts),
                                                 self.orbitals_max_order, max(self.sph_counts),
                                                 clebsch_gordan=None, mix_orders=False,
                                                 bias=True, zero_init=True, normalize=self.normalize)
        print('self.pred_radial_coeffs', self.pred_radial_coeffs)
        if self.pred_radial_coeffs:
            self.radial_width = nn.ModuleList([nn.Linear(self.num_features, self.rad_counts[L])
                                               for L in range(self.orbitals_max_order + 1)])
            self.radial_scale = nn.ModuleList([nn.Linear(self.num_features, self.rad_counts[L])
                                               for L in range(self.orbitals_max_order + 1)])
        if init_coeffs is not None:
            self.init_L0_coeffs()
        if coeff_weights is not None:
            self.init_coeff_weights()

        self.init_radial_coeffs()
        self.reset_parameters()

    def reset_parameters(self):
        """
        Sets the initial L=0 coefficients for the model, which are used as baseline for the
        predicted coefficients to speed up convergence

        outputs:
            init_sph: Initial L=0 spherical harmonic coefficients
            init_scale: Initial L=0 radial scale coefficients
            init_width: Initial L=0 radial width coefficients
        """
        if self.pred_radial_coeffs:
            for L in range(len(self.radial_width)):
                nn.init.zeros_(self.radial_width[L].weight)
                nn.init.zeros_(self.radial_scale[L].weight)

    def init_radial_coeffs(self):
        init_width = {}
        init_scale = {}
        for z in self.orbital_basis.keys():
            # print('init coeffs i', i)
            rad_count = [0] * (self.orbitals_max_order + 1)
            for j in range(len(self.spherical_spec[z])):
                orb = self.spherical_spec[z][j]
                L = orb[2]
                key = (z, L)
                if self.radial_coeffs is not None:
                    init_width[key] = torch.zeros(1, 1, self.r_max[key], orb[1])
                else:
                    init_width[key] = torch.ones(1, 1, self.r_max[key], orb[1])
                init_scale[key] = torch.zeros(1, 1, self.r_max[key], orb[1])
            if self.radial_coeffs is not None:
                for j in range(len(self.orbital_basis[z])):
                    # print('init coeffs j', j)
                    orb = self.orbital_basis[z][j]
                    L = orb[2]
                    key = (z, L)
                    # print('init coeffs orb', orb)
                    n_coeff = len(self.radial_coeffs[z][j][0])
                    # print('rad_count L', rad_count[L])
                    # print('init coeffs ncoeff', n_coeff)
                    init_width[key][..., :n_coeff, rad_count[L]] += torch.Tensor(self.radial_coeffs[z][j][0])
                    init_scale[key][..., :n_coeff, rad_count[L]] += torch.Tensor(self.radial_coeffs[z][j][1])
                    rad_count[L] += 1

        for key in init_width.keys():
            self.register_buffer('init_radial_width_{}_{}'.format(key[0], key[1]), init_width[key])
            self.register_buffer('init_radial_scale_{}_{}'.format(key[0], key[1]), init_scale[key])

        return init_width, init_scale

    def init_radial_width(self, key):
        return getattr(self, 'init_radial_width_{}_{}'.format(key[0], key[1]))

    def init_radial_scale(self, key):
        return getattr(self, 'init_radial_scale_{}_{}'.format(key[0], key[1]))

    def init_L0_coeffs(self):
        init_sph = {}
        init_width = {}
        init_scale = {}
        for z in self.orbital_basis.keys():
            for j in range(len(self.orbital_basis[z])):
                orb = self.orbital_basis[z][j]
                L = orb[2]
                key = (z, L)
                if L == 0:
                    init_sph[key] = self.init_basis['spherical_coeffs'][z]
                    init_width[key] = self.init_basis['radial_width'][z]
                    init_scale[key] = self.init_basis['radial_scale'][z]
                    self.register_buffer('init_sph_{}_{}'.format(key[0], key[1]), init_sph[key])
                    self.register_buffer('init_width_{}_{}'.format(key[0], key[1]), init_width[key])
                    self.register_buffer('init_scale_{}_{}'.format(key[0], key[1]), init_scale[key])

        return init_sph, init_width, init_scale

    def init_sph(self, key):
        return getattr(self, 'init_sph_{}_{}'.format(key[0], key[1]))

    def init_width(self, key):
        return getattr(self, 'init_width_{}_{}'.format(key[0], key[1]))

    def init_scale(self, key):
        return getattr(self, 'init_scale_{}_{}'.format(key[0], key[1]))

    def init_coeff_weights(self):
        coeff_weights = {}
        for z in self.orbital_basis.keys():
            # print('init coeffs i', i)
            rad_count = [0] * (self.orbitals_max_order + 1)
            for j in range(len(self.spherical_spec[z])):
                orb = self.spherical_spec[z][j]
                L = orb[2]
                key = (z, L)
                coeff_weights[key] = torch.zeros(1, 1, self.r_max[key], orb[1])
            for j in range(len(self.orbital_basis[z])):
                # print('init coeffs j', j)
                orb = self.orbital_basis[z][j]
                L = orb[2]
                key = (z, L)
                # print('init coeffs orb', orb)
                if self.coeff_weights is not None:
                    n_coeff = len(self.coeff_weights[z][j])
                    # print('rad_count L', rad_count[L])
                    # print('init coeffs ncoeff', n_coeff)
                    coeff_weights[key][..., :n_coeff, rad_count[L]] += torch.Tensor(self.coeff_weights[z][j])
                    rad_count[L] += 1

        for key in coeff_weights.keys():
            self.register_buffer('coeff_weights_{}_{}'.format(key[0], key[1]), coeff_weights[key])

        return coeff_weights

    def coeff_weight(self, key):
        return getattr(self, 'coeff_weights_{}_{}'.format(key[0], key[1]))

    """
    Collects spherical harmonics features into orbital coefficients of the appropriate size

    inputs:
        fs: List of spherical harmonics features of different orders, each element of shape [batch_size, 3, 2*order + 1, num_features]
    outputs:
        matrix: Array of orbital coefficients of shape [batch_size, num_orbitals]
    """

    def extract_coefficients(self, sph_fs, rad_width, rad_scale, atoms):
        atom_numbers = atoms['batch_atom_numbers']
        atom_num = sph_fs[0].shape[1]
        spherical_coeffs = [None] * atom_num
        radial_width = [None] * atom_num
        radial_scale = [None] * atom_num
        coeff_weights = [None] * atom_num
        # print('len radial width', len(radial_width))
        # print('len rad width', len(rad_width))
        for i in range(atom_num):
            pos = atoms['batch_positions'][:, [i]]
            spherical_coeffs[i] = {}
            radial_width[i] = {}
            radial_scale[i] = {}
            coeff_weights[i] = {}
            z = int(max(atom_numbers[:, i]))
            atom_mask = atoms['batch_atom_mask'][:, i].to(pos)
            dim_diff = 4 - atom_mask.dim()
            if dim_diff > 0:
                atom_mask = atom_mask.reshape(atom_mask.shape + (1,) * dim_diff)
            if z == 0:
                continue
            for orb in self.spherical_spec[z]:
                L = orb[2]
                key = (z, L)
                inds = self.sph_dict[key]
                sph_fs_i = sph_fs[L][:, [i], :, :]
                # print('sph l=', L, 'shape:', sph_fs[L].shape)
                # print('inds', inds)
                spherical_coeffs[i][key] = sph_fs_i[..., inds]
                if self.coeff_weights is not None:
                    coeff_weights[i][key] = torch.sum(self.coeff_weight(key), dim=-2, keepdim=True)
                    coeff_weights[i][key] = coeff_weights[i][key].expand(-1, -1, spherical_coeffs[i][key].shape[-2], -1).clone()
                # spherical_coeffs[i][key] = torch.ones_like(sph_fs_i[..., inds])
                # print('spherical coeffs shape', spherical_coeffs[i][key].shape)
                # print('i', i)
                # print('L', L)
                radial_width[i][key] = torch.zeros(*sph_fs_i.shape[:2], self.r_max[key], orb[1]).to(sph_fs_i)
                radial_scale[i][key] = torch.zeros(*sph_fs_i.shape[:2], self.r_max[key], orb[1]).to(sph_fs_i)
                if self.pred_radial_coeffs and z != 0:
                    inds = self.rad_dict[key]
                    rad_w_i = rad_width[L][:, [i], :, :]
                    rad_s_i = rad_scale[L][:, [i], :, :]
                    # print('rad fs l=', L, 'shape:', rad_w_i.shape)
                    # print('inds', inds)
                    r_curr = 0
                    for k, r_num in enumerate(self.radial_count[z][L]):
                        # print('radial width ', i, key, 'shape', radial_width[i][key].shape)
                        rad_inds = inds[r_curr: r_curr + r_num]
                        r_curr += r_num
                        radial_width[i][key][..., :r_num, k] = rad_w_i[..., 0, rad_inds]
                        radial_scale[i][key][..., :r_num, k] = rad_s_i[..., 0, rad_inds]
                # print('radial width shape', radial_width[i][key].shape)
                # print('self.init coeffs', self.init_coeffs)
                if self.init_coeffs is not None and L == 0:
                    # print('spherical_coeffs[i][key] before shape', spherical_coeffs[i][key].shape)
                    # print('self.init_sph(i, key)', self.init_sph(i, key))
                    spherical_coeffs[i][key] = spherical_coeffs[i][key] + self.init_sph(key)
                    if self.pred_radial_coeffs:
                        radial_width[i][key] = torch.clamp(radial_width[i][key] + self.init_width(key), -0.999999, 0.99999)
                        radial_scale[i][key] = radial_scale[i][key] + self.init_scale(key)
                    # print('spherical_coeffs[i][key] after shape', spherical_coeffs[i][key].shape)
                # print(f'key:{key}, radial width: {radial_width[i][key][0]}')
                if self.radial_coeffs is not None:
                    # the original range of the radial width is [-1, 1]
                    curr_width = (radial_width[i][key] + 1) * self.ml_width_scale + self.ml_width_offset
                    radial_width[i][key] = curr_width * self.init_radial_width(key)
                    # print(f'key:{key}, init width ', self.init_radial_width(key))
                    # print(f'key:{key}, radial width after update: {radial_width[i][key][0]}')
                    # if self.integral_constraint is True or self.integral_constraint == 'coeffs':
                    #     width = torch.clamp(width, 1e-1, 1e+5)
                    if self.positive_coeffs:
                        curr_scale = (radial_scale[i][key] + 1) * self.ml_width_scale + self.ml_width_offset
                        radial_scale[i][key] = (curr_scale * self.init_radial_scale(key)) * atom_mask
                    else:
                        radial_scale[i][key] = (radial_scale[i][key] + self.init_radial_scale(key)) * atom_mask

        return spherical_coeffs, radial_width, radial_scale, coeff_weights



    """
    Counts how many features of each order are needed to collect the orbital coefficients
    outputs:
        matrix: Number of features required for each orbital order
    """

    def compute_orbital_features_num(self):
        # counts the number of orbitals of each order across all atoms for the given basis
        sph_counts = [0 for L in range(self.orbitals_max_order + 1)]
        rad_counts = [0 for L in range(self.orbitals_max_order + 1)]
        # contains maximum number of radial components for each order across all atoms for the given basis
        r_max = {}
        spherical_dict = {}
        radial_dict = {}
        for z in self.spherical_spec.keys():
            rad_c = self.radial_count[z]
            for j in range(len(self.spherical_spec[z])):
                orb = self.spherical_spec[z][j]
                L = orb[2]
                n = orb[1]
                key = (z, L)
                if key not in spherical_dict:
                    # print('lcounts range', rad_counts[L], rad_counts[L] + n)
                    spherical_dict[key] = torch.arange(sph_counts[L], sph_counts[L] + n)
                    sph_counts[L] += n
            for j in range(len(self.radial_spec[z])):
                orb = self.radial_spec[z][j]
                L = orb[2]
                n = orb[1]
                key = (z, L)
                if key not in radial_dict:
                    # print('lcounts range', rad_counts[L], rad_counts[L] + n)
                    radial_dict[key] = torch.arange(rad_counts[L], rad_counts[L] + n)
                    rad_counts[L] += n
                    if len(rad_c[L]) < 1:
                        max_rad_c = 0
                    else:
                        max_rad_c = max(rad_c[L])
                    if key in r_max.keys():
                        r_max[key] = max(max_rad_c, r_max[key])
                    else:
                        r_max[key] = max_rad_c

        return sph_counts, rad_counts, r_max, spherical_dict, radial_dict

    """
    Counts how many features of each order are needed to collect the orbital coefficients
    outputs:
        matrix: Number of features required for each orbital order
    """
    def compute_orbital_features_num_compressed(self):
        L_counts = [0 for L in range(self.orbitals_max_order + 1)]
        r_max = [0 for L in range(self.orbitals_max_order + 1)]
        orbital_dict = {}
        for z in self.radial_spec.keys():
            for j in range(len(self.radial_spec[z])):
                orb = self.radial_spec[z][j]
                rad_c = self.radial_count[z][j]
                L = orb[2]
                n = orb[1]
                key = (z, L)
                if key not in orbital_dict:
                    orbital_dict[key] = range(n)
                    if L_counts[L] < n:
                        L_counts[L] = n
                    r_max[L] = max(rad_c[L], r_max[L])
        return L_counts, r_max, orbital_dict

    """
    Computes the spherical harmonics coefficients for the density

    inputs:
        R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
    outputs:
        C: Spherical harmonics coefficients
    """

    def forward(self, atoms):
        if self.memory:
            print('density coeffs forward start:')
            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
            print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        start = time.time()
        atoms['sph_repr_batch'] = [repr * 1 for repr in atoms['sph_repr']]
        atoms = batch_compressed_atoms(atoms, ['sph_repr_batch'])
        fs = atoms['sph_repr_batch']
        if self.verbose > 3:
            print('distances', atoms['distances'])
            print('fs[0]:', fs[0][:, 0, :, :10])
            print('fs[1]:', fs[1][:, 0, :, :10])
        out_sph = self.spherical_output(fs)
        if self.linear_out:
            out_sph = self.linear_output(out_sph)
        if self.scale_sph_order:
            for L in range(len(out_sph)):
                out_sph[L] = out_sph[L] * 10**(-L)
        if self.positive_coeffs:
            out_sph[0] = F.softplus(out_sph[0])
        out_width = []
        out_scale = []
        if self.pred_radial_coeffs:
            for L in range(len(self.radial_width)):
                out_width.append(torch.tanh(self.radial_width[L](fs[0])))
                # out_width[L] = out_width[L].view(*out_width[L].shape[:-2], self.r_max[L], self.L_counts[L])
                if self.positive_coeffs:
                    out_scale.append(torch.tanh(self.radial_scale[L](fs[0])))
                else:
                    out_scale.append(self.radial_scale[L](fs[0]))
        if self.memory:
            print('density coeffs forward outputs:')
            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
            print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        coeff_weighting = self.coeff_weights is not None
        # if self.timing:
        #     print('density coeffs setup time:', time.time() - start)
        # extract_start = time.time()
        if 'spherical_coeffs' not in atoms.keys():
            atoms['spherical_coeffs'], atoms['radial_width'], atoms['radial_scale'], atoms['coeff_weights'] =\
                self.extract_coefficients(out_sph, out_width, out_scale, atoms)
        # if self.timing:
        #     print('density coeffs extract time:', time.time() - extract_start)
            all_coeffs = coeffs_dict_to_vector(atoms, self.orbital_basis,
                                               atoms['batch_atom_numbers'],
                                               radial_coeffs=False, coeff_weighting=coeff_weighting,
                                               convert_to_pyscf=True)
            # df_start = time.time()
            atoms['df_coeffs'] = all_coeffs['spherical_coeffs']
            if coeff_weighting:
                atoms['df_weights'] = all_coeffs['coeff_weights']
        # if self.timing:
        #     print('density coeffs df vector time:', time.time() - df_start)
        else:
            core_coeffs = {}
            core_coeffs['spherical_coeffs'], core_coeffs['radial_width'], core_coeffs['radial_scale'], core_coeffs['coeff_weights'] =\
                self.extract_coefficients(out_sph, out_width, out_scale, atoms)
            for key in core_coeffs.keys():
                for i in range(len(core_coeffs[key])):
                    for orb in core_coeffs[key][i].keys():
                        atoms[key][i][orb] = torch.cat([core_coeffs[key][i][orb], atoms[key][i][orb]], dim=-1)
        if self.memory:
            print('density coeffs forward extract coeffs:')
            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
            print('Memory cached', torch.cuda.memory_cached() / 1024**2)

        if fs[0].get_device() >= 0:
            atoms['rad_dict'] = {key: self.rad_dict[key].to(fs[0].get_device()) for key in self.rad_dict.keys()}
            atoms['sph_dict'] = {key: self.sph_dict[key].to(fs[0].get_device()) for key in self.sph_dict.keys()}
        else:
            atoms['rad_dict'] = {key: self.rad_dict[key].to('cpu') for key in self.rad_dict.keys()}
            atoms['sph_dict'] = {key: self.sph_dict[key].to('cpu') for key in self.sph_dict.keys()}
        if self.timing:
            print('density coeffs time:', time.time() - start)
        if self.memory:
            print('density coeffs forward end:')
            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
            print('Memory cached', torch.cuda.memory_cached() / 1024**2)

        return atoms


class DensityExpansion(nn.Module):
    """
    Module for expanding density coefficients into a density sampled on a given grid
    """

    def __init__(self, orbital_basis,
                 expansion_constraint=None,
                 integral_constraint=None,
                 softmax_norm=False,
                 n_electrons=None,
                 integral_scale=False,
                 verbose=0,
                 timing=False,
                 memory=False,
                 grid_scaling_factor=False,
                 ):
        super().__init__()
        self.orbital_basis = orbital_basis
        self.expansion_constraint = expansion_constraint
        self.integral_constraint = integral_constraint
        self.softmax_norm = softmax_norm
        self.timing = timing
        self.memory = memory
        self.verbose = verbose
        if grid_scaling_factor:
            self.register_buffer('grid_scaling_factor', torch.ones(size=(1,)))
        else:
            self.grid_scaling_factor = 0
        if integral_scale:
            self.register_parameter('integral_scale', nn.Parameter(torch.ones(size=(1,))))
        else:
            self.register_buffer('integral_scale', torch.ones(size=(1,)))
        if self.verbose:
            print('expansion constraint', self.expansion_constraint)
            print('integral constraint', self.integral_constraint)

        self.orbitals_max_order_dict = get_max_order(self.orbital_basis, per_atom=True)
        self.orbitals_max_order = max(self.orbitals_max_order_dict.values())

        self.spherical_spec, self.radial_spec, radial_counts = combine_orbital_basis(self.orbital_basis, self.orbitals_max_order)
        self.r_max = {}
        radial_dict = {}
        for z in self.spherical_spec.keys():
            rad_c = radial_counts[z]
            for j in range(len(self.radial_spec[z])):
                orb = self.radial_spec[z][j]
                L = orb[2]
                key = (z, L)
                if key not in radial_dict:
                    if len(rad_c[L]) < 1:
                        max_rad_c = 0
                    else:
                        max_rad_c = max(rad_c[L])
                    if key in self.r_max.keys():
                        self.r_max[key] = max(max_rad_c, self.r_max[key])
                    else:
                        self.r_max[key] = max_rad_c

    def forward(self, atoms, eval_atoms=None, eval_L=None):
        n_eval = len(atoms['spherical_coeffs'])
        start = time.time()
        if eval_atoms is None:
            eval_atoms = list(range(n_eval))
        if eval_L is None:
            eval_L = list(range(self.orbitals_max_order + 1))
        atoms['density'] = 0
        L0_coeffs = []
        L0_sph = []
        L0_d = []
        L0_i = []
        L0_width = []
        n_electrons = get_n_electrons(atoms['atom_numbers'])
        for i in range(n_eval):
            if self.verbose > 1 and self.memory:
                print('Atom', i)
                print('density density expansion:')
                print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
                print('Memory cached', torch.cuda.memory_cached() / 1024**2)
            z = int(max(atoms['batch_atom_numbers'][:, i]))
            if z == 0:
                continue
            pos = atoms['batch_positions'][:, [i]]
            d, u = calculate_distances_and_directions(atoms['coords'], center=pos)
            s = spherical_harmonics(self.orbitals_max_order_dict[z], u)
            for L in range(len(s)):
                zeros = torch.zeros_like(s[L])
                s[L] = torch.where(torch.isnan(s[L]), zeros, s[L])  # making sure there are no nans to avoid NaNs
            for j in range(len(self.spherical_spec[z])):
                orb = self.spherical_spec[z][j]
                L = orb[2]
                key = (z, L)
                width = atoms['radial_width'][i][key]
                scale = atoms['radial_scale'][i][key]
                if self.verbose > 3:
                    print('width', width)
                    print('scale', scale)
                sph_coeff = atoms['spherical_coeffs'][i][key]
                dim_diff = s[L].dim() - sph_coeff.dim()
                if dim_diff >= 0:
                    sph_coeff = sph_coeff.reshape(sph_coeff.shape[:2] +
                                                  (1,) * (dim_diff + 1) +
                                                  sph_coeff.shape[2:])
                    width = width.reshape(width.shape[:2] +
                                          (1,) * (dim_diff + 1) +
                                          width.shape[2:])
                    scale = scale.reshape(scale.shape[:2] +
                                          (1,) * (dim_diff + 1) +
                                          scale.shape[2:])
                if L == 0:
                    L0_coeff = sph_coeff * scale
                    L0_coeffs.append(L0_coeff)
                    L0_sph.append(s[L])
                    L0_d.append(d)
                    L0_i.append(i)
                    L0_width.append(width)
                    continue
                sph = s[L].unsqueeze(-1) * sph_coeff
                rbf = gaussian_rbf(d.unsqueeze(-1), width, scale, L)
                if self.verbose > 2:
                    print('L', L)
                    print('rbf integral', torch.sum(rbf * atoms['coord_weights'].unsqueeze(-1).unsqueeze(-1), dim=(-2, -3)))
                    print('abs sph', torch.sum(torch.abs(sph) * atoms['coord_weights'].unsqueeze(-1).unsqueeze(-1), dim=(-2, -3)))
                if i in eval_atoms and L in eval_L:
                    atoms['density'] += torch.sum(rbf * sph, dim=(-2, -1))
        if self.verbose > 0:
            print('Density shape', atoms['density'].shape)
        L0_coeffs_comb = torch.cat([coeff.view((coeff.shape[0], -1)) for coeff in L0_coeffs], dim=1)
        L0_widths_comb = torch.cat([width.view((width.shape[0], -1)) for width in L0_width], dim=1)
        atoms['L0_coeffs'] = L0_coeffs_comb
        atoms['L0_widths'] = L0_coeffs_comb
        if self.integral_constraint is True or self.integral_constraint == 'coeffs':
            if self.softmax_norm:
                L0_coeffs_comb = F.softmax(L0_coeffs_comb, dim=1)
                L0_coeffs_comb = L0_coeffs_comb * n_electrons
                L0_coeffs_comb = L0_coeffs_comb * torch.clamp(self.integral_scale, 0.5, 1.5)
            else:
                norms = 1/gto_norm(0, L0_widths_comb)
                coeffs_sum = torch.sum(L0_coeffs_comb * norms / pyscf_gto_factor, dim=1, keepdim=True)
                scale_factor = n_electrons / coeffs_sum
                L0_coeffs_comb = L0_coeffs_comb * scale_factor
                L0_coeffs_comb = L0_coeffs_comb * torch.clamp(self.integral_scale, 0.5, 1.5)
        coeffs_pointer = 0
        L0_integrals = []
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
                normalize = False
                if (self.integral_constraint is True or self.integral_constraint == 'coeffs') and self.softmax_norm:
                    normalize = True
                rbf = gaussian_rbf(L0_d[i].unsqueeze(-1), L0_width[i], curr_coeffs, 0, normalize=normalize)
                sph = L0_sph[i].unsqueeze(-1)
                if self.verbose > 2:
                    print('L', 0)
                    print('rbf integral', torch.sum(rbf * atoms['coord_weights'].unsqueeze(-1).unsqueeze(-1), dim=(-2, -3)))
                    print('abs sph', torch.sum(torch.abs(sph) * atoms['coord_weights'].unsqueeze(-1).unsqueeze(-1), dim=(-2, -3)))
                if L0_i[i] in eval_atoms:
                    L0_dens = torch.sum(rbf * sph, dim=(-1, -2))
                    L0_int = torch.sum(L0_dens * atoms['coord_weights'], dim=-1)
                    L0_integrals.append(L0_int)
                    if self.verbose > 2:
                        print('L0_dens integral', L0_int)
                        print('l0 dens shape', L0_dens.shape)
                        print('atoms density shape', atoms['density'].shape)
                    atoms['density'] += L0_dens
        if self.verbose > 0:
            print('L0_int sum', torch.sum(torch.cat(L0_integrals)))
            print('sum neg integrals', torch.sum((atoms['density'] * atoms['coord_weights'])[atoms['density'] < 0], dim=-1))
        if self.expansion_constraint == 'sq':
            atoms['density'] = atoms['density']**2
        if self.expansion_constraint == 'abs':
            atoms['density'] = torch.sqrt(atoms['density']**2)
        if self.expansion_constraint == 'sp':
            atoms['density'] = F.softplus(atoms['density'], beta=100000000) + 1e-30
        if self.integral_constraint == 'grid':
            grid_scaling = n_electrons / torch.sum(atoms['density'] * atoms['coord_weights'], dim=1, keepdim=True)
            if self.verbose > 0:
                print('grid_scaling', grid_scaling)
                print('grid_scaling_factor', (1 - self.grid_scaling_factor))
            grid_scaling = grid_scaling * (1 - self.grid_scaling_factor) + self.grid_scaling_factor
            atoms['density'] = (atoms['density'] * grid_scaling)
        if self.timing:
            print('density expansion time:', time.time() - start)
        return atoms
