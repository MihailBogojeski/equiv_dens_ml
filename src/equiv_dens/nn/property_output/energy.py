import torch
import torch.nn as nn
from equiv_dens.nn.modules.network_blocks import ModularBlock, ResidualBlock, layer_norm_orders
from equiv_dens.nn.modules.radial_basis_functions import BernsteinRadialBasisFunctions,\
    GaussianRadialBasisFunctions, ExponentialBernsteinRadialBasisFunctions, ExponentialGaussianRadialBasisFunctions
from equiv_dens.nn.modules.activations import Swish, ShiftedSoftplus
from equiv_dens.utils.orbitals import combine_orbital_basis, get_invariant_features,\
        get_max_order, coeffs_dict_to_tensors, gto_norm, pyscf_gto_factor, gto_norm_pyscf
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.modules.spherical_harmonic_layers import SphericalLinear, apply_L0, order_scale
from equiv_dens.nn.modules.embeddings import SphericalEmbedding
import time
import numpy as np


def abs_(x):
    """|x| with the same derivatives as torch.abs (the double backward of torch.abs specializes dynamic shapes in torch.compile)."""
    return x * torch.sign(x).detach()


def accumulate_features(fs, ys, normalize):
    """Running sum of the output features of the modules (packed), rescaled per order if normalizing."""
    if fs is None:
        return ys
    D = min(fs.shape[-2], ys.shape[-2])
    fs_low, ys_low = fs[..., :D, :], ys[..., :D, :]
    scale = order_scale(fs_low, ys_low, normalize)
    low = ys_low * scale + fs_low * scale if normalize else ys_low + fs_low
    rest = fs[..., D:, :] if fs.shape[-2] > D else ys[..., D:, :]
    return torch.cat([low, rest], dim=-2) if rest.shape[-2] else low


class SphericalHarmonicsEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """
    def __init__(self,
                 orbital_basis=None,
                 order=1,  # maximum order of spherical harmonics features
                 mixing_order=None,
                 num_features=32,  # dimensionality of the feature space
                 num_basis_functions=32,  # number of basis functions for featurizing distances
                 num_radial_components=32,  # number of basis functions for the radial component of the density
                 # how many modules are stacked for calculating atomic features (iterations)
                 num_modules=1,
                 # number of residual blocks applied to atomic features before interaction layer
                 num_residual_pre_x=1,
                 # number of residual blocks applied to atomic features after interaction layer
                 num_residual_post_x=1,
                 # number of residual blocks applied to atomic features i before computing interaction features
                 num_residual_pre_vi=1,
                 # number of residual blocks applied to atomic features j before computing interaction features
                 num_residual_pre_vj=1,
                 # number of residual blocks applied to interaction features after combining atomic features i / j
                 num_residual_post_v=1,
                 # number of residual blocks applied to atomic features before collecting output atomic features
                 num_residual_output=1,
                 # type of radial basis functions (exp - gaussian / exp - bernstein / gaussian / bernstein)
                 basis_functions='exp-bernstein',
                 cutoff=15.0,  # cutoff distance (default is 15 Bohr)
                 # type of activation function used (swish / ssp)
                 activation='swish',
                 clebsch_gordan=None,  # instance of the clebsch gordan matrix
                 calculate_forces=False,
                 compressed_extraction=False,
                 verbose=0,
                 timing=False,
                 pred_radial_coeffs=True,
                 num_neighbours=1,
                 normalize=0,
                 parity=False,
                 atom_dens=None,
                 L0_start=True,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces

        # store hyperparameter values
        self.orbital_basis = orbital_basis
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_modules = num_modules
        self.num_residual_pre_x = num_residual_pre_x
        self.num_residual_post_x = num_residual_post_x
        self.num_residual_pre_vi = num_residual_pre_vi
        self.num_residual_pre_vj = num_residual_pre_vj
        self.num_residual_post_v = num_residual_post_v
        self.num_residual_output = num_residual_output
        self.num_radial_components = num_radial_components
        self.basis_functions = basis_functions
        self.cutoff = cutoff
        self.activation = activation
        self.compressed_extraction = compressed_extraction
        self.verbose = verbose
        self.timing = timing
        self.mixing_order = mixing_order
        self.pred_radial_coeffs = pred_radial_coeffs
        self.num_neighbours = num_neighbours
        self.normalize = normalize
        self.atom_dens = atom_dens
        self.L0_start = L0_start


        if self.mixing_order is None:
            self.mixing_order = self.order

        if not isinstance(self.order, list):
            self.order = [self.order] * self.num_modules

        if not isinstance(self.mixing_order, list):
            self.mixing_order = [self.mixing_order] * self.num_modules

        if len(self.order) != self.num_modules:
            print('Order needs to be an integer or a list of integers with length equal to num_modules.' +
                  ' Taking last order element and using it for all modules.')
            self.order = [self.order[-1]] * self.num_modules

        if len(self.mixing_order) != self.num_modules:
            print('Mixing order needs to be an integer or a list of integers with length equal to num_modules.' +
                  ' Taking last order element and using it for all modules.')
            self.mixing_order = [self.mixing_order[-1]] * self.num_modules

        self.orbitals_max_order = get_max_order(self.orbital_basis)
        self.spherical_spec, _, _ = combine_orbital_basis(self.orbital_basis, self.orbitals_max_order)
        self.dens_features = [0] * (self.orbitals_max_order + 1)
        seen_z = []

        self._order = [ord for ord in self.order]
        if self.L0_start:
            self._order[0] = 0
        
        if self.atom_dens is not None:
            # free-atom DF coefficients (s-type) of each element, in front of its L=0 channels: element z with n_z
            # predicted L=0 channels and m_z free-atom ones occupies [off_z, off_z + m_z + n_z), the free-atom ones first
            n0 = {z: sum(orb[1] for orb in spec if orb[2] == 0) for z, spec in self.spherical_spec.items()}
            m0 = {z: self.atom_dens[z]['df_coeffs'].shape[1] for z in self.spherical_spec}
            C = sum(n0[z] + m0[z] for z in self.spherical_spec)
            l0_map = []
            tables = torch.zeros(3, max(self.spherical_spec) + 1, C, dtype=torch.float64)
            off = 0
            for z in self.spherical_spec:
                widths = [shell[1][0] for shell in self.atom_dens[z]['df_basis'][z]]
                tables[0, z, off:off + m0[z]] = self.atom_dens[z]['df_coeffs'].reshape(-1).double()
                tables[1, z, off:off + m0[z]] = torch.tensor(widths)  # float32 widths, as the old code had them
                tables[2, z, off:off + m0[z]] = torch.tensor([gto_norm_pyscf(0, w) for w in widths], dtype=torch.float64)
                l0_map.extend(range(off + m0[z], off + m0[z] + n0[z]))
                off += m0[z] + n0[z]
            self.register_buffer('atom_df_l0_map', torch.tensor(l0_map), persistent=False)
            self.register_buffer('atom_df_coeffs', tables[0], persistent=False)
            self.register_buffer('atom_df_widths', tables[1], persistent=False)
            self.register_buffer('atom_df_scales', tables[2], persistent=False)
            self.atom_df = True
        else:
            self.atom_df = None

        for key in self.spherical_spec.keys():
            curr_feats = [0] * (self.orbitals_max_order + 1)
            for orb in self.spherical_spec[key]:
                L = orb[2]
                curr_feats[L] += orb[1]
                z = orb[0]
                if self.atom_dens is not None and L==0:
                    curr_feats[0] += self.atom_dens[z]['df_coeffs'].shape[1]

            if self.compressed_extraction:
                for L in range(len(curr_feats)):
                    if curr_feats[L] > self.dens_features[L]:
                        self.dens_features[L] = curr_feats[L]
            else:
                if z not in seen_z:
                    seen_z.append(z)
                    for L in range(len(curr_feats)):
                        self.dens_features[L] += curr_feats[L]

        self.order_max = max(self.mixing_order)
        if clebsch_gordan is None:
            self.clebsch_gordan = ClebschGordanMatrix()
        else:
            self.clebsch_gordan = clebsch_gordan
        # extract nuclear charges from orbitals, determine maximum order, and
        # build the occupation mask (for extracting occupied orbitals in energy prediction)
        if self.basis_functions == 'exp-gaussian':
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'exp-bernstein':
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff, normalize=self.normalize)
        elif self.basis_functions == 'gaussian':
            self.radial_basis_functions = GaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff, normalize=self.normalize)
        else:
            print("basis function type:",
                  self.basis_functions, "is not supported")

        if self.pred_radial_coeffs:
            for L in range(len(self.dens_features)):
                name = "radial_scale_filters_{}".format(L)
                self.register_parameter(name, nn.Parameter(torch.ones(1, self.dens_features[L])))
                name = "radial_width_filters_{}".format(L)
                self.register_parameter(name, nn.Parameter(torch.ones(1, self.dens_features[L])))
        if self.L0_start:
            self.radial_L0_map = nn.ModuleList([nn.Linear(df_num, self.dens_features[0]) for df_num in self.dens_features])

            # self.radial_scale_filters = nn.ParameterList([nn.Parameter(torch.ones(1, df_num)) for df_num in self.dens_features])
            # self.radial_width_filters = nn.ParameterList([nn.Parameter(torch.ones(1, df_num)) for df_num in self.dens_features])
        self.input_layer = nn.ModuleList([nn.Linear(df_num, self.num_features) for df_num in self.dens_features])

        modules = [ModularBlock(self._order[0], self.num_features, self.num_basis_functions,
                                self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                self.clebsch_gordan, True, self.mixing_order[0], self.orbitals_max_order, self.activation,
                                self.num_neighbours, normalize, parity=parity)]
        modules.extend([ModularBlock(self._order[i], self.num_features, self.num_basis_functions,
                                     self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                     self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                     self.clebsch_gordan, True, self.mixing_order[i], self._order[i - 1],
                                     self.activation, self.num_neighbours, normalize, parity=parity) for i in range(1, self.num_modules)])
        self.module = nn.ModuleList(modules)

        self.order_change = []
        if self.L0_start:
            self.L0_change = ResidualBlock(self.orbitals_max_order, self.num_features,
                                                   clebsch_gordan=self.clebsch_gordan,
                                                   activation=self.activation,
                                                   order_out=0, normalize=normalize,
                                                   parity=parity)
            self.order_change.append(nn.Identity())
        elif self.orbitals_max_order != self._order[0]:
            print('orbitals max order', self.orbitals_max_order)
            self.order_change.append(ResidualBlock(self.orbitals_max_order, self.num_features,
                                                   clebsch_gordan=self.clebsch_gordan,
                                                   activation=self.activation,
                                                   order_out=self._order[0], normalize=normalize,
                                                   parity=parity))
        else:
            self.order_change.append(nn.Identity())
        for i in range(1, self.num_modules):
            if self._order[i] != self._order[i - 1]:
                self.order_change.append(ResidualBlock(self._order[i - 1], self.num_features,
                                                       clebsch_gordan=self.clebsch_gordan,
                                                       activation=self.activation,
                                                       order_out=self._order[i], normalize=normalize,
                                                       parity=parity))
            else:
                self.order_change.append(nn.Identity())
        self.order_change = nn.ModuleList(self.order_change)

        if self.activation == 'swish':
            self.coeff_activation = nn.ModuleList([Swish(df_num) for df_num in self.dens_features])
            self.out_activation = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.coeff_activation = nn.ModuleList([ShiftedSoftplus(df_num) for df_num in self.dens_features])
            self.out_activation = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        self.energy_output = SphericalLinear(self._order[-1], self.num_features,
                                             0, 1, self.clebsch_gordan, parity=parity)

    def radial_scale_filters(self, L):
        return getattr(self, "radial_scale_filters_{}".format(L))

    def radial_width_filters(self, L):
        return getattr(self, "radial_width_filters_{}".format(L))

    def forward(self, atoms):
        idx_i = atoms['idx_i']
        idx_j = atoms['idx_j']
        atom_df = None
        if self.atom_df is not None:
            atom_df = (self.atom_df_l0_map, self.atom_df_coeffs, self.atom_df_widths, self.atom_df_scales)
        sph_fs, scale_fs, width_fs = coeffs_dict_to_tensors(atoms, radial_coeffs=self.pred_radial_coeffs, atom_df=atom_df)
        for i in range(len(sph_fs)):
            sph_fs[i] = sph_fs[i].view(1, -1, *sph_fs[i].shape[2:]).index_select(1, atoms['batch_idx_pos'])
            scale_fs[i] = scale_fs[i].view(1, -1, *scale_fs[i].shape[2:]).index_select(1, atoms['batch_idx_pos'])
            width_fs[i] = width_fs[i].view(1, -1, *width_fs[i].shape[2:]).index_select(1, atoms['batch_idx_pos'])
        dij = atoms['distances']
        sph = atoms['sph']
        rbf = self.radial_basis_functions(dij).unsqueeze(-2)  # unsqueeze for broadcasting
        xs = []
        radial_all = 0
        if self.pred_radial_coeffs:
            for L in range(len(scale_fs)):
                scale_fs[L] = scale_fs[L] * self.radial_scale_filters(L)
                width_fs[L] = width_fs[L] * self.radial_width_filters(L)
                radial_comb = self.coeff_activation[L](scale_fs[L] * width_fs[L])
                radial_comb = radial_comb / torch.sqrt(abs_(radial_comb) + 1e-8)
                radial_comb = radial_comb.sum(-2, keepdim=True)
                if self.L0_start:
                    radial_all = radial_all + self.radial_L0_map[L](radial_comb)
                    xs.append(sph_fs[L])
                else:
                    xs.append(sph_fs[L] * radial_comb)
            if self.L0_start:
                xs[0] = xs[0] * radial_all
                xs[0] = xs[0] / torch.sqrt(abs_(xs[0]) + 1e-8)

        else:
            xs = sph_fs

        xs = torch.cat([self.input_layer[L](xs[L]) for L in range(len(xs))], dim=-2)

        if self.L0_start:
            xs = self.L0_change(xs)

        if self.normalize > 1 or self.L0_start:
            xs = layer_norm_orders(xs)
        # perform iterations over modular building blocks to get environment - dependent features
        fs = None
        for i, module in enumerate(self.module):
            xs = self.order_change[i](xs)
            xs, ys = module(xs, rbf, sph, idx_i, idx_j)
            fs = accumulate_features(fs, ys, self.normalize)
        fs = apply_L0(self.out_activation, fs)

        atom_en = self.energy_output(fs).squeeze(-1).squeeze(-1)

        energy = torch.zeros(1, atoms['batch_atom_numbers'].shape[0], device=atoms['positions'].device, dtype=atoms['positions'].dtype)
        energy = energy.scatter_add(1, atoms['atom_batch_idx'], atom_en)
        energy = torch.t(energy)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        return atoms

class SphericalLinearEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """
    def __init__(self,
                 orbital_basis=None,
                 order=1,  # maximum order of spherical harmonics features
                 num_features=32,  # dimensionality of the feature space
                 # how many modules are stacked for calculating atomic features (iterations)
                 num_modules=1,
                 activation='swish',
                 clebsch_gordan=None,  # instance of the clebsch gordan matrix
                 calculate_forces=False,
                 compressed_extraction=False,
                 verbose=0,
                 timing=False,
                 pred_radial_coeffs=True,
                 normalize=0,
                 parity=False,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces

        # store hyperparameter values
        self.orbital_basis = orbital_basis
        self.order = order
        self.num_features = num_features
        self.num_modules = num_modules
        self.activation = activation
        self.compressed_extraction = compressed_extraction
        self.verbose = verbose
        self.timing = timing
        self.pred_radial_coeffs = pred_radial_coeffs
        self.normalize = normalize


        if not isinstance(self.order, list):
            self.order = [self.order] * self.num_modules

        self.orbitals_max_order = get_max_order(self.orbital_basis)
        self.spherical_spec, _, _ = combine_orbital_basis(self.orbital_basis, self.orbitals_max_order)
        self.dens_features = [0] * (self.orbitals_max_order + 1)
        seen_z = []
        for key in self.spherical_spec.keys():
            curr_feats = [0] * (self.orbitals_max_order + 1)
            z = self.spherical_spec[key][0][0]
            for orb in self.spherical_spec[key]:
                L = orb[2]
                curr_feats[L] += orb[1]

            if self.compressed_extraction:
                for L in range(len(curr_feats)):
                    if curr_feats[L] > self.dens_features[L]:
                        self.dens_features[L] = curr_feats[L]
            else:
                if z not in seen_z:
                    seen_z.append(z)
                    for L in range(len(curr_feats)):
                        self.dens_features[L] += curr_feats[L]

        self.order_max = max(self.order)

        if clebsch_gordan is None:
            self.clebsch_gordan = ClebschGordanMatrix()
        else:
            self.clebsch_gordan = clebsch_gordan

        if self.activation == 'swish':
            self.coeff_activation = nn.ModuleList([Swish(df_num) for df_num in self.dens_features])
            self.out_activation = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.coeff_activation = nn.ModuleList([ShiftedSoftplus(df_num) for df_num in self.dens_features])
            self.out_activation = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()
            
        if self.pred_radial_coeffs:
            for L in range(len(self.dens_features)):
                name = "radial_scale_filters_{}".format(L)
                self.register_parameter(name, nn.Parameter(torch.ones(1, self.dens_features[L])))
                name = "radial_width_filters_{}".format(L)
                self.register_parameter(name, nn.Parameter(torch.ones(1, self.dens_features[L])))
            # self.radial_scale_filters = nn.ParameterList([nn.Parameter(torch.ones(1, df_num)) for df_num in self.dens_features])
            # self.radial_width_filters = nn.ParameterList([nn.Parameter(torch.ones(1, df_num)) for df_num in self.dens_features])
        self.input_layer = nn.ModuleList([nn.Linear(df_num, self.num_features) for df_num in self.dens_features])

        self.order_change = []
        if self.orbitals_max_order != self.order[0]:
            print('orbitals max order', self.orbitals_max_order)
            self.order_change.append(ResidualBlock(self.orbitals_max_order, self.num_features,
                                                     clebsch_gordan=self.clebsch_gordan,
                                                     activation=self.activation,
                                                     order_out=self.order[0]))
        else:
            self.order_change.append(nn.Identity())
        for i in range(1, self.num_modules):
            if self.order[i] != self.order[i - 1]:
                self.order_change.append(ResidualBlock(self.order[i - 1], self.num_features,
                                                       clebsch_gordan=self.clebsch_gordan,
                                                       activation=self.activation,
                                                       order_out=self.order[i], normalize=self.normalize,
                                                       parity=parity))
            else:
                self.order_change.append(nn.Identity())
        self.order_change = nn.ModuleList(self.order_change)

        modules = [ResidualBlock(self.order[0], self.num_features, clebsch_gordan=self.clebsch_gordan,
                                 activation=self.activation,
                                 normalize=self.normalize, parity=parity)]
        modules.extend([ResidualBlock(self.order[i], self.num_features,
                                     clebsch_gordan=self.clebsch_gordan,
                                     activation=self.activation,
                                     normalize=self.normalize, parity=parity) for i in range(1, self.num_modules)])
        self.module = nn.ModuleList(modules)

        self.energy_output = SphericalLinear(self.order[-1], self.num_features, 0, 1,
                                             self.clebsch_gordan, parity=parity)

    def radial_scale_filters(self, L):
        return getattr(self, "radial_scale_filters_{}".format(L))

    def radial_width_filters(self, L):
        return getattr(self, "radial_width_filters_{}".format(L))

    def forward(self, atoms):
        idx_i = atoms['idx_i']
        idx_j = atoms['idx_j']
        sph_fs, scale_fs, width_fs = coeffs_dict_to_tensors(atoms, radial_coeffs=self.pred_radial_coeffs)
        for i in range(len(sph_fs)):
            sph_fs[i] = sph_fs[i].view(1, -1, *sph_fs[i].shape[2:]).index_select(1, atoms['batch_idx_pos'])
            scale_fs[i] = scale_fs[i].view(1, -1, *scale_fs[i].shape[2:]).index_select(1, atoms['batch_idx_pos'])
            width_fs[i] = width_fs[i].view(1, -1, *width_fs[i].shape[2:]).index_select(1, atoms['batch_idx_pos'])
        xs = []
        if self.pred_radial_coeffs:
            for L in range(len(scale_fs)):
                scale_fs[L] = scale_fs[L] * self.radial_scale_filters(L)
                width_fs[L] = width_fs[L] * self.radial_width_filters(L)
                radial_comb = self.coeff_activation[L](scale_fs[L] * width_fs[L])
                radial_comb = radial_comb.sum(-2, keepdim=True)
                xs.append(sph_fs[L] * radial_comb)
        else:
            xs = sph_fs

        xs = torch.cat([self.input_layer[L](xs[L]) for L in range(len(xs))], dim=-2)
        fs = xs
        for i, module in enumerate(self.module):
            xs = self.order_change[i](xs)
            xs = module(xs)
            fs = accumulate_features(fs, xs, self.normalize)

        fs = apply_L0(self.out_activation, fs)
        atom_en = self.energy_output(xs).squeeze(-1).squeeze(-1)

        energy = torch.sum(atom_en, dim=1, keepdim=True)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        return atoms

class RepresentationEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """
    def __init__(self,
                 order=1,  # maximum order of spherical harmonics features
                 num_features=32,  # dimensionality of the feature space
                 activation='swish',
                 clebsch_gordan=None,  # instance of the clebsch gordan matrix
                 calculate_forces=False,
                 verbose=0,
                 timing=False,
                 parity=False,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces

        self.order = order
        self.num_features = num_features
        self.activation = activation
        self.verbose = verbose
        self.timing = timing

        if not isinstance(self.order, list):
            self.order = [self.order] * 1 

        if clebsch_gordan is None:
            self.clebsch_gordan = ClebschGordanMatrix()
        else:
            self.clebsch_gordan = clebsch_gordan
        # extract nuclear charges from orbitals, determine maximum order, and
        # build the occupation mask (for extracting occupied orbitals in energy prediction)
        if self.activation == 'swish':
            self.out_activation = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.out_activation = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        self.energy_output = SphericalLinear(self.order[-1], self.num_features,
                                             0, 1, self.clebsch_gordan, parity=parity)

    def radial_scale_filters(self, L):
        return getattr(self, "radial_scale_filters_{}".format(L))

    def radial_width_filters(self, L):
        return getattr(self, "radial_width_filters_{}".format(L))

    def forward(self, atoms):
        fs = apply_L0(self.out_activation, atoms['sph_repr'])

        atom_en = self.energy_output(fs).squeeze(-1).squeeze(-1)

        energy = torch.zeros(1, atoms['batch_atom_numbers'].shape[0], device=atoms['positions'].device, dtype=atoms['positions'].dtype)
        energy = energy.scatter_add(1, atoms['atom_batch_idx'], atom_en)
        energy = torch.t(energy)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        return atoms

class SphericalHarmonicsEmbeddingEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """
    def __init__(self,
                 orbital_basis=None,
                 order=1,  # maximum order of spherical harmonics features
                 mixing_order=None,
                 num_features=32,  # dimensionality of the feature space
                 num_basis_functions=32,  # number of basis functions for featurizing distances
                 num_radial_components=32,  # number of basis functions for the radial component of the density
                 # how many modules are stacked for calculating atomic features (iterations)
                 num_modules=1,
                 # number of residual blocks applied to atomic features before interaction layer
                 num_residual_pre_x=1,
                 # number of residual blocks applied to atomic features after interaction layer
                 num_residual_post_x=1,
                 # number of residual blocks applied to atomic features i before computing interaction features
                 num_residual_pre_vi=1,
                 # number of residual blocks applied to atomic features j before computing interaction features
                 num_residual_pre_vj=1,
                 # number of residual blocks applied to interaction features after combining atomic features i / j
                 num_residual_post_v=1,
                 # number of residual blocks applied to atomic features before collecting output atomic features
                 num_residual_output=1,
                 # type of radial basis functions (exp - gaussian / exp - bernstein / gaussian / bernstein)
                 basis_functions='exp-bernstein',
                 cutoff=15.0,  # cutoff distance (default is 15 Bohr)
                 # type of activation function used (swish / ssp)
                 activation='swish',
                 clebsch_gordan=None,  # instance of the clebsch gordan matrix
                 calculate_forces=False,
                 compressed_extraction=False,
                 verbose=0,
                 timing=False,
                 pred_radial_coeffs=True,
                 num_neighbours=1,
                 normalize=0,
                 parity=False,
                 Zmax=87,
                 atom_dens=None,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces

        # store hyperparameter values
        self.orbital_basis = orbital_basis
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_modules = num_modules
        self.num_residual_pre_x = num_residual_pre_x
        self.num_residual_post_x = num_residual_post_x
        self.num_residual_pre_vi = num_residual_pre_vi
        self.num_residual_pre_vj = num_residual_pre_vj
        self.num_residual_post_v = num_residual_post_v
        self.num_residual_output = num_residual_output
        self.num_radial_components = num_radial_components
        self.basis_functions = basis_functions
        self.cutoff = cutoff
        self.activation = activation
        self.compressed_extraction = compressed_extraction
        self.verbose = verbose
        self.timing = timing
        self.mixing_order = mixing_order
        self.pred_radial_coeffs = pred_radial_coeffs
        self.num_neighbours = num_neighbours
        self.normalize = normalize
        self.Zmax = Zmax
        self.atom_dens = atom_dens

        if self.mixing_order is None:
            self.mixing_order = self.order

        if not isinstance(self.order, list):
            self.order = [self.order] * self.num_modules

        if not isinstance(self.mixing_order, list):
            self.mixing_order = [self.mixing_order] * self.num_modules

        if len(self.order) != self.num_modules:
            print('Order needs to be an integer or a list of integers with length equal to num_modules.' +
                  ' Taking last order element and using it for all modules.')
            self.order = [self.order[-1]] * self.num_modules

        if len(self.mixing_order) != self.num_modules:
            print('Mixing order needs to be an integer or a list of integers with length equal to num_modules.' +
                  ' Taking last order element and using it for all modules.')
            self.mixing_order = [self.mixing_order[-1]] * self.num_modules

        self.orbitals_max_order = get_max_order(self.orbital_basis)
        self.spherical_spec, _, _ = combine_orbital_basis(self.orbital_basis, self.orbitals_max_order)
        self.dens_features = [0] * (self.orbitals_max_order + 1)
        seen_z = []
        for key in self.spherical_spec.keys():
            curr_feats = [0] * (self.orbitals_max_order + 1)
            z = self.spherical_spec[key][0][0]
            for orb in self.spherical_spec[key]:
                L = orb[2]
                curr_feats[L] += orb[1]

            if self.compressed_extraction:
                for L in range(len(curr_feats)):
                    if curr_feats[L] > self.dens_features[L]:
                        self.dens_features[L] = curr_feats[L]
            else:
                if z not in seen_z:
                    seen_z.append(z)
                    for L in range(len(curr_feats)):
                        self.dens_features[L] += curr_feats[L]

        self.order_max = max(self.mixing_order)
        if clebsch_gordan is None:
            self.clebsch_gordan = ClebschGordanMatrix()
        else:
            self.clebsch_gordan = clebsch_gordan
        self.embedding = SphericalEmbedding(
            self.orbitals_max_order, self.num_features, self.Zmax)
        if self.basis_functions == 'exp-gaussian':
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'exp-bernstein':
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff, normalize=self.normalize)
        elif self.basis_functions == 'gaussian':
            self.radial_basis_functions = GaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff, normalize=self.normalize)
        else:
            print("basis function type:",
                  self.basis_functions, "is not supported")

        if self.pred_radial_coeffs:
            for L in range(len(self.dens_features)):
                name = "radial_scale_filters_{}".format(L)
                self.register_parameter(name, nn.Parameter(torch.ones(1, self.dens_features[L])))
                name = "radial_width_filters_{}".format(L)
                self.register_parameter(name, nn.Parameter(torch.ones(1, self.dens_features[L])))
            # self.radial_scale_filters = nn.ParameterList([nn.Parameter(torch.ones(1, df_num)) for df_num in self.dens_features])
            # self.radial_width_filters = nn.ParameterList([nn.Parameter(torch.ones(1, df_num)) for df_num in self.dens_features])
        self.input_layer = nn.ModuleList([nn.Linear(df_num, self.num_features) for df_num in self.dens_features])

        modules = [ModularBlock(self.order[0], self.num_features, self.num_basis_functions,
                                self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                self.clebsch_gordan, True, self.mixing_order[0], self.orbitals_max_order, self.activation,
                                self.num_neighbours, normalize, parity=parity)]
        modules.extend([ModularBlock(self.order[i], self.num_features, self.num_basis_functions,
                                     self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                     self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                     self.clebsch_gordan, True, self.mixing_order[i], self.order[i - 1],
                                     self.activation, self.num_neighbours, normalize, parity=parity) for i in range(1, self.num_modules)])
        self.module = nn.ModuleList(modules)

        self.order_change = []
        if self.orbitals_max_order != self.order[0]:
            print('orbitals max order', self.orbitals_max_order)
            self.order_change.append(ResidualBlock(self.orbitals_max_order, self.num_features,
                                                   clebsch_gordan=self.clebsch_gordan,
                                                   activation=self.activation,
                                                   order_out=self.order[0], normalize=normalize,
                                                   parity=parity))
        else:
            self.order_change.append(nn.Identity())
        for i in range(1, self.num_modules):
            if self.order[i] != self.order[i - 1]:
                self.order_change.append(ResidualBlock(self.order[i - 1], self.num_features,
                                                       clebsch_gordan=self.clebsch_gordan,
                                                       activation=self.activation,
                                                       order_out=self.order[i], normalize=normalize,
                                                       parity=parity))
            else:
                self.order_change.append(nn.Identity())
        self.order_change = nn.ModuleList(self.order_change)

        if self.activation == 'swish':
            self.coeff_activation = nn.ModuleList([Swish(df_num) for df_num in self.dens_features])
            self.out_activation = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.coeff_activation = nn.ModuleList([ShiftedSoftplus(df_num) for df_num in self.dens_features])
            self.out_activation = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        self.energy_output = SphericalLinear(self.order[-1], self.num_features,
                                             0, 1, self.clebsch_gordan, parity=parity)

    def radial_scale_filters(self, L):
        return getattr(self, "radial_scale_filters_{}".format(L))

    def radial_width_filters(self, L):
        return getattr(self, "radial_width_filters_{}".format(L))

    def forward(self, atoms):
        idx_i = atoms['idx_i']
        idx_j = atoms['idx_j']
        sph_fs, scale_fs, width_fs = coeffs_dict_to_tensors(atoms, radial_coeffs=self.pred_radial_coeffs)
        for i in range(len(sph_fs)):
            sph_fs[i] = sph_fs[i].view(1, -1, *sph_fs[i].shape[2:]).index_select(1, atoms['batch_idx_pos'])
            scale_fs[i] = scale_fs[i].view(1, -1, *scale_fs[i].shape[2:]).index_select(1, atoms['batch_idx_pos'])
            width_fs[i] = width_fs[i].view(1, -1, *width_fs[i].shape[2:]).index_select(1, atoms['batch_idx_pos'])
        dij = atoms['distances']
        sph = atoms['sph']
        rbf = self.radial_basis_functions(dij).unsqueeze(-2)  # unsqueeze for broadcasting
        xs = self.embedding(atoms['atom_numbers'])
        dens_fs = []
        if self.pred_radial_coeffs:
            for L in range(len(scale_fs)):
                scale_fs[L] = scale_fs[L] * self.radial_scale_filters(L)
                width_fs[L] = width_fs[L] * self.radial_width_filters(L)
                radial_comb = self.coeff_activation[L](scale_fs[L] * width_fs[L])
                radial_comb = radial_comb.sum(-2, keepdim=True)
                dens_fs.append(sph_fs[L] * radial_comb)
        else:
            dens_fs = sph_fs

        D = min(xs.shape[-2], sum(2 * L + 1 for L in range(len(dens_fs))))
        dens_fs = torch.cat([self.input_layer[L](dens_fs[L]) for L in range(len(dens_fs))], dim=-2)

        if self.normalize > 1:
            dens_fs = layer_norm_orders(dens_fs)

        xs = xs[..., :D, :] + dens_fs[..., :D, :]
        fs = xs
        for i, module in enumerate(self.module):
            xs = self.order_change[i](xs)
            xs, ys = module(xs, rbf, sph, idx_i, idx_j)
            fs = accumulate_features(fs, ys, self.normalize)
        fs = apply_L0(self.out_activation, fs)

        atom_en = self.energy_output(fs).squeeze(-1).squeeze(-1)

        energy = torch.zeros(1, atoms['batch_atom_numbers'].shape[0], device=atoms['positions'].device, dtype=atoms['positions'].dtype)
        energy = energy.scatter_add(1, atoms['atom_batch_idx'], atom_en)
        energy = torch.t(energy)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        return atoms

