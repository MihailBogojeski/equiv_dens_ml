import torch
import torch.nn as nn
from equiv_dens.nn.modules.network_blocks import ModularBlock, ResidualBlock
from equiv_dens.nn.modules.radial_basis_functions import BernsteinRadialBasisFunctions,\
    GaussianRadialBasisFunctions, ExponentialBernsteinRadialBasisFunctions, ExponentialGaussianRadialBasisFunctions
from equiv_dens.nn.modules.activations import Swish, ShiftedSoftplus
from equiv_dens.utils.orbitals import combine_orbitals, combine_orbital_basis, get_invariant_features, get_max_order, coeffs_dict_to_tensors
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.modules.spherical_harmonic_layers import SphericalLinear
import time


class ComplexEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """

    def __init__(self,
                 orbitals=None,
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
                 calculate_forces=False,
                 compressed_extraction=False,
                 verbose=0,
                 timing=False,
                 pred_radial_coeffs=True,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces

        # store hyperparameter values
        self.orbitals = orbitals
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
        self.pred_radial_coeffs = pred_radial_coeffs

        N = len(self.orbitals)
        idx_i = torch.arange(N, dtype=torch.int64).view(-1, 1).repeat(1, N).view(-1)
        idx_j = torch.arange(N, dtype=torch.int64).view(1, -1).repeat(N, 1).view(-1)
        # exclude self - interactions
        idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j]
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)

        self.orbitals_max_order = get_max_order(self.orbitals)
        self.orbital_spec, _, _ = combine_orbitals(self.orbitals, self.orbitals_max_order)
        self.dens_features = 0
        seen_zs = []
        for i in range(len(self.orbital_spec)):
            curr_feats = 0
            for orb in self.orbital_spec[i]:
                curr_feats += orb[1]
            if self.compressed_extraction:
                if curr_feats > self.dens_features:
                    self.dens_features = curr_feats
            else:
                if orb[0] not in seen_zs:
                    self.dens_features += curr_feats
                    seen_zs.append(orb[0])
        if self.pred_radial_coeffs:
            self.dens_features *= 3
        if self.num_features is None:
            self.num_features = self.dens_features
        # extract nuclear charges from orbitals, determine maximum order, and
        # build the occupation mask (for extracting occupied orbitals in energy prediction)
        if self.basis_functions == 'exp-gaussian':
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'exp-bernstein':
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'gaussian':
            self.radial_basis_functions = GaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        else:
            print("basis function type:",
                  self.basis_functions, "is not supported")

        if self.num_features != self.dens_features:
            self.input_layer = nn.Linear(self.dens_features, self.num_features)
        self.module = nn.ModuleList([ModularBlock(0, self.num_features, self.num_basis_functions,
                                                  self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                                  self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                                  None, False, None, None, self.activation) for i in range(self.num_modules)])

        if self.activation == 'swish':
            self.out_activation = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.out_activation = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        self.energy_output = nn.Linear(self.num_features, 1)

    def forward(self, atoms):
        # initialize atomic features to embeddings
        start = time.time()
        xs = get_invariant_features(atoms, permutational_invariance=False,
                                    keep_dims=True, radial_coeffs=self.pred_radial_coeffs)
        dij = atoms['distances']
        sph = atoms['sph']
        # print('dij shape', dij.shape)
        # print('uij shape', uij.shape)
        # print('R shape', R.shape)
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2)  # unsqueeze for broadcasting

        if self.num_features != self.dens_features:
            xs = self.input_layer(xs)
        xs = [xs.unsqueeze(-2)]
        # print('xs[0] shape', xs[0].shape)

        # perform iterations over modular building blocks to get environment - dependent features
        fs = [torch.zeros_like(x) for x in xs]  # output features
        for module in self.module:
            xs, ys = module(xs, rbf, sph, self.idx_i, self.idx_j)
            fs[0] += ys[0]  # add contributions to output features

        atom_en = self.energy_output(self.out_activation(fs[0])).squeeze(-1).squeeze(-1)

        energy = torch.sum(atom_en, dim=1, keepdim=True)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        if self.timing:
            print('simple energy time', time.time() - start)
        return atoms


class SimpleEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """

    def __init__(self,
                 orbitals=None,
                 num_features=32,
                 num_layers=5,
                 calculate_forces=False,
                 activation='swish',
                 compressed_extraction=False,
                 verbose=0,
                 timing=False,
                 pred_radial_coeffs=True,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces
        self.compressed_extraction = compressed_extraction

        # store hyperparameter values
        self.orbitals = orbitals
        self.num_features = num_features

        self.orbitals_max_order = get_max_order(self.orbitals)
        self.activation = activation

        self.orbital_spec, _, _ = combine_orbitals(self.orbitals, self.orbitals_max_order)
        self.dens_features = 0
        self.verbose = verbose
        self.timing = timing
        self.pred_radial_coeffs = pred_radial_coeffs

        seen_zs = []
        for i in range(len(self.orbital_spec)):
            curr_feats = 0
            for orb in self.orbital_spec[i]:
                curr_feats += orb[1]
            if self.compressed_extraction:
                if curr_feats > self.dens_features:
                    self.dens_features = curr_feats
            else:
                if orb[0] not in seen_zs:
                    self.dens_features += curr_feats
                    seen_zs.append(orb[0])
        if self.pred_radial_coeffs:
            self.dens_features *= 3
        if self.num_features is None:
            self.num_features = self.dens_features

        if self.num_features != self.dens_features:
            self.input_layer = nn.Linear(self.dens_features, self.num_features)

        self.transformation_layers = nn.ModuleList([nn.Linear(self.num_features, self.num_features) for i in range(num_layers)])

        if self.activation == 'swish':
            self.out_activation = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.out_activation = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        self.energy_output = nn.Linear(self.num_features, 1)

    def forward(self, atoms):
        start = time.time()
        # initialize atomic features to embeddings
        # print('sph coeffs', atoms['spherical_coeffs'][0][(8, 0)])
        fs = get_invariant_features(atoms, permutational_invariance=False,
                                    keep_dims=True, radial_coeffs=self.pred_radial_coeffs)
        if self.verbose > 3:
            print('fs.shape', fs.shape)

        if self.num_features != self.dens_features:
            fs = self.input_layer(fs)
        for layer in self.transformation_layers:
            # print('fs intermediate', fs)
            fs = self.out_activation(layer(fs))
        atom_en = self.energy_output(self.out_activation(fs)).squeeze(-1)

        if self.verbose > 3:
            print('atom en shape', atom_en.shape)

        energy = torch.sum(atom_en, dim=1, keepdim=True)

        # print('energy shape', energy.shape)
        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        if self.timing:
            print('simple energy time', time.time() - start)
        return atoms


class SphericalHarmonicsEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """
    def __init__(self,
                 orbitals=None,
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
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces

        # store hyperparameter values
        self.orbitals = orbitals
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

        if self.mixing_order is None:
            self.mixing_order = self.order

        print('self order', self.order)
        if not isinstance(self.order, list):
            self.order = [self.order] * self.num_modules
        print('self order', self.order)

        print('self mixing_order', self.mixing_order)
        if not isinstance(self.mixing_order, list):
            self.mixing_order = [self.mixing_order] * self.num_modules
        print('self mixing_order', self.mixing_order)

        if len(self.order) != self.num_modules:
            print('Order needs to be an integer or a list of integers with length equal to num_modules.' +
                  ' Taking last order element and using it for all modules.')
            self.order = [self.order[-1]] * self.num_modules

        if len(self.mixing_order) != self.num_modules:
            print('Mixing order needs to be an integer or a list of integers with length equal to num_modules.' +
                  ' Taking last order element and using it for all modules.')
            self.mixing_order = [self.mixing_order[-1]] * self.num_modules

        N = len(self.orbitals)
        idx_i = torch.arange(N, dtype=torch.int64).view(-1, 1).repeat(1, N).view(-1)
        idx_j = torch.arange(N, dtype=torch.int64).view(1, -1).repeat(N, 1).view(-1)
        # exclude self - interactions
        idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j]
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)

        self.orbitals_max_order = get_max_order(self.orbitals)
        self.orbital_spec, _, _ = combine_orbitals(self.orbitals, self.orbitals_max_order)
        self.dens_features = [0] * (self.orbitals_max_order + 1)
        seen_z = []
        for i in range(len(self.orbital_spec)):
            curr_feats = [0] * (self.orbitals_max_order + 1)
            z = self.orbital_spec[i][0][0]
            for orb in self.orbital_spec[i]:
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
        # extract nuclear charges from orbitals, determine maximum order, and
        # build the occupation mask (for extracting occupied orbitals in energy prediction)
        if self.basis_functions == 'exp-gaussian':
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'exp-bernstein':
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'gaussian':
            self.radial_basis_functions = GaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
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
                                self.clebsch_gordan, True, self.mixing_order[0], 0, self.activation)]
        modules.extend([ModularBlock(self.order[i], self.num_features, self.num_basis_functions,
                                     self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                     self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                     self.clebsch_gordan, True, self.mixing_order[i], self.order[i - 1],
                                     self.activation) for i in range(1, self.num_modules)])
        self.module = nn.ModuleList(modules)

        self.order_change = [nn.Identity()]
        for i in range(1, self.num_modules):
            if self.order[i] != self.order[i - 1]:
                self.order_change.append(SphericalLinear(self.order[i - 1], self.num_features,
                                                         self.order[i], self.num_features,
                                                         clebsch_gordan=self.clebsch_gordan,
                                                         bias=False))
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

        self.energy_output = SphericalLinear(self.order[-1], self.num_features, 0, 1, self.clebsch_gordan)

    def radial_scale_filters(self, L):
        return getattr(self, "radial_scale_filters_{}".format(L))

    def radial_width_filters(self, L):
        return getattr(self, "radial_width_filters_{}".format(L))

    def forward(self, atoms):
        start = time.time()
        # initialize atomic features to embeddings
        sph_fs, scale_fs, width_fs = coeffs_dict_to_tensors(atoms, radial_coeffs=self.pred_radial_coeffs)
        dij = atoms['distances']
        sph = atoms['sph']
        # print('dij shape', dij.shape)
        # print('uij shape', uij.shape)
        # print('R shape', R.shape)
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2)  # unsqueeze for broadcasting
        xs = []
        # print('dens features', self.dens_features)
        if self.pred_radial_coeffs:
            for L in range(len(scale_fs)):
                # print('L', L)
                # print('scale fs L', scale_fs[L].shape)
                # print('self.radial_scale_filters[L]', self.radial_scale_filters(L).shape)
                scale_fs[L] = scale_fs[L] * self.radial_scale_filters(L)
                width_fs[L] = width_fs[L] * self.radial_width_filters(L)
                radial_comb = self.coeff_activation[L](scale_fs[L] * width_fs[L])
                radial_comb = radial_comb.sum(-2, keepdim=True)
                xs.append(sph_fs[L] * radial_comb)
        else:
            xs = sph_fs

        for L in range(len(xs)):
            xs[L] = self.input_layer[L](xs[L])

        # perform iterations over modular building blocks to get environment - dependent features
        fs = [torch.zeros_like(x) for x in xs]  # output features
        for i, module in enumerate(self.module):
            xs = self.order_change[i](xs)
            xs, ys = module(xs, rbf, sph, self.idx_i, self.idx_j)
            for L in range(self.order[i] + 1):
                fs[L] += ys[L]  # add contributions to output features
        fs[0] = self.out_activation(fs[0])

        atom_en = self.energy_output(fs)[0].squeeze(-1).squeeze(-1)
        print('atomic energies', atom_en)

        energy = torch.sum(atom_en, dim=1, keepdim=True)

        atoms['energy'] = energy
        if self.calculate_forces:
            print('forces create graph', self.training)
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        if self.timing:
            print('simple energy time', time.time() - start)
        return atoms


class TransferableSphericalHarmonicsEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """
    def __init__(self,
                 orbitals=None,
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
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces

        # store hyperparameter values
        self.orbitals = orbitals
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

        self.orbital_basis = {}
        for orb in orbitals:
            z = orb[0][0]
            if z not in self.orbital_basis.keys():
                self.orbital_basis[z] = orb

        if self.mixing_order is None:
            self.mixing_order = self.order

        if not isinstance(self.order, list):
            self.order = [self.order] * self.num_modules

        if not isinstance(self.mixing_order, list):
            self.mixing_order = [self.mixing_order] * self.num_modules

        print('self energy order', self.order)
        print('self energy mixing order', self.order)
        print('self energy num features', self.num_features)

        if len(self.order) != self.num_modules:
            print('Order needs to be an integer or a list of integers with length equal to num_modules.' +
                  ' Taking last order element and using it for all modules.')
            self.order = [self.order[-1]] * self.num_modules

        if len(self.mixing_order) != self.num_modules:
            print('Mixing order needs to be an integer or a list of integers with length equal to num_modules.' +
                  ' Taking last order element and using it for all modules.')
            self.mixing_order = [self.mixing_order[-1]] * self.num_modules

        self.orbitals_max_order = get_max_order(self.orbitals)
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
        # extract nuclear charges from orbitals, determine maximum order, and
        # build the occupation mask (for extracting occupied orbitals in energy prediction)
        if self.basis_functions == 'exp-gaussian':
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'exp-bernstein':
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'gaussian':
            self.radial_basis_functions = GaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
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
                                self.clebsch_gordan, True, self.mixing_order[0], 0, self.activation)]
        modules.extend([ModularBlock(self.order[i], self.num_features, self.num_basis_functions,
                                     self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                     self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                     self.clebsch_gordan, True, self.mixing_order[i], self.order[i - 1],
                                     self.activation) for i in range(1, self.num_modules)])
        self.module = nn.ModuleList(modules)

        self.order_change = [nn.Identity()]
        for i in range(1, self.num_modules):
            if self.order[i] != self.order[i - 1]:
                self.order_change.append(SphericalLinear(self.order[i - 1], self.num_features,
                                                         self.order[i], self.num_features,
                                                         clebsch_gordan=self.clebsch_gordan,
                                                         bias=False))
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

        self.energy_output = SphericalLinear(self.order[-1], self.num_features, 0, 1, self.clebsch_gordan)

    def radial_scale_filters(self, L):
        return getattr(self, "radial_scale_filters_{}".format(L))

    def radial_width_filters(self, L):
        return getattr(self, "radial_width_filters_{}".format(L))

    def forward(self, atoms):
        start = time.time()
        R = atoms['positions']
        N = R.shape[1]
        batch_size = R.shape[0]
        idx_i = torch.arange(N).view(-1, 1).repeat(1, N).view(-1).to(R).type(torch.int64)
        idx_j = torch.arange(N).view(1, -1).repeat(N, 1).view(-1).to(R).type(torch.int64)
        neighbor_mask = atoms['atom_mask'].view(batch_size, 1, -1).repeat(1, N, 1).view(batch_size, -1)
        # exclude self - interactions
        neighbor_mask = neighbor_mask[:, idx_i != idx_j]
        idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j]
        # initialize atomic features to embeddings
        sph_fs, scale_fs, width_fs = coeffs_dict_to_tensors(atoms, radial_coeffs=self.pred_radial_coeffs)
        dij = atoms['distances']
        sph = atoms['sph']
        # print('dij shape', dij.shape)
        # print('uij shape', uij.shape)
        # print('R shape', R.shape)
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2)  # unsqueeze for broadcasting
        xs = []
        # print('dens features', self.dens_features)
        if self.pred_radial_coeffs:
            for L in range(len(scale_fs)):
                # print('L', L)
                # print('scale fs L', scale_fs[L].shape)
                # print('self.radial_scale_filters[L]', self.radial_scale_filters(L).shape)
                scale_fs[L] = scale_fs[L] * self.radial_scale_filters(L)
                width_fs[L] = width_fs[L] * self.radial_width_filters(L)
                radial_comb = self.coeff_activation[L](scale_fs[L] * width_fs[L])
                radial_comb = radial_comb.sum(-2, keepdim=True)
                xs.append(sph_fs[L] * radial_comb)
        else:
            xs = sph_fs

        for L in range(len(xs)):
            xs[L] = self.input_layer[L](xs[L])

        mask_dim = neighbor_mask.dim()
        dim_diff = xs[0].dim() - mask_dim
        neighbor_mask = neighbor_mask.to(xs[0]).reshape(rbf.shape[:mask_dim] + (1,) * dim_diff)
        # perform iterations over modular building blocks to get environment - dependent features
        fs = [torch.zeros_like(x) for x in xs]  # output features
        for i, module in enumerate(self.module):
            xs = self.order_change[i](xs)
            xs, ys = module(xs, rbf, sph, idx_i, idx_j, neighbor_mask=neighbor_mask)
            for L in range(self.order[i] + 1):
                fs[L] += ys[L]  # add contributions to output features
        fs[0] = self.out_activation(fs[0])

        atom_en = self.energy_output(fs)[0].squeeze(-1).squeeze(-1)
        atom_en = atom_en * atoms['atom_mask']

        energy = torch.sum(atom_en, dim=1, keepdim=True)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        if self.timing:
            print('simple energy time', time.time() - start)
        return atoms


class SimpleEnergyNetworkv2(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """

    def __init__(self,
                 order=2,
                 orbitals=None,
                 num_features=32,
                 # num_layers=5,
                 calculate_forces=False,
                 activation='swish',
                 compressed_extraction=False,
                 verbose=0,
                 timing=False,
                 clebsch_gordan=None,
                 pred_radial_coeffs=True,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces
        self.order = order
        self.compressed_extraction = compressed_extraction

        print('self order', self.order)
        if isinstance(self.order, list):
            self.order = self.order[-1]
        print('self order', self.order)
        # store hyperparameter values
        self.orbitals = orbitals
        self.num_features = num_features

        self.orbitals_max_order = get_max_order(self.orbitals)
        self.activation = activation

        self.orbital_spec, _, _ = combine_orbitals(self.orbitals, self.orbitals_max_order)
        self.dens_features = 0
        self.verbose = verbose
        self.timing = timing
        self.pred_radial_coeffs = pred_radial_coeffs

        self.dens_features = [0] * (self.orbitals_max_order + 1)
        seen_z = []
        for i in range(len(self.orbital_spec)):
            curr_feats = [0] * (self.orbitals_max_order + 1)
            z = self.orbital_spec[i][0][0]
            for orb in self.orbital_spec[i]:
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

        if clebsch_gordan is None:
            self.clebsch_gordan = ClebschGordanMatrix()
        else:
            self.clebsch_gordan = clebsch_gordan

        if self.pred_radial_coeffs:
            for L in range(len(self.dens_features)):
                name = "radial_scale_filters_{}".format(L)
                self.register_parameter(name, nn.Parameter(torch.ones(1, self.dens_features[L])))
                name = "radial_width_filters_{}".format(L)
                self.register_parameter(name, nn.Parameter(torch.ones(1, self.dens_features[L])))

        # self.radial_scale_filters = nn.ParameterList([nn.Parameter(torch.ones(1, df_num)) for df_num in self.dens_features])
        # self.radial_width_filters = nn.ParameterList([nn.Parameter(torch.ones(1, df_num)) for df_num in self.dens_features])
        self.input_layer = nn.ModuleList([nn.Linear(df_num, self.num_features) for df_num in self.dens_features])

        self.linear = SphericalLinear(
            self.order,
            self.num_features,
            self.order,
            self.num_features,
            clebsch_gordan,
            True,
        )
        self.energy_out = SphericalLinear(
            self.order,
            self.num_features,
            1,
            1,
            clebsch_gordan,
            True,
            zero_init=True,
        )

        if self.activation == 'swish':
            self.pre_activation = Swish(self.num_features)
            self.post_activation = Swish(self.num_features)
            self.coeff_activation = nn.ModuleList([Swish(df_num) for df_num in self.dens_features])
        elif self.activation == 'ssp':
            self.pre_activation = ShiftedSoftplus(self.num_features)
            self.post_activation = ShiftedSoftplus(self.num_features)
            self.coeff_activation = nn.ModuleList([Swish(df_num) for df_num in self.dens_features])
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        self.energy_output = nn.Linear(self.num_features, 1)

    def radial_scale_filters(self, L):
        return getattr(self, "radial_scale_filters_{}".format(L))

    def radial_width_filters(self, L):
        return getattr(self, "radial_width_filters_{}".format(L))

    def forward(self, atoms):
        start = time.time()
        # initialize atomic features to embeddings
        # print('sph coeffs', atoms['spherical_coeffs'][0][(8, 0)])
        sph_fs, scale_fs, width_fs = coeffs_dict_to_tensors(atoms, radial_coeffs=self.pred_radial_coeffs)
        # print('dij shape', dij.shape)
        # print('uij shape', uij.shape)
        # print('R shape', R.shape)
        xs = []
        # print('dens features', self.dens_features)
        if self.pred_radial_coeffs:
            for L in range(len(scale_fs)):
                # print('L', L)
                # print('scale fs L', scale_fs[L].shape)
                # print('self.radial_scale_filters[L]', self.radial_scale_filters(L).shape)
                scale_fs[L] = (scale_fs[L] * self.radial_scale_filters(L))
                width_fs[L] = (width_fs[L] * self.radial_width_filters(L))
                radial_comb = self.coeff_activation[L](scale_fs[L] * width_fs[L])
                radial_comb = radial_comb.sum(-2, keepdim=True)
                xs.append(sph_fs[L] * radial_comb)
        else:
            xs = sph_fs

        for L in range(len(xs)):
            xs[L] = self.input_layer[L](xs[L])

            # print('fs intermediate', fs)
        xs[0] = self.pre_activation(xs[0])
        xs = self.linear(xs)
        xs[0] = self.post_activation(xs[0])
        atom_en = self.energy_out(xs)[0].squeeze(-1).squeeze(-1)

        if self.verbose > 3:
            print('atom en shape', atom_en.shape)

        energy = torch.sum(atom_en, dim=1, keepdim=True)

        # print('energy shape', energy.shape)
        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        if self.timing:
            print('simple energy time', time.time() - start)
        return atoms


class RepresentationEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """
    def __init__(self,
                 orbitals=None,
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
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces

        # store hyperparameter values
        self.orbitals = orbitals
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
        if self.mixing_order is None:
            self.mixing_order = self.order

        print('self order', self.order)
        if not isinstance(self.order, list):
            self.order = [self.order] * self.num_modules
        print('self order', self.order)

        print('self mixing_order', self.mixing_order)
        if not isinstance(self.mixing_order, list):
            self.mixing_order = [self.mixing_order] * self.num_modules
        print('self mixing_order', self.mixing_order)

        if len(self.order) != self.num_modules:
            print('Order needs to be an integer or a list of integers with length equal to num_modules.' +
                  ' Taking last order element and using it for all modules.')
            self.order = [self.order[-1]] * self.num_modules

        if len(self.mixing_order) != self.num_modules:
            print('Mixing order needs to be an integer or a list of integers with length equal to num_modules.' +
                  ' Taking last order element and using it for all modules.')
            self.mixing_order = [self.mixing_order[-1]] * self.num_modules

        N = len(self.orbitals)
        idx_i = torch.arange(N, dtype=torch.int64).view(-1, 1).repeat(1, N).view(-1)
        idx_j = torch.arange(N, dtype=torch.int64).view(1, -1).repeat(N, 1).view(-1)
        # exclude self - interactions
        idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j]
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)

        self.orbitals_max_order = get_max_order(self.orbitals)
        self.orbital_spec, _, _ = combine_orbitals(self.orbitals, self.orbitals_max_order)

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
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'gaussian':
            self.radial_basis_functions = GaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        else:
            print("basis function type:",
                  self.basis_functions, "is not supported")

        modules = [ModularBlock(self.order[0], self.num_features, self.num_basis_functions,
                                self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                self.clebsch_gordan, True, self.mixing_order[0], 0, self.activation)]
        modules.extend([ModularBlock(self.order[i], self.num_features, self.num_basis_functions,
                                     self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                     self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                     self.clebsch_gordan, True, self.mixing_order[i], self.order[i - 1],
                                     self.activation) for i in range(1, self.num_modules)])
        self.module = nn.ModuleList(modules)

        self.order_change = [nn.Identity()]
        for i in range(1, self.num_modules):
            if self.order[i] != self.order[i - 1]:
                self.order_change.append(SphericalLinear(self.order[i - 1], self.num_features,
                                                         self.order[i], self.num_features,
                                                         clebsch_gordan=self.clebsch_gordan,
                                                         bias=False))
            else:
                self.order_change.append(nn.Identity())
        self.order_change = nn.ModuleList(self.order_change)

        if self.activation == 'swish':
            self.out_activation = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.out_activation = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        self.energy_output = SphericalLinear(self.order[-1], self.num_features, 0, 1, self.clebsch_gordan)

    def radial_scale_filters(self, L):
        return getattr(self, "radial_scale_filters_{}".format(L))

    def radial_width_filters(self, L):
        return getattr(self, "radial_width_filters_{}".format(L))

    def forward(self, atoms):
        start = time.time()
        # initialize atomic features to embeddings
        xs = atoms['sph_repr']
        dij = atoms['distances']
        sph = atoms['sph']
        # print('dij shape', dij.shape)
        # print('uij shape', uij.shape)
        # print('R shape', R.shape)
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2)  # unsqueeze for broadcasting

        # perform iterations over modular building blocks to get environment - dependent features
        fs = [torch.zeros_like(x) for x in xs]  # output features
        for i, module in enumerate(self.module):
            xs = self.order_change[i](xs)
            xs, ys = module(xs, rbf, sph, self.idx_i, self.idx_j)
            for L in range(self.order[i] + 1):
                fs[L] += ys[L]  # add contributions to output features
        fs[0] = self.out_activation(fs[0])

        atom_en = self.energy_output(fs)[0].squeeze(-1).squeeze(-1)

        energy = torch.sum(atom_en, dim=1, keepdim=True)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        if self.timing:
            print('simple energy time', time.time() - start)
        return atoms


class SimpleRepresentationEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """
    def __init__(self,
                 orbitals=None,
                 order=1,  # maximum order of spherical harmonics features
                 num_features=32,  # dimensionality of the feature space
                 # number of residual blocks applied to atomic features before interaction layer
                 activation='swish',
                 clebsch_gordan=None,  # instance of the clebsch gordan matrix
                 calculate_forces=False,
                 compressed_extraction=False,
                 verbose=0,
                 timing=False,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces

        # store hyperparameter values
        self.orbitals = orbitals
        self.order = order
        self.num_features = num_features
        self.activation = activation
        self.compressed_extraction = compressed_extraction
        self.verbose = verbose
        self.timing = timing

        print('self order', self.order)
        if not isinstance(self.order, list):
            self.order = [self.order]
        print('self order', self.order)

        N = len(self.orbitals)
        idx_i = torch.arange(N, dtype=torch.int64).view(-1, 1).repeat(1, N).view(-1)
        idx_j = torch.arange(N, dtype=torch.int64).view(1, -1).repeat(N, 1).view(-1)
        # exclude self - interactions
        idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j]
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)

        self.orbitals_max_order = get_max_order(self.orbitals)
        self.orbital_spec, _, _ = combine_orbitals(self.orbitals, self.orbitals_max_order)

        if clebsch_gordan is None:
            self.clebsch_gordan = ClebschGordanMatrix()
        else:
            self.clebsch_gordan = clebsch_gordan
        if self.activation == 'swish':
            self.out_activation = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.out_activation = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        self.residual_block = ResidualBlock(self.order[-1], self.num_features, clebsch_gordan=self.clebsch_gordan)

        self.energy_output = SphericalLinear(self.order[-1], self.num_features, 0, 1, self.clebsch_gordan)

    def radial_scale_filters(self, L):
        return getattr(self, "radial_scale_filters_{}".format(L))

    def radial_width_filters(self, L):
        return getattr(self, "radial_width_filters_{}".format(L))

    def forward(self, atoms):
        start = time.time()
        # initialize atomic features to embeddings
        xs = atoms['sph_repr']
        print('order', self.order)
        print('len sph_repr', len(atoms['sph_repr']))

        # perform iterations over modular building blocks to get environment - dependent features
        xs = self.residual_block(xs)
        xs[0] = self.out_activation(xs[0])

        atom_en = self.energy_output(xs)[0].squeeze(-1).squeeze(-1)

        energy = torch.sum(atom_en, dim=1, keepdim=True)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        if self.timing:
            print('simple energy time', time.time() - start)
        return atoms

class SphericalLinearEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """
    def __init__(self,
                 orbitals=None,
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
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)

        self.calculate_forces = calculate_forces

        # store hyperparameter values
        self.orbitals = orbitals
        self.order = order
        self.num_features = num_features
        self.num_modules = num_modules
        self.activation = activation
        self.compressed_extraction = compressed_extraction
        self.verbose = verbose
        self.timing = timing
        self.pred_radial_coeffs = pred_radial_coeffs

        self.orbital_basis = {}
        for orb in orbitals:
            z = orb[0][0]
            if z not in self.orbital_basis.keys():
                self.orbital_basis[z] = orb

        if not isinstance(self.order, list):
            self.order = [self.order] * self.num_modules

        self.orbitals_max_order = get_max_order(self.orbitals)
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

        self.order_change = [nn.Identity()]
        for i in range(1, self.num_modules):
            if self.order[i] != self.order[i - 1]:
                self.order_change.append(ResidualBlock(self.order[i - 1], self.num_features,
                                                         clebsch_gordan=self.clebsch_gordan,
                                                         activation=self.activation,
                                                         order_out=self.order[i]))
            else:
                self.order_change.append(nn.Identity())
        self.order_change = nn.ModuleList(self.order_change)

        modules = [ResidualBlock(self.order[0], self.num_features, clebsch_gordan=self.clebsch_gordan,
                                 activation=self.activation)]
        modules.extend([ResidualBlock(self.order[i], self.num_features,
                                     clebsch_gordan=self.clebsch_gordan,
                                     activation=self.activation) for i in range(1, self.num_modules)])
        self.module = nn.ModuleList(modules)

        self.energy_output = SphericalLinear(self.order[-1], self.num_features, 0, 1, self.clebsch_gordan)

    def radial_scale_filters(self, L):
        return getattr(self, "radial_scale_filters_{}".format(L))

    def radial_width_filters(self, L):
        return getattr(self, "radial_width_filters_{}".format(L))

    def forward(self, atoms):
        start = time.time()
        R = atoms['positions']
        N = R.shape[1]
        batch_size = R.shape[0]

        sph_fs, scale_fs, width_fs = coeffs_dict_to_tensors(atoms, radial_coeffs=self.pred_radial_coeffs)

        xs = []
        # print('dens features', self.dens_features)
        if self.pred_radial_coeffs:
            for L in range(len(scale_fs)):
                # print('L', L)
                # print('scale fs L', scale_fs[L].shape)
                # print('self.radial_scale_filters[L]', self.radial_scale_filters(L).shape)
                scale_fs[L] = scale_fs[L] * self.radial_scale_filters(L)
                width_fs[L] = width_fs[L] * self.radial_width_filters(L)
                radial_comb = self.coeff_activation[L](scale_fs[L] * width_fs[L])
                radial_comb = radial_comb.sum(-2, keepdim=True)
                xs.append(sph_fs[L] * radial_comb)
        else:
            xs = sph_fs

        # perform iterations over modular building blocks to get environment - dependent features
        fs = [1*x for x in xs]  # output features
        for i, module in enumerate(self.module):
            xs = self.order_change[i](xs)
            xs = module(xs)
            for L in range(self.order[i] + 1):
                fs[L] += xs[L]  # add contributions to output features

        fs[0] = self.out_activation(fs[0])

        atom_en = self.energy_output(xs)[0].squeeze(-1).squeeze(-1)

        energy = torch.sum(atom_en, dim=1, keepdim=True)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        if self.timing:
            print('simple energy time', time.time() - start)
        return atoms
