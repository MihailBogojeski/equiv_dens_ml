import torch
import torch.nn as nn
from equiv_dens.nn.modules.network_blocks import ModularBlock, ResidualBlock, layer_norm
from equiv_dens.nn.modules.radial_basis_functions import BernsteinRadialBasisFunctions,\
    GaussianRadialBasisFunctions, ExponentialBernsteinRadialBasisFunctions, ExponentialGaussianRadialBasisFunctions
from equiv_dens.nn.modules.activations import Swish, ShiftedSoftplus
from equiv_dens.utils.orbitals import combine_orbital_basis, get_invariant_features, get_max_order, coeffs_dict_to_tensors
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.modules.spherical_harmonic_layers import SphericalLinear
import time
import numpy as np


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
            # self.radial_scale_filters = nn.ParameterList([nn.Parameter(torch.ones(1, df_num)) for df_num in self.dens_features])
            # self.radial_width_filters = nn.ParameterList([nn.Parameter(torch.ones(1, df_num)) for df_num in self.dens_features])
        self.input_layer = nn.ModuleList([nn.Linear(df_num, self.num_features) for df_num in self.dens_features])

        modules = [ModularBlock(self.order[0], self.num_features, self.num_basis_functions,
                                self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                self.clebsch_gordan, True, self.mixing_order[0], 0, self.activation,
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
        start = time.time()
        R = atoms['positions']
        N = R.shape[1]
        batch_size = R.shape[0]
        idx_i = atoms['idx_i']
        idx_j = atoms['idx_j']
        neighbor_mask = 1
        # exclude self - interactions
        # initialize atomic features to embeddings
        sph_fs, scale_fs, width_fs = coeffs_dict_to_tensors(atoms, radial_coeffs=self.pred_radial_coeffs)
        for i in range(len(sph_fs)):
            sph_fs[i] = sph_fs[i].view(1, -1, *sph_fs[i].shape[2:])
            sph_fs[i] = sph_fs[i][:, atoms['atom_mask']]
            scale_fs[i] = scale_fs[i].view(1, -1, *scale_fs[i].shape[2:])
            scale_fs[i] = scale_fs[i][:, atoms['atom_mask']]
            width_fs[i] = width_fs[i].view(1, -1, *width_fs[i].shape[2:])
            width_fs[i] = width_fs[i][:, atoms['atom_mask']]
        dij = atoms['distances']
        sph = atoms['sph']
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
        # print('xs energy norm before:', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])

        for L in range(len(xs)):
            xs[L] = self.input_layer[L](xs[L])

        if self.normalize > 1:
            for L in range(len(xs)):
                xs[L] = layer_norm(xs[L], dims=(-2, -1)) 
        # print('xs energy norm after input layer:', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
        # perform iterations over modular building blocks to get environment - dependent features
        fs = [0 for _ in range(max(self.order_max, self.orbitals_max_order) + 1)]  # output features
        for i in range(len(xs)):
            fs[i] = fs[i] + xs[i]
        # print('fs norm start :', [float(torch.mean(fs[L]**2)) for L in range(len(fs))])
        # fs = [torch.zeros_like(x) for x in xs]  # output features

        for i, module in enumerate(self.module):
            xs = self.order_change[i](xs)
            # print('xs norm module ', i, ':', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
            xs, ys = module(xs, rbf, sph, idx_i, idx_j, neighbor_mask=neighbor_mask)
            for L in range(self.order[i] + 1):
                if not self.normalize or torch.mean(fs[L]**2) == 0 or torch.mean(ys[L]**2) == 0:
                    scale = 1
                else:
                    scale = np.sqrt(1/2)
                    # print('self normalize en', self.normalize)
                fs[L] = ys[L] * scale + fs[L] * scale
                # print('module', i, 'fs[0]', fs[0])
            # print('fs norm ', i, ':', [float(torch.mean(fs[L]**2)) for L in range(len(fs))])
        fs[0] = self.out_activation(fs[0])

        atom_en = self.energy_output(fs)[0].squeeze(-1).squeeze(-1)

        energy = torch.zeros(1, atoms['batch_atom_numbers'].shape[0]).to(atoms['positions'])
        energy = energy.scatter_add(1, atoms['atom_batch_idx'], atom_en)
        energy = torch.t(energy)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        # raise Exception('Random exception')

        if self.timing:
            print('simple energy time', time.time() - start)
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

        for L in range(len(xs)):
            xs[L] = self.input_layer[L](xs[L])
        # perform iterations over modular building blocks to get environment - dependent features
        # print('self order', self.order)
        fs = [0 for _ in range(max(self.order_max, self.orbitals_max_order) + 1)]  # output features
        for i in range(len(xs)):
            fs[i] = fs[i] + xs[i]
        # print('xs shapes', [x.shape for x in xs])
        for i, module in enumerate(self.module):
            # print('xs len before order change', len(xs))
            xs = self.order_change[i](xs)
            # print('xs len after order change', len(xs))
            # print('order ', i, ':', self.order[i] + 1)
            xs = module(xs)
            for L in range(self.order[i] + 1):
                if not self.normalize or torch.mean(fs[L]**2) == 0 or torch.mean(xs[L]**2) == 0:
                    scale = 1
                else:
                    scale = 1/np.sqrt(2)
                fs[L] = xs[L] * scale + fs[L] * scale

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
        start = time.time()
        fs = atoms['sph_repr']
        fs[0] = self.out_activation(fs[0])

        atom_en = self.energy_output(fs)[0].squeeze(-1).squeeze(-1)

        # print('atom_en shape', atom_en.shape)
        # print('positions shape', atoms['positions'].shape)
        energy = torch.zeros(1, atoms['batch_atom_numbers'].shape[0]).to(atoms['positions'])
        energy = energy.scatter_add(1, atoms['atom_batch_idx'], atom_en)
        energy = torch.t(energy)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        # raise Exception('Random exception')

        if self.timing:
            print('simple energy time', time.time() - start)
        return atoms


