import torch
import torch.nn as nn
from equiv_dens.nn.modules.network_blocks import ModularBlock
from equiv_dens.nn.modules.radial_basis_functions import BernsteinRadialBasisFunctions,\
    GaussianRadialBasisFunctions, ExponentialBernsteinRadialBasisFunctions, ExponentialGaussianRadialBasisFunctions
from equiv_dens.nn.modules.activations import Swish, ShiftedSoftplus
from equiv_dens.utils.orbitals import combine_orbitals, get_invariant_features, get_max_order


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
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)
        self.create_graph = True  # can be set to False if the NN is only used for inference

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

        N = len(self.orbitals)
        idx_i = torch.arange(N, dtype=torch.int64).view(-1, 1).repeat(1, N).view(-1)
        idx_j = torch.arange(N, dtype=torch.int64).view(1, -1).repeat(N, 1).view(-1)
        # exclude self - interactions
        idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j]
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)

        self.order_max = get_max_order()
        self.orbital_spec, _ = combine_orbitals(self.orbitals, self.order_max)
        self.dens_features = 0
        for i in range(len(self.orbital_spec)):
            curr_feats = 0
            for orb in self.orbital_spec[i]:
                curr_feats += orb[1]
            if curr_feats > self.dens_features:
                self.dens_features = curr_feats
        self.dens_features *= 3

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
                                                  None, False, self.activation) for i in range(self.num_modules)])

        if self.activation == 'swish':
            self.out_activation = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.out_activation = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        self.energy_output = nn.Linear(self.num_features, 1)

    def forward(self, atoms, coeffs):
        # initialize atomic features to embeddings
        xs = get_invariant_features(coeffs, permutational_invariance=False, keep_dims=True)
        dij = atoms['distances']
        sph = atoms['sph']
        # print('dij shape', dij.shape)
        # print('uij shape', uij.shape)
        # print('R shape', R.shape)
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2)  # unsqueeze for broadcasting
        # print('rbf shape', rbf.shape)
        # print('sph shape', sph[0].shape)
        for L in range(1):
            sph[L].unsqueeze_(-1)  # unsqueeze for broadcasting

        if self.num_features != self.dens_features:
            xs = self.input_layer(xs)
        xs = [xs]

        # perform iterations over modular building blocks to get environment - dependent features
        fs = [torch.zeros_like(x) for x in xs]  # output features
        for module in self.module:
            xs, ys = module(xs, rbf, sph, self.idx_i, self.idx_j)
            fs[0] += ys[0]  # add contributions to output features

        atom_en = self.energy_output(self.out_activation(fs[0]))

        energy = torch.sum(atom_en, dim=1)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.create_graph)[0]
            atoms['forces'] = forces

        return atoms


class SimpleEnergyNetwork(nn.Module):
    """
    Neural network for computing the energy of a molecule based on the density coefficients
    """

    def __init__(self,
                 orbitals=None,
                 num_features=32,
                 calculate_forces=False,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)
        self.create_graph = True  # can be set to False if the NN is only used for inference

        self.calculate_forces = calculate_forces

        # store hyperparameter values
        self.orbitals = orbitals
        self.num_features = num_features

        self.orbital_spec, _ = combine_orbitals()
        self.dens_features = 0
        for i in range(len(self.orbital_spec)):
            curr_feats = 0
            for orb in self.orbital_spec[i]:
                curr_feats += orb[1]
            if curr_feats > self.dens_features:
                self.dens_features = curr_feats
        self.dens_features *= 3

        if self.num_features != self.dens_features:
            self.input_layer = nn.Linear(self.dens_features, self.num_features)

        if self.activation == 'swish':
            self.out_activation = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.out_activation = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        self.energy_output = nn.Linear(self.num_features, 1)

    def forward(self, atoms, coeffs):
        # initialize atomic features to embeddings
        fs = get_invariant_features(coeffs, permutational_invariance=False, keep_dims=True)

        atom_en = self.energy_output(self.out_activation(fs))

        energy = torch.sum(atom_en, dim=1)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.create_graph)[0]
            atoms['forces'] = forces

        return atoms
