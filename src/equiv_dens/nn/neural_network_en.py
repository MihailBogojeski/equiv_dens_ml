import torch
import torch.nn as nn
from .modules.bernstein_radial_basis_functions import BernsteinRadialBasisFunctions
from .modules.gaussian_radial_basis_functions import GaussianRadialBasisFunctions
from .modules.exponential_bernstein_radial_basis_functions import ExponentialBernsteinRadialBasisFunctions
from .modules.exponential_gaussian_radial_basis_functions import ExponentialGaussianRadialBasisFunctions
from .modules.modular_block import ModularBlock
from .modules.swish import Swish
from .modules.shifted_softplus import ShiftedSoftplus
from equiv_dens.gradient_learning.utils import get_invariant_features
from .spherical_harmonics import spherical_harmonics

"""
Neural network for computing Hamiltonian / Overlap matrices in a rotationally equivariant way
"""


class EnergyNetwork(nn.Module):
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
                 load_from=None,  # if this is given the network is loaded from the specified .pth file and all other arguments are ignored
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)
        self.create_graph = True  # can be set to False if the NN is only used for inference

        # load state from a file (if load_from is given) and overwrite hyperparameters
        if load_from is not None:
            saved_state = torch.load(load_from, map_location='cpu')
            orbitals = saved_state['orbitals']
            num_features = saved_state['num_features']
            num_basis_functions = saved_state['num_basis_functions']
            num_modules = saved_state['num_modules']
            num_residual_pre_x = saved_state['num_residual_pre_x']
            num_residual_post_x = saved_state['num_residual_post_x']
            num_residual_pre_vi = saved_state['num_residual_pre_vi']
            num_residual_pre_vj = saved_state['num_residual_pre_vj']
            num_residual_post_v = saved_state['num_residual_post_v']
            num_residual_output = saved_state['num_residual_output']
            num_radial_components = saved_state['num_radial_components']
            basis_functions = saved_state['basis_functions']
            cutoff = saved_state['cutoff']
            activation = saved_state['activation']

        self.calculate_forces = False

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

        self.order_max = 0
        self.Nfeats = 0
        for i in range(len(self.orbitals)):
            for z, _, l in self.orbitals[i]:
                if l > self.order_max:
                    self.order_max = l

        self.orbital_spec, _ = self.combine_orbitals()
        self.dens_features = 0
        for i in range(len(self.orbital_spec)):
            curr_feats = 0
            for orb in self.orbital_spec[i]:
                curr_feats += orb[1]
            if curr_feats > self.dens_features:
                self.dens_features = curr_feats
        self.dens_features *= 3
        # for calculating nucleus - nucleus repulsion

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

        # determine minimum number of output features based on orbitals
        # and generate dictionaries (irreps_ii / irreps_ij) that store indices
        # for collecting the correct irreproducible representations from features
        # diagonal blocks
        # keeps track of how many irreps of each order there are already
        # initialize parameters
        if load_from is not None:
            self.load_state_dict(saved_state['state_dict'], strict=False)

    """
    saves the model to a file given by PATH (including all values of the hyperparameters)
    (this file can be passed to the load_from value in the initialization in order to construct
    the model from the saved state)
    """

    def save(self, PATH):
        torch.save({
            'state_dict': self.state_dict(),
            'num_features': self.num_features,
            'num_basis_functions': self.num_basis_functions,
            'num_radial_components': self.num_radial_components,
            'num_modules': self.num_modules,
            'num_residual_pre_x': self.num_residual_pre_x,
            'num_residual_post_x': self.num_residual_post_x,
            'num_residual_pre_vi': self.num_residual_pre_vi,
            'num_residual_pre_vj': self.num_residual_pre_vj,
            'num_residual_post_v': self.num_residual_post_v,
            'num_residual_output': self.num_residual_output,
            'basis_functions': self.basis_functions,
            'cutoff': self.cutoff,
            'activation': self.activation,
        }, PATH)

    """
    Just for easily printing out the total number of parameters
    """

    def get_number_of_parameters(self):
        num = 0
        for param in self.parameters():
            if param.requires_grad:
                num += param.numel()
        return num

    def calculate_distances_and_directions(self, R, idx_i, idx_j):
        # print('R shape', R.shape)
        Ri = torch.gather(R, -2, idx_i.view(*(1,) * len(R.shape[: -2]), -1, 1).repeat(*R.shape[: -2], 1, R.size(-1)))
        Rj = torch.gather(R, -2, idx_j.view(*(1,) * len(R.shape[: -2]), -1, 1).repeat(*R.shape[: -2], 1, R.size(-1)))
        # print('Ri shape', Ri.shape)
        # print('Rj shape', Rj.shape)
        rij = Rj - Ri  # displacement vectors
        dij = torch.norm(rij, dim=-1, keepdim=True)  # distances
        uij = rij / dij  # unit displacement vectors
        return dij, uij

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

    """
    Computes the spherical harmonics coefficients for the density

    inputs:
        R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
    outputs:
        C: Spherical harmonics coefficients
    """

    def forward(self, R, coeffs):
        # initialize atomic features to embeddings
        xs = get_invariant_features(coeffs, permutational_invariance=False, keep_dims=True)
        dij, uij = self.calculate_distances_and_directions(R, self.idx_i, self.idx_j)
        # print('dij shape', dij.shape)
        # print('uij shape', uij.shape)
        # print('R shape', R.shape)
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2)  # unsqueeze for broadcasting
        # print('rbf shape', rbf.shape)
        sph = spherical_harmonics(0,  uij)
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

        results = {'energy': energy}
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), R, create_graph=self.create_graph)[0]
            results['forces'] = forces
            R.requires_grad = False

        return results
