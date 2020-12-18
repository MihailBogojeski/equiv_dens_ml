import torch
import torch.nn.functional as F
import torch.nn as nn
from .modules.clebsch_gordan import ClebschGordanMatrix
from .modules.bernstein_radial_basis_functions import BernsteinRadialBasisFunctions
from .modules.gaussian_radial_basis_functions import GaussianRadialBasisFunctions
from .modules.exponential_bernstein_radial_basis_functions import ExponentialBernsteinRadialBasisFunctions
from .modules.exponential_gaussian_radial_basis_functions import ExponentialGaussianRadialBasisFunctions
from .modules.spherical_linear import SphericalLinear
from .modules.spherical_embedding import SphericalEmbedding
from .modules.modular_block import ModularBlock
from .spherical_harmonics import spherical_harmonics

"""
Neural network for computing Hamiltonian / Overlap matrices in a rotationally equivariant way
"""


class DensityNetwork(nn.Module):
    def __init__(self,
                 orbitals=None,  # orbitals of atoms, defines layout and shape of output matrix
                 order=1,  # maximum order of spherical harmonics features
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
                 positive_coeffs=False,
                 energy_offset=False,
                 cutoff=15.0,  # cutoff distance (default is 15 Bohr)
                 # type of activation function used (swish / ssp)
                 activation='swish',
                 load_from=None,  # if this is given the network is loaded from the specified .pth file and all other arguments are ignored
                 Zmax=87):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        self.calculate_forces = False

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)
        self.create_graph = True  # can be set to False if the NN is only used for inference

        # load state from a file (if load_from is given) and overwrite hyperparameters
        if load_from is not None:
            saved_state = torch.load(load_from, map_location='cpu')
            print('saved state', saved_state.keys())
            orbitals = saved_state['orbitals']
            order = saved_state['order']
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
            positive_coeffs = saved_state['positive_coeffs']
            basis_functions = saved_state['basis_functions']
            cutoff = saved_state['cutoff']
            activation = saved_state['activation']
            Zmax = saved_state['Zmax']
            energy_offset = saved_state['energy_offset']

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
        self.positive_coeffs = positive_coeffs
        self.cutoff = cutoff
        self.activation = activation
        self.energy_offset = energy_offset
        self.Zmax = Zmax

        if energy_offset:
            self.en_offset = nn.Parameter(torch.zeros((1,)))
        else:
            self.register_buffer('en_offset', torch.zeros((1,)))

        N = len(self.orbitals)
        idx_i = torch.arange(N, dtype=torch.int64).view(-1, 1).repeat(1, N).view(-1)
        idx_j = torch.arange(N, dtype=torch.int64).view(1, -1).repeat(N, 1).view(-1)
        # exclude self - interactions
        idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j]
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)

        # extract nuclear charges from orbitals, determine maximum order, and
        # build the occupation mask (for extracting occupied orbitals in energy prediction)
        Zl = []
        self.order_max = 0
        self.Norb = 0
        for i in range(len(self.orbitals)):
            Zl.append(self.orbitals[i][0][0])
            for z, _, l in self.orbitals[i]:
                self.Norb += 2 * l + 1
                assert z == Zl[i]  # check that Z is the same for all orbitals
                if l > self.order_max:
                    self.order_max = l
        Z = torch.tensor(Zl, dtype=torch.int64).unsqueeze(0)
        # for calculating nucleus - nucleus repulsion
        self.register_buffer('Z', Z)  # for gathering embeddings

        # error checking
        if self.order < self.order_max:
            print("An orbital with L={} was found, but the neural network was initialized with L={}".format(
                self.order_max, self.order))
            print("The neural network MUST have at least the same order as all orbitals!")
            quit()
        if self.order < 2 * self.order_max:
            print("An orbital with L={} was found, but the neural network was initialized with L={}".format(
                self.order_max, self.order))
            print("The neural network SHOULD have at least twice the order of the maximum order orbital for good results!")
            # don't quit here, maybe someone wants to do it like this

        self.orbital_spec, self.radial_count = self.combine_orbitals()
        print('orbital_spec', self.orbital_spec)

        # declare modules and parameters
        self.clebsch_gordan = ClebschGordanMatrix()
        self.embedding = SphericalEmbedding(
            self.order, self.num_features, self.Zmax)
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
        self.module = nn.ModuleList([ModularBlock(self.order, self.num_features, self.num_basis_functions,
                                                  self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                                  self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                                  self.clebsch_gordan, True, self.activation) for i in range(self.num_modules)])
        self.angular_fn = SphericalLinear(self.order, 1, self.order, self.num_features, self.clebsch_gordan, mix_orders=False)

        self.L_counts, self.r_max, self.L_dict = self.compute_orbital_features_num()
        print('L_counts', self.L_counts)
        print('L_dict', self.L_dict)
        print('max lcounts', max(self.L_counts))
        self.spherical_output = SphericalLinear(self.order, self.num_features, self.order_max + 1, max(self.L_counts), self.clebsch_gordan)
        self.radial_width = nn.ModuleList([nn.Linear(self.num_features, self.L_counts[L] * self.r_max[L])
                                           for L in range(self.order_max + 2)])
        self.radial_scale = nn.ModuleList([nn.Linear(self.num_features, self.L_counts[L] * self.r_max[L])
                                           for L in range(self.order_max + 2)])

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
            'orbitals': self.orbitals,
            'order': self.order,
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
            'positive_coeffs': self.positive_coeffs,
            'energy_offset': self.energy_offset,
            'cutoff': self.cutoff,
            'activation': self.activation,
            'Zmax': self.Zmax
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

    """
    Given the Cartesian coordinates and index lists, calculates pairwise distances and unit displacement
    vectors. Each distance / vector is specified by a pair of atom indices i and j (i != j). The total
    number of interactions is num_interactions=num_atoms * (num_atoms - 1) when all pairwise distances are
    calculated.

    inputs:
        R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
        idx_i: indices of atoms i of shape [num_interactions] for collecting Cartesian coordinates
        idx_j: indices of atoms j of shape [num_interactions] for collecting Cartesian coordinates
    outputs:
        dij: pairwise distances of shape [batch_size, num_interactions, 1]
        uij: unit displacement vectors of shape [batch_size, num_interactions, 3]
    """

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
        # orbital_counts = torch.zeros(size=(self.order + 1, ))
        L_counts = [0 for L in range(2 * self.order + 1)]
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
    Computes the spherical harmonics coefficients for the density

    inputs:
        R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
    outputs:
        C: Spherical harmonics coefficients
    """

    def forward(self, R):
        if self.calculate_forces:
            R.requires_grad = True
        # compute radial basis functions and spherical harmonics
        # print('idx_i', self.idx_i)
        dij, uij = self.calculate_distances_and_directions(R, self.idx_i, self.idx_j)
        # print('dij shape', dij.shape)
        # print('uij shape', uij.shape)
        # print('R shape', R.shape)
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2)  # unsqueeze for broadcasting
        sph = spherical_harmonics(self.order, uij)
        # print('sph shape', sph[0].shape)
        for L in range(self.order + 1):
            sph[L].unsqueeze_(-1)  # unsqueeze for broadcasting
        # print('sph shape', sph[0].shape)

        # initialize atomic features to embeddings
        # repeat Z along batch dimension
        xs = self.embedding(self.Z.repeat(R.size(0), 1))

        # perform iterations over modular building blocks to get environment - dependent features
        fs = [torch.zeros_like(x) for x in xs]  # output features
        for module in self.module:
            xs, ys = module(xs, rbf, sph, self.idx_i, self.idx_j)
            for L in range(self.order + 1):
                fs[L] += ys[L]  # add contributions to output features

        out_sph = self.spherical_output(fs)
        if self.positive_coeffs:
            out_sph[0] = F.softplus(out_sph[0])
        out_width = []
        out_scale = []
        for L in range(len(self.radial_width)):
            out_width.append(F.tanh(self.radial_width[L](fs[0])))
            out_width[L] = out_width[L].view(*out_width[L].shape[:-2], self.r_max[L], self.L_counts[L])
            if self.positive_coeffs:
                out_scale.append(F.softplus(self.radial_scale[L](fs[0])))
            else:
                out_scale.append(self.radial_scale[L](fs[0]))
            out_scale[L] = out_scale[L].view(*out_scale[L].shape[:-2], self.r_max[L], self.L_counts[L])
        results = {}
        results['spherical_coeffs'], results['radial_width'], results['radial_scale'] =\
            self.extract_coefficients(out_sph, out_width, out_scale)

        return results
