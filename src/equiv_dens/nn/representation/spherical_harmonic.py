import torch
import torch.nn as nn
import torch.nn.functional as F
from equiv_dens.nn.modules.radial_basis_functions import BernsteinRadialBasisFunctions, GaussianRadialBasisFunctions,\
    ExponentialBernsteinRadialBasisFunctions, ExponentialGaussianRadialBasisFunctions
from equiv_dens.nn.modules.embeddings import SphericalEmbedding
from equiv_dens.nn.modules.network_blocks import ModularBlock
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
from equiv_dens.utils.base import calculate_distances_and_directions
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
import time


class EquivariantSphericalHarmonics(nn.Module):
    """
    Neural network for computing Hamiltonian / Overlap matrices in a rotationally equivariant way
    """

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
                 cutoff=15.0,  # cutoff distance (default is 15 Bohr)
                 # type of activation function used (swish / ssp)
                 activation='swish',
                 clebsch_gordan=None,  # instance of the clebsch gordan matrix
                 Zmax=87,
                 timing=False,
                 verbose=0,
                 ):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian / core_hamiltonian / overlap_matrix / energy / forces?)
        self.create_graph = True  # can be set to False if the NN is only used for inference

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
        self.cutoff = cutoff
        self.activation = activation
        self.Zmax = Zmax
        self.timing = timing
        self.verbose = verbose

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
        for i in range(len(self.orbitals)):
            Zl.append(self.orbitals[i][0][0])
            for z, _, l in self.orbitals[i]:
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

        # declare modules and parameters
        if clebsch_gordan is None:
            self.clebsch_gordan = ClebschGordanMatrix()
        else:
            self.clebsch_gordan = clebsch_gordan
        self.embedding = SphericalEmbedding(
            self.order, self.num_features, self.Zmax)
        if basis_functions == 'exp-gaussian':
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif basis_functions == 'exp-bernstein':
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif basis_functions == 'gaussian':
            self.radial_basis_functions = GaussianRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        elif basis_functions == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(
                self.num_basis_functions, self.cutoff)
        else:
            print("basis function type:",
                  basis_functions, "is not supported")
        self.module = nn.ModuleList([ModularBlock(self.order, self.num_features, self.num_basis_functions,
                                                  self.num_residual_pre_x, self.num_residual_post_x, self.num_residual_pre_vi,
                                                  self.num_residual_pre_vj, self.num_residual_post_v, self.num_residual_output,
                                                  self.clebsch_gordan, True, self.activation) for i in range(self.num_modules)])

    def forward(self, atoms):
        """
        Computes the spherical harmonic atomistic features for the molecule

        inputs:
            R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
        outputs:
            C: Spherical harmonics coefficients
        """
        start = time.time()
        R = atoms['positions']
        # compute radial basis functions and spherical harmonics
        # print('idx_i', self.idx_i)
        dij, uij = calculate_distances_and_directions(R, self.idx_i, self.idx_j)
        # print('dij shape', dij.shape)
        # print('uij shape', uij.shape)
        # print('R shape', R.shape)
        # print('dij', dij)
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2)  # unsqueeze for broadcasting
        # print('rbf shape', rbf.shape)
        # print('rbf', rbf)
        rbf = rbf**10
        # print('rbf power', rbf)
        rbf = rbf / torch.sum(rbf, dim=-1, keepdim=True)
        # print('rbf power scale', rbf)
        # print('rbf shape', rbf.shape)
        # print('rbf sum', torch.sum(rbf, dim=-1))
        # print('rbf softmax', F.softmax(rbf, dim=-1))
        sph = spherical_harmonics(self.order, uij)
        # print('sph', sph)
        atoms['distances'] = dij
        atoms['directions'] = uij
        # print('sph shape', sph[0].shape)
        for L in range(self.order + 1):
            sph[L].unsqueeze_(-1)  # unsqueeze for broadcasting
        # print('sph shape', sph[0].shape)
        atoms['sph'] = sph
        # initialize atomic features to embeddings
        # repeat Z along batch dimension
        xs = self.embedding(atoms['atom_numbers'].repeat(R.size(0), 1))

        # perform iterations over modular building blocks to get environment - dependent features
        fs = [torch.zeros_like(x) for x in xs]  # output features
        for module in self.module:
            xs, ys = module(xs, rbf, sph, self.idx_i, self.idx_j)
            for L in range(self.order + 1):
                fs[L] += ys[L]  # add contributions to output features

        atoms['sph_repr'] = fs
        if self.timing:
            print('sph repr time', time.time() - start)
        return atoms
