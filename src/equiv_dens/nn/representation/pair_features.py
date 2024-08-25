import torch
import torch.nn as nn
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordan
from equiv_dens.nn.modules.embeddings import SphericalEmbedding
from equiv_dens.nn.modules.radial_basis_functions import *
from equiv_dens.nn.modules.network_blocks import *

class AtomicPairFeatures(nn.Module):
    """
    Computes pair features (self-interaction and pair-interaction) from atomic features.
    """

    def __init__(self,
            orbitals             = None, #orbitals of atoms, used to get number of atoms
            order                = 1,  #maximum order of spherical harmonics features
            num_features         = 32, #dimensionality of the feature space
            num_basis_functions  = 32, #number of basis functions for featurizing distances
            num_residual_pc      = 1, #number of residual blocks applied to output atomic features for constructing pair features (central atoms)
            num_residual_pn      = 1, #number of residual blocks applied to output atomic features for constructing pair features (neighboring atoms)
            num_residual_ii      = 1, #number of residual blocks applied to output atomic features for predicting irreps of diagonal blocks (shared)
            num_residual_ij      = 1, #number of residual blocks applied to pair features for predicting irreps of off-diagonal blocks (shared)
            basis_functions      = 'exp-bernstein', #type of radial basis functions (exp-gaussian/exp-bernstein/gaussian/bernstein)
            cutoff               = 15.0, #cutoff distance (default is 15 Bohr)
            activation           = 'swish', #type of activation function used (swish/ssp)
            output_property_name = 'ao_matrix', # key of atoms dict, e.g. atoms['hamiltonian_matrix']
            load_from            = None, #if this is given the network is loaded from the specified .pth file and all other arguments are ignored
            #Zmax                 = 87 #maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default 
    ):
        super(AtomicPairFeatures, self).__init__()
        
        self.create_graph = True  #can be set to False if the NN is only used for inference

        #store hyperparameter values
        self.orbitals = orbitals
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_residual_pc      = num_residual_pc
        self.num_residual_pn      = num_residual_pn
        self.num_residual_ii      = num_residual_ii
        self.num_residual_ij      = num_residual_ij
        self.basis_functions = basis_functions
        self.cutoff = cutoff
        self.activation = activation
        self.output_property_name = output_property_name
        #self.Zmax = Zmax

        #generate index lists for computing pairwise distances
        N = len(self.orbitals)
        idx_i = torch.arange(N, dtype=torch.int64).view(-1,1).repeat(1,N).view(-1)
        idx_j = torch.arange(N, dtype=torch.int64).view(1,-1).repeat(N,1).view(-1)
        idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j] #exclude self-interactions
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)

        #generate index lists for asymmetrizing pair interactions
        idx_pi = []
        idx_pj = []
        for ni, ij1 in enumerate(zip(idx_i, idx_j)):
            i1 = ij1[0].item()
            j1 = ij1[1].item()
            for nj, ij2 in enumerate(zip(idx_i, idx_j)):
                i2 = ij2[0].item()
                j2 = ij2[1].item()
                if ((i1 == i2) and (not j1 == j2)):
                    idx_pi.append(ni)
                    idx_pj.append(nj)
        self.register_buffer('idx_pi', torch.tensor(idx_pi, dtype=torch.int64))
        self.register_buffer('idx_pj', torch.tensor(idx_pj, dtype=torch.int64))

        #declare modules and parameters
        self.clebsch_gordan = ClebschGordan()
        if self.basis_functions == 'exp-gaussian':
            self.radial_basis_functions = ExponentialGaussianRadialBasisFunctions(self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'exp-bernstein':
            self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'gaussian':
            self.radial_basis_functions = GaussianRadialBasisFunctions(self.num_basis_functions, self.cutoff)
        elif self.basis_functions == 'bernstein':
            self.radial_basis_functions = BernsteinRadialBasisFunctions(self.num_basis_functions, self.cutoff)
        else:
            print("basis function type:", self.basis_functions, "is not supported")

        self.mix_ij = PairMixing(self.order, self.order, self.order, self.num_basis_functions, self.num_features, self.clebsch_gordan)
        self.radial_ii = nn.ModuleList([nn.Linear(self.num_basis_functions, self.num_features, bias=False)
            for L in range(self.order+1)])
        self.radial_ij = nn.ModuleList([nn.Linear(self.num_basis_functions, self.num_features, bias=False)
            for L in range(self.order+1)])
        self.residual_pc = ResidualStack(self.num_residual_pc, self.order, self.num_features, self.clebsch_gordan, True, self.activation)
        self.residual_pn = ResidualStack(self.num_residual_pn, self.order, self.num_features, self.clebsch_gordan, True, self.activation)
        self.residual_ii = ResidualStack(self.num_residual_ii, self.order, self.num_features, self.clebsch_gordan, True, self.activation)
        self.residual_ij = ResidualStack(self.num_residual_ij, self.order, self.num_features, self.clebsch_gordan, True, self.activation)


    def forward(self, atoms):

        dij = atoms['distances']
        fs = atoms['sph_repr']  # equivariant spherical harmonics representation
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2) #unsqueeze for broadcasting
        
        fpc = self.residual_pc(fs) #central pair features
        fpn = self.residual_pn(fs) #neighbor pair features

        #compute pair features for self-interactions
        fii = [1*x for x in fpc]
        for L in range(self.order+1): #add influence of neighbouring atoms to pairs
            idx_j  = self.idx_j.view(*(1,)*len(fpn[L].shape[:-3]),-1,1,1).repeat(*fpn[L].shape[:-3], 1, *fpn[L].shape[-2:])
            fpn_j  = self.radial_ii[L](rbf)*torch.gather(fpn[L], 1, idx_j)
            fii[L] = fii[L].index_add(1, self.idx_i, fpn_j)

        #compute output features (irreducible representations) for self-interactions
        fii = self.residual_ii(fii)

        #gather atomic pairs
        fi = []
        fj = []
        for L in range(self.order+1):
            i = self.idx_i.view(*(1,)*len(fpc[L].shape[:-3]),-1,1,1).repeat(*fpc[L].shape[:-3], 1, *fpc[L].shape[-2:])
            j = self.idx_j.view(*(1,)*len(fpc[L].shape[:-3]),-1,1,1).repeat(*fpc[L].shape[:-3], 1, *fpc[L].shape[-2:])
            fi.append(torch.gather(fpc[L], 1, i))
            fj.append(torch.gather(fpc[L], 1, j))

        #compute pair features for ordinary interactions
        fij = self.mix_ij(fi, fj, rbf) #mix pairs
        for L in range(self.order+1): #add influence of neighbouring atoms to pairs
            idx_j  = self.idx_j.view(*(1,)*len(fpn[L].shape[:-3]),-1,1,1).repeat(*fpn[L].shape[:-3], 1, *fpn[L].shape[-2:])
            fpn_j  = self.radial_ij[L](rbf)*torch.gather(fpn[L], 1, idx_j)
            idx_pj = self.idx_pj.view(*(1,)*len(fpn_j.shape[:-3]),-1,1,1).repeat(*fpn_j.shape[:-3], 1, *fpn_j.shape[-2:])
            fij[L] = fij[L].index_add(1, self.idx_pi, torch.gather(fpn_j, 1, idx_pj))

        #compute output features (irreducible representations) for pair-interactions
        fij = self.residual_ij(fij)

        atoms['pair_features'] = fii, fij

        return atoms