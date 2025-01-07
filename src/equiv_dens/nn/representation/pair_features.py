import torch
import torch.nn as nn
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.modules.embeddings import SphericalEmbedding
from equiv_dens.nn.modules.radial_basis_functions import *
from equiv_dens.nn.modules.network_blocks import *
from equiv_dens.utils.orbitals import get_max_order


class PairFeatures(nn.Module):
    """
    Computes pair features (self-interaction and pair-interaction) from atomic features.
    """

    def __init__(self,
            orbital_basis        = None, #orbitals of atoms, used to get number of atoms
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
            load_from            = None, #if this is given the network is loaded from the specified .pth file and all other arguments are ignored
            #Zmax                 = 87 #maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default 
    ):
        super().__init__()

        self.create_graph = True  #can be set to False if the NN is only used for inference

        #store hyperparameter values
        max_order_per_atom = get_max_order(orbital_basis, per_atom=True)

        order_max = max(max_order_per_atom.values())

        self.orbital_basis = orbital_basis
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
        #self.Zmax = Zmax

        #error checking
        if self.order < order_max:
            print("An orbital with L={} was found, but the neural network was initialized with L={}".format(order_max, self.order))
            print("The neural network MUST have at least the same order as all orbitals!")
            quit()
        if self.order < 2*order_max:
            print("An orbital with L={} was found, but the neural network was initialized with L={}".format(order_max, self.order))
            print("The neural network SHOULD have at least twice the order of the maximum order orbital for good results!")
            #don't quit here, maybe someone wants to do it like this

        #declare modules and parameters
        self.clebsch_gordan = ClebschGordanMatrix()
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

        R = atoms['positions']
        dij = atoms['distances']
        fs = atoms['sph_repr']  # equivariant spherical harmonics atomic representation
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2) #unsqueeze for broadcasting

        fpc = self.residual_pc(fs) #central pair features
        fpn = self.residual_pn(fs) #neighbor pair features


        # #compute pair features for self-interactions
        fii = [1*x for x in fpc]
        for L in range(self.order+1): #add influence of neighbouring atoms to pairs
            idx_j  = atoms['idx_j'].view(*(1,)*len(fpn[L].shape[:-3]),-1,1,1).repeat(*fpn[L].shape[:-3], 1, *fpn[L].shape[-2:])
            fpn_j  = self.radial_ii[L](rbf)*torch.gather(fpn[L], 1, idx_j)
            fii[L] = fii[L].index_add(1, atoms['idx_i'], fpn_j)

        #gather atomic pairs
        fi = []
        fj = []
        for L in range(self.order+1):
            i = atoms['idx_i'].view(*(1,)*len(fpc[L].shape[:-3]),-1,1,1).repeat(*fpc[L].shape[:-3], 1, *fpc[L].shape[-2:])
            j = atoms['idx_j'].view(*(1,)*len(fpc[L].shape[:-3]),-1,1,1).repeat(*fpc[L].shape[:-3], 1, *fpc[L].shape[-2:])
            fi.append(torch.gather(fpc[L], 1, i))
            fj.append(torch.gather(fpc[L], 1, j))

        #compute output features (irreducible representations) for self-interactions
        fii = self.residual_ii(fii)

        #generate index lists for asymmetrizing pair interactions
        idx_pi = []
        idx_pj = []
        for ni, ij1 in enumerate(zip(atoms['idx_i'], atoms['idx_j'])):
            i1 = ij1[0].item()
            j1 = ij1[1].item()
            for nj, ij2 in enumerate(zip(atoms['idx_i'], atoms['idx_j'])):
                i2 = ij2[0].item()
                j2 = ij2[1].item()
                if ((i1 == i2) and (not j1 == j2)):
                    idx_pi.append(ni)
                    idx_pj.append(nj)
        idx_pi = torch.tensor(idx_pi, dtype=torch.int64, device=R.device)
        idx_pj = torch.tensor(idx_pj, dtype=torch.int64, device=R.device)

        #compute pair features for ordinary interactions
        fij = self.mix_ij(fi, fj, rbf) #mix pairs
        for L in range(self.order+1): #add influence of neighbouring atoms to pairs
            idx_j  = atoms['idx_j'].view(*(1,)*len(fpn[L].shape[:-3]),-1,1,1).repeat(*fpn[L].shape[:-3], 1, *fpn[L].shape[-2:])
            fpn_j  = self.radial_ij[L](rbf)*torch.gather(fpn[L], 1, idx_j)
            idx_pj_L = idx_pj.view(*(1,)*len(fpn_j.shape[:-3]),-1,1,1).repeat(*fpn_j.shape[:-3], 1, *fpn_j.shape[-2:])
            fij[L] = fij[L].index_add(1, idx_pi, torch.gather(fpn_j, 1, idx_pj_L))

        #compute output features (irreducible representations) for pair-interactions
        fij = self.residual_ij(fij)

        atoms['pair_features'] = fii, fij

        return atoms


class PairFeaturesV2(nn.Module):
    """
    Computes pair features (self-interaction and pair-interaction) from atomic features.
    """

    def __init__(self,
            orbital_basis        = None, #orbitals of atoms, used to get number of atoms
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
            num_hidden_att_mlp   = 128, #hidden size of the MLP used for computing attentive scores
            num_hidden_rbf_mlp   = 128, #hidden size of the MLP used for transforming rbf
            num_hidden_normgate_mlp = 128, #hidden size of the MLP used in normgate
            load_from            = None, #if this is given the network is loaded from the specified .pth file and all other arguments are ignored
            #Zmax                 = 87 #maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default 
    ):
        super().__init__()

        self.create_graph = True  #can be set to False if the NN is only used for inference

        #store hyperparameter values
        max_order_per_atom = get_max_order(orbital_basis, per_atom=True)

        order_max = max(max_order_per_atom.values())

        self.orbital_basis = orbital_basis
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
        self.num_hidden_att_mlp = num_hidden_att_mlp
        self.num_hidden_rbf_mlp = num_hidden_rbf_mlp
        self.num_hidden_normgate_mlp = num_hidden_normgate_mlp
        #self.Zmax = Zmax

        #error checking
        if self.order < order_max:
            print("An orbital with L={} was found, but the neural network was initialized with L={}".format(order_max, self.order))
            print("The neural network MUST have at least the same order as all orbitals!")
            quit()
        if self.order < 2*order_max:
            print("An orbital with L={} was found, but the neural network was initialized with L={}".format(order_max, self.order))
            print("The neural network SHOULD have at least twice the order of the maximum order orbital for good results!")
            #don't quit here, maybe someone wants to do it like this

        #declare modules and parameters
        self.clebsch_gordan = ClebschGordanMatrix()
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

        self.diagonal_pair = DiagonalPair(self.order,
                                          self.num_features,
                                          self.num_basis_functions,
                                          self.clebsch_gordan,
                                          True,
                                          self.activation)
        self.off_diagonal_pair = OffDiagonalPair(self.order,
                                                 self.num_features,
                                                 self.num_basis_functions,
                                                 self.clebsch_gordan,
                                                 True,
                                                 self.activation,
                                                 num_hidden_att_mlp=self.num_hidden_att_mlp,
                                                 num_hidden_rbf_mlp=self.num_hidden_rbf_mlp,
                                                 num_hidden_normgate_mlp=self.num_hidden_normgate_mlp)

    def forward(self, atoms):

        R = atoms['positions']
        dij = atoms['distances']
        fs = atoms['sph_repr']  # equivariant spherical harmonics atomic representation
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2) #unsqueeze for broadcasting

        # print("\n".join(f"fs[{l}]: {fs[l].shape}" for l in range(len(fs))))
        # print(f"rbf: {rbf.shape}")

        fpc = self.residual_pc(fs) #central pair features
        # fpn = self.residual_pn(fs) #neighbor pair features

        # #compute pair features for self-interactions
        # fii = [1*x for x in fpc]
        # for L in range(self.order+1): #add influence of neighbouring atoms to pairs
        #     idx_j  = atoms['idx_j'].view(*(1,)*len(fpn[L].shape[:-3]),-1,1,1).repeat(*fpn[L].shape[:-3], 1, *fpn[L].shape[-2:])
        #     fpn_j  = self.radial_ii[L](rbf)*torch.gather(fpn[L], 1, idx_j)
        #     fii[L] = fii[L].index_add(1, atoms['idx_i'], fpn_j)

        #gather atomic pairs
        fi = []
        fj = []
        for L in range(self.order+1):
            i = atoms['idx_i'].view(*(1,)*len(fpc[L].shape[:-3]),-1,1,1).repeat(*fpc[L].shape[:-3], 1, *fpc[L].shape[-2:])
            j = atoms['idx_j'].view(*(1,)*len(fpc[L].shape[:-3]),-1,1,1).repeat(*fpc[L].shape[:-3], 1, *fpc[L].shape[-2:])
            fi.append(torch.gather(fpc[L], 1, i))
            fj.append(torch.gather(fpc[L], 1, j))

        # print("\n".join(f"fi[{l}]: {fi[l].shape}" for l in range(len(fi))))

        # print("before diagonal_pair")
        # print("\n".join(f"fi[{l}]: {fi[l].shape}" for l in range(len(fi))))
        fii = self.diagonal_pair(fi)
        # print("after diagonal_pair")
        # print("\n".join(f"fii[{l}]: {fii[l].shape}" for l in range(len(fii))))

        #compute output features (irreducible representations) for self-interactions
        # fii = self.residual_ii(fii)

        #generate index lists for asymmetrizing pair interactions
        # idx_pi = []
        # idx_pj = []
        # for ni, ij1 in enumerate(zip(atoms['idx_i'], atoms['idx_j'])):
        #     i1 = ij1[0].item()
        #     j1 = ij1[1].item()
        #     for nj, ij2 in enumerate(zip(atoms['idx_i'], atoms['idx_j'])):
        #         i2 = ij2[0].item()
        #         j2 = ij2[1].item()
        #         if ((i1 == i2) and (not j1 == j2)):
        #             idx_pi.append(ni)
        #             idx_pj.append(nj)
        # idx_pi = torch.tensor(idx_pi, dtype=torch.int64, device=R.device)
        # idx_pj = torch.tensor(idx_pj, dtype=torch.int64, device=R.device)

        #compute pair features for ordinary interactions
        # fij = self.mix_ij(fi, fj, rbf) #mix pairs
        # for L in range(self.order+1): #add influence of neighbouring atoms to pairs
        #     idx_j  = atoms['idx_j'].view(*(1,)*len(fpn[L].shape[:-3]),-1,1,1).repeat(*fpn[L].shape[:-3], 1, *fpn[L].shape[-2:])
        #     fpn_j  = self.radial_ij[L](rbf)*torch.gather(fpn[L], 1, idx_j)
        #     idx_pj_L = idx_pj.view(*(1,)*len(fpn_j.shape[:-3]),-1,1,1).repeat(*fpn_j.shape[:-3], 1, *fpn_j.shape[-2:])
        #     fij[L] = fij[L].index_add(1, idx_pi, torch.gather(fpn_j, 1, idx_pj_L))

        #compute output features (irreducible representations) for pair-interactions
        # fij = self.residual_ij(fij)

        fij = self.off_diagonal_pair(fi, fj, rbf)

        atoms['pair_features'] = fii, fij

        return atoms
    

class DiagonalPair(nn.Module):

    def __init__(self,
                 order,
                 num_features,
                 num_basis_functions,
                 clebsch_gordan=None,
                 mix_order=True,
                 activation='swish',
                 normalize=0,
                 order_out=None,
                 parity=False,
                 bias=True):
        super().__init__()

        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.clebsch_gordan = clebsch_gordan
        self.normalize = normalize
        self.mix_orders = mix_order

        if order_out is None:
            self.order_out = self.order
        else:
            self.order_out = order_out

        if self.mix_orders:
            assert clebsch_gordan is not None

        if activation == "swish":
            self.activation_pre = Swish(self.num_features)
        elif activation == "ssp":
            self.activation_pre = ShiftedSoftplus(self.num_features)
        else:
            raise ValueError("Unsupported activation function:", activation)
        
        self.simple_residual_l = SimplifiedResidualBlock(order=self.order,
                                                         num_features=self.num_features,
                                                         clebsch_gordan=self.clebsch_gordan, 
                                                         mix_orders=True,
                                                         activation=activation,
                                                         normalize=normalize,
                                                         order_out=self.order_out,
                                                         parity=parity,
                                                         bias=bias)
        self.simple_residual_r = SimplifiedResidualBlock(order=self.order,
                                                         num_features=self.num_features,
                                                         clebsch_gordan=self.clebsch_gordan, 
                                                         mix_orders=True,
                                                         activation=activation,
                                                         normalize=normalize,
                                                         order_out=self.order_out,
                                                         parity=parity,
                                                         bias=bias)
        
        self.mix_lr = PairMixing(self.order, self.order, self.order, self.num_basis_functions, self.num_features, self.clebsch_gordan, distance_dependent=False)

        self.simple_residual_out = SimplifiedResidualBlock(order=self.order,
                                                         num_features=self.num_features,
                                                         clebsch_gordan=self.clebsch_gordan, 
                                                         mix_orders=True,
                                                         activation=activation,
                                                         normalize=normalize,
                                                         order_out=self.order_out,
                                                         parity=parity,
                                                         bias=bias)

    def forward(self, x):

        xl = self.simple_residual_l(x)
        xr = self.simple_residual_r(x)

        # print("\n".join(f"xl[{l}]: {xl[l].shape}" for l in range(len(xl))))
        # print("\n".join(f"xr[{l}]: {xr[l].shape}" for l in range(len(xr))))
        
        fii = self.mix_lr(xl, xr, None)

        fii = [fii[l] + x[l] for l in range(len(x))]  # residual connection

        fii = self.simple_residual_out(fii)

        return fii
    

class OffDiagonalPair(nn.Module):

    def __init__(self,
                 order,
                 num_features,
                 num_basis_functions,
                 clebsch_gordan=None,
                 mix_order=True,
                 activation='swish',
                 normalize=0,
                 order_out=None,
                 parity=False,
                 bias=True,
                 num_hidden_att_mlp=128,
                 num_hidden_rbf_mlp=128,
                 num_hidden_normgate_mlp=128):
        super().__init__()

        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.clebsch_gordan = clebsch_gordan
        self.normalize = normalize
        self.mix_orders = mix_order
        self.num_hidden_att_mlp = num_hidden_att_mlp
        self.num_hidden_rbf_mlp = num_hidden_rbf_mlp
        self.num_hidden_normgate_mlp = num_hidden_normgate_mlp

        if order_out is None:
            self.order_out = self.order
        else:
            self.order_out = order_out

        if self.mix_orders:
            assert clebsch_gordan is not None

        if activation == "swish":
            self.activation_rbf_mlp = Swish(self.num_features)
        elif activation == "ssp":
            self.activation_rbf_mlp = ShiftedSoftplus(self.num_features)
        else:
            raise ValueError("Unsupported activation function:", activation)

        self.simple_residual_i = SimplifiedResidualBlock(order=self.order,
                                                         num_features=self.num_features,
                                                         clebsch_gordan=self.clebsch_gordan, 
                                                         mix_orders=self.mix_orders,
                                                         activation=activation,
                                                         normalize=self.normalize,
                                                         order_out=self.order_out,
                                                         parity=parity,
                                                         bias=bias,
                                                         num_hidden_normgate_mlp=self.num_hidden_normgate_mlp)
        
        self.simple_residual_j = SimplifiedResidualBlock(order=self.order,
                                                         num_features=self.num_features,
                                                         clebsch_gordan=self.clebsch_gordan, 
                                                         mix_orders=self.mix_orders,
                                                         activation=activation,
                                                         normalize=self.normalize,
                                                         order_out=self.order_out,
                                                         parity=parity,
                                                         bias=bias,
                                                         num_hidden_normgate_mlp=self.num_hidden_normgate_mlp)
        
        self.att_scores = AttentiveScores(order=self.order,
                                          num_features=self.num_features,
                                          clebsch_gordan=self.clebsch_gordan,
                                          mix_order=self.mix_orders,
                                          activation=activation,
                                          normalize=self.normalize,
                                          order_out=self.order_out,
                                          parity=parity,
                                          bias=bias,
                                          num_hidden_att_mlp=self.num_hidden_att_mlp)
        
        self.rbf_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.num_hidden_rbf_mlp),
            self.activation_rbf_mlp,
            nn.Linear(self.num_hidden_rbf_mlp, self.num_features))

        self.pairmix = PairMixing(self.order, self.order, self.order, self.num_basis_functions, self.num_features, self.clebsch_gordan)
        
        self.simple_residual_out = SimplifiedResidualBlock(order=self.order,
                                                           num_features=self.num_features,
                                                           clebsch_gordan=self.clebsch_gordan, 
                                                           mix_orders=self.mix_orders,
                                                           activation=activation,
                                                           normalize=self.normalize,
                                                           order_out=self.order_out,
                                                           parity=parity,
                                                           bias=bias,
                                                           num_hidden_normgate_mlp=self.num_hidden_normgate_mlp)

    def forward(self, xi, xj, rbf):

        x_i = self.simple_residual_i(xi)
        x_j = self.simple_residual_j(xj)

        # print("\n".join(f"x_i[{l}]: {x_i[l].shape}" for l in range(len(x_i))))
        # print("\n".join(f"x_j[{l}]: {x_j[l].shape}" for l in range(len(x_j))))

        aij = self.att_scores(xi, xj)  # (1, 30, 6, 128)
        # print(f"aij: {aij.shape}")
        rbf_mlp = self.rbf_mlp(rbf)  # (1, 30, 1, 128)
        # print(f"rbf_mlp: {rbf_mlp.shape}")

        filter = aij * rbf_mlp
        # print(f"filter: {filter.shape}")

        # tensor product x_i, x_j, filter
        fij = self.pairmix(x_i, x_j, filter)
        # print("\n".join(f"fij[{l}]: {fij[l].shape}" for l in range(len(fij))))

        fij = self.simple_residual_out(fij)

        return fij