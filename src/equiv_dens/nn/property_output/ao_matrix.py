import math
import torch
import torch.nn as nn
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.modules.radial_basis_functions import *
from equiv_dens.nn.modules.network_blocks import *


class AOMatrixFromAtomFeatures(nn.Module):
    """
    Neural network for computing atomic orbital matrices like the hamiltonian and the density matrix
    from spherical harmonic features in a blockwise rotationally equivariant way
    """

    def __init__(self,
            orbital_basis        = None, #orbitals of atoms, defines layout and shape of output matrix
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
            output_property_name = 'ao_matrix', # output property key of atoms dict, e.g. atoms['hamiltonian_matrix']
            load_from            = None, #if this is given the network is loaded from the specified .pth file and all other arguments are ignored
            #Zmax                 = 87 #maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default 
    ):
        super(AOMatrixFromAtomFeatures, self).__init__()

        self.create_graph = True  #can be set to False if the NN is only used for inference

        #store hyperparameter values
        # TEMP transform orbital basis from equiv dens structure to hamiltonian code structure
        ham_orbital_basis = []
        for z, orb in orbital_basis.items():
            atom_orbital_basis = []
            for _, _, l in orb:
                if l <= 2:  # TODO remove. this is for troubleshooting
                    atom_orbital_basis.append((z, l))
            ham_orbital_basis.append(tuple(atom_orbital_basis))
        ham_orbital_basis = tuple(ham_orbital_basis)
        # TODO keep dictionary structure, delete [1] entry

        print(f"Transformed orbital basis from\n{orbital_basis}\nto\n{ham_orbital_basis}")

        self.orbital_basis = ham_orbital_basis
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
        N = len(self.orbital_basis)
        idx_i = torch.arange(N, dtype=torch.int64).view(-1,1).repeat(1,N).view(-1)
        idx_j = torch.arange(N, dtype=torch.int64).view(1,-1).repeat(N,1).view(-1)
        idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j] #exclude self-interactions
        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_j', idx_j)
        print(f"idx_i: {self.idx_i}")
        print(f"idx_j: {self.idx_j}")
        # TODO use atoms dict in forward

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
        # TODO move to forward

        #extract nuclear charges from orbitals, determine maximum order, and
        #build the occupation mask (for extracting occupied orbitals in energy prediction)
        Zl = []
        order_max = 0
        self.Norb = 0
        for i in range(len(self.orbital_basis)):
            Zl.append(self.orbital_basis[i][0][0])
            for z, l in self.orbital_basis[i]:
                self.Norb += 2*l+1
                assert z == Zl[i] #check that Z is the same for all orbitals
                if l > order_max:
                    order_max = l
        # TODO use orbital.get_max_order function
        # TODO don't need Zl

        #(unsqueeze for batch dimension)
        #occupation = torch.tensor([1 if n < sum(Zl)//2 else 0 for n in range(self.Norb)], dtype=torch.float64).unsqueeze(0)
        #Zf = torch.tensor(Zl, dtype=torch.float64).unsqueeze(0)
        #Z  = torch.tensor(Zl, dtype=torch.int64).unsqueeze(0)   
        #self.register_buffer('ZiZj',Zf[:,idx_i]*Zf[:,idx_j]) #for calculating nucleus-nucleus repulsion
        #self.register_buffer('Z', Z)                         #for gathering embeddings
        #self.register_buffer('occupation', occupation)       #for masking out unoccupied orbitals

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
        #self.embedding = SphericalEmbedding(self.order, self.num_features, self.Zmax)
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
        # self.mix_s  = PairMixing(self.order, self.order, self.order, self.num_basis_functions, self.num_features, self.clebsch_gordan)
        self.mix_ij = PairMixing(self.order, self.order, self.order, self.num_basis_functions, self.num_features, self.clebsch_gordan)
        self.radial_ii = nn.ModuleList([nn.Linear(self.num_basis_functions, self.num_features, bias=False)
            for L in range(self.order+1)])
        self.radial_ij = nn.ModuleList([nn.Linear(self.num_basis_functions, self.num_features, bias=False)
            for L in range(self.order+1)])
        self.residual_pc = ResidualStack(self.num_residual_pc, self.order, self.num_features, self.clebsch_gordan, True, self.activation)
        self.residual_pn = ResidualStack(self.num_residual_pn, self.order, self.num_features, self.clebsch_gordan, True, self.activation)
        self.residual_ii = ResidualStack(self.num_residual_ii, self.order, self.num_features, self.clebsch_gordan, True, self.activation)
        self.residual_ij = ResidualStack(self.num_residual_ij, self.order, self.num_features, self.clebsch_gordan, True, self.activation)

        #determine minimum number of output features based on orbitals
        #and generate dictionaries (irreps_ii/irreps_ij) that store indices 
        #for collecting the correct irreproducible representations from features
        #diagonal blocks
        number_L = [0 for L in range(2*order_max+1)] #keeps track of how many irreps of each order there are already
        self.irreps_ii = {}
        for i in range(len(self.orbital_basis)):
            self.irreps_ii, number_L = self.compute_matrix_irreps(
                self.orbital_basis[i], self.orbital_basis[i], self.irreps_ii, number_L)
        #print('ii', number_L)

        #off-diagonal blocks
        number_L = [0 for L in range(2*order_max+1)] #keeps track of how many irreps of each order there are already
        self.irreps_ij = {}
        for i in range(len(self.orbital_basis)):
            for j in range(len(self.orbital_basis)):
                if i == j:
                    continue
                self.irreps_ij, number_L = self.compute_matrix_irreps(
                    self.orbital_basis[i], self.orbital_basis[j], self.irreps_ij, number_L)
        # TODO adapt to dictionary structure, loop over keys, no atom indices
        #print('ij', number_L)

        #initialize parameters
        #if load_from is not None:
        #    self.load_state_dict(saved_state['state_dict'], strict=False)
        #else:
        #    self.reset_parameters()


    def compute_matrix_irreps(self, orbitals_i, orbitals_j, irreps, number_L):
        """
        Given the orbitals of atom i and atom j, computes how many irreducible representations
        of each order are necessary for constructing the corresponding off-diagonal block of the matrix

        inputs:
            orbitals_i: Tuple or list of tuples with integer entries (Z, L) that define the orbitals of atom i
            orbitals_j: Tuple or list of tuples with integer entries (Z, L) that define the orbitals of atom j
            irreps: Dictionary that stores the feature indices for collecting irreducible representations
            number_L: List of length L+1 with integer entries that stores how many irreducible representations of 
                    each order are already in use
        outputs:
            irreps: Updated input dictionary
            number_L: Updated input list
        """
        for n_i, orb_i in enumerate(orbitals_i):
            z_i, l_i = orb_i
            for n_j, orb_j in enumerate(orbitals_j):
                z_j, l_j = orb_j 
                for L in range(abs(l_i-l_j), l_i+l_j+1):
                    key = (z_i, z_j, n_i, n_j, L)
                    if key not in irreps.keys():
                        irreps[key] = number_L[L]
                        number_L[L] += 1
        return irreps, number_L


    def matrix_block(self, row, col, irreps, batch_size, j_gt_i, device='cpu', dtype=torch.float32):
        """
        Given the orbitals in the row and column, constructs a block of the Hamiltonian/Overlap matrix
        from the input irreducible representations

        inputs:
            row: Tuple or list of tuples with integer entries (Z, L) that define the orbitals in the row
            col: Tuple or list of tuples with integer entries (Z, L) that define the orbitals in the column
            irreps: list of irreducible representations of shape [batch_size, 2*L+1]
            batch_size: how many matrices are in this batch (needed to initialize matrix subblock)
        outputs:
            block: batch of matrix blocks of shape [batch_size, nrow, ncol] (nrow/ncol depends on row/col inputs)
        """
        nrow = sum((2*l+1) for _, l in row) #number of rows in the block
        ncol = sum((2*l+1) for _, l in col) #number of columns in the block
        block = torch.zeros(batch_size, nrow, ncol, device=device, dtype=dtype)

        idx = 0 #index for accessing the correct irreps
        start_i = 0
        for _, l_i in row:
            n_i = 2*l_i+1
            start_j = 0
            for _, l_j in col:
                n_j = 2*l_j+1
                for L in range(abs(l_i-l_j), l_i+l_j+1):
                    #compute inverse spherical tensor product             
                    # TODO carefully useing new cg matrix output, shape!!!
                    cg = math.sqrt(2*L+1)*self.clebsch_gordan(l_i, l_j, L)[0].unsqueeze(0)
                    product = (cg*irreps[idx].unsqueeze(-2).unsqueeze(-2)).sum(-1)

                    #add product to appropriate part of the block
                    blockpart = block.narrow(-2,start_i,n_i).narrow(-1,start_j,n_j)
                    blockpart += product

                    idx += 1
                start_j += n_j
            start_i += n_i
        return block
    

    """
    Given the Cartesian coordinates and index lists, calculates pairwise distances and unit displacement 
    vectors. Each distance/vector is specified by a pair of atom indices i and j (i != j). The total
    number of interactions is num_interactions=num_atoms*(num_atoms-1) when all pairwise distances are 
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
        Ri  = torch.gather(R, -2, idx_i.view(*(1,)*len(R.shape[:-2]),-1,1).repeat(*R.shape[:-2],1,R.size(-1)))
        Rj  = torch.gather(R, -2, idx_j.view(*(1,)*len(R.shape[:-2]),-1,1).repeat(*R.shape[:-2],1,R.size(-1)))
        rij = Rj-Ri #displacement vectors
        dij = torch.norm(rij, dim=-1, keepdim=True) #distances
        uij = rij/dij #unit displacement vectors
        return dij, uij

    
    def forward(self, atoms):
        """
        Computes the Hamiltonian/Density matrix.
        
        inputs:
            fs: spherical harmonics representation of shape [batch_size, num_atoms, 3]
        outputs:
            matrix: Hamiltonian/Density matrix of shape [batch_size, num_orbitals, num_orbitals]
        """

        print(f"atoms idx_i: {atoms['idx_i']}")
        print(f"atoms idx_j: {atoms['idx_j']}")
        print(f"atoms _idx: {atoms['_idx']}")
        print(f"atoms neighbor_batch_idx: {atoms['neighbor_batch_idx']}")
        print(f"atoms idxi pure: {atoms['idx_i_pure']}")
        print(f"atoms idxj pure: {atoms['idx_j_pure']}")
        R = atoms['positions']
        dij = atoms['distances']
        print(f"dij shape: {dij.shape}")
        fs = atoms['sph_repr']  # equivariant spherical harmonics atomic representation
        print(f"fs shape: {fs[0].shape}")
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2) #unsqueeze for broadcasting
        
        fpc = self.residual_pc(fs) #central pair features
        fpn = self.residual_pn(fs) #neighbor pair features

        #compute pair features for self-interactions
        fii = [1*x for x in fpc]
        for L in range(self.order+1): #add influence of neighbouring atoms to pairs
            idx_j  = self.idx_j.view(*(1,)*len(fpn[L].shape[:-3]),-1,1,1).repeat(*fpn[L].shape[:-3], 1, *fpn[L].shape[-2:])
            radiil = self.radial_ii[L](rbf)
            fpnlj = torch.gather(fpn[L], 1, idx_j)
            print(f"L: {L}")
            print(f"radiil shape: {radiil.shape}")
            print(f"fpnlj shape: {fpnlj.shape}")
            print(f"fpn[L] shape: {fpn[L].shape}")
            print(f"idx_j shape: {idx_j.shape}")
            fpn_j = radiil*fpnlj
            #fpn_j  = self.radial_ii[L](rbf)*torch.gather(fpn[L], 1, idx_j)
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
        # TODO use idx lists from atoms dict

        #compute pair features for ordinary interactions
        fij = self.mix_ij(fi, fj, rbf) #mix pairs
        for L in range(self.order+1): #add influence of neighbouring atoms to pairs
            idx_j  = self.idx_j.view(*(1,)*len(fpn[L].shape[:-3]),-1,1,1).repeat(*fpn[L].shape[:-3], 1, *fpn[L].shape[-2:])
            fpn_j  = self.radial_ij[L](rbf)*torch.gather(fpn[L], 1, idx_j)
            idx_pj = self.idx_pj.view(*(1,)*len(fpn_j.shape[:-3]),-1,1,1).repeat(*fpn_j.shape[:-3], 1, *fpn_j.shape[-2:])
            fij[L] = fij[L].index_add(1, self.idx_pi, torch.gather(fpn_j, 1, idx_pj))

        #compute output features (irreducible representations) for pair-interactions
        fij = self.residual_ij(fij)

        #construct batch of matrices of shape [batch_size, num_orbitals, num_orbitals]
        idx = 0 #initialize interaction index to 0 (gets incremented)
        batch_size = fii[0].size(0)  # TODO atoms['batch_positions'].shape[0]
         #atoms in batch TODO atoms['batch_positions'].shape[1]
        matrix_rows = [] # TODO empty array length atoms in batch

        for i in range(len(self.orbital_basis)): #loop over rows
            current_row = []# TODO empty array length atoms in batch
            for j in range(len(self.orbital_basis)): #loop over columns
                #collect irreps from output features (their shape after squeezing is [batch_size, 2*L+1])
                #features have shape [batch_size,num_atoms/num_interactions,2*L+1,num_features]
                #dimension -3 corresponds to atom/interaction indices
                #dimension -1 corresponds to the feature dimension
                #irreps have shape [batch_size, 2*L+1] (after squeezing)
                irreps = []
                if i == j: #diagonal block
                    for n_i, orb_i in enumerate(self.orbital_basis[i]):
                        z_i, l_i = orb_i
                        for n_j, orb_j in enumerate(self.orbital_basis[j]):
                            z_j, l_j = orb_j
                            for L in range(abs(l_i-l_j), l_i+l_j+1):
                                #self.irreps_ii is a dictionary that stores the index ii of the irrep
                                ii = self.irreps_ii[(z_i, z_j, n_i, n_j, L)]
                                irreps.append(fii[L].narrow(-3,i,1).narrow(-1,ii,1).squeeze(-3).squeeze(-1))
                else: #off-diagonal block
                    for n_i, orb_i in enumerate(self.orbital_basis[i]):
                        z_i, l_i = orb_i
                        for n_j, orb_j in enumerate(self.orbital_basis[j]):
                            z_j, l_j = orb_j
                            for L in range(abs(l_i-l_j), l_i+l_j+1):
                                #self.irreps_ij is a dictionary that stores the index ij of the irrep
                                ij = self.irreps_ij[(z_i, z_j, n_i, n_j, L)]
                                irreps.append(fij[L].narrow(-3,idx,1).narrow(-1,ij,1).squeeze(-3).squeeze(-1))
                    idx += 1 #increment interaction index
                # TODO use correct index instead of append j index pure
                current_row.append(self.matrix_block(self.orbital_basis[i], self.orbital_basis[j], irreps, batch_size, j>i, device=R.device, dtype=R.dtype))
            # TODO use correct index instead of append i index pure
            matrix_rows.append(torch.cat(current_row, -1))

        matrix = torch.cat(matrix_rows,-2)
        matrix = matrix + matrix.transpose(-2,-1) #symmetrize

        atoms[self.output_property_name] = matrix

        return atoms
    

class AOMatrixFromPairFeatures(nn.Module):

    # TODO use sequential with (repr, matrix output)
    def __init__(self,
            orbital_basis        = None, #orbitals of atoms, defines layout and shape of output matrix
            order                = 1,  #maximum order of spherical harmonics features
            num_features         = 32, #dimensionality of the feature space
            num_basis_functions  = 32, #number of basis functions for featurizing distances
            num_residual_ao_ii  = 1, #number of residual blocks applied to output atomic features for predicting irreps of diagonal blocks (output matrix)
            num_residual_ao_ij  = 1, #number of residual blocks applied to pair features for predicting irreps of off-diagonal blocks (output matrix)
            basis_functions      = 'exp-bernstein', #type of radial basis functions (exp-gaussian/exp-bernstein/gaussian/bernstein)
            cutoff               = 15.0, #cutoff distance (default is 15 Bohr)
            activation           = 'swish', #type of activation function used (swish/ssp)
            output_property_name = 'ao_matrix', # output property key of atoms dict, e.g. atoms['hamiltonian_matrix']
            load_from            = None, #if this is given the network is loaded from the specified .pth file and all other arguments are ignored
            #Zmax                 = 87 #maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default 
    ):
        super(AOMatrixFromPairFeatures, self).__init__()

        self.create_graph = True  #can be set to False if the NN is only used for inference

        #store hyperparameter values
        self.orbital_basis = orbital_basis
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_residual_ao_ii  = num_residual_ao_ii
        self.num_residual_ao_ij  = num_residual_ao_ij
        self.basis_functions = basis_functions
        self.cutoff = cutoff
        self.activation = activation
        self.output_property_name = output_property_name
        #self.Zmax = Zmax

        #extract nuclear charges from orbitals, determine maximum order, and
        #build the occupation mask (for extracting occupied orbitals in energy prediction)
        Zl = []
        order_max = 0
        self.Norb = 0
        for i in range(len(self.orbital_basis)):
            Zl.append(self.orbital_basis[i][0][0])
            for z, l in self.orbital_basis[i]:
                self.Norb += 2*l+1
                assert z == Zl[i] #check that Z is the same for all orbitals
                if l > order_max:
                    order_max = l

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
        self.residual_ao_ii = ResidualStack(self.num_residual_ao_ii, self.order, self.num_features, self.clebsch_gordan, True, self.activation)
        self.residual_ao_ij = ResidualStack(self.num_residual_ao_ij, self.order, self.num_features, self.clebsch_gordan, True, self.activation)
        if self.activation == 'swish':
            self.activation_ao_ii = Swish(self.num_features)
            self.activation_ao_ij = Swish(self.num_features)
        elif activation == 'ssp':
            self.activation_ao_ii = ShiftedSoftplus(self.num_features)
            self.activation_ao_ij = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", self.activation)
            quit()

        #determine minimum number of output features based on orbitals
        #and generate dictionaries (irreps_ii/irreps_ij) that store indices 
        #for collecting the correct irreproducible representations from features
        #diagonal blocks
        number_L = [0 for L in range(2*order_max+1)] #keeps track of how many irreps of each order there are already
        self.irreps_ii = {}
        for i in range(len(self.orbital_basis)):
            self.irreps_ii, number_L = self.compute_matrix_irreps(
                self.orbital_basis[i], self.orbital_basis[i], self.irreps_ii, number_L)
        self.output_ii = SphericalLinear(self.order, self.num_features, 2*order_max, max(number_L), self.clebsch_gordan, zero_init=True)
        #print('ii', number_L)

        #off-diagonal blocks
        number_L = [0 for L in range(2*order_max+1)] #keeps track of how many irreps of each order there are already
        self.irreps_ij = {}
        for i in range(len(self.orbital_basis)):
            for j in range(len(self.orbital_basis)):
                if i == j:
                    continue
                self.irreps_ij, number_L = self.compute_matrix_irreps(
                    self.orbital_basis[i], self.orbital_basis[j], self.irreps_ij, number_L)
        self.output_ij = SphericalLinear(self.order, self.num_features, 2*order_max, max(number_L), self.clebsch_gordan, zero_init=True)
        #print('ij', number_L)


    def compute_matrix_irreps(self, orbitals_i, orbitals_j, irreps, number_L):
        """
        Given the orbitals of atom i and atom j, computes how many irreducible representations
        of each order are necessary for constructing the corresponding off-diagonal block of the matrix

        inputs:
            orbitals_i: Tuple or list of tuples with integer entries (Z, L) that define the orbitals of atom i
            orbitals_j: Tuple or list of tuples with integer entries (Z, L) that define the orbitals of atom j
            irreps: Dictionary that stores the feature indices for collecting irreducible representations
            number_L: List of length L+1 with integer entries that stores how many irreducible representations of 
                    each order are already in use
        outputs:
            irreps: Updated input dictionary
            number_L: Updated input list
        """
        for n_i, orb_i in enumerate(orbitals_i):
            z_i, l_i = orb_i
            for n_j, orb_j in enumerate(orbitals_j):
                z_j, l_j = orb_j 
                for L in range(abs(l_i-l_j), l_i+l_j+1):
                    key = (z_i, z_j, n_i, n_j, L)
                    if key not in irreps.keys():
                        irreps[key] = number_L[L]
                        number_L[L] += 1
        return irreps, number_L


    def matrix_block(self, row, col, irreps, batch_size, j_gt_i, device='cpu', dtype=torch.float32):
        """
        Given the orbitals in the row and column, constructs a block of the Hamiltonian/Overlap matrix
        from the input irreducible representations

        inputs:
            row: Tuple or list of tuples with integer entries (Z, L) that define the orbitals in the row
            col: Tuple or list of tuples with integer entries (Z, L) that define the orbitals in the column
            irreps: list of irreducible representations of shape [batch_size, 2*L+1]
            batch_size: how many matrices are in this batch (needed to initialize matrix subblock)
        outputs:
            block: batch of matrix blocks of shape [batch_size, nrow, ncol] (nrow/ncol depends on row/col inputs)
        """
        nrow = sum((2*l+1) for _, l in row) #number of rows in the block
        ncol = sum((2*l+1) for _, l in col) #number of columns in the block
        block = torch.zeros(batch_size, nrow, ncol, device=device, dtype=dtype)

        idx = 0 #index for accessing the correct irreps
        start_i = 0
        for _, l_i in row:
            n_i = 2*l_i+1
            start_j = 0
            for _, l_j in col:
                n_j = 2*l_j+1
                for L in range(abs(l_i-l_j), l_i+l_j+1):
                    #compute inverse spherical tensor product             
                    # TODO carefully useing new cg matrix output, shape!!!
                    cg = math.sqrt(2*L+1)*self.clebsch_gordan(l_i, l_j, L)[0].unsqueeze(0)
                    product = (cg*irreps[idx].unsqueeze(-2).unsqueeze(-2)).sum(-1)

                    #add product to appropriate part of the block
                    blockpart = block.narrow(-2,start_i,n_i).narrow(-1,start_j,n_j)
                    blockpart += product

                    idx += 1
                start_j += n_j
            start_i += n_i
        return block

    
    def forward(self, atoms):
        fii, fij = atoms['pair_features']

        # additional layer to refine pair features for specific output matrix
        fii    = self.residual_ao_ij(fii)
        fii[0] = self.activation_ao_ij(fii[0])
        fii    = self.output_ij(fii)

        fij    = self.residual_ao_ij(fij)
        fij[0] = self.activation_ao_ij(fij[0])
        fij    = self.output_ij(fij)

        #construct batch of matrices of shape [batch_size, num_orbitals, num_orbitals]
        idx = 0 #initialize interaction index to 0 (gets incremented)
        batch_size = fii[0].size(0)
        matrix_rows = []

        for i in range(len(self.orbital_basis)): #loop over rows
            current_row = []
            for j in range(len(self.orbital_basis)): #loop over columns
                #collect irreps from output features (their shape after squeezing is [batch_size, 2*L+1])
                #features have shape [batch_size,num_atoms/num_interactions,2*L+1,num_features]
                #dimension -3 corresponds to atom/interaction indices
                #dimension -1 corresponds to the feature dimension
                #irreps have shape [batch_size, 2*L+1] (after squeezing)
                irreps = []
                if i == j: #diagonal block
                    for n_i, orb_i in enumerate(self.orbital_basis[i]):
                        z_i, l_i = orb_i
                        for n_j, orb_j in enumerate(self.orbital_basis[j]):
                            z_j, l_j = orb_j
                            for L in range(abs(l_i-l_j), l_i+l_j+1):
                                #self.irreps_ii is a dictionary that stores the index ii of the irrep
                                ii = self.irreps_ii[(z_i, z_j, n_i, n_j, L)]
                                irreps.append(fii[L].narrow(-3,i,1).narrow(-1,ii,1).squeeze(-3).squeeze(-1))
                else: #off-diagonal block
                    for n_i, orb_i in enumerate(self.orbital_basis[i]):
                        z_i, l_i = orb_i
                        for n_j, orb_j in enumerate(self.orbital_basis[j]):
                            z_j, l_j = orb_j
                            for L in range(abs(l_i-l_j), l_i+l_j+1):
                                #self.irreps_ij is a dictionary that stores the index ij of the irrep
                                ij = self.irreps_ij[(z_i, z_j, n_i, n_j, L)]
                                irreps.append(fij[L].narrow(-3,idx,1).narrow(-1,ij,1).squeeze(-3).squeeze(-1))
                    idx += 1 #increment interaction index
                current_row.append(self.matrix_block(self.orbital_basis[i], self.orbital_basis[j], irreps, batch_size, j>i, device=R.device, dtype=R.dtype))
            matrix_rows.append(torch.cat(current_row, -1))

        matrix = torch.cat(matrix_rows,-2)
        matrix = matrix + matrix.transpose(-2,-1) #symmetrize

        atoms[self.output_property_name] = matrix

        return atoms