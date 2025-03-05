import math
import torch
import torch.nn as nn
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.nn.modules.radial_basis_functions import *
from equiv_dens.nn.modules.network_blocks import *
from equiv_dens.utils.orbitals import get_max_order
from equiv_dens.utils.base import remap_pair_idxs_for_padding
from equiv_dens.utils.orbital_conversions import convert_ao_matrix
# import time


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
        super().__init__()

        self.create_graph = True  #can be set to False if the NN is only used for inference

        #store hyperparameter values
        order_max = get_max_order(orbital_basis)

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
        self.output_property_name = output_property_name
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

        #determine minimum number of output features based on orbitals
        #and generate dictionaries (irreps_ii/irreps_ij) that store indices 
        #for collecting the correct irreproducible representations from features
        #diagonal blocks
        number_L = [0 for L in range(2*order_max+1)] #keeps track of how many irreps of each order there are already
        self.irreps_ii = {}
        for z in self.orbital_basis.keys():
            self.irreps_ii, number_L = self.compute_matrix_irreps(
                self.orbital_basis[z], self.orbital_basis[z], self.irreps_ii, number_L)

        #off-diagonal blocks
        number_L = [0 for L in range(2*order_max+1)] #keeps track of how many irreps of each order there are already
        self.irreps_ij = {}
        for z1 in self.orbital_basis.keys():
            for z2 in self.orbital_basis.keys():
                self.irreps_ij, number_L = self.compute_matrix_irreps(
                    self.orbital_basis[z1], self.orbital_basis[z2], self.irreps_ij, number_L)


    def compute_matrix_irreps(self, orbitals_i, orbitals_j, irreps, number_L):
        """
        Given the orbitals of atom i and atom j, computes how many irreducible representations
        of each order are necessary for constructing the corresponding off-diagonal block of the matrix

        inputs:
            orbitals_i: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals of atom i
            orbitals_j: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals of atom j
            irreps: Dictionary that stores the feature indices for collecting irreducible representations
            number_L: List of length L+1 with integer entries that stores how many irreducible representations of 
                    each order are already in use
        outputs:
            irreps: Updated input dictionary
            number_L: Updated input list
        """
        for n_i, orb_i in enumerate(orbitals_i):
            z_i, _, l_i = orb_i
            for n_j, orb_j in enumerate(orbitals_j):
                z_j, _, l_j = orb_j
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
            row: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals in the row
            col: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals in the column
            irreps: list of irreducible representations of shape [batch_size, 2*L+1]
            batch_size: how many matrices are in this batch (needed to initialize matrix subblock)
        outputs:
            block: batch of matrix blocks of shape [batch_size, nrow, ncol] (nrow/ncol depends on row/col inputs)
        """
        nrow = sum((2*l+1) for _, _, l in row) #number of rows in the block
        ncol = sum((2*l+1) for _, _, l in col) #number of columns in the block
        block = torch.zeros(batch_size, nrow, ncol, device=device, dtype=dtype)

        idx = 0 #index for accessing the correct irreps
        start_i = 0
        for _, _, l_i in row:
            n_i = 2*l_i+1
            start_j = 0
            for _, _, l_j in col:
                n_j = 2*l_j+1
                for L in range(abs(l_i-l_j), l_i+l_j+1):

                    #compute inverse spherical tensor product
                    cg_matrix, _ = self.clebsch_gordan(l_i, l_j, L)  # holds coefficients up to(!) order l_i,l_j,L -> get only coefficients for order l_i,l_j,L
                    cg = cg_matrix[
                        l_i ** 2:(l_i + 1) ** 2,
                        l_j ** 2:(l_j + 1) ** 2,
                        L ** 2:(L + 1) ** 2,
                    ]
                    cg = math.sqrt(2*L+1)*cg.unsqueeze(0)
                    product = (cg*irreps[idx].unsqueeze(-2).unsqueeze(-2)).sum(-1)

                    #add product to appropriate part of the block
                    blockpart = block.narrow(-2,start_i,n_i).narrow(-1,start_j,n_j)
                    blockpart += product

                    idx += 1
                start_j += n_j
            start_i += n_i
        return block


    def forward(self, atoms):
        """
        Computes the Hamiltonian/Density matrix.

        inputs:
            fs: spherical harmonics representation of shape [batch_size, num_atoms, 3]
        outputs:
            matrix: Hamiltonian/Density matrix of shape [batch_size, num_orbitals, num_orbitals]
        """

        R = atoms['positions']
        dij = atoms['distances']
        fs = atoms['sph_repr']  # equivariant spherical harmonics atomic representation
        rbf = self.radial_basis_functions(dij).unsqueeze_(-2) #unsqueeze for broadcasting

        fpc = self.residual_pc(fs) #central pair features
        fpn = self.residual_pn(fs) #neighbor pair features

        #compute pair features for self-interactions
        fii = [1*x for x in fpc]
        for L in range(self.order+1): #add influence of neighbouring atoms to pairs
            idx_j  = atoms['idx_j'].view(*(1,)*len(fpn[L].shape[:-3]),-1,1,1).repeat(*fpn[L].shape[:-3], 1, *fpn[L].shape[-2:])
            fpn_j  = self.radial_ii[L](rbf)*torch.gather(fpn[L], 1, idx_j)
            fii[L] = fii[L].index_add(1, atoms['idx_i'], fpn_j)

        #compute output features (irreducible representations) for self-interactions
        fii = self.residual_ii(fii)

        #gather atomic pairs
        fi = []
        fj = []
        for L in range(self.order+1):
            i = atoms['idx_i'].view(*(1,)*len(fpc[L].shape[:-3]),-1,1,1).repeat(*fpc[L].shape[:-3], 1, *fpc[L].shape[-2:])
            j = atoms['idx_j'].view(*(1,)*len(fpc[L].shape[:-3]),-1,1,1).repeat(*fpc[L].shape[:-3], 1, *fpc[L].shape[-2:])
            fi.append(torch.gather(fpc[L], 1, i))
            fj.append(torch.gather(fpc[L], 1, j))

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

        batch_size = atoms['batch_positions'].shape[0]
        num_atoms_in_batch = atoms['batch_positions'].shape[1]
        max_atom_numbers = torch.max(atoms['batch_atom_numbers'], dim=0)[0]
        num_orbitals_per_atom = {z.item(): sum((2*l+1) for _, _, l in self.orbital_basis[z.item()]) for z in torch.unique(max_atom_numbers)}
        num_orbitals = sum(num_orbitals_per_atom[z.item()] for z in max_atom_numbers)
        matrix = torch.zeros((batch_size, num_orbitals, num_orbitals), device=R.device)

        # todo do not use the same neighbor list as used for building representation and features (different cutoffs)
        # use idx_i/j_ao_matrix and neighbor_ao_matrix_batch_idx instead (usually smaller cutoff)
        # print(f"cutoff: {self.cutoff}")
        # print(f"idx_i: {atoms['idx_i']}")
        # print(f"idx_j: {atoms['idx_j']}")
        # print(f"neighbor_batch_idx: {atoms['neighbor_batch_idx']}")
        # print()
        # print(f"ao_matrix_cutoff not an output module argument")
        # print(f"idx_i_ao_matrix: {atoms['idx_i_ao_matrix']}")
        # print(f"idx_j_ao_matrix: {atoms['idx_j_ao_matrix']}")
        # print(f"neighbor_ao_matrix_batch_idx: {atoms['neighbor_ao_matrix_batch_idx']}")

        # 0: H
        # 1: H
        # 2: O

        idx_i_batch, idx_j_batch = remap_pair_idxs_for_padding(n_atoms=num_atoms_in_batch,
                                                               batch_idx_pos=atoms['batch_idx_pos'],
                                                               idx_i=atoms['idx_i_ao_matrix'],
                                                               idx_j=atoms['idx_j_ao_matrix'])
        
        # print(f"idx_i_batch: {idx_i_batch}")
        # print(f"idx_j_batch: {idx_j_batch}")


        # ao offset for each atom in batch
        ao_offsets = {0: 0}
        for i in range(1, num_atoms_in_batch):
            z_previous = atoms['batch_atom_numbers'][0][i-1].item()
            ao_offsets[i] = ao_offsets[i-1] + num_orbitals_per_atom[z_previous]

        idx = 0
        for s in range(batch_size):
            s_atom_numbers = atoms['batch_atom_numbers'][s]
            s_idx_i = idx_i_batch[atoms['neighbor_ao_matrix_batch_idx'] == s]
            s_idx_j = idx_j_batch[atoms['neighbor_ao_matrix_batch_idx'] == s]
            s_batch_idx = atoms['atom_batch_idx'][0] == s
            s_batch_idx_in_fii = torch.where(s_batch_idx)[0]  # idx for fii
            s_batch_pos = atoms['batch_idx_pos'][s_batch_idx] - (s * num_atoms_in_batch)

            # off-diagonal matrix blocks
            for i, j in zip(s_idx_i, s_idx_j):

                # print(f"off-diagonal: {i, j}")

                irreps = []

                for n_i, orb_i in enumerate(self.orbital_basis[s_atom_numbers[i].item()]):
                    z_i, _, l_i = orb_i

                    for n_j, orb_j in enumerate(self.orbital_basis[s_atom_numbers[j].item()]):
                        z_j, _, l_j = orb_j

                        for L in range(abs(l_i - l_j), l_i+l_j+1):
                            ij = self.irreps_ij[(z_i, z_j, n_i, n_j, L)]
                            irreps.append(fij[L].narrow(-3, idx, 1).narrow(-1, ij, 1).squeeze(-3).squeeze(-1))

                mblock = self.matrix_block(row=self.orbital_basis[z_i],
                                            col=self.orbital_basis[z_j],
                                            irreps=irreps,
                                            batch_size=1,
                                            j_gt_i=j>i,
                                            device=R.device,
                                            dtype=R.dtype)

                row_offset, col_offset = ao_offsets[i.item()], ao_offsets[j.item()]

                row_end = row_offset + mblock.shape[-2]
                col_end = col_offset + mblock.shape[-1]
                matrix[s, row_offset:row_end, col_offset:col_end] = mblock

                idx += 1

            # diagonal matrix blocks
            for i in range(len(s_batch_pos)):
                pos_in_batch = s_batch_pos[i]
                pos_in_fii = s_batch_idx_in_fii[i]

                # print(f"diagonal: {i}")

                irreps = []

                for n_i , orb_i in enumerate(self.orbital_basis[s_atom_numbers[i].item()]):
                    z_i, _, l_i = orb_i

                    for n_j , orb_j in enumerate(self.orbital_basis[s_atom_numbers[i].item()]):
                        z_j, _, l_j = orb_j

                        for L in range(abs(l_i - l_j), l_i + l_j + 1):
                            ii = self.irreps_ii[(z_i, z_j, n_i, n_j, L)]
                            irreps.append(fii[L].narrow(-3, pos_in_fii, 1).narrow(-1, ii, 1).squeeze(-3).squeeze(-1))

                mblock = self.matrix_block(row=self.orbital_basis[z_i],
                                            col=self.orbital_basis[z_j],
                                            irreps=irreps,
                                            batch_size=1,
                                            j_gt_i=j>i,
                                            device=R.device,
                                            dtype=R.dtype)

                row_offset, col_offset = ao_offsets[pos_in_batch.item()], ao_offsets[pos_in_batch.item()]

                row_end = row_offset + mblock.shape[-2]
                col_end = col_offset + mblock.shape[-1]
                matrix[s, row_offset:row_end, col_offset:col_end] = mblock

        matrix = matrix + matrix.transpose(-2,-1) #symmetrize

        matrix = convert_ao_matrix(ao_matrix=matrix,
                                   atom_numbers=atoms['batch_atom_numbers'],
                                   convention='pyscf_augccpvdz')

        atoms[self.output_property_name] = matrix

        return atoms


class AOMatrixFromPairFeatures(nn.Module):
    """
    Neural network for computing atomic orbital matrices like the hamiltonian and the density matrix
    from atom pair features in a blockwise rotationally equivariant way
    """

    def __init__(self,
            orbital_basis        = None, #orbitals of atoms, defines layout and shape of output matrix
            order                = 1,  #maximum order of spherical harmonics features
            num_features         = 32, #dimensionality of the feature space
            num_basis_functions  = 32, #number of basis functions for featurizing distances
            num_residual_ao_ii  = 1, #number of residual blocks applied to output atomic features for predicting irreps of diagonal blocks (output matrix)
            num_residual_ao_ij  = 1, #number of residual blocks applied to pair features for predicting irreps of off-diagonal blocks (output matrix)
            basis_functions      = 'exp-bernstein', #type of radial basis functions (exp-gaussian/exp-bernstein/gaussian/bernstein)
            activation           = 'swish', #type of activation function used (swish/ssp)
            output_property_name = 'ao_matrix', # output property key of atoms dict, e.g. atoms['hamiltonian_matrix']
            load_from            = None, #if this is given the network is loaded from the specified .pth file and all other arguments are ignored
            #Zmax                 = 87 #maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default 
    ):
        super().__init__()

        self.create_graph = True  #can be set to False if the NN is only used for inference

        #store hyperparameter values
        order_max = get_max_order(orbital_basis)

        self.orbital_basis = orbital_basis
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_residual_ao_ii = num_residual_ao_ii
        self.num_residual_ao_ij = num_residual_ao_ij
        self.basis_functions = basis_functions
        self.activation = activation
        self.output_property_name = output_property_name
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
        self.mix_ij = PairMixing(self.order, self.order, self.order, self.num_basis_functions, self.num_features, self.clebsch_gordan)
        self.radial_ii = nn.ModuleList([nn.Linear(self.num_basis_functions, self.num_features, bias=False)
            for L in range(self.order+1)])
        self.radial_ij = nn.ModuleList([nn.Linear(self.num_basis_functions, self.num_features, bias=False)
            for L in range(self.order+1)])
        self.residual_ao_ii = ResidualStack(self.num_residual_ao_ii, self.order, self.num_features, self.clebsch_gordan, True, self.activation)
        self.residual_ao_ij = ResidualStack(self.num_residual_ao_ij, self.order, self.num_features, self.clebsch_gordan, True, self.activation)

        if self.activation == 'swish':
            self.activation_ao_ii = Swish(self.num_features)
            self.activation_ao_ij = Swish(self.num_features)
        elif self.activation == 'ssp':
            self.activation_ao_ii = ShiftedSoftplus(self.num_features)
            self.activation_ao_ij = ShiftedSoftplus(self.num_features)
        else:
            print("Unsupported activation function:", activation)
            quit()

        #determine minimum number of output features based on orbitals
        #and generate dictionaries (irreps_ii/irreps_ij) that store indices 
        #for collecting the correct irreproducible representations from features
        #diagonal blocks
        number_L = [0 for L in range(2*order_max+1)] #keeps track of how many irreps of each order there are already
        self.irreps_ii = {}
        for z in self.orbital_basis.keys():
            self.irreps_ii, number_L = self.compute_matrix_irreps(
                self.orbital_basis[z], self.orbital_basis[z], self.irreps_ii, number_L)
        self.output_ii = SphericalLinear(self.order, self.num_features, 2*order_max, max(number_L), self.clebsch_gordan, zero_init=True)

        #off-diagonal blocks
        number_L = [0 for L in range(2*order_max+1)] #keeps track of how many irreps of each order there are already
        self.irreps_ij = {}
        for z1 in self.orbital_basis.keys():
            for z2 in self.orbital_basis.keys():
                self.irreps_ij, number_L = self.compute_matrix_irreps(
                    self.orbital_basis[z1], self.orbital_basis[z2], self.irreps_ij, number_L)
        self.output_ij = SphericalLinear(self.order, self.num_features, 2*order_max, max(number_L), self.clebsch_gordan, zero_init=True)



    def compute_matrix_irreps(self, orbitals_i, orbitals_j, irreps, number_L):
        """
        Given the orbitals of atom i and atom j, computes how many irreducible representations
        of each order are necessary for constructing the corresponding off-diagonal block of the matrix

        inputs:
            orbitals_i: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals of atom i
            orbitals_j: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals of atom j
            irreps: Dictionary that stores the feature indices for collecting irreducible representations
            number_L: List of length L+1 with integer entries that stores how many irreducible representations of 
                    each order are already in use
        outputs:
            irreps: Updated input dictionary
            number_L: Updated input list
        """
        for n_i, orb_i in enumerate(orbitals_i):
            z_i, _, l_i = orb_i
            for n_j, orb_j in enumerate(orbitals_j):
                z_j, _, l_j = orb_j
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
            row: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals in the row
            col: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals in the column
            irreps: list of irreducible representations of shape [batch_size, 2*L+1]
            batch_size: how many matrices are in this batch (needed to initialize matrix subblock)
        outputs:
            block: batch of matrix blocks of shape [batch_size, nrow, ncol] (nrow/ncol depends on row/col inputs)
        """
        nrow = sum((2*l+1) for _, _, l in row) #number of rows in the block
        ncol = sum((2*l+1) for _, _, l in col) #number of columns in the block
        block = torch.zeros(batch_size, nrow, ncol, device=device, dtype=dtype)

        idx = 0 #index for accessing the correct irreps
        start_i = 0
        for _, _, l_i in row:
            n_i = 2*l_i+1
            start_j = 0
            for _, _, l_j in col:
                n_j = 2*l_j+1
                for L in range(abs(l_i-l_j), l_i+l_j+1):

                    #compute inverse spherical tensor product
                    cg_matrix, _ = self.clebsch_gordan(l_i, l_j, L)  # holds coefficients up to(!) order l_i,l_j,L -> get only coefficients for order l_i,l_j,L
                    cg = cg_matrix[
                        l_i ** 2:(l_i + 1) ** 2,
                        l_j ** 2:(l_j + 1) ** 2,
                        L ** 2:(L + 1) ** 2,
                    ]
                    cg = math.sqrt(2*L+1)*cg.unsqueeze(0)
                    product = (cg*irreps[idx].unsqueeze(-2).unsqueeze(-2)).sum(-1)

                    #add product to appropriate part of the block
                    blockpart = block.narrow(-2,start_i,n_i).narrow(-1,start_j,n_j)
                    blockpart += product

                    idx += 1
                start_j += n_j
            start_i += n_i
        return block

    
    def forward(self, atoms):

        R = atoms['positions']
        fii, fij = atoms['pair_features']

        # additional layer to refine pair features for specific output matrix
        fii    = self.residual_ao_ii(fii)
        fii[0] = self.activation_ao_ii(fii[0])
        fii    = self.output_ii(fii)

        fij    = self.residual_ao_ij(fij)
        fij[0] = self.activation_ao_ij(fij[0])
        fij    = self.output_ij(fij)

        batch_size = atoms['batch_positions'].shape[0]
        num_atoms_in_batch = atoms['batch_positions'].shape[1]
        max_atom_numbers = torch.max(atoms['batch_atom_numbers'], dim=0)[0]
        num_orbitals_per_atom = {z.item(): sum((2*l+1) for _, _, l in self.orbital_basis[z.item()]) for z in torch.unique(max_atom_numbers)}
        num_orbitals = sum(num_orbitals_per_atom[z.item()] for z in max_atom_numbers)
        matrix = torch.zeros((batch_size, num_orbitals, num_orbitals), device=R.device)
        # matrix_old = torch.zeros((batch_size, num_orbitals, num_orbitals), device=R.device)

        # time.sleep(10)

        idx_i_batch, idx_j_batch = remap_pair_idxs_for_padding(n_atoms=num_atoms_in_batch,
                                                               batch_idx_pos=atoms['batch_idx_pos'],
                                                               idx_i=atoms['idx_i_ao_matrix'],
                                                               idx_j=atoms['idx_j_ao_matrix'])

        # ao offset for each atom in batch
        ao_offsets = {0: 0}
        for i in range(1, num_atoms_in_batch):
            z_previous = max_atom_numbers[i-1].item()
            ao_offsets[i] = ao_offsets[i-1] + num_orbitals_per_atom[z_previous]


        # print(f"atom_numbers: {atoms['atom_numbers']}")
        # print(f"batch_atom_numbers: {atoms['batch_atom_numbers']}")
        # print(f"batch_atom_mask: {atoms['batch_atom_mask']}")
        # print(f"max_atom_numbers: {max_atom_numbers}")
        # print(f"idx_i_ao_matrix: {atoms['idx_i_ao_matrix']}")
        # print(f"idx_j_ao_matrix: {atoms['idx_j_ao_matrix']}")
        # print(f"idx_i_batch: {idx_i_batch}")
        # print(f"idx_j_batch: {idx_j_batch}")
        # print(f"neighbor_ao_matrix_batch_idx: {atoms['neighbor_ao_matrix_batch_idx']}")
        # print(f"atom_batch_idx: {atoms['atom_batch_idx']}")
        # print(f"orbital_basis: {self.orbital_basis}")

        # print(f"batch_idx_pos: {atoms['batch_idx_pos']}")
        # print(f"atoms keys: {atoms.keys()}")


        # TODO: off-diagonal matrix blocks V2
        # [x] get list of unique interacting atom pairs [(i,j)..] from idx_i_batch and idx_j_batch (implicitly i!=j)
        # [x] get indexes of unique interacting atom pairs (i, j) -> "atom i and j interact at indexes [idx_0, idx_1, idx_2, ...]"
        # [x] get molecule indexes in which the unique interacting atom pairs are located from neighbor_ao_matrix_batch_idx
        # [x] for each unique (i,j) pair, get fij, e.g. torch.gather(fij[L], dim=1, index=interaction_indexes[(i,j)])
        #     [x] narrow dim=-1 according to ij (irreps count)
        #     [x] construct matrix block (batch_size = len(interaction_indexes[(i,j)]))
        #     [x] add matrix block to matrix via index_add (index=list of indexes the current atom pair interacts in)



        ############################
        # batch_wise
        ############################
        # t0 = time.time()

        # off-diagonal matrix blocks
        # get unique interacting atom pairs in batch
        unique_atom_pairs_in_batch, atom_pair_index_in_batch_inv = torch.unique(torch.stack([idx_i_batch, idx_j_batch], dim=1), dim=0, return_inverse=True)
        atom_pair_index_in_batch = [torch.nonzero(atom_pair_index_in_batch_inv == i, as_tuple=True)[0] for i in range(len(unique_atom_pairs_in_batch))]
        atom_pair_molecule_index = [atoms['neighbor_ao_matrix_batch_idx'][atom_pair_index] for atom_pair_index in atom_pair_index_in_batch]

        # print(f"unique_atom_pairs_in_batch: {unique_atom_pairs_in_batch}")
        # print(f"atom_pair_index_in_batch: {atom_pair_index_in_batch}")
        # print(f"atom_pair_molecule_index: {atom_pair_molecule_index}")

        for interaction_idx, (i, j) in enumerate(unique_atom_pairs_in_batch):
            i, j = i.item(), j.item()

            irreps = []

            for n_i, orb_i in enumerate(self.orbital_basis[max_atom_numbers[i].item()]):
                z_i, _, l_i = orb_i

                for n_j, orb_j in enumerate(self.orbital_basis[max_atom_numbers[j].item()]):
                    z_j, _, l_j = orb_j

                    for L in range(abs(l_i - l_j), l_i+l_j+1):
                        ij = self.irreps_ij[(z_i, z_j, n_i, n_j, L)]
                        fij_atom_pairs_in_batch = torch.index_select(fij[L], -3, atom_pair_index_in_batch[interaction_idx])
                        ir = fij_atom_pairs_in_batch.narrow(-1, ij, 1).squeeze(-1).squeeze(0)
                        irreps.append(ir)
            
            # Calculates potentially less then batch_size many matrix blocks,
            # e.g. if atom pair interacts not in every molecule in batch
            mblock = self.matrix_block(row=self.orbital_basis[z_i.item()],
                                       col=self.orbital_basis[z_j.item()],
                                       irreps=irreps,
                                       batch_size=len(atom_pair_index_in_batch[interaction_idx]),
                                       j_gt_i=j>i,
                                       device=R.device,
                                       dtype=R.dtype)
            
            row_offset, col_offset = ao_offsets[i], ao_offsets[j]

            matrixpart = matrix.narrow(-2, row_offset, mblock.shape[-2]).narrow(-1, col_offset, mblock.shape[-1])
            matrixpart.index_add_(0, atom_pair_molecule_index[interaction_idx], mblock)


        # diagonal matrix blocks
        # 
        # [x] indexes for each atom in batch from batch_idx_pos, i.e. location in fii per atom
        #   apply atom_mask to max_atom_numbers?
        #
        # [x] get index list that holds in which molecules of the batch each atom is contained
        #
        # sandbox
        # which molecules contain atom i? use batch_atom_mask
        # 

        # atoms['batch_atom_mask'][3, 0] = False
        # atoms['atom_batch_idx'] = torch.cat((atoms['atom_batch_idx'][:, :9], atoms['atom_batch_idx'][:, 10:]), dim=1)
        # print(f"atom_batch_idx: {atoms['atom_batch_idx']}")

        atom_index = torch.where(atoms['batch_atom_mask'])[1]
        # print(f"atom_index: {atom_index}")

        unique_atoms_in_batch, atom_index_in_batch_inv = torch.unique(atom_index, return_inverse=True)
        atom_index_in_batch = [torch.nonzero(atom_index_in_batch_inv == i, as_tuple=True)[0] for i in range(len(unique_atoms_in_batch))]
        atom_molecule_index = [atoms['atom_batch_idx'][0][atom_idx] for atom_idx in atom_index_in_batch]
        # print(f"unique_atoms_in_batch: {unique_atoms_in_batch}")
        # print(f"atom_index_in_batch: {atom_index_in_batch}")
        # print(f"atom_molecule_index: {atom_molecule_index}")

        for atom_idx, i in enumerate(unique_atoms_in_batch):
            i = i.item()

            irreps = []

            for n_i, orb_i in enumerate(self.orbital_basis[max_atom_numbers[i].item()]):
                z_i, _, l_i = orb_i

                for n_j, orb_j in enumerate(self.orbital_basis[max_atom_numbers[i].item()]):
                    z_j, _, l_j = orb_j

                    for L in range(abs(l_i - l_j), l_i+l_j+1):
                        ii = self.irreps_ii[(z_i, z_j, n_i, n_j, L)]
                        fii_atom_in_batch = torch.index_select(fii[L], -3, atom_index_in_batch[atom_idx])
                        ir = fii_atom_in_batch.narrow(-1, ii, 1).squeeze(-1).squeeze(0)
                        irreps.append(ir)

            # Calculates potentially less then batch_size many matrix blocks,
            # e.g. if atom not in every molecule in batch
            mblock = self.matrix_block(row=self.orbital_basis[z_i.item()],
                                       col=self.orbital_basis[z_j.item()],
                                       irreps=irreps,
                                       batch_size=len(atom_index_in_batch[atom_idx]),
                                       j_gt_i=j>i,
                                       device=R.device,
                                       dtype=R.dtype)
            
            row_offset, col_offset = ao_offsets[i], ao_offsets[i]

            matrixpart = matrix.narrow(-2, row_offset, mblock.shape[-2]).narrow(-1, col_offset, mblock.shape[-1])
            matrixpart.index_add_(0, atom_molecule_index[atom_idx], mblock)

        # print(f"time batch-wise: {time.time() - t0}")
        # quit()


        ############################
        # molecule-wise
        ############################
        # t0 = time.time()
        # idx = 0
        # for s in range(batch_size):
        #     s_atom_numbers = atoms['batch_atom_numbers'][s]
        #     s_idx_i = idx_i_batch[atoms['neighbor_ao_matrix_batch_idx'] == s]
        #     s_idx_j = idx_j_batch[atoms['neighbor_ao_matrix_batch_idx'] == s]
        #     s_batch_idx = atoms['atom_batch_idx'][0] == s
        #     s_batch_idx_in_fii = torch.where(s_batch_idx)[0]  # idx for fii
        #     s_batch_pos = atoms['batch_idx_pos'][s_batch_idx] - (s * num_atoms_in_batch)

        #     for i, j in zip(s_idx_i, s_idx_j):

        #         irreps = []

        #         for n_i, orb_i in enumerate(self.orbital_basis[s_atom_numbers[i].item()]):
        #             z_i, _, l_i = orb_i

        #             for n_j, orb_j in enumerate(self.orbital_basis[s_atom_numbers[j].item()]):
        #                 z_j, _, l_j = orb_j

        #                 for L in range(abs(l_i - l_j), l_i+l_j+1):
        #                     ij = self.irreps_ij[(z_i, z_j, n_i, n_j, L)]
        #                     ir = fij[L].narrow(-3, idx, 1).narrow(-1, ij, 1).squeeze(-3).squeeze(-1)
        #                     irreps.append(ir)

        #         mblock = self.matrix_block(row=self.orbital_basis[z_i],
        #                                     col=self.orbital_basis[z_j],
        #                                     irreps=irreps,
        #                                     batch_size=1,
        #                                     j_gt_i=j>i,
        #                                     device=R.device,
        #                                     dtype=R.dtype)

        #         row_offset, col_offset = ao_offsets[i.item()], ao_offsets[j.item()]

        #         row_end = row_offset + mblock.shape[-2]
        #         col_end = col_offset + mblock.shape[-1]
        #         matrix_old[s, row_offset:row_end, col_offset:col_end] = mblock

        #         idx += 1

        #         # diagonal matrix blocks
        #         for i in range(len(s_batch_pos)):
        #             pos_in_batch = s_batch_pos[i]
        #             pos_in_fii = s_batch_idx_in_fii[i]

        #             irreps = []

        #             for n_i , orb_i in enumerate(self.orbital_basis[s_atom_numbers[i].item()]):
        #                 z_i, _, l_i = orb_i

        #                 for n_j , orb_j in enumerate(self.orbital_basis[s_atom_numbers[i].item()]):
        #                     z_j, _, l_j = orb_j

        #                     for L in range(abs(l_i - l_j), l_i + l_j + 1):
        #                         ii = self.irreps_ii[(z_i, z_j, n_i, n_j, L)]
        #                         irreps.append(fii[L].narrow(-3, pos_in_fii, 1).narrow(-1, ii, 1).squeeze(-3).squeeze(-1))

        #             mblock = self.matrix_block(row=self.orbital_basis[z_i],
        #                                         col=self.orbital_basis[z_j],
        #                                         irreps=irreps,
        #                                         batch_size=1,
        #                                         j_gt_i=j>i,
        #                                         device=R.device,
        #                                         dtype=R.dtype)

        #             row_offset, col_offset = ao_offsets[pos_in_batch.item()], ao_offsets[pos_in_batch.item()]

        #             row_end = row_offset + mblock.shape[-2]
        #             col_end = col_offset + mblock.shape[-1]
        #             matrix_old[s, row_offset:row_end, col_offset:col_end] = mblock

        # print(f"time molecule-wise: {time.time() - t0}")

        # print(f"matrix_old zero?: {torch.allclose(matrix_old, torch.zeros_like(matrix_old))}")
        # print(f"matrix zero?: {torch.allclose(matrix, torch.zeros_like(matrix))}")
        # print(f"similiarity: {torch.allclose(matrix_old, matrix)}")

        # train (bs=3*5) : 1.0425s -> 0.08875s => 8.513% of previous wall clock time => 11.75x speedup
        # valid (bs=3*10): 1.3976s -> 0.06110s => 4.372% of previous wall clock time => 22.87x speedup

        # matrix = matrix_old
        matrix = matrix + matrix.transpose(-2,-1) #symmetrize

        matrix = convert_ao_matrix(ao_matrix=matrix,
                                   atom_numbers=atoms['batch_atom_numbers'],
                                   convention=None)

        atoms[self.output_property_name] = matrix

        return atoms


class AOMatrixFromPairFeaturesV2(nn.Module):
    """
    Neural network for computing atomic orbital matrices like the hamiltonian and the density matrix
    from atom pair features in a blockwise rotationally equivariant way
    """

    def __init__(self,
            orbital_basis        = None, #orbitals of atoms, defines layout and shape of output matrix
            order                = 1,  #maximum order of spherical harmonics features
            num_features         = 32, #dimensionality of the feature space
            num_basis_functions  = 32, #number of basis functions for featurizing distances
            num_residual_ao_ii  = 1, #number of residual blocks applied to output atomic features for predicting irreps of diagonal blocks (output matrix)
            num_residual_ao_ij  = 1, #number of residual blocks applied to pair features for predicting irreps of off-diagonal blocks (output matrix)
            basis_functions      = 'exp-bernstein', #type of radial basis functions (exp-gaussian/exp-bernstein/gaussian/bernstein)
            activation           = 'swish', #type of activation function used (swish/ssp)
            num_hidden_normgate_mlp = 128, #number of hidden features in the normgate MLP
            output_property_name = 'ao_matrix', # output property key of atoms dict, e.g. atoms['hamiltonian_matrix']
            load_from            = None, #if this is given the network is loaded from the specified .pth file and all other arguments are ignored
            #Zmax                 = 87 #maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default 
    ):
        super().__init__()

        self.create_graph = True  #can be set to False if the NN is only used for inference

        #store hyperparameter values
        order_max = get_max_order(orbital_basis)

        self.orbital_basis = orbital_basis
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_residual_ao_ii = num_residual_ao_ii
        self.num_residual_ao_ij = num_residual_ao_ij
        self.basis_functions = basis_functions
        self.activation = activation
        self.num_hidden_normgate_mlp = num_hidden_normgate_mlp
        self.output_property_name = output_property_name
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
        self.mix_ij = PairMixing(self.order, self.order, self.order, self.num_basis_functions, self.num_features, self.clebsch_gordan)
        self.radial_ii = nn.ModuleList([nn.Linear(self.num_basis_functions, self.num_features, bias=False)
            for L in range(self.order+1)])
        self.radial_ij = nn.ModuleList([nn.Linear(self.num_basis_functions, self.num_features, bias=False)
            for L in range(self.order+1)])
        self.residual_ao_ii = ResidualStack(self.num_residual_ao_ii, self.order, self.num_features, self.clebsch_gordan, True, self.activation, use_V2=True, num_hidden_normgate_mlp=self.num_hidden_normgate_mlp)
        self.residual_ao_ij = ResidualStack(self.num_residual_ao_ij, self.order, self.num_features, self.clebsch_gordan, True, self.activation, use_V2=True, num_hidden_normgate_mlp=self.num_hidden_normgate_mlp)

        #determine minimum number of output features based on orbitals
        #and generate dictionaries (irreps_ii/irreps_ij) that store indices 
        #for collecting the correct irreproducible representations from features
        #diagonal blocks
        number_L = [0 for L in range(2*order_max+1)] #keeps track of how many irreps of each order there are already
        self.irreps_ii = {}
        for z in self.orbital_basis.keys():
            self.irreps_ii, number_L = self.compute_matrix_irreps(
                self.orbital_basis[z], self.orbital_basis[z], self.irreps_ii, number_L)
        self.output_ii = SphericalLinear(self.order, self.num_features, 2*order_max, max(number_L), self.clebsch_gordan, zero_init=True, use_V2=True)

        #off-diagonal blocks
        number_L = [0 for L in range(2*order_max+1)] #keeps track of how many irreps of each order there are already
        self.irreps_ij = {}
        for z1 in self.orbital_basis.keys():
            for z2 in self.orbital_basis.keys():
                self.irreps_ij, number_L = self.compute_matrix_irreps(
                    self.orbital_basis[z1], self.orbital_basis[z2], self.irreps_ij, number_L)
                
        # print(f"init output ij sphlineasr with self.order={self.order}, order_max={order_max}")
        self.output_ij = SphericalLinear(self.order, self.num_features, 2*order_max, max(number_L), self.clebsch_gordan, zero_init=True, use_V2=True)



    def compute_matrix_irreps(self, orbitals_i, orbitals_j, irreps, number_L):
        """
        Given the orbitals of atom i and atom j, computes how many irreducible representations
        of each order are necessary for constructing the corresponding off-diagonal block of the matrix

        inputs:
            orbitals_i: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals of atom i
            orbitals_j: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals of atom j
            irreps: Dictionary that stores the feature indices for collecting irreducible representations
            number_L: List of length L+1 with integer entries that stores how many irreducible representations of 
                    each order are already in use
        outputs:
            irreps: Updated input dictionary
            number_L: Updated input list
        """
        for n_i, orb_i in enumerate(orbitals_i):
            z_i, _, l_i = orb_i
            for n_j, orb_j in enumerate(orbitals_j):
                z_j, _, l_j = orb_j
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
            row: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals in the row
            col: Tuple or list of tuples with integer entries (Z, _, L) that define the orbitals in the column
            irreps: list of irreducible representations of shape [batch_size, 2*L+1]
            batch_size: how many matrices are in this batch (needed to initialize matrix subblock)
        outputs:
            block: batch of matrix blocks of shape [batch_size, nrow, ncol] (nrow/ncol depends on row/col inputs)
        """
        nrow = sum((2*l+1) for _, _, l in row) #number of rows in the block
        ncol = sum((2*l+1) for _, _, l in col) #number of columns in the block
        block = torch.zeros(batch_size, nrow, ncol, device=device, dtype=dtype)

        idx = 0 #index for accessing the correct irreps
        start_i = 0
        for _, _, l_i in row:
            n_i = 2*l_i+1
            start_j = 0
            for _, _, l_j in col:
                n_j = 2*l_j+1
                for L in range(abs(l_i-l_j), l_i+l_j+1):

                    #compute inverse spherical tensor product
                    cg_matrix, _ = self.clebsch_gordan(l_i, l_j, L)  # holds coefficients up to(!) order l_i,l_j,L -> get only coefficients for order l_i,l_j,L
                    cg = cg_matrix[
                        l_i ** 2:(l_i + 1) ** 2,
                        l_j ** 2:(l_j + 1) ** 2,
                        L ** 2:(L + 1) ** 2,
                    ]
                    cg = math.sqrt(2*L+1)*cg.unsqueeze(0)
                    product = (cg*irreps[idx].unsqueeze(-2).unsqueeze(-2)).sum(-1)

                    #add product to appropriate part of the block
                    blockpart = block.narrow(-2,start_i,n_i).narrow(-1,start_j,n_j)
                    blockpart += product

                    idx += 1
                start_j += n_j
            start_i += n_i
        return block

    
    def forward(self, atoms):

        R = atoms['positions']
        fii, fij = atoms['pair_features']

        # additional layer to refine pair features for specific output matrix
        fii    = self.residual_ao_ii(fii)
        fii    = self.output_ii(fii)

        fij    = self.residual_ao_ij(fij)
        fij    = self.output_ij(fij)


        batch_size = atoms['batch_positions'].shape[0]
        num_atoms_in_batch = atoms['batch_positions'].shape[1]
        max_atom_numbers = torch.max(atoms['batch_atom_numbers'], dim=0)[0]
        num_orbitals_per_atom = {z.item(): sum((2*l+1) for _, _, l in self.orbital_basis[z.item()]) for z in torch.unique(max_atom_numbers)}
        num_orbitals = sum(num_orbitals_per_atom[z.item()] for z in max_atom_numbers)
        matrix = torch.zeros((batch_size, num_orbitals, num_orbitals), device=R.device)
        # print(f"max_atom_numbers: {max_atom_numbers}")
        # print(f"matrix shape: {matrix.shape}")
        # matrix_old = torch.zeros((batch_size, num_orbitals, num_orbitals), device=R.device)

        # time.sleep(10)

        idx_i_batch, idx_j_batch = remap_pair_idxs_for_padding(n_atoms=num_atoms_in_batch,
                                                               batch_idx_pos=atoms['batch_idx_pos'],
                                                               idx_i=atoms['idx_i_ao_matrix'],
                                                               idx_j=atoms['idx_j_ao_matrix'])
        
        # print(f"atom_numbers: {atoms['atom_numbers']}")
        # print(f"batch_atom_numbers: {atoms['batch_atom_numbers']}")
        # print(f"idx_i_batch: {idx_i_batch}")
        # print(f"idx_j_batch: {idx_j_batch}")

        # TODO if first water, than ethanol in batch, atom order is [1, 1, 1, 1, 1, 1, 8, 6, 6]
        # TODO if first ethanol, than water in batch, atom order is [1, 1, 1, 1, 1, 1, 6, 6, 8]
        # atom order determines ao offset for each atom in batch, i.e. where to place matrix block
        # ! if atom order changes, matrix block is placed at different positions in each batch
        # ! padding is performed batch-wise. check if padding in dynamic per batch but consequent within the batch

        # ao offset for each atom in batch
        ao_offsets = {0: 0}
        for i in range(1, num_atoms_in_batch):
            z_previous = max_atom_numbers[i-1].item()
            ao_offsets[i] = ao_offsets[i-1] + num_orbitals_per_atom[z_previous]


        # print(f"atom_numbers: {atoms['atom_numbers']}")
        # print(f"batch_atom_numbers: {atoms['batch_atom_numbers']}")
        # print(f"batch_atom_mask: {atoms['batch_atom_mask']}")
        # print(f"max_atom_numbers: {max_atom_numbers}")
        # print(f"idx_i_ao_matrix: {atoms['idx_i_ao_matrix']}")
        # print(f"idx_j_ao_matrix: {atoms['idx_j_ao_matrix']}")
        # print(f"idx_i_batch: {idx_i_batch}")
        # print(f"idx_j_batch: {idx_j_batch}")
        # print(f"neighbor_ao_matrix_batch_idx: {atoms['neighbor_ao_matrix_batch_idx']}")
        # print(f"atom_batch_idx: {atoms['atom_batch_idx']}")
        # print(f"orbital_basis: {self.orbital_basis}")

        # print(f"batch_idx_pos: {atoms['batch_idx_pos']}")
        # print(f"atoms keys: {atoms.keys()}")


        # TODO: off-diagonal matrix blocks V2
        # [x] get list of unique interacting atom pairs [(i,j)..] from idx_i_batch and idx_j_batch (implicitly i!=j)
        # [x] get indexes of unique interacting atom pairs (i, j) -> "atom i and j interact at indexes [idx_0, idx_1, idx_2, ...]"
        # [x] get molecule indexes in which the unique interacting atom pairs are located from neighbor_ao_matrix_batch_idx
        # [x] for each unique (i,j) pair, get fij, e.g. torch.gather(fij[L], dim=1, index=interaction_indexes[(i,j)])
        #     [x] narrow dim=-1 according to ij (irreps count)
        #     [x] construct matrix block (batch_size = len(interaction_indexes[(i,j)]))
        #     [x] add matrix block to matrix via index_add (index=list of indexes the current atom pair interacts in)



        ############################
        # batch_wise
        ############################
        # t0 = time.time()

        # off-diagonal matrix blocks
        # get unique interacting atom pairs in batch
        unique_atom_pairs_in_batch, atom_pair_index_in_batch_inv = torch.unique(torch.stack([idx_i_batch, idx_j_batch], dim=1), dim=0, return_inverse=True)
        atom_pair_index_in_batch = [torch.nonzero(atom_pair_index_in_batch_inv == i, as_tuple=True)[0] for i in range(len(unique_atom_pairs_in_batch))]
        atom_pair_molecule_index = [atoms['neighbor_ao_matrix_batch_idx'][atom_pair_index] for atom_pair_index in atom_pair_index_in_batch]

        # print(f"unique_atom_pairs_in_batch: {unique_atom_pairs_in_batch}")
        # print(f"atom_pair_index_in_batch: {atom_pair_index_in_batch}")
        # print(f"atom_pair_molecule_index: {atom_pair_molecule_index}")

        for interaction_idx, (i, j) in enumerate(unique_atom_pairs_in_batch):
            i, j = i.item(), j.item()

            irreps = []

            for n_i, orb_i in enumerate(self.orbital_basis[max_atom_numbers[i].item()]):
                z_i, _, l_i = orb_i

                for n_j, orb_j in enumerate(self.orbital_basis[max_atom_numbers[j].item()]):
                    z_j, _, l_j = orb_j

                    for L in range(abs(l_i - l_j), l_i+l_j+1):
                        ij = self.irreps_ij[(z_i, z_j, n_i, n_j, L)]
                        fij_atom_pairs_in_batch = torch.index_select(fij[L], -3, atom_pair_index_in_batch[interaction_idx])
                        ir = fij_atom_pairs_in_batch.narrow(-1, ij, 1).squeeze(-1).squeeze(0)
                        irreps.append(ir)
            
            # Calculates potentially less then batch_size many matrix blocks,
            # e.g. if atom pair interacts not in every molecule in batch
            mblock = self.matrix_block(row=self.orbital_basis[z_i.item()],
                                       col=self.orbital_basis[z_j.item()],
                                       irreps=irreps,
                                       batch_size=len(atom_pair_index_in_batch[interaction_idx]),
                                       j_gt_i=None,
                                       device=R.device,
                                       dtype=R.dtype)
            
            row_offset, col_offset = ao_offsets[i], ao_offsets[j]

            matrixpart = matrix.narrow(-2, row_offset, mblock.shape[-2]).narrow(-1, col_offset, mblock.shape[-1])
            matrixpart.index_add_(0, atom_pair_molecule_index[interaction_idx], mblock)


        # diagonal matrix blocks
        # 
        # [x] indexes for each atom in batch from batch_idx_pos, i.e. location in fii per atom
        #   apply atom_mask to max_atom_numbers?
        #
        # [x] get index list that holds in which molecules of the batch each atom is contained
        #
        # sandbox
        # which molecules contain atom i? use batch_atom_mask
        # 

        # atoms['batch_atom_mask'][3, 0] = False
        # atoms['atom_batch_idx'] = torch.cat((atoms['atom_batch_idx'][:, :9], atoms['atom_batch_idx'][:, 10:]), dim=1)
        # print(f"atom_batch_idx: {atoms['atom_batch_idx']}")

        atom_index = torch.where(atoms['batch_atom_mask'])[1]
        # print(f"atom_index: {atom_index}")

        unique_atoms_in_batch, atom_index_in_batch_inv = torch.unique(atom_index, return_inverse=True)
        atom_index_in_batch = [torch.nonzero(atom_index_in_batch_inv == i, as_tuple=True)[0] for i in range(len(unique_atoms_in_batch))]
        atom_molecule_index = [atoms['atom_batch_idx'][0][atom_idx] for atom_idx in atom_index_in_batch]
        # print(f"unique_atoms_in_batch: {unique_atoms_in_batch}")
        # print(f"atom_index_in_batch: {atom_index_in_batch}")
        # print(f"atom_molecule_index: {atom_molecule_index}")

        for atom_idx, i in enumerate(unique_atoms_in_batch):
            i = i.item()

            irreps = []

            for n_i, orb_i in enumerate(self.orbital_basis[max_atom_numbers[i].item()]):
                z_i, _, l_i = orb_i

                for n_j, orb_j in enumerate(self.orbital_basis[max_atom_numbers[i].item()]):
                    z_j, _, l_j = orb_j

                    for L in range(abs(l_i - l_j), l_i+l_j+1):
                        ii = self.irreps_ii[(z_i, z_j, n_i, n_j, L)]
                        fii_atom_in_batch = torch.index_select(fii[L], -3, atom_index_in_batch[atom_idx])
                        ir = fii_atom_in_batch.narrow(-1, ii, 1).squeeze(-1).squeeze(0)
                        irreps.append(ir)

            # Calculates potentially less then batch_size many matrix blocks,
            # e.g. if atom not in every molecule in batch
            mblock = self.matrix_block(row=self.orbital_basis[z_i.item()],
                                       col=self.orbital_basis[z_j.item()],
                                       irreps=irreps,
                                       batch_size=len(atom_index_in_batch[atom_idx]),
                                       j_gt_i=None,
                                       device=R.device,
                                       dtype=R.dtype)
            
            row_offset, col_offset = ao_offsets[i], ao_offsets[i]

            matrixpart = matrix.narrow(-2, row_offset, mblock.shape[-2]).narrow(-1, col_offset, mblock.shape[-1])
            matrixpart.index_add_(0, atom_molecule_index[atom_idx], mblock)

        # print(f"time batch-wise: {time.time() - t0}")
        # quit()


        ############################
        # molecule-wise
        ############################
        # t0 = time.time()
        # idx = 0
        # for s in range(batch_size):
        #     s_atom_numbers = atoms['batch_atom_numbers'][s]
        #     s_idx_i = idx_i_batch[atoms['neighbor_ao_matrix_batch_idx'] == s]
        #     s_idx_j = idx_j_batch[atoms['neighbor_ao_matrix_batch_idx'] == s]
        #     s_batch_idx = atoms['atom_batch_idx'][0] == s
        #     s_batch_idx_in_fii = torch.where(s_batch_idx)[0]  # idx for fii
        #     s_batch_pos = atoms['batch_idx_pos'][s_batch_idx] - (s * num_atoms_in_batch)

        #     for i, j in zip(s_idx_i, s_idx_j):

        #         irreps = []

        #         for n_i, orb_i in enumerate(self.orbital_basis[s_atom_numbers[i].item()]):
        #             z_i, _, l_i = orb_i

        #             for n_j, orb_j in enumerate(self.orbital_basis[s_atom_numbers[j].item()]):
        #                 z_j, _, l_j = orb_j

        #                 for L in range(abs(l_i - l_j), l_i+l_j+1):
        #                     ij = self.irreps_ij[(z_i, z_j, n_i, n_j, L)]
        #                     ir = fij[L].narrow(-3, idx, 1).narrow(-1, ij, 1).squeeze(-3).squeeze(-1)
        #                     irreps.append(ir)

        #         mblock = self.matrix_block(row=self.orbital_basis[z_i],
        #                                     col=self.orbital_basis[z_j],
        #                                     irreps=irreps,
        #                                     batch_size=1,
        #                                     j_gt_i=j>i,
        #                                     device=R.device,
        #                                     dtype=R.dtype)

        #         row_offset, col_offset = ao_offsets[i.item()], ao_offsets[j.item()]

        #         row_end = row_offset + mblock.shape[-2]
        #         col_end = col_offset + mblock.shape[-1]
        #         matrix_old[s, row_offset:row_end, col_offset:col_end] = mblock

        #         idx += 1

        #         # diagonal matrix blocks
        #         for i in range(len(s_batch_pos)):
        #             pos_in_batch = s_batch_pos[i]
        #             pos_in_fii = s_batch_idx_in_fii[i]

        #             irreps = []

        #             for n_i , orb_i in enumerate(self.orbital_basis[s_atom_numbers[i].item()]):
        #                 z_i, _, l_i = orb_i

        #                 for n_j , orb_j in enumerate(self.orbital_basis[s_atom_numbers[i].item()]):
        #                     z_j, _, l_j = orb_j

        #                     for L in range(abs(l_i - l_j), l_i + l_j + 1):
        #                         ii = self.irreps_ii[(z_i, z_j, n_i, n_j, L)]
        #                         irreps.append(fii[L].narrow(-3, pos_in_fii, 1).narrow(-1, ii, 1).squeeze(-3).squeeze(-1))

        #             mblock = self.matrix_block(row=self.orbital_basis[z_i],
        #                                         col=self.orbital_basis[z_j],
        #                                         irreps=irreps,
        #                                         batch_size=1,
        #                                         j_gt_i=j>i,
        #                                         device=R.device,
        #                                         dtype=R.dtype)

        #             row_offset, col_offset = ao_offsets[pos_in_batch.item()], ao_offsets[pos_in_batch.item()]

        #             row_end = row_offset + mblock.shape[-2]
        #             col_end = col_offset + mblock.shape[-1]
        #             matrix_old[s, row_offset:row_end, col_offset:col_end] = mblock

        # print(f"time molecule-wise: {time.time() - t0}")

        # print(f"matrix_old zero?: {torch.allclose(matrix_old, torch.zeros_like(matrix_old))}")
        # print(f"matrix zero?: {torch.allclose(matrix, torch.zeros_like(matrix))}")
        # print(f"similiarity: {torch.allclose(matrix_old, matrix)}")

        # train (bs=3*5) : 1.0425s -> 0.08875s => 8.513% of previous wall clock time => 11.75x speedup
        # valid (bs=3*10): 1.3976s -> 0.06110s => 4.372% of previous wall clock time => 22.87x speedup

        # matrix = matrix_old
        matrix = matrix + matrix.transpose(-2,-1) #symmetrize

        # TODO no conversion ,i.e. convention=None
        matrix = convert_ao_matrix(ao_matrix=matrix,
                                   atom_numbers=atoms['batch_atom_numbers'],
                                   convention=None)

        atoms[self.output_property_name] = matrix

        return atoms
