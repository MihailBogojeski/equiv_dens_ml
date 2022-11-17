import os
import torch
import torch.nn as nn
import numpy as np
from itertools import permutations


class ClebschGordan(nn.Module):
    """
    Helper class that stores Clebsch - Gordan coefficients
    """

    def __init__(self):
        super(ClebschGordan, self).__init__()
        tmp = np.load(os.path.join(os.path.dirname(__file__), 'clebsch_gordan_coefficients_L10.npz'), allow_pickle=True)['cg'][()]
        # add permutations (the npz file only stores coefficients for l1 <= l2 <= l3) and register buffers
        for l123 in tmp.keys():
            for a, b, c in permutations((0, 1, 2)):
                name = 'cg_{}_{}_{}'.format(l123[a], l123[b], l123[c])
                if name not in dir(self):
                    self.register_buffer(name, torch.tensor(tmp[l123].transpose(a, b, c)))

    def forward(self, l1, l2, l3):
        return getattr(self, 'cg_{}_{}_{}'.format(l1, l2, l3))


class ClebschGordanMatrix(nn.Module):
    # pack the clebsch gordan coefficients into a single matrix to enable faster tensor operations
    def __init__(self, order_max=10):
        super().__init__()
        cg_matrix, cg_masked = construct_cg_matrix(order_max)
        self.register_buffer('cg_matrix', cg_matrix)
        self.register_buffer('cg_masked', cg_masked)

    def forward(self, l1, l2, l3):
        dim1 = (l1 + 1)**2
        dim2 = (l2 + 1)**2
        dim3 = (l3 + 1)**2
        return self.cg_matrix[:dim1, :dim2, :dim3], self.cg_masked[:dim1, :dim2, :dim3]


def construct_cg_matrix(order_max):
    cg_matrix = torch.zeros((order_max + 1)**2,
                            (order_max + 1)**2,
                            (order_max + 1)**2, dtype=torch.float64)

    cg_mask = torch.zeros_like(cg_matrix)

    # load clebsch gordan coefficients from file
    tmp = np.load(os.path.join(os.path.dirname(__file__),
                               'clebsch_gordan_coefficients_L10.npz'),
                  allow_pickle=True)['cg'][()]

    print('cg_matrix shape', cg_matrix.shape)
    for l123 in tmp.keys():
        if max(l123) > order_max:
            continue
        for a, b, c in permutations((0, 1, 2)):
            # place the coefficients in the appropriate place in the matrix
            l1 = l123[a]
            l2 = l123[b]
            l3 = l123[c]
            start1 = (l1)**2
            start2 = (l2)**2
            start3 = (l3)**2
            end1 = (l1 + 1)**2
            end2 = (l2 + 1)**2
            end3 = (l3 + 1)**2
            cg_matrix[start1:end1, start2:end2, start3:end3] = torch.tensor(tmp[l123].transpose(a, b, c))

    # create a mask matrix to mask out coefficients not needed for self mixing
    for l1 in range(order_max + 1):
        start1 = (l1)**2
        for l2 in range(l1 + 1, order_max + 1):
            start2 = (l2)**2
            for l3 in range(abs(l1 - l2), min(l1 + l2, order_max) + 1):
                start3 = (l3)**2
                end1 = (l1 + 1)**2
                end2 = (l2 + 1)**2
                end3 = (l3 + 1)**2
                cg_mask[start1:end1, start2:end2, start3:end3] = 1

    # create self mixing matrix by masking the origin CG matrix
    masked_cg = cg_matrix * cg_mask

    return cg_matrix, masked_cg


def sparsify_cg_matrix(
    cg: torch.Tensor,
    ):
    """
    Convert Clebsch-Gordon tensor to sparse format.
    Args:
        cg: dense tensor Clebsch-Gordon coefficients
            [(lmax_1+1)^2, (lmax_2+1)^2, (lmax_out+1)^2]
    Returns:
        cg_sparse: vector of non-zeros CG coefficients
        idx_in_1: indices for first set of irreps
        idx_in_2: indices for second set of irreps
        idx_out: indices for output set of irreps
    """
    idx = torch.nonzero(cg)
    # print('idx', idx)
    idx_in_1, idx_in_2, idx_out = torch.split(idx, 1, dim=1)
    idx_in_1, idx_in_2, idx_out = (
        idx_in_1[:, 0],
        idx_in_2[:, 0],
        idx_out[:, 0],
    )
    # print('idx_in_1', idx_in_1)
    # print('idx_in_2', idx_in_2)
    # print('idx_out', idx_out)
    cg_sparse = cg[idx_in_1, idx_in_2, idx_out]
    return cg_sparse, idx_in_1, idx_in_2, idx_out
