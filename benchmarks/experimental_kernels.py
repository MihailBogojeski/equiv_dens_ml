"""Rejected H3-B1 experiment, retained outside production for reproducibility.

R1 CUDA measurements found this exact matrix contraction slower. Production
keeps the original broadcast kernels. These benchmark adapters share the
production state schema and retain the original coupling/normalization rules.
"""
import numpy as np
import torch
from equiv_dens.nn.modules.spherical_harmonic_layers import (
    SelfMixing as BroadcastSelfMixing, PairMixing as BroadcastPairMixing,
)

def _contract_cg_product(tp, cg, backend):
    """Contract the two input magnetic indices without changing CG conventions.

    The matrix backend avoids materializing the output-magnetic broadcast
    intermediate. ``tp`` remains shared across each pair's output degrees.
    Broadcast is retained for old models and matched performance comparisons.
    """
    if backend == 'broadcast':
        return (cg * tp).sum(-3).sum(-3)
    if backend != 'matmul':
        raise ValueError(f'Unknown spherical contraction backend: {backend}')
    product = tp.squeeze(-2).flatten(-3, -2).transpose(-2, -1)
    matrix = cg.reshape(tp.shape[-4] * tp.shape[-3], cg.shape[-2])
    dtype = torch.promote_types(product.dtype, matrix.dtype)
    return torch.matmul(product.to(dtype=dtype), matrix.to(dtype=dtype)).transpose(-2, -1)


class SelfMixing(BroadcastSelfMixing):
    def __init__(self, *args, contraction_backend="matmul", **kwargs):
        super().__init__(*args, **kwargs)
        if contraction_backend not in ("broadcast", "matmul"):
            raise ValueError(f"Unknown spherical contraction backend: {contraction_backend}")
        self.contraction_backend = contraction_backend

    def forward(self, xs):
        """
            Compute the spherical tensor product between the different orders of the same atomic features.

            Args:
                x1 (list of torch.Tensor): set of spherical features
            Returns:
                ys (list of torch.Tensor): features after self-mixing
            """
        ys = [self.keepcoeff(L) * xs[L] if L <= self.order_in else torch.zeros_like(xs[0]).repeat(*(1,) * len(xs[0].shape[:-2]), 2 * L + 1, 1) for L in range(self.order_out + 1)]
        _, cg_matrix = self.clebsch_gordan(self.order_in, self.order_in, self.order_out)
        for l1 in range(self.order_in + 1):
            x1 = xs[l1].view(*xs[l1].shape[:-2], xs[l1].size(-2), 1, 1, self.num_features)
            for l2 in range(l1 + 1, self.order_in + 1):
                x2 = xs[l2].view(*xs[l2].shape[:-2], 1, xs[l2].size(-2), 1, self.num_features)
                tp = x1 * x2
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    cg = cg_matrix[l1 ** 2:(l1 + 1) ** 2, l2 ** 2:(l2 + 1) ** 2, L ** 2:(L + 1) ** 2]
                    cg = cg.view((*(1,) * len(tp.shape[:-4]), *cg.shape, 1))
                    coeff = self.mixcoeff(l1, l2, L).view(*(1,) * len(tp.shape[:-4]), 1, -1)
                    ys[L] = ys[L] + coeff * _contract_cg_product(tp, cg, getattr(self, 'contraction_backend', 'broadcast'))
        return ys


class PairMixing(BroadcastPairMixing):
    def __init__(self, *args, contraction_backend="matmul", **kwargs):
        super().__init__(*args, **kwargs)
        if contraction_backend not in ("broadcast", "matmul"):
            raise ValueError(f"Unknown spherical contraction backend: {contraction_backend}")
        self.contraction_backend = contraction_backend

    def forward(self, x1s, x2s, rbf):
        """
            Compute the spherical tensor product between two spherical features by mixing orders.

            Args:
                x1s (list of torch.Tensor): first set of spherical features
                x2s (list of torch.Tensor): second set of spherical features
                rbf (torch.Tensor): radial basis function
            Returns:
                ys (list of torch.Tensor): features after pair interaction
            """
        ys = [torch.zeros_like(x1s[0]).repeat(*(1,) * len(x1s[0].shape[:-2]), 2 * L + 1, 1) for L in range(self.order_out + 1)]
        if self.clebsch_gordan is not None:
            cg_matrix, _ = self.clebsch_gordan(self.order_in1, self.order_in2, self.order_out)
        else:
            cg_matrix = torch.ones((1, 1, 1), device=x1s[0].device, dtype=x1s[0].dtype)
        for l1 in range(self.order_in1 + 1):
            x1 = x1s[l1].view(*x1s[l1].shape[:-2], x1s[l1].size(-2), 1, 1, self.num_features)
            for l2 in range(self.order_in2 + 1):
                x2 = x2s[l2].view(*x2s[l2].shape[:-2], 1, x2s[l2].size(-2), 1, self.num_features)
                tp = x1 * x2
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    cg = cg_matrix[l1 ** 2:(l1 + 1) ** 2, l2 ** 2:(l2 + 1) ** 2, L ** 2:(L + 1) ** 2]
                    cg = cg.view((*(1,) * len(tp.shape[:-4]), *cg.shape, 1))
                    if self.normalize:
                        norm_factor = np.sqrt(L + 1) * np.sqrt(self.num_features / self.num_basis_functions)
                    else:
                        norm_factor = 1
                    ys[L] = ys[L] + self.coeff(l1, l2, L)(rbf) * norm_factor * (_contract_cg_product(tp, cg, getattr(self, 'contraction_backend', 'broadcast')) / np.sqrt(self.L_count[L]))
        return ys


def install_matrix_experiment(model):
    """Opt in only from a benchmark; add no parameters/state-dict entries."""
    for module in model.modules():
        if isinstance(module, BroadcastSelfMixing):
            module.__class__ = SelfMixing
            module.contraction_backend = "matmul"
        elif isinstance(module, BroadcastPairMixing):
            module.__class__ = PairMixing
            module.contraction_backend = "matmul"
    return model
