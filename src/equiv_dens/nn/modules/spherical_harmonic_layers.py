import torch
import torch.nn as nn
import numpy as np


class SphericalLinear(nn.Module):
    """
    Like a linear layer, but acting on spherical harmonic features (optionally mixes features)
    """

    def __init__(
        self,
        order_in,
        num_in,
        order_out,
        num_out,
        clebsch_gordan=None,
        mix_orders=True,
        bias=True,
        zero_init=False,
    ):
        super(SphericalLinear, self).__init__()
        self.order_in = order_in
        self.num_in = num_in
        self.order_out = order_out
        self.num_out = num_out
        self.bias = bias
        self.mix_orders = mix_orders
        self.zero_init = zero_init
        if self.mix_orders:
            assert (
                clebsch_gordan is not None
            )  # Clebsch-Gordan coefficients are necessary for mixing
            self.mixing = SelfMixing(
                self.order_in, self.order_out, self.num_in, clebsch_gordan
            )
        else:  # order can only be changed if mixing is enabled
            assert order_in == order_out
        self.linear = nn.ModuleList(
            [
                nn.Linear(self.num_in, self.num_out, bias=(self.bias and L == 0))
                for L in range(self.order_out + 1)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.zero_init:
            for L in range(self.order_out + 1):
                nn.init.zeros_(self.linear[L].weight)
        else:
            for L in range(self.order_out + 1):
                nn.init.orthogonal_(self.linear[L].weight)
        if self.bias:
            nn.init.zeros_(self.linear[0].bias)

    def forward(self, xs):
        if self.mix_orders:
            ys = self.mixing(xs)
            for L in range(self.order_out + 1):
                ys[L] = self.linear[L](ys[L])
        else:
            ys = []
            for x, linear in zip(xs, self.linear):
                ys.append(linear(x))
        return ys


class SelfMixing(nn.Module):
    """
    Mixes features of different orders
    """

    def __init__(self, order_in, order_out, num_features, clebsch_gordan):
        super(SelfMixing, self).__init__()
        self.order_in = order_in
        self.order_out = order_out
        self.num_features = num_features
        self.clebsch_gordan = clebsch_gordan
        # coefficients for mixing
        for l1 in range(self.order_in + 1):
            for l2 in range(l1 + 1, self.order_in + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    name = "mixcoeff_{}_{}_{}".format(l1, l2, L)
                    self.register_parameter(
                        name, nn.Parameter(torch.Tensor(self.num_features))
                    )
        for L in range(min(self.order_in, self.order_out) + 1):
            name = "keepcoeff_{}".format(L)
            self.register_parameter(name, nn.Parameter(torch.Tensor(self.num_features)))
        self.reset_parameters()

    def reset_parameters(self):
        count = [0 for L in range(self.order_out + 1)]
        for L in range(min(self.order_in, self.order_out) + 1):
            count[L] += 1
        for l1 in range(self.order_in + 1):
            for l2 in range(l1 + 1, self.order_in + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    count[L] += 1

        for L in range(min(self.order_in, self.order_out) + 1):
            nn.init.uniform_(
                self.keepcoeff(L), a=-np.sqrt(3 / count[L]), b=np.sqrt(3 / count[L])
            )

        for l1 in range(self.order_in + 1):
            for l2 in range(l1 + 1, self.order_in + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    nn.init.uniform_(
                        self.mixcoeff(l1, l2, L),
                        a=-np.sqrt(3 / count[L]),
                        b=np.sqrt(3 / count[L]),
                    )

    def keepcoeff(self, L):
        return getattr(self, "keepcoeff_{}".format(L))

    def mixcoeff(self, l1, l2, L):
        return getattr(self, "mixcoeff_{}_{}_{}".format(l1, l2, L))

    def forward(self, xs):
        # initialize output
        ys = [
            self.keepcoeff(L) * xs[L]
            if L <= self.order_in
            else torch.zeros_like(xs[0]).repeat(
                *(1,) * len(xs[0].shape[:-2]), 2 * L + 1, 1
            )
            for L in range(self.order_out + 1)
        ]
        _, cg_matrix = self.clebsch_gordan(self.order_in, self.order_in, self.order_out)
        # loop over all combinations of orders
        for l1 in range(self.order_in + 1):
            # get view of x[l1] that enables broadcasting to compute the spherical tensor product
            x1 = xs[l1].view(
                *xs[l1].shape[:-2], xs[l1].size(-2), 1, 1, self.num_features
            )
            for l2 in range(l1 + 1, self.order_in + 1):
                # get view of x[l2] that enables broadcasting to compute the spherical tensor product
                x2 = xs[l2].view(
                    *xs[l2].shape[:-2], 1, xs[l2].size(-2), 1, self.num_features
                )
                # compute spherical tensor product
                tp = x1 * x2
                # decompose tensor product into irreducible representations and collect contributions
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    # get Clebsch - Gordan coefficients in broadcastable form
                    cg = cg_matrix[
                        l1 ** 2:(l1 + 1) ** 2,
                        l2 ** 2:(l2 + 1) ** 2,
                        L ** 2:(L + 1) ** 2,
                    ]
                    cg = cg.view((*(1,) * len(tp.shape[:-4]), *cg.shape, 1))
                    # get coefficients in broadcastable form
                    coeff = self.mixcoeff(l1, l2, L).view(
                        *(1,) * len(tp.shape[:-4]), 1, -1
                    )
                    # contract and add
                    ys[L] = ys[L] + coeff * ((cg * tp).sum(-3).sum(-3))
        return ys


class PairMixing(nn.Module):
    """
    Mixes pairs of atomic features in a distance dependent way using a learnable radial function
    and outputs pair features
    """

    def __init__(
        self,
        order_in1,
        order_in2,
        order_out,
        num_basis_functions,
        num_features,
        clebsch_gordan,
    ):
        super(PairMixing, self).__init__()
        self.order_in1 = order_in1
        self.order_in2 = order_in2
        self.order_out = order_out
        self.num_basis_functions = num_basis_functions
        self.num_features = num_features
        self.clebsch_gordan = clebsch_gordan
        # distance - dependent coefficients for mixing
        for l1 in range(self.order_in1 + 1):
            for l2 in range(self.order_in2 + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    name = "coeff_{}_{}_{}".format(l1, l2, L)
                    self.add_module(
                        name,
                        nn.Linear(
                            self.num_basis_functions, self.num_features, bias=False
                        ),
                    )
        self.reset_parameters()

    def reset_parameters(self):
        for l1 in range(self.order_in1 + 1):
            for l2 in range(self.order_in2 + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    nn.init.orthogonal_(self.coeff(l1, l2, L).weight)

    def coeff(self, l1, l2, L):
        return getattr(self, "coeff_{}_{}_{}".format(l1, l2, L))

    def forward(self, x1s, x2s, rbf):
        # initialize output to zeros
        ys = [
            torch.zeros_like(x1s[0]).repeat(
                *(1,) * len(x1s[0].shape[:-2]), 2 * L + 1, 1
            )
            for L in range(self.order_out + 1)
        ]
        if self.clebsch_gordan is not None:
            cg_matrix, _ = self.clebsch_gordan(
                self.order_in1, self.order_in1, self.order_out
            )
        else:
            cg_matrix = torch.ones((1, 1, 1)).to(x1s[0])
        # loop over all combinations of orders
        for l1 in range(self.order_in1 + 1):
            # get view of x1s[l1] that enables broadcasting to compute the spherical tensor product
            x1 = x1s[l1].view(
                *x1s[l1].shape[:-2], x1s[l1].size(-2), 1, 1, self.num_features
            )
            for l2 in range(self.order_in2 + 1):
                # get view of x2s[l2] that enables broadcasting to compute the spherical tensor product
                x2 = x2s[l2].view(
                    *x2s[l2].shape[:-2], 1, x2s[l2].size(-2), 1, self.num_features
                )
                # compute spherical tensor product
                tp = x1 * x2
                # decompose tensor product into irreducible representations and collect contributions
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    # get Clebsch - Gordan coefficients in broadcastable form
                    cg = cg_matrix[
                        l1 ** 2:(l1 + 1) ** 2,
                        l2 ** 2:(l2 + 1) ** 2,
                        L ** 2:(L + 1) ** 2,
                    ]
                    cg = cg.view((*(1,) * len(tp.shape[:-4]), *cg.shape, 1))
                    # contract and add
                    ys[L] = ys[L] + self.coeff(l1, l2, L)(rbf) * (
                        (cg * tp).sum(-3).sum(-3)
                    )
        return ys
