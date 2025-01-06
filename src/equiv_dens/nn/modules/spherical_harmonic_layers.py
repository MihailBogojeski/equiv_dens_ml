import torch
import torch.nn as nn
import numpy as np
# import equiv_dens.utils.base as utils
# from equiv_dens.nn.modules.clebsch_gordan import sparsify_cg_matrix


class SphericalLinear(nn.Module):
    """Like a linear layer, but acting on spherical harmonic features (optionally mixes features)."""

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
        normalize=0,
        parity=False,
        use_V2=False,
    ):
        """
        Initialize a SphericalLinear layer.

        Args:
            order_in (int): order of the input spherical features
            num_in (int): dimensionality of the input features
            order_out (int): order of the output spherical features
            num_out (int): dimensionality of the output features
            clebsch_gordan (torch.Tensor): Clebsch-Gordan coefficients
            mix_orders (bool): whether to mix input features
            bias (bool): whether to use bias
            zero_init (bool): whether to initialize with zeros
            normalize (int): normalization of the output features
            parity (bool): whether to preserve parity of spherical features
            use_V2 (bool): whether to use the SelfMixingV2 layer
        """
        super().__init__()
        self.order_in = order_in
        self.num_in = num_in
        self.order_out = order_out
        self.num_out = num_out
        self.bias = bias
        self.mix_orders = mix_orders
        self.zero_init = zero_init
        self.parity = parity
        if self.mix_orders:
            assert (
                clebsch_gordan is not None
            )  # Clebsch-Gordan coefficients are necessary for mixing
            if use_V2:
                self.mixing = SelfMixingV2(
                    self.order_in, self.order_out, self.num_in, clebsch_gordan, normalize, parity,
                )
            else:
                self.mixing = SelfMixing(
                    self.order_in, self.order_out, self.num_in, clebsch_gordan, normalize, parity,
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
        """Initialize/reset the layer's parameters."""
        if self.zero_init:
            for L in range(self.order_out + 1):
                nn.init.zeros_(self.linear[L].weight)
        else:
            for L in range(self.order_out + 1):
                nn.init.orthogonal_(self.linear[L].weight)
        if self.bias:
            nn.init.zeros_(self.linear[0].bias)

    def forward(self, xs):
        """
        Forward pass of the layer.

        Args:
            xs (list of torch.Tensor): input features
        Returns:
            ys (list of torch.Tensor): output features
        """
        if self.mix_orders:
            ys = self.mixing(xs)
            for L in range(self.order_out + 1):
                ys[L] = self.linear[L](ys[L])
        else:
            ys = []
            for x, linear in zip(xs, self.linear):
                ys.append(linear(x) * np.sqrt(self.num_out))
        return ys


class SelfMixing(nn.Module):
    """Mixes spherical features of different orders."""

    def __init__(self, order_in,
                 order_out, num_features,
                 clebsch_gordan, normalize=0, parity=False):
        """
        Initialize a SelfMixing layer.

        Args:
            order_in (int): order of the input spherical features
            order_out (int): order of the output spherical features
            num_features (int): dimensionality of the input features
            clebsch_gordan (torch.Tensor): Clebsch-Gordan coefficients
            normalize (int): normalization of the output features
            parity (bool): whether to preserve parity of spherical features
        """
        super().__init__()
        self.order_in = order_in
        self.order_out = order_out
        self.num_features = num_features
        self.clebsch_gordan = clebsch_gordan
        self.normalize = normalize
        self.parity = parity
        # coefficients for mixing
        for l1 in range(self.order_in + 1):
            for l2 in range(l1 + 1, self.order_in + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    name = "mixcoeff_{}_{}_{}".format(l1, l2, L)
                    self.register_parameter(
                        name, nn.Parameter(torch.Tensor(self.num_features))
                    )
        for L in range(min(self.order_in, self.order_out) + 1):
            name = "keepcoeff_{}".format(L)
            self.register_parameter(name, nn.Parameter(torch.Tensor(self.num_features)))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize/reset the layer's parameters."""
        count = [1 for L in range(self.order_out + 1)]
        # for L in range(min(self.order_in, self.order_out) + 1):
        #     count[L] += 1
        for l1 in range(self.order_in + 1):
            for l2 in range(l1 + 1, self.order_in + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    count[L] += 1

        # print('count L', count)
        for L in range(min(self.order_in, self.order_out) + 1):
            if self.normalize:
                norm_factor = (L + 1)
            else:
                norm_factor = 1
            nn.init.uniform_(
                self.keepcoeff(L), a=-np.sqrt(3 * norm_factor/count[L]), b=np.sqrt(3 * norm_factor/count[L])
            )

        for l1 in range(self.order_in + 1):
            for l2 in range(l1 + 1, self.order_in + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    if self.normalize:
                        norm_factor = (L + 1)
                    else:
                        norm_factor = 1
                    nn.init.uniform_(
                        self.mixcoeff(l1, l2, L),
                        a=-np.sqrt(3 * norm_factor/count[L]),
                        b=np.sqrt(3 * norm_factor/count[L]),
                    )

    def keepcoeff(self, L):
        """
        Get the mixing coefficients for the given order.

        Args:
            L (int): output order
        Returns:
            keepcoeff (torch.Tensor): mixing coefficients
        """
        return getattr(self, "keepcoeff_{}".format(L))

    def mixcoeff(self, l1, l2, L):
        """
        Get the mixing coefficients for the given orders.

        Args:
            l1 (int): first order
            l2 (int): second order
            L (int): output order
        Returns:
            mixcoeff (torch.Tensor): mixing coefficients
        """
        return getattr(self, "mixcoeff_{}_{}_{}".format(l1, l2, L))

    def forward(self, xs):
        """
        Compute the spherical tensor product between the different orders of the same atomic features.

        Args:
            x1 (list of torch.Tensor): set of spherical features
        Returns:
            ys (list of torch.Tensor): features after self-mixing
        """
        # print('in', self.order_in, 'out', self.order_out)
        # print('xs norm selfmix', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
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
        # print('ys', [float(torch.mean(ys[L]**2)) for L in range(len(ys))])
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
                # print('tp norm', float(torch.mean(tp ** 2)))
                # decompose tensor product into irreducible representations and collect contributions
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
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
                    # print('L', L)
                    # print('cg * tp norm', float(torch.mean((coeff * (cg * tp).sum(-3).sum(-3)) ** 2)))
                    # print('norm coeff', float(torch.mean((coeff / np.sqrt(2 * L + 1)) ** 2)))
                    ys[L] = ys[L] + coeff * ((cg * tp).sum(-3).sum(-3))
        # print('ys norm selfmix', [float(torch.mean(ys[L]**2)) for L in range(len(ys))])
        return ys
    

class SelfMixingV2(nn.Module):
    """Mixes spherical features of different orders."""

    def __init__(self, order_in,
                 order_out, num_features,
                 clebsch_gordan, normalize=0, parity=False):
        """
        Initialize a SelfMixing layer with mixing coefficients as matrix.

        Args:
            order_in (int): order of the input spherical features
            order_out (int): order of the output spherical features
            num_features (int): dimensionality of the input features
            clebsch_gordan (torch.Tensor): Clebsch-Gordan coefficients
            normalize (int): normalization of the output features
            parity (bool): whether to preserve parity of spherical features
        """
        super().__init__()
        self.order_in = order_in
        self.order_out = order_out
        self.num_features = num_features
        self.clebsch_gordan = clebsch_gordan
        self.normalize = normalize
        self.parity = parity
        # coefficients for mixing
        for l1 in range(self.order_in + 1):
            for l2 in range(l1 + 1, self.order_in + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    name = "mixcoeff_{}_{}_{}".format(l1, l2, L)
                    self.add_module(
                        name,
                        nn.Linear(
                            self.num_features, self.num_features, bias=False
                        ),
                    )
        for L in range(min(self.order_in, self.order_out) + 1):
            name = "keepcoeff_{}".format(L)
            self.register_parameter(name, nn.Parameter(torch.Tensor(self.num_features)))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize/reset the layer's parameters."""
        count = [1 for L in range(self.order_out + 1)]
        # for L in range(min(self.order_in, self.order_out) + 1):
        #     count[L] += 1
        for l1 in range(self.order_in + 1):
            for l2 in range(l1 + 1, self.order_in + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    count[L] += 1

        # print('count L', count)
        for L in range(min(self.order_in, self.order_out) + 1):
            if self.normalize:
                norm_factor = (L + 1)
            else:
                norm_factor = 1
            nn.init.uniform_(
                self.keepcoeff(L), a=-np.sqrt(3 * norm_factor/count[L]), b=np.sqrt(3 * norm_factor/count[L])
            )

        for l1 in range(self.order_in + 1):
            for l2 in range(l1 + 1, self.order_in + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    if self.normalize:
                        norm_factor = (L + 1)
                    else:
                        norm_factor = 1
                    # nn.init.uniform_(
                    #     self.mixcoeff(l1, l2, L),
                    #     a=-np.sqrt(3 * norm_factor/count[L]),
                    #     b=np.sqrt(3 * norm_factor/count[L]),
                    # )
                    nn.init.orthogonal_(self.mixcoeff(l1, l2, L).weight)

    def keepcoeff(self, L):
        """
        Get the mixing coefficients for the given order.

        Args:
            L (int): output order
        Returns:
            keepcoeff (torch.Tensor): mixing coefficients
        """
        return getattr(self, "keepcoeff_{}".format(L))

    def mixcoeff(self, l1, l2, L):
        """
        Get the mixing coefficients for the given orders.

        Args:
            l1 (int): first order
            l2 (int): second order
            L (int): output order
        Returns:
            mixcoeff (torch.Tensor): mixing coefficients
        """
        return getattr(self, "mixcoeff_{}_{}_{}".format(l1, l2, L))

    def forward(self, xs):
        """
        Compute the spherical tensor product between the different orders of the same atomic features.

        Args:
            x1 (list of torch.Tensor): set of spherical features
        Returns:
            ys (list of torch.Tensor): features after self-mixing
        """
        # print('in', self.order_in, 'out', self.order_out)
        # print('xs norm selfmix', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
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
        # print('ys', [float(torch.mean(ys[L]**2)) for L in range(len(ys))])
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
                # print('tp norm', float(torch.mean(tp ** 2)))
                # decompose tensor product into irreducible representations and collect contributions
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    # get Clebsch - Gordan coefficients in broadcastable form
                    cg = cg_matrix[
                        l1 ** 2:(l1 + 1) ** 2,
                        l2 ** 2:(l2 + 1) ** 2,
                        L ** 2:(L + 1) ** 2,
                    ]
                    cg = cg.view((*(1,) * len(tp.shape[:-4]), *cg.shape, 1))
                    # get coefficients in broadcastable form
                    # coeff = self.mixcoeff(l1, l2, L).view(
                    #     *(1,) * len(tp.shape[:-4]), 1, -1
                    # )

                    # contract and add
                    # print('L', L)
                    # print('cg * tp norm', float(torch.mean((coeff * (cg * tp).sum(-3).sum(-3)) ** 2)))
                    # print('norm coeff', float(torch.mean((coeff / np.sqrt(2 * L + 1)) ** 2)))
                    # ys[L] = ys[L] + coeff * ((cg * tp).sum(-3).sum(-3))
                    ys[L] = ys[L] + self.mixcoeff(l1, l2, L)((cg * tp).sum(-3).sum(-3))
        # print('ys norm selfmix', [float(torch.mean(ys[L]**2)) for L in range(len(ys))])
        return ys


class PairMixing(nn.Module):
    """Mixes pairs of atomic features in an optionally distance dependent way and outputs pair features."""

    def __init__(
        self,
        order_in1,
        order_in2,
        order_out,
        num_basis_functions,
        num_features,
        clebsch_gordan,
        normalize=0,
        parity=False,
        distance_dependent=True,
    ):
        """
        Initialize the PairMixing object.

        Args:
            order_in1 (int): order of first input spherical harmonics
            order_in2 (int): order of second input spherical harmonics
            order_out (int): order of output spherical harmonics
            num_basis_functions (int): number of basis functions
            num_features (int): dimensionality of the feature space
            clebsch_gordan (torch.Tensor): Clebsch-Gordan coefficients
            normalize (int): whether to normalize the output features
            parity (bool): whether to preserve parity of spherical features
            distance_dependent (bool): whether to use distance dependent coefficients. Set to False for mixing atomic features of the same atom as in `DiagonalPair`.
        """
        super().__init__()
        self.order_in1 = order_in1
        self.order_in2 = order_in2
        self.order_out = order_out
        self.num_basis_functions = num_basis_functions
        self.num_features = num_features
        self.clebsch_gordan = clebsch_gordan
        self.normalize = normalize
        self.parity = parity
        self.distance_dependent = distance_dependent
        # distance - dependent coefficients for mixing
        for l1 in range(self.order_in1 + 1):
            for l2 in range(self.order_in2 + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    name = "coeff_{}_{}_{}".format(l1, l2, L)
                    if self.distance_dependent:
                        self.add_module(
                            name,
                            nn.Linear(
                                self.num_basis_functions, self.num_features, bias=False
                            ),
                        )
                    else:
                        self.add_module(
                            name,
                            nn.Linear(
                                self.num_features, self.num_features, bias=False
                            ),
                        )
        self.L_count = [0 for L in range(self.order_out + 1)]
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize/reset the layer's parameters."""
        for l1 in range(self.order_in1 + 1):
            for l2 in range(self.order_in2 + 1):
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    nn.init.orthogonal_(self.coeff(l1, l2, L).weight)
                    self.L_count[L] += 1

    def coeff(self, l1, l2, L):
        """
        Retrieve the learneable mixing coefficients.

        Args:
            l1 (int): order of first input spherical harmonics
            l2 (int): order of second input spherical harmonics
            L (int): order of output spherical harmonics
        Returns:
            coeff (torch.Tensor): mixing coefficients
        """
        return getattr(self, "coeff_{}_{}_{}".format(l1, l2, L))

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
        # initialize output to zeros
        ys = [
            torch.zeros_like(x1s[0]).repeat(
                *(1,) * len(x1s[0].shape[:-2]), 2 * L + 1, 1
            )
            for L in range(self.order_out + 1)
        ]
        if self.clebsch_gordan is not None:
            cg_matrix, _ = self.clebsch_gordan(
                self.order_in1, self.order_in2, self.order_out
            )
        else:
            cg_matrix = torch.ones((1, 1, 1)).to(x1s[0])
        # loop over all combinations of orders
        # print('xs1 pairmix norm', [float(torch.mean(x1s[L] ** 2)) for L in range(len(x1s))])
        # print('xs2 pairmix norm', [float(torch.mean(x2s[L] ** 2)) for L in range(len(x2s))])
        # print('L_counts', self.L_count)
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
                # print('ls', l1, l2)
                # print('x1 norm', float(torch.mean(x1 ** 2)))
                # print('x2 norm', float(torch.mean(x2 ** 2)))
                tp = x1 * x2
                # print('tp norm', float(torch.mean(tp ** 2)))
                # decompose tensor product into irreducible representations and collect contributions
                for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
                    if l1 + l2 % 2 != L % 2 and self.parity:
                        continue
                    # get Clebsch - Gordan coefficients in broadcastable form

                    cg = cg_matrix[
                        l1 ** 2:(l1 + 1) ** 2,
                        l2 ** 2:(l2 + 1) ** 2,
                        L ** 2:(L + 1) ** 2,
                    ]
                    cg = cg.view((*(1,) * len(tp.shape[:-4]), *cg.shape, 1))
                    # contract and addi
                    # print("L", L)
                    # print('cg * tp norm', float(torch.mean((self.coeff(l1, l2, L)(rbf) * np.sqrt(2* L + 1) * (cg * tp).sum(-3).sum(-3)) ** 2)))
                    # print('coeff norm', float(torch.mean((self.coeff(l1, l2, L)(rbf) * np.sqrt(self.num_features/self.num_basis_functions)) ** 2)))
                    # print('num basis functions', self.num_basis_functions)
                    # print('num features', self.num_features)
                    if self.distance_dependent:
                        if self.normalize:
                            norm_factor = np.sqrt(L + 1) * np.sqrt(self.num_features / self.num_basis_functions)
                        else:
                            norm_factor = 1

                        if rbf.shape[-2] == 1:
                            rbf = rbf.repeat(1, 1, len(ys), 1)

                        ys[L] = ys[L] + self.coeff(l1, l2, L)(rbf[:, :, [L], :]) * (norm_factor) * (
                            (cg * tp).sum(-3).sum(-3) / np.sqrt(self.L_count[L])
                        )
                        # TODO no normalize, no sqrt also for distant dependent path?
                    else:
                        ys[L] = ys[L] + self.coeff(l1, l2, L)((cg * tp).sum(-3).sum(-3))

        # print('ys pairmix norm', [float(torch.mean(ys[L] ** 2)) for L in range(len(ys))])
        return ys


class PairInteraction(nn.Module):
    """Lets a pair of atomic features interact in a distance dependent way, without mixing different orders."""

    def __init__(
        self,
        order,
        num_basis_functions,
        num_features,
        clebsch_gordan,
        normalize=0,
    ):
        """
        Initialize the PairInteraction object.

        Args:
            order (int): order of spherical harmonics
            num_basis_functions (int): number of basis functions
            num_features (int): dimensionality of the feature space
            clebsch_gordan (torch.Tensor): Clebsch-Gordan coefficients
            normalize (int): whether to normalize the output features
        """
        super().__init__()
        self.order = order
        self.num_basis_functions = num_basis_functions
        self.num_features = num_features
        self.clebsch_gordan = clebsch_gordan
        self.normalize = normalize
        # distance - dependent coefficients for mixing
        for L in range(self.order + 1):
            name = "coeff_{}".format(L)
            self.add_module(
                name,
                nn.Linear(
                    self.num_basis_functions, self.num_features, bias=False
                ),
            )
        self.L_count = [0 for L in range(self.order + 1)]
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize/reset the layer's parameters."""
        for L in range(self.order + 1):
            nn.init.orthogonal_(self.coeff(L).weight)
            self.L_count[L] += 1

    def coeff(self, L):
        """
        Retrieve the learneable coefficients for the order L.

        Args:
            L (int): order of spherical harmonics
        Returns:
            coeff (torch.Tensor): mixing coefficients
        """
        return getattr(self, "coeff_{}".format(L))

    def forward(self, x1s, x2s, rbf):
        """
        Compute the spherical tensor product between the same orders only.

        Args:
            x1s (list of torch.Tensor): first set of spherical features
            x2s (list of torch.Tensor): second set of spherical features
            rbf (torch.Tensor): radial basis function
        Returns:
            ys (list of torch.Tensor): features after pair interaction
        """
        # initialize output to zeros
        ys = [
            torch.zeros_like(x1s[0]).repeat(
                *(1,) * len(x1s[0].shape[:-2]), 2 * L + 1, 1
            )
            for L in range(self.order + 1)
        ]
        if self.clebsch_gordan is not None:
            cg_matrix, _ = self.clebsch_gordan(
                self.order, self.order, self.order
            )
        else:
            cg_matrix = torch.ones((1, 1, 1)).to(x1s[0])
        # loop over all combinations of orders
        # print('xs1 pairmix norm', [float(torch.mean(x1s[L] ** 2)) for L in range(len(x1s))])
        # print('xs2 pairmix norm', [float(torch.mean(x2s[L] ** 2)) for L in range(len(x2s))])
        # print('L_counts', self.L_count)
        for L in range(self.order + 1):
            # get view of x1s[l1] that enables broadcasting to compute the spherical tensor product
            x1 = x1s[L].view(
                *x1s[L].shape[:-2], x1s[L].size(-2), 1, 1, self.num_features
            )
            # get view of x2s[l2] that enables broadcasting to compute the spherical tensor product
            x2 = x2s[L].view(
                *x2s[L].shape[:-2], 1, x2s[L].size(-2), 1, self.num_features
            )
            # compute spherical tensor product
            # print('ls', l1, l2)
            # print('x1 norm', float(torch.mean(x1 ** 2)))
            # print('x2 norm', float(torch.mean(x2 ** 2)))
            tp = x1 * x2
            # print('tp norm', float(torch.mean(tp ** 2)))
            # decompose tensor product into irreducible representations and collect contributions

            cg = cg_matrix[
                L ** 2:(L + 1) ** 2,
                L ** 2:(L + 1) ** 2,
                L ** 2:(L + 1) ** 2,
            ]
            cg = cg.view((*(1,) * len(tp.shape[:-4]), *cg.shape, 1))
            # contract and addi
            # print("L", L)
            # print('cg * tp norm', float(torch.mean((self.coeff(l1, l2, L)(rbf) * np.sqrt(2* L + 1) * (cg * tp).sum(-3).sum(-3)) ** 2)))
            # print('coeff norm', float(torch.mean((self.coeff(l1, l2, L)(rbf) * np.sqrt(self.num_features/self.num_basis_functions)) ** 2)))
            # print('num basis functions', self.num_basis_functions)
            # print('num features', self.num_features)
            if self.normalize:
                norm_factor = np.sqrt(L + 1) * np.sqrt(self.num_features / self.num_basis_functions)
            else:
                norm_factor = 1
            ys[L] = ys[L] + self.coeff(L)(rbf) * (norm_factor) * (
                (cg * tp).sum(-3).sum(-3) / np.sqrt(self.L_count[L])
            )
        # print('ys pairmix norm', [float(torch.mean(ys[L] ** 2)) for L in range(len(ys))])
        return ys
# class SelfMixing_new(nn.Module):
#     """
#     Mixes features of different orders
#     """
#
#     def __init__(self, order_in, order_out, num_features, clebsch_gordan, normalize=0):
#         super().__init__()
#         self.order_in = order_in
#         self.order_out = order_out
#         self.num_features = num_features
#         self.clebsch_gordan = clebsch_gordan
#         self.normalize = normalize
#         # coefficients for mixing
#         for l1 in range(self.order_in + 1):
#             for l2 in range(l1 + 1, self.order_in + 1):
#                 for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
#                     name = "mixcoeff_{}_{}_{}".format(l1, l2, L)
#                     self.register_parameter(
#                         name, nn.Parameter(torch.Tensor(self.num_features))
#                     )
#         for L in range(min(self.order_in, self.order_out) + 1):
#             name = "keepcoeff_{}".format(L)
#             self.register_parameter(name, nn.Parameter(torch.Tensor(self.num_features)))
#         _, cg_matrix = self.clebsch_gordan(self.order_in, self.order_in, self.order_out)
#         cg_matrix, in_idx_1, in_idx_2, out_idx = sparsify_cg_matrix(cg_matrix) 
#         self.register_buffer("in_idx_1", in_idx_1, persistent=False)
#         self.register_buffer("in_idx_2", in_idx_2, persistent=False)
#         self.register_buffer("out_idx",  out_idx, persistent=False)
#         self.register_buffer("cg_matrix", cg_matrix, persistent=False)
#         mix_idx_1 = torch.floor(torch.sqrt(in_idx_1)).to(torch.long)
#         mix_idx_2 = torch.floor(torch.sqrt(in_idx_2)).to(torch.long)
#         mix_idx_out = torch.floor(torch.sqrt(out_idx)).to(torch.long)
#         self.register_buffer("mix_idx_1", mix_idx_1, persistent=False)
#         self.register_buffer("mix_idx_2", mix_idx_2, persistent=False)
#         self.register_buffer("mix_idx_out",  mix_idx_out, persistent=False)
#         self.register_parameter(
#             'mixcoeffs', nn.Parameter(torch.Tensor(self.order_in + 1, self.order_in + 1, self.order_out + 1, self.num_features))
#         )
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         count = [1 for L in range(self.order_out + 1)]
#         # for L in range(min(self.order_in, self.order_out) + 1):
#         #     count[L] += 1
#         for l1 in range(self.order_in + 1):
#             for l2 in range(l1 + 1, self.order_in + 1):
#                 for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
#                     count[L] += 1
#
#         # print('count L', count)
#         for L in range(min(self.order_in, self.order_out) + 1):
#             if self.normalize:
#                 norm_factor = (L + 1)
#             else:
#                 norm_factor = 1
#             nn.init.uniform_(
#                 self.keepcoeff(L), a=-np.sqrt(3 * norm_factor/count[L]), b=np.sqrt(3 * norm_factor/count[L])
#             )
#
#         for l1 in range(self.order_in + 1):
#             for l2 in range(l1 + 1, self.order_in + 1):
#                 for L in range(abs(l1 - l2), min(l1 + l2, self.order_out) + 1):
#                     if self.normalize:
#                         norm_factor = (L + 1)
#                     else:
#                         norm_factor = 1
#                     # nn.init.uniform_(
#                     #     self.mixcoeff(l1, l2, L),
#                     #     a=-np.sqrt(3 * norm_factor/count[L]),
#                     #     b=np.sqrt(3 * norm_factor/count[L]),
#                     # )
#                     nn.init.uniform_(
#                         self.mixcoeffs[l1, l2, L, :],
#                         a=-np.sqrt(3 * norm_factor/count[L]),
#                         b=np.sqrt(3 * norm_factor/count[L]),
#                     )
#
#     def keepcoeff(self, L):
#         return getattr(self, "keepcoeff_{}".format(L))
#
#     def mixcoeff(self, l1, l2, L):
#         return getattr(self, "mixcoeff_{}_{}_{}".format(l1, l2, L))
#
#     def forward(self, xs):
#         # print('in', self.order_in, 'out', self.order_out)
#         # print('xs norm selfmix', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
#         # initialize output
#         ys = [
#             self.keepcoeff(L) * xs[L]
#             if L <= self.order_in
#             else torch.zeros_like(xs[0]).repeat(
#                 *(1,) * len(xs[0].shape[:-2]), 2 * L + 1, 1
#             )
#             for L in range(self.order_out + 1)
#         ]
#         xs = torch.cat(xs, dim=2)
#         ys = torch.cat(ys, dim=2)
#         # print('ys', [float(torch.mean(ys[L]**2)) for L in range(len(ys))])
#         # loop over all combinations of orders
#         # print('ys norm selfmix', [float(torch.mean(ys[L]**2)) for L in range(len(ys))])
#         x1 = xs[:, :, self.in_idx_1, :]
#         x2 = xs[:, :, self.in_idx_2, :]
#         # print(self.cg_matrix.shape)
#         mixcoeff = self.mixcoeffs[self.mix_idx_1, self.mix_idx_2, self.mix_idx_out]
#         # print(mixcoeff.shape)
#         y = x1 * x2 * self.cg_matrix[None, None, :, None] * mixcoeff[None, None, :]
#         ys += utils.scatter_add(y, self.out_idx, dim_size=(self.order_out + 1) ** 2, dim=2)
#
#         return [ys[:, :, l ** 2:(l + 1) ** 2] for l in range(self.order_out + 1)]
