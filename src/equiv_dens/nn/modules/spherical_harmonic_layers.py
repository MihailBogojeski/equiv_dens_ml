import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from equiv_dens.nn.modules.cg_product import CGProduct

"""
Spherical features are packed tensors of shape [..., (order + 1)**2, num_features].
The m axis is ordered by L: [L=0 | L=1 (3) | L=2 (5) | ...], the same ordering as ClebschGordanMatrix.
"""


_l_of_m_cache = {}


def l_of_m(D, device):
    """Order L of every entry of a packed m axis of length D (cached in eager mode, saves 3 kernels per call)."""
    if torch.compiler.is_compiling():
        return torch.arange(D, device=device).float().sqrt().floor().long()
    key = (D, str(device))
    if key not in _l_of_m_cache:
        _l_of_m_cache[key] = torch.arange(D, device=device).float().sqrt().floor().long()
    return _l_of_m_cache[key]


def order_of(D):
    return int(round(D ** 0.5)) - 1


def unpack_orders(x):
    """List of the per order slices [..., 2L + 1, F] (views) of a packed tensor."""
    return [x[..., L ** 2:(L + 1) ** 2, :] for L in range(order_of(x.shape[-2]) + 1)]


def fit_orders(x, D):
    """Cut or zero-pad the packed m axis of x to length D."""
    if x.shape[-2] > D:
        return x[..., :D, :]
    if x.shape[-2] < D:
        return F.pad(x, (0, 0, 0, D - x.shape[-2]))
    return x


def apply_L0(fn, x):
    """Apply fn to the L=0 slice of a packed tensor."""
    if x.shape[-2] == 1:
        return fn(x)
    return torch.cat([fn(x[..., :1, :]), x[..., 1:, :]], dim=-2)


def order_scale(x, y, normalize):
    """Residual scale per order: sqrt(1/2), or 1 if x or y is exactly zero for that order."""
    if not normalize:
        return 1
    D = x.shape[-2]
    l = l_of_m(D, x.device)
    nL = order_of(D) + 1
    dims = [d for d in range(x.dim()) if d != x.dim() - 2]
    nz_x = torch.zeros(nL, device=x.device, dtype=x.dtype).index_add(0, l, x.pow(2).sum(dim=dims)) != 0
    nz_y = torch.zeros(nL, device=x.device, dtype=x.dtype).index_add(0, l, y.pow(2).sum(dim=dims)) != 0
    scale = torch.where(nz_x & nz_y, x.new_full((), np.sqrt(1 / 2)), x.new_full((), 1.0))
    return scale[l].unsqueeze(-1)


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
        """
        super().__init__()
        self.order_in = int(order_in)
        self.num_in = int(num_in)
        self.order_out = int(order_out)
        self.num_out = int(num_out)
        self.bias = bias
        self.mix_orders = mix_orders
        self.zero_init = zero_init
        self.parity = parity
        if self.mix_orders:
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
        self.register_buffer("l_idx", l_of_m((self.order_out + 1) ** 2, "cpu"), persistent=False)
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

    def forward(self, x):
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): packed input features [..., (order_in + 1)**2, num_in]
        Returns:
            y (torch.Tensor): packed output features [..., (order_out + 1)**2, num_out]
        """
        D = (self.order_out + 1) ** 2
        if self.mix_orders:
            x = self.mixing(x)
        else:
            x = x[..., :D, :]
        weight = torch.stack([lin.weight for lin in self.linear])[self.l_idx]  # [D, num_out, num_in]
        y = torch.einsum('...mi,moi->...mo', x, weight)
        if self.bias:
            y = y + F.pad(self.linear[0].bias.unsqueeze(0), (0, 0, 0, D - 1))
        if not self.mix_orders:
            y = y * np.sqrt(self.num_out)
        return y


def mixing_paths(order_in1, order_in2, order_out, parity, only_upper=False):
    """List of (l1, l2, L) paths of a tensor product, in the order the coefficients are registered."""
    paths = []
    for l1 in range(order_in1 + 1):
        for l2 in range(l1 + 1 if only_upper else 0, order_in2 + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, order_out) + 1):
                # NOTE: parsed as l1 + (l2 % 2), a precedence bug that is kept on purpose so that
                # trained parity models stay compatible
                if l1 + l2 % 2 != L % 2 and parity:
                    continue
                paths.append((l1, l2, L))
    return paths


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
        self.order_in = int(order_in)
        self.order_out = int(order_out)
        self.num_features = int(num_features)
        self.normalize = normalize
        self.parity = parity
        self.paths = mixing_paths(order_in, order_in, order_out, parity, only_upper=True)
        self.order_keep = min(order_in, order_out)
        # coefficients for mixing
        for l1, l2, L in self.paths:
            name = "mixcoeff_{}_{}_{}".format(l1, l2, L)
            self.register_parameter(
                name, nn.Parameter(torch.Tensor(self.num_features))
            )
        for L in range(self.order_keep + 1):
            name = "keepcoeff_{}".format(L)
            self.register_parameter(name, nn.Parameter(torch.Tensor(self.num_features)))
        self.register_buffer("l_keep", l_of_m((self.order_keep + 1) ** 2, "cpu"), persistent=False)
        if self.paths:
            cg_matrix, _ = clebsch_gordan(order_in, order_in, order_out)
            self.product = CGProduct(cg_matrix, self.paths)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize/reset the layer's parameters."""
        count = [1 for L in range(self.order_out + 1)]
        for l1, l2, L in self.paths:
            count[L] += 1

        for L in range(self.order_keep + 1):
            if self.normalize:
                norm_factor = (L + 1)
            else:
                norm_factor = 1
            nn.init.uniform_(
                self.keepcoeff(L), a=-np.sqrt(3 * norm_factor/count[L]), b=np.sqrt(3 * norm_factor/count[L])
            )

        for l1, l2, L in self.paths:
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

    def forward(self, x):
        """
        Compute the spherical tensor product between the different orders of the same atomic features.

        Args:
            x (torch.Tensor): packed spherical features [..., (order_in + 1)**2, num_features]
        Returns:
            y (torch.Tensor): packed features after self-mixing [..., (order_out + 1)**2, num_features]
        """
        x = x[..., :(self.order_in + 1) ** 2, :]
        keep = torch.stack([self.keepcoeff(L) for L in range(self.order_keep + 1)])[self.l_keep]
        y = fit_orders(keep * x[..., :(self.order_keep + 1) ** 2, :], (self.order_out + 1) ** 2)
        if self.paths:
            w = torch.stack([self.mixcoeff(*path) for path in self.paths])
            y = y + self.product(x, x, w)
        return y


class PairMixing(nn.Module):
    """Mixes pairs of atomic features in a distance dependent way and outputs pair features."""

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
        """
        super().__init__()
        self.order_in1 = int(order_in1)
        self.order_in2 = int(order_in2)
        self.order_out = int(order_out)
        self.num_basis_functions = int(num_basis_functions)
        self.num_features = int(num_features)
        self.normalize = normalize
        self.parity = parity
        self.paths = mixing_paths(order_in1, order_in2, order_out, parity)
        # distance - dependent coefficients for mixing
        for l1, l2, L in self.paths:
            name = "coeff_{}_{}_{}".format(l1, l2, L)
            self.add_module(
                name,
                nn.Linear(
                    self.num_basis_functions, self.num_features, bias=False
                ),
            )
        self.L_count = [0 for L in range(self.order_out + 1)]
        for l1, l2, L in self.paths:
            self.L_count[L] += 1
        # per path factor that is applied to the coefficients
        path_scale = []
        for l1, l2, L in self.paths:
            norm_factor = 1
            if self.normalize:
                norm_factor = np.sqrt(L + 1) * np.sqrt(self.num_features / self.num_basis_functions)
            path_scale.append(norm_factor / np.sqrt(self.L_count[L]))
        self.register_buffer("path_scale", torch.tensor(path_scale, dtype=torch.float64), persistent=False)
        cg_matrix, _ = clebsch_gordan(order_in1, order_in2, order_out)
        self.product = CGProduct(cg_matrix, self.paths)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize/reset the layer's parameters."""
        for l1, l2, L in self.paths:
            nn.init.orthogonal_(self.coeff(l1, l2, L).weight)

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

    def forward(self, x1, x2, rbf):
        """
        Compute the spherical tensor product between two spherical features by mixing orders.

        Args:
            x1 (torch.Tensor): first set of packed spherical features [..., (order_in1 + 1)**2, F]
            x2 (torch.Tensor): second set of packed spherical features [..., (order_in2 + 1)**2, F]
            rbf (torch.Tensor): radial basis function [..., 1, num_basis_functions]
        Returns:
            y (torch.Tensor): packed features after pair interaction [..., (order_out + 1)**2, F]
        """
        x1 = x1[..., :(self.order_in1 + 1) ** 2, :]
        x2 = x2[..., :(self.order_in2 + 1) ** 2, :]
        weight = torch.cat([self.coeff(*path).weight for path in self.paths])  # [num_paths * F, num_basis]
        weight = weight.view(len(self.paths), self.num_features, -1) * self.path_scale.to(weight.dtype).view(-1, 1, 1)
        coeff = F.linear(rbf, weight.view(-1, weight.shape[-1]))  # [..., 1, num_paths * F]
        coeff = coeff.view(*coeff.shape[:-2], len(self.paths), self.num_features)
        return self.product(x1, x2, coeff)


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
        self.order = int(order)
        self.num_basis_functions = int(num_basis_functions)
        self.num_features = int(num_features)
        self.normalize = normalize
        self.paths = [(L, L, L) for L in range(self.order + 1)]
        # distance - dependent coefficients for mixing
        for L in range(self.order + 1):
            name = "coeff_{}".format(L)
            self.add_module(
                name,
                nn.Linear(
                    self.num_basis_functions, self.num_features, bias=False
                ),
            )
        self.L_count = [1 for L in range(self.order + 1)]
        path_scale = []
        for L in range(self.order + 1):
            norm_factor = 1
            if self.normalize:
                norm_factor = np.sqrt(L + 1) * np.sqrt(self.num_features / self.num_basis_functions)
            path_scale.append(norm_factor)
        self.register_buffer("path_scale", torch.tensor(path_scale, dtype=torch.float64), persistent=False)
        cg_matrix, _ = clebsch_gordan(order, order, order)
        self.product = CGProduct(cg_matrix, self.paths)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize/reset the layer's parameters."""
        for L in range(self.order + 1):
            nn.init.orthogonal_(self.coeff(L).weight)

    def coeff(self, L):
        """
        Retrieve the learneable coefficients for the order L.

        Args:
            L (int): order of spherical harmonics
        Returns:
            coeff (torch.Tensor): mixing coefficients
        """
        return getattr(self, "coeff_{}".format(L))

    def forward(self, x1, x2, rbf):
        """
        Compute the spherical tensor product between the same orders only.

        Args:
            x1 (torch.Tensor): first set of packed spherical features [..., (order + 1)**2, F]
            x2 (torch.Tensor): second set of packed spherical features [..., (order + 1)**2, F]
            rbf (torch.Tensor): radial basis function [..., 1, num_basis_functions]
        Returns:
            y (torch.Tensor): packed features after pair interaction [..., (order + 1)**2, F]
        """
        x1 = x1[..., :(self.order + 1) ** 2, :]
        x2 = x2[..., :(self.order + 1) ** 2, :]
        weight = torch.cat([self.coeff(L).weight for L in range(self.order + 1)])
        weight = weight.view(self.order + 1, self.num_features, -1) * self.path_scale.to(weight.dtype).view(-1, 1, 1)
        coeff = F.linear(rbf, weight.view(-1, weight.shape[-1]))
        coeff = coeff.view(*coeff.shape[:-2], self.order + 1, self.num_features)
        return self.product(x1, x2, coeff)
