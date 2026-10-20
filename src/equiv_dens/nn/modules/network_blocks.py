import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import Swish, ShiftedSoftplus
from .spherical_harmonic_layers import PairMixing, PairInteraction, SphericalLinear, \
    l_of_m, order_of, fit_orders, apply_L0, order_scale
import numpy as np

class ModularBlock(nn.Module):
    """
    Basic building block of the neural network which refines atomic features in an iterative way
    """

    def __init__(
        self,
        order,
        num_features,
        num_basis_functions,
        num_residual_pre_x,
        num_residual_post_x,
        num_residual_pre_vi,
        num_residual_pre_vj,
        num_residual_post_v,
        num_residual_output,
        clebsch_gordan=None,
        mix_orders=True,
        mixing_order=None,
        input_order=None,
        activation="swish",
        num_neighbours=1,
        normalize=0,
        parity=False,
    ):
        super(ModularBlock, self).__init__()
        # initialize attributes
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_residual_pre_x = num_residual_pre_x
        self.num_residual_post_x = num_residual_post_x
        self.num_residual_pre_vi = num_residual_pre_vi
        self.num_residual_pre_vj = num_residual_pre_vj
        self.num_residual_post_v = num_residual_post_v
        self.num_residual_output = num_residual_output
        self.mixing_order = mixing_order
        if self.mixing_order is None:
            self.mixing_order = self.order
        self.input_order = input_order
        if self.input_order is None:
            self.input_order = self.order
        # initialize modules
        self.interaction = InteractionBlock(
            order=self.order,
            num_features=self.num_features,
            num_basis_functions=self.num_basis_functions,
            num_residual_pre_vi=self.num_residual_pre_vi,
            num_residual_pre_vj=self.num_residual_pre_vj,
            num_residual_post_v=self.num_residual_post_v,
            clebsch_gordan=clebsch_gordan,
            mix_orders=mix_orders,
            mixing_order=self.mixing_order,
            input_order=self.input_order,
            activation=activation,
            num_neighbours=num_neighbours,
            normalize=normalize,
            parity=parity,
        )
        self.residual_pre_x = ResidualStack(
            num_blocks=self.num_residual_pre_x,
            order=min(2 * self.input_order, self.order),
            num_features=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=mix_orders,
            activation=activation,
            normalize=normalize,
            parity=parity,
        )
        self.residual_post_x = ResidualStack(
            num_blocks=self.num_residual_post_x,
            order=self.order,
            num_features=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=mix_orders,
            activation=activation,
            normalize=normalize,
            parity=parity,
        )
        self.residual_out = ResidualStack(
            num_blocks=self.num_residual_output,
            order=self.order,
            num_features=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=mix_orders,
            activation=activation,
            normalize=normalize,
            parity=parity,
        )

    def forward(self, xs, rbf, sph, idx_i, idx_j, seg=None):
        xs = self.residual_pre_x(xs, seg)
        xs = self.interaction(xs, rbf, sph, idx_i, idx_j, seg)
        xs = self.residual_post_x(xs, seg)
        ys = self.residual_out(xs, seg)
        return xs, ys


class InteractionBlock(nn.Module):
    """
    Refines atomic features by interacting with its neighbors
    """

    def __init__(
        self,
        order,
        num_features,
        num_basis_functions,
        num_residual_pre_vi,
        num_residual_pre_vj,
        num_residual_post_v,
        clebsch_gordan=None,
        mix_orders=True,
        mixing_order=None,
        input_order=None,
        activation="swish",
        num_neighbours=1,
        normalize=0,
        parity=False,
    ):
        super(InteractionBlock, self).__init__()
        # initialiye attributes
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_residual_pre_vi = num_residual_pre_vi
        self.num_residual_pre_vj = num_residual_pre_vj
        self.num_residual_post_v = num_residual_post_v
        self.mixing_order = mixing_order
        self.num_neighbours = num_neighbours
        self.normalize=normalize
        if self.mixing_order is None:
            self.mixing_order = self.order
        self.input_order = input_order
        if self.input_order is None:
            self.input_order = self.order
        # initialize activation function
        if activation == "swish":
            self.activation_i = Swish(self.num_features)
            self.activation_j = Swish(self.num_features)
            self.activation_v = Swish(self.num_features)
        elif activation == "ssp":
            self.activation_i = ShiftedSoftplus(self.num_features)
            self.activation_j = ShiftedSoftplus(self.num_features)
            self.activation_v = ShiftedSoftplus(self.num_features)
        else:
            raise ValueError("Unsupported activation function:", activation)
        # initialize modules
        self.angular_fn1 = SphericalLinear(
            order_in=self.mixing_order,
            num_in=1,
            order_out=self.mixing_order,
            num_out=self.num_features,
            clebsch_gordan=None,
            mix_orders=False,
            normalize=normalize,
            parity=parity,
        )
        self.angular_fn2 = SphericalLinear(
            order_in=self.mixing_order,
            num_in=1,
            order_out=self.mixing_order,
            num_out=self.num_features,
            clebsch_gordan=None,
            mix_orders=False,
            normalize=normalize,
            parity=parity,
        )
        self.radial_fn = nn.ModuleList(
            [
                nn.Linear(self.num_basis_functions, self.num_features, bias=False)
                for L in range(self.mixing_order + 1)
            ]
        )
        self.mixing = PairMixing(
            order_in1=min(2 * self.input_order, self.order),
            order_in2=self.mixing_order,
            order_out=self.mixing_order,
            num_basis_functions=self.num_basis_functions,
            num_features=self.num_features,
            clebsch_gordan=clebsch_gordan,
            normalize=0,
            parity=False,
        )
        self.linear_i = SphericalLinear(
            order_in=min(2 * self.input_order, self.order),
            num_in=self.num_features,
            order_out=min(2 * self.input_order, self.order),
            num_out=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=mix_orders,
            normalize=normalize,
            parity=parity,
        )
        self.linear_j = SphericalLinear(
            order_in=min(2 * self.input_order, self.order),
            num_in=self.num_features,
            order_out=min(2 * self.input_order, self.order),
            num_out=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=mix_orders,
            normalize=normalize,
            parity=parity,
        )
        self.linear_v = SphericalLinear(
            order_in=self.order,
            num_in=self.num_features,
            order_out=self.order,
            num_out=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=mix_orders,
            normalize=normalize,
            parity=parity,
        )
        if self.mixing_order != self.order:
            self.linear_contract = SphericalLinear(
                order_in=self.mixing_order,
                num_in=self.num_features,
                order_out=self.order,
                num_out=self.num_features,
                clebsch_gordan=clebsch_gordan,
                mix_orders=mix_orders,
                normalize=normalize,
                parity=parity,
            )
        self.residual_pre_vi = ResidualStack(
            num_blocks=self.num_residual_pre_vi,
            order=min(2 * self.input_order, self.order),
            num_features=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=mix_orders,
            activation=activation,
            normalize=normalize,
            parity=parity,
        )
        self.residual_pre_vj = ResidualStack(
            num_blocks=self.num_residual_pre_vj,
            order=min(2 * self.input_order, self.order),
            num_features=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=mix_orders,
            activation=activation,
            normalize=normalize,
            parity=parity,
        )
        self.residual_post_v = ResidualStack(
            num_blocks=self.num_residual_post_v,
            order=self.order,
            num_features=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=mix_orders,
            activation=activation,
            normalize=normalize,
            parity=parity,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for L in range(self.order + 1):
            nn.init.orthogonal_(self.radial_fn[L].weight)

    def forward(self, xs, rbf, sph, idx_i, idx_j, seg=None):
        # path for atoms i
        yi = self.residual_pre_vi(xs, seg)
        yi = apply_L0(self.activation_i, yi)
        if self.normalize > 1:
            yi = layer_norm_orders(yi, seg)
        yi = self.linear_i(yi)
        yi = fit_orders(yi, (self.mixing_order + 1) ** 2)

        # path for atoms j
        yj = self.residual_pre_vj(xs, seg)
        yj = apply_L0(self.activation_j, yj)
        if self.normalize == 1:
            yj = apply_L0(lambda t: layer_norm_orders(t, seg), yj)
        elif self.normalize > 1:
            yj = layer_norm_orders(yj, seg)
        yj = self.linear_j(yj)
        # interaction function
        yj = yj.index_select(1, idx_j)

        ang = self.angular_fn1(sph)
        vs = self.mixing(yj, ang, rbf)
        a = self.angular_fn2(sph)
        if self.normalize:
            norm_rbf = np.sqrt(self.num_features / self.num_basis_functions)
            norm_sph = np.sqrt(self.mixing_order + 1)
        else:
            norm_rbf = 1
            norm_sph = 1
        radial = F.linear(rbf, torch.cat([fn.weight for fn in self.radial_fn]) * norm_rbf)
        radial = radial.view(*radial.shape[:-2], self.mixing_order + 1, self.num_features)
        radial = radial[..., l_of_m((self.mixing_order + 1) ** 2, rbf.device), :]
        scale = order_scale(yi, vs, self.normalize)
        vs = vs + radial * a * yj[..., :1, :]
        if self.normalize:
            vs = (yi * scale).index_add(1, idx_i, scale * vs) * norm_sph / self.num_neighbours
        else:
            vs = yi.index_add(1, idx_i, vs)

        if self.normalize > 1:
            vs = layer_norm_orders(vs, seg)
        if self.mixing_order != self.order:
            vs = self.linear_contract(vs)

        # interaction refinement
        vs = self.residual_post_v(vs, seg)
        vs = apply_L0(self.activation_v, vs)
        vs = self.linear_v(vs)
        return fit_orders(xs, vs.shape[-2]) + vs


class NonmixingInteractionBlock(nn.Module):
    """Creates new atomic features by interacting with its neighbors, without mixing different spherical orders."""

    def __init__(
        self,
        order,
        num_features,
        num_basis_functions,
        num_residual_pre_vi,
        num_residual_pre_vj,
        num_residual_post_v,
        clebsch_gordan=None,
        activation="swish",
        num_neighbours=1,
        normalize=0,
        residual=True,
    ):

        super(NonmixingInteractionBlock, self).__init__()
        # initialiye attributes
        self.order = order
        self.num_features = num_features
        self.num_basis_functions = num_basis_functions
        self.num_residual_pre_vi = num_residual_pre_vi
        self.num_residual_pre_vj = num_residual_pre_vj
        self.num_residual_post_v = num_residual_post_v
        self.num_neighbours = num_neighbours
        self.normalize = normalize
        self.residual = residual
        # initialize activation function
        if activation == "swish":
            self.activation_i = Swish(self.num_features)
            self.activation_j = Swish(self.num_features)
            self.activation_v = Swish(self.num_features)
        elif activation == "ssp":
            self.activation_i = ShiftedSoftplus(self.num_features)
            self.activation_j = ShiftedSoftplus(self.num_features)
            self.activation_v = ShiftedSoftplus(self.num_features)
        else:
            raise ValueError("Unsupported activation function:", activation)
        # initialize modules
        self.angular_fn1 = SphericalLinear(
            order_in=self.order,
            num_in=1,
            order_out=self.order,
            num_out=self.num_features,
            clebsch_gordan=None,
            mix_orders=False,
            normalize=normalize,
            parity=False,
        )
        self.angular_fn2 = SphericalLinear(
            order_in=self.order,
            num_in=1,
            order_out=self.order,
            num_out=self.num_features,
            clebsch_gordan=None,
            mix_orders=False,
            normalize=normalize,
            parity=False,
        )
        self.radial_fn = nn.ModuleList(
            [
                nn.Linear(self.num_basis_functions, self.num_features, bias=False)
                for L in range(self.order + 1)
            ]
        )
        self.mixing = PairInteraction(
            order=self.order,
            num_basis_functions=self.num_basis_functions,
            num_features=self.num_features,
            clebsch_gordan=clebsch_gordan,
            normalize=normalize,
        )
        self.linear_i = SphericalLinear(
            order_in=self.order,
            num_in=self.num_features,
            order_out=self.order,
            num_out=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=False,
            normalize=normalize,
            parity=False,
        )
        self.linear_j = SphericalLinear(
            order_in=self.order,
            num_in=self.num_features,
            order_out=self.order,
            num_out=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=False,
            normalize=normalize,
            parity=False,
        )
        self.linear_v = SphericalLinear(
            order_in=self.order,
            num_in=self.num_features,
            order_out=self.order,
            num_out=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=False,
            normalize=normalize,
            parity=False,
            bias=self.residual,
        )
        self.residual_pre_vi = ResidualStack(
            num_blocks=self.num_residual_pre_vi,
            order=self.order,
            num_features=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=False,
            activation=activation,
            normalize=normalize,
            parity=False,
        )
        self.residual_pre_vj = ResidualStack(
            num_blocks=self.num_residual_pre_vj,
            order=self.order,
            num_features=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=False,
            activation=activation,
            normalize=normalize,
            parity=False,
        )
        if self.residual:
            self.residual_post_v = ResidualStack(
                num_blocks=self.num_residual_post_v,
                order=self.order,
                num_features=self.num_features,
                clebsch_gordan=clebsch_gordan,
                mix_orders=False,
                activation=activation,
                normalize=normalize,
                parity=False,
            )
        self.reset_parameters()

    def reset_parameters(self):
        for L in range(self.order + 1):
            nn.init.orthogonal_(self.radial_fn[L].weight)

    def forward(self, xs, rbf, sph, idx_i, idx_j, seg=None):
        # path for atoms j
        yj = self.residual_pre_vj(xs, seg)
        yj = apply_L0(self.activation_j, yj)
        if self.normalize == 1:
            yj = apply_L0(lambda t: layer_norm_orders(t, seg), yj)
        elif self.normalize > 1:
            yj = layer_norm_orders(yj, seg)
        yj = self.linear_j(yj)
        # interaction function
        yj = yj.index_select(1, idx_j)

        ang = self.angular_fn1(sph)
        vs = self.mixing(yj, ang, rbf)
        a = self.angular_fn2(sph)
        if self.normalize:
            norm_rbf = np.sqrt(self.num_features / self.num_basis_functions)
            norm_sph = np.sqrt(self.order + 1)
        else:
            norm_rbf = 1
            norm_sph = 1
        radial = F.linear(rbf, torch.cat([fn.weight for fn in self.radial_fn]) * norm_rbf)
        radial = radial.view(*radial.shape[:-2], self.order + 1, self.num_features)
        radial = radial[..., l_of_m((self.order + 1) ** 2, rbf.device), :]
        vs = vs + radial * a * yj[..., :1, :]
        vs = torch.zeros_like(xs).index_add(1, idx_i, vs)
        if self.normalize:
            vs = vs * norm_sph / self.num_neighbours
        vs = self.linear_v(vs)
        if self.residual:
            vs = self.residual_post_v(vs, seg)
            vs = apply_L0(self.activation_v, vs)
            vs = self.linear_v(vs)
            return fit_orders(xs, vs.shape[-2]) + vs
        else:
            return vs


class ResidualStack(nn.Module):
    """
    Stack of pre-activation residual blocks
    """

    def __init__(
        self,
        num_blocks,
        order,
        num_features,
        clebsch_gordan=None,
        mix_orders=True,
        activation="swish",
        normalize=0,
        parity=False,
        bias=True,
    ):
        super(ResidualStack, self).__init__()
        self.num_blocks = num_blocks
        self.order = order
        self.num_features = num_features
        self.stack = nn.ModuleList(
            [
                ResidualBlock(
                    order=self.order,
                    num_features=self.num_features,
                    clebsch_gordan=clebsch_gordan,
                    mix_orders=mix_orders,
                    activation=activation,
                    normalize=normalize,
                    parity=parity,
                    bias=bias,
                )
                for i in range(self.num_blocks)
            ]
        )

    def forward(self, xs, seg=None):
        for residual_block in self.stack:
            xs = residual_block(xs, seg)
        return xs


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block
    """

    def __init__(
        self,
        order,
        num_features,
        clebsch_gordan=None,
        mix_orders=True,
        activation="swish",
        normalize=0,
        order_out=None,
        parity=False,
        bias=True,
    ):
        super(ResidualBlock, self).__init__()
        self.order = order
        self.num_features = num_features
        self.normalize = normalize
        self.mix_orders = mix_orders
        if order_out is None:
            self.order_out = self.order
        else:
            self.order_out = order_out
        if self.mix_orders:
            assert clebsch_gordan is not None
        if activation == "swish":
            self.activation_pre = Swish(self.num_features)
            self.activation_post = Swish(self.num_features)
        elif activation == "ssp":
            self.activation_pre = ShiftedSoftplus(self.num_features)
            self.activation_post = ShiftedSoftplus(self.num_features)
        else:
            raise ValueError("Unsupported activation function:", activation)
        self.linear1 = SphericalLinear(
            order_in=self.order,
            num_in=self.num_features,
            order_out=self.order_out,
            num_out=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=self.mix_orders,
            normalize=normalize,
            parity=parity,
            bias=bias,
        )
        self.linear2 = SphericalLinear(
            order_in=self.order_out,
            num_in=self.num_features,
            order_out=self.order_out,
            num_out=self.num_features,
            clebsch_gordan=clebsch_gordan,
            mix_orders=self.mix_orders,
            zero_init=True,
            normalize=normalize,
            parity=parity,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, xs, seg=None):
        ys = apply_L0(self.activation_pre, xs)
        ys = self.linear1(ys)
        ys = apply_L0(self.activation_post, ys)
        if self.normalize > 1:
            ys = layer_norm_orders(ys, seg)
        ys = self.linear2(ys)
        xs = fit_orders(xs, ys.shape[-2])
        scale = order_scale(xs, ys, self.normalize)
        return ys * scale + xs * scale if self.normalize else ys + xs


class Identity(nn.Module):
    def forward(self, xs, seg=None):
        return xs


def batch_segments(atoms):
    """Molecule index of every atom of the packed batch and the number of molecules, for per-molecule statistics."""
    return atoms['atom_batch_idx'][0], atoms['batch_atom_numbers'].shape[0]


def layer_norm_orders(x, seg=None):
    """layer_norm over (atoms, m, features) separately for every order of a packed tensor [B, N, D, F], and for every
    molecule if seg = (molecule index of every atom [N], number of molecules) is given."""
    D = x.shape[-2]
    l = l_of_m(D, x.device)
    nL = order_of(D) + 1
    n = torch.zeros(nL, device=x.device, dtype=x.dtype).index_add(0, l, torch.ones(D, device=x.device, dtype=x.dtype))
    if seg is None:
        n = n * (x.shape[-3] * x.shape[-1])

        def per_order_sum(t):
            return torch.zeros(t.shape[0], nL, device=x.device, dtype=x.dtype).index_add(1, l, t.sum(dim=(-3, -1)))

        expand = lambda t: t[:, l][:, None, :, None]
    else:
        mol, n_mol = seg
        n = torch.zeros(n_mol, device=x.device, dtype=x.dtype).index_add(0, mol, torch.ones_like(mol, dtype=x.dtype))[:, None] * n * x.shape[-1]

        def per_order_sum(t):
            per_atom = torch.zeros(*t.shape[:2], nL, device=x.device, dtype=x.dtype).index_add(2, l, t.sum(dim=-1))
            return torch.zeros(t.shape[0], n_mol, nL, device=x.device, dtype=x.dtype).index_add(1, mol, per_atom)

        expand = lambda t: t[:, mol][:, :, l, None]

    mean = per_order_sum(x) / n
    xc = x - expand(mean)
    var = per_order_sum(xc * xc) / (n - 1)
    positive = var > 0
    std = torch.where(positive, torch.sqrt(torch.where(positive, var, torch.ones_like(var))), torch.zeros_like(var))
    return xc / (expand(std) + 1e-8)
