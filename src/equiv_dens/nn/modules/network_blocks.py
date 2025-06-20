import torch
import torch.nn as nn
from .activations import Swish, ShiftedSoftplus
from .spherical_harmonic_layers import PairMixing, PairInteraction, SphericalLinear
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

    def forward(self, xs, rbf, sph, idx_i, idx_j, neighbor_mask=1):
        # print('xs norm modular', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
        xs = self.residual_pre_x(xs)
        # print('xs norm modular residual pre', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
        xs = self.interaction(xs, rbf, sph, idx_i, idx_j, neighbor_mask=neighbor_mask)
        # print('xs norm modular interaction', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
        xs = self.residual_post_x(xs)
        # print('xs norm modular residual post', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
        ys = self.residual_out(xs)
        # print('ys norm modular residual out', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
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

    def forward(self, xs, rbf, sph, idx_i, idx_j, neighbor_mask=1):
        ys = [1 * x for x in xs]
        # path for atoms i
        yi = self.residual_pre_vi(ys)
        yi[0] = self.activation_i(yi[0])
        # print('yi norm:', float(torch.mean(yi[0]**2)))
        if self.normalize > 1:
            # print('yi before norm:', [float(torch.mean(yi[k]**2)) for k in range(len(yi))])
            for L in range(len(yi)):
                yi[L] = layer_norm(yi[L], dims=(-3, -2, -1)) 
            # yi[0] = layer_norm(yi[0], dims=(-2, -1)) 
            # print('yi after norm:', [float(torch.mean(yi[k]**2)) for k in range(len(yi))])
        # print('yi norm:', float(torch.mean(yi[0]**2)))
        # print('yi norm:', torch.mean(yi[0]**2, dim=(-2,-1)))
        yi = self.linear_i(yi)

        for L in range(len(yi), self.mixing_order + 1):
            yi.append(torch.zeros(*yi[0].shape[:2], (2 * L) + 1, yi[0].shape[-1]).to(yi[0]))

        # path for atoms j
        yj = self.residual_pre_vj(ys)
        yj[0] = self.activation_j(yj[0])
        if self.normalize == 1:
            yj[0] = layer_norm(yj[0], dims=(-3, -2, -1)) 
        elif self.normalize > 1:
            # print('yj before norm:', [float(torch.mean(yj[k]**2)) for k in range(len(yj))])
            for L in range(len(yj)):
                yj[L] = layer_norm(yj[L], dims=(-3, -2, -1)) 
            # print('yj after norm:', [float(torch.mean(yj[k]**2)) for k in range(len(yj))])
        # print('yj norm:', float(torch.mean(yj[0]**2)))
        yj = self.linear_j(yj)
        # interaction function
        for L in range(min(2 * self.input_order, self.order) + 1):
            idx = idx_j.view(*(1,) * len(yj[L].shape[:-3]), -1, 1, 1).repeat(
                *yj[L].shape[:-3], 1, *yj[L].shape[-2:]
            )
            yj[L] = torch.gather(yj[L], 1, idx) * neighbor_mask

        # print('yj norm:', [float(torch.mean(yj[L]**2)) for L in range(len(yj))])
        # print('rbf norm:', float(torch.mean(rbf**2)))
        # print('rbf shape', rbf.shape)
        # print('rbf sum:', float(torch.mean(torch.sum(rbf, dim=-1))))
        ang = self.angular_fn1(sph)
        # print('sph norm:', [float(torch.mean(sph[L]**2)) for L in range(len(sph))])
        # print('angular norm:', [float(torch.mean(ang[L]**2)) for L in range(len(ang))])
        vs = self.mixing(yj, ang, rbf)
        # print('vs norm:', [float(torch.mean(vs[L]**2)) for L in range(len(vs))])
        a = self.angular_fn2(sph)
        # print('a norm:', [float(torch.mean(a[L]**2)) for L in range(len(a))])
        for L in range(self.mixing_order + 1):
            # idx_i_scat = idx_i.view(*(1,) * len(vs[L].shape[:-3]), -1, 1, 1).repeat(
            # *vs[L].shape[:-3], 1, *vs[L].shape[-2:])
            # print('idx i scat', idx_i_scat.shape)
            # if L == 0:
            # #     print('idx_i', idx_i)
            # #     print('vs[0]', vs[L])
            # #     print('index add', torch.zeros_like(yi[L]).index_add(
            # #         1, idx_i, vs[L] + self.radial_fn[L](rbf) * a[L] * yj[0]
            # #         )
            # #     )
            #     print('vs 0', vs[L].shape)
            #     print('a 0', a[L].shape)
            #     print('y 0', yj[L].shape)
            #     print('scatter add', torch.scatter_reduce(
            #         vs[L] + self.radial_fn[L](rbf) * a[L] * yj[0], 1, idx_i_scat, 
            #         'mean')
            #     )
            #     print('index add', torch.zeros_like(yj[L]).index_add(1, idx_i,
            #         vs[L] + self.radial_fn[L](rbf) * a[L] * yj[0])
            #     )
            if not self.normalize or torch.mean(yi[L]**2) == 0 or torch.mean(vs[L]**2) == 0:
                scale = 1
            else:
                scale = np.sqrt(1/2)
            if self.normalize:
                norm_rbf = np.sqrt(self.num_features / self.num_basis_functions)
                norm_sph = np.sqrt(self.mixing_order + 1)
            else:
                norm_rbf = 1
                norm_sph = 1
            vs[L] = (yi[L] * scale).index_add(
                1, idx_i, scale * (vs[L] + self.radial_fn[L](rbf) * norm_rbf * a[L] * yj[0])
            )
            # print('vs L norm 1:', float(torch.mean(vs[L]**2)))
            # print('num neighbours', self.num_neighbours)
            # print('norm sph', norm_sph)
            if self.normalize:
                vs[L] = vs[L] * norm_sph / self.num_neighbours
            # print('vs L norm 2:', float(torch.mean(vs[L]**2)))
            # print('vs ' + str(L) + ' norm:', float(torch.mean(vs[L]**2)))
            # vs[L] = yi[L] + torch.scatter_reduce(
            #         vs[L] + self.radial_fn[L](rbf) * a[L] * yj[0], 1, idx_i_scat, 
            #         'mean')
            #     print('yi[0]', yi[L])
            # if L == 0:
            #     print('vs[0]', vs[L])
        # print('vs norm:', [float(torch.mean(vs[L]**2)) for L in range(len(vs))])

        # print('vs norm after mixing:', [float(torch.mean(vs[L]**2)) for L in range(len(vs))])

        if self.normalize > 1:
            # print('vs before norm:', [float(torch.mean(vs[k]**2)) for k in range(len(vs))])
            for L in range(len(vs)):
                vs[L] = layer_norm(vs[L], dims=(-3, -2, -1))
            # print('vs after norm:', [float(torch.mean(vs[k]**2)) for k in range(len(vs))])
        if self.mixing_order != self.order:
            vs = self.linear_contract(vs)        # interaction refinement

        # print('vs norm before residual:', [float(torch.mean(vs[L]**2)) for L in range(len(vs))])
        vs = self.residual_post_v(vs)
        vs[0] = self.activation_v(vs[0])
        # print('vs norm after residual:', [float(torch.mean(vs[L]**2)) for L in range(len(vs))])
        vs = self.linear_v(vs)
        for L in range(len(xs), len(vs)):
            xs.append(torch.zeros_like(vs[L]))
        # print('vs norm end interaction:', [float(torch.mean(vs[L]**2)) for L in range(len(vs))])
        # print('xs norm end interaction:', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
        return [x + v for x, v in zip(xs, vs)]


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

    def forward(self, xs, rbf, sph, idx_i, idx_j, neighbor_mask=1):
        ys = [1 * x for x in xs]
        # path for atoms i
        yi = [torch.zeros_like(y) for y in ys]
        # yi = self.residual_pre_vi(ys)
        # yi[0] = self.activation_i(yi[0])
        # print('yi norm:', float(torch.mean(yi[0]**2)))
        # if self.normalize > 1:
        #     for L in range(len(yi)):
        #         yi[L] = layer_norm(yi[L], dims=(-2, -1))
        # print('yi norm:', float(torch.mean(yi[0]**2)))
        # print('yi norm:', torch.mean(yi[0]**2, dim=(-2,-1)))
        # yi = self.linear_i(yi)

        # for L in range(len(yi), self.order + 1):
        #     yi.append(torch.zeros(*yi[0].shape[:2], (2 * L) + 1, yi[0].shape[-1]).to(yi[0]))

        # path for atoms j
        yj = self.residual_pre_vj(ys)
        yj[0] = self.activation_j(yj[0])
        if self.normalize == 1:
            yj[0] = layer_norm(yj[0], dims=(-3, -2, -1))
        elif self.normalize > 1:
            # print('yj before norm:', [float(torch.mean(yj[k]**2)) for k in range(len(yj))])
            for L in range(len(yj)):
                yj[L] = layer_norm(yj[L], dims=(-3, -2, -1))
            # print('yj after norm:', [float(torch.mean(yj[k]**2)) for k in range(len(yj))])
        # print('yj norm:', float(torch.mean(yj[0]**2)))
        yj = self.linear_j(yj)
        # interaction function
        for L in range(min(2 * self.order, self.order) + 1):
            idx = idx_j.view(*(1,) * len(yj[L].shape[:-3]), -1, 1, 1).repeat(
                *yj[L].shape[:-3], 1, *yj[L].shape[-2:]
            )
            yj[L] = torch.gather(yj[L], 1, idx) * neighbor_mask

        # print('yj norm:', [float(torch.mean(yj[L]**2)) for L in range(len(yj))])
        # print('rbf norm:', float(torch.mean(rbf**2)))
        # print('rbf shape', rbf.shape)
        # print('rbf sum:', float(torch.mean(torch.sum(rbf, dim=-1))))
        ang = self.angular_fn1(sph)
        # print('sph norm:', [float(torch.mean(sph[L]**2)) for L in range(len(sph))])
        # print('angular norm:', [float(torch.mean(ang[L]**2)) for L in range(len(ang))])
        # print('rbf', rbf[:, :, :, 0])
        vs = self.mixing(yj, ang, rbf)
        # print('vs 0', vs[0][:, :, :, 0])
        # print('idx i', idx_i)
        # print('vs norm:', [float(torch.mean(vs[L]**2)) for L in range(len(vs))])
        a = self.angular_fn2(sph)
        # print('a norm:', [float(torch.mean(a[L]**2)) for L in range(len(a))])
        for L in range(self.order + 1):
            # idx_i_scat = idx_i.view(*(1,) * len(vs[L].shape[:-3]), -1, 1, 1).repeat(
            # *vs[L].shape[:-3], 1, *vs[L].shape[-2:])
            # print('idx i scat', idx_i_scat.shape)
            # if L == 0:
            # #     print('idx_i', idx_i)
            # #     print('vs[0]', vs[L])
            # #     print('index add', torch.zeros_like(yi[L]).index_add(
            # #         1, idx_i, vs[L] + self.radial_fn[L](rbf) * a[L] * yj[0]
            # #         )
            # #     )
            #     print('vs 0', vs[L].shape)
            #     print('a 0', a[L].shape)
            #     print('y 0', yj[L].shape)
            #     print('scatter add', torch.scatter_reduce(
            #         vs[L] + self.radial_fn[L](rbf) * a[L] * yj[0], 1, idx_i_scat, 
            #         'mean')
            #     )
            #     print('index add', torch.zeros_like(yj[L]).index_add(1, idx_i,
            #         vs[L] + self.radial_fn[L](rbf) * a[L] * yj[0])
            #     )
            if not self.normalize or torch.mean(yi[L]**2) == 0 or torch.mean(vs[L]**2) == 0:
                scale = 1
            else:
                scale = np.sqrt(1/2)
            if self.normalize:
                norm_rbf = np.sqrt(self.num_features / self.num_basis_functions)
                norm_sph = np.sqrt(self.order + 1)
            else:
                norm_rbf = 1
                norm_sph = 1
            vs[L] = (yi[L] * scale).index_add(
                1, idx_i, scale * (vs[L] + self.radial_fn[L](rbf) * norm_rbf * a[L] * yj[0])
            )
            # print('vs L norm 1:', float(torch.mean(vs[L]**2)))
            # print('num neighbours', self.num_neighbours)
            # print('norm sph', norm_sph)
            if self.normalize:
                vs[L] = vs[L] * norm_sph / self.num_neighbours
            # print('vs L norm 2:', float(torch.mean(vs[L]**2)))
            # print('vs ' + str(L) + ' norm:', float(torch.mean(vs[L]**2)))
            # vs[L] = yi[L] + torch.scatter_reduce(
            #         vs[L] + self.radial_fn[L](rbf) * a[L] * yj[0], 1, idx_i_scat, 
            #         'mean')
            #     print('yi[0]', yi[L])
            # if L == 0:
            #     print('vs[0]', vs[L])
        print('vs norm:', [float(torch.mean(vs[L]**2)) for L in range(len(vs))])

        vs = self.linear_v(vs)
        if self.residual:
            vs = self.residual_post_v(vs)
            vs[0] = self.activation_v(vs[0])
            vs = self.linear_v(vs)
            for L in range(len(xs), len(vs)):
                xs.append(torch.zeros_like(vs[L]))
            print('vs norm:', [float(torch.mean(vs[L]**2)) for L in range(len(vs))])
            print('xs norm:', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
            return [x + v for x, v in zip(xs, vs)]
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

    def forward(self, xs):
        if self.num_blocks > 0:
            for residual_block in self.stack:
                xs = residual_block(xs)
            return xs
        else:  # to prevent inplace modification
            return [1 * x for x in xs]


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

    def forward(self, xs):
        # for L in range(len(xs)):
        #     print('L', L)
        #     print('xs[L] pre-residual', xs[L])
        # print('xs pre residual norm:', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
        ys = [1 * x for x in xs]
        ys[0] = self.activation_pre(ys[0])
        ys = self.linear1(ys)
        # for L in range(len(xs)):
        #     print('L', L)
        #     print('xs[L] post_linear1', xs[L])
        ys[0] = self.activation_post(ys[0])
        if self.normalize > 1:
            # print('ys before norm:', [float(torch.mean(ys[k]**2)) for k in range(len(ys))])
            for L in range(len(ys)):
                ys[L] = layer_norm(ys[L], dims=(-3, -2, -1))
            # print('ys after norm:', [float(torch.mean(ys[k]**2)) for k in range(len(ys))])
        #     # ys[0] = layer_norm(ys[0], dims=(-2, -1)) 
        # for L in range(len(ys)):
        #     if torch.mean(ys[L]**2) != 0:
        #         ys[L] = ys[L] / torch.sqrt(torch.mean(ys[L]**2))
        # if torch.mean(ys[0]**2) != 0:
        #     ys[0] = ys[0] / torch.sqrt(torch.mean(ys[0]**2))
        # print('ys residual norm:', [float(torch.mean(ys[L]**2)) for L in range(len(ys))])
        ys = self.linear2(ys)
        if len(ys) > len(xs):
            for L in range(len(xs), len(ys)):
                xs.append(torch.zeros_like(ys[L]))

        # print('xs post residual norm:', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
        # print('ys pre residual norm:', [float(torch.mean(ys[L]**2)) for L in range(len(ys))])
        for L in range(self.order_out + 1):
            # print('L', L)
            # print('xs[L] residual', xs[L])
            # print('ys[L]', ys[L])
            # print('xs[L]', xs[L])
            if not self.normalize or torch.mean(xs[L]**2) == 0 or torch.mean(ys[L]**2) == 0:
                scale = 1
            else:
                scale = np.sqrt(1/2)
            ys[L] = ys[L] * scale + xs[L] * scale
        # print('ys post residual norm:', [float(torch.mean(ys[L]**2)) for L in range(len(ys))])
        return ys


def layer_norm(x, dims):
    x_mean = torch.mean(x, dim=dims, keepdim=True)
    x_var = torch.var(x, dim=dims, keepdim=True)
    if torch.sum(x_var) == 0:
        x_std = 0
    else:
        x_std = torch.sqrt(x_var)
    # print('x_std', x_std)
    eps = 1e-8
    return (x - x_mean) / (x_std + eps)
