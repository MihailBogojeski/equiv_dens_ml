"""Generator-only patch: Torch layer_norm_orders broadcasts its per-sample statistics with
unsqueeze(-3) on a [B, D] tensor (gives [1, B, D]); the intended [B, 1, D, 1] is restored here so the
references keep the per-order, per-sample (-3, -2, -1) layer_norm semantics the JAX port implements."""
import torch
from equiv_dens.nn.modules import network_blocks
from equiv_dens.nn.modules.spherical_harmonic_layers import l_of_m, order_of


def layer_norm_orders(x):
    D = x.shape[-2]
    l = l_of_m(D, x.device)
    nL = order_of(D) + 1
    n = torch.zeros(nL, device=x.device, dtype=x.dtype).index_add(0, l, torch.ones(D, device=x.device, dtype=x.dtype))
    n = n * (x.shape[-3] * x.shape[-1])

    def per_order_sum(t):
        return torch.zeros(t.shape[0], nL, device=x.device, dtype=x.dtype).index_add(1, l, t.sum(dim=(-3, -1)))

    mean = per_order_sum(x) / n
    xc = x - mean[:, l][:, None, :, None]
    var = per_order_sum(xc * xc) / (n - 1)
    positive = var > 0
    std = torch.where(positive, torch.sqrt(torch.where(positive, var, torch.ones_like(var))), torch.zeros_like(var))
    return xc / (std[:, l][:, None, :, None] + 1e-8)


def apply():
    network_blocks.layer_norm_orders = layer_norm_orders
    from equiv_dens.nn.property_output import energy
    energy.layer_norm_orders = layer_norm_orders
