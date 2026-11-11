"""layer_norm_orders (packed [1, N, D, F] tensors) must normalise each molecule of a batch on its own, per order,
when seg = (molecule index of every atom, number of molecules) is given; see network_blocks.batch_segments."""

import torch

from equiv_dens.nn.modules.network_blocks import layer_norm_orders

D = 1 + 3 + 5  # orders 0, 1, 2 packed along one axis


def _batch(sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    parts = [torch.randn(1, n, D, 6, generator=g) * (k + 1) + k for k, n in enumerate(sizes)]
    idx = torch.cat([torch.full((n,), k, dtype=torch.long) for k, n in enumerate(sizes)])
    return parts, (idx, len(sizes))


def test_batched_equals_each_structure_alone():
    parts, seg = _batch([5, 7, 4])
    alone = torch.cat([layer_norm_orders(p) for p in parts], dim=1)
    batched = layer_norm_orders(torch.cat(parts, dim=1), seg)
    assert torch.allclose(batched, alone, atol=1e-5)


def test_single_structure_matches_pooled():
    parts, seg = _batch([9])
    assert torch.allclose(layer_norm_orders(parts[0], seg), layer_norm_orders(parts[0]), atol=1e-6)


def test_pooled_without_segments_is_per_order():
    parts, _ = _batch([5, 7])
    x = torch.cat(parts, dim=1)
    y = layer_norm_orders(x)
    for lo, hi in ((0, 1), (1, 4), (4, 9)):
        block = x[:, :, lo:hi]
        ref = (block - block.mean()) / (block.var().sqrt() + 1e-8)
        assert torch.allclose(y[:, :, lo:hi], ref, atol=1e-5)


def test_zero_variance_structure_has_finite_gradient():
    parts, seg = _batch([4, 3])
    x = torch.cat([torch.zeros_like(parts[0]), parts[1]], dim=1).requires_grad_(True)
    y = layer_norm_orders(x, seg)
    y.pow(2).sum().backward()
    assert torch.all(y[:, :4] == 0)
    assert torch.isfinite(x.grad).all()
