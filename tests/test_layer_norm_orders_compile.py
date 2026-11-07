"""Per-molecule layer_norm_orders under torch.compile.

seg = (molecule index of every atom, number of molecules) is an ordinary tensor argument, so a compiled
representation (--compile, dynamic shapes) must honour the segments of the batch being traced *and* of every later
batch: a later batch of the same size but different molecule boundaries must be normalised with its own statistics.
Each call below is checked against the eager, per-molecule result.
"""

import pytest
import torch

from equiv_dens.nn.modules.network_blocks import layer_norm_orders

D = 1 + 3 + 5
# Inductor's CPU C++ codegen (torch 2.13) fails on the index_add scatter of the per-molecule statistics; what this
# test checks is dynamo's tracing of the segment argument, so on CPU the graph is run through aot_eager instead.
BACKEND = "inductor" if torch.cuda.is_available() else "aot_eager"


def _batch(sizes, seed):
    g = torch.Generator().manual_seed(seed)
    parts = [torch.randn(1, n, D, 6, generator=g) * (k + 1) + k for k, n in enumerate(sizes)]
    idx = torch.cat([torch.full((n,), k, dtype=torch.long) for k, n in enumerate(sizes)])
    return torch.cat(parts, dim=1), (idx, len(sizes)), parts


def _eager(parts):
    return torch.cat([layer_norm_orders(p) for p in parts], dim=1)


@pytest.mark.parametrize("dynamic", [True, False])
def test_compiled_layer_norm_orders_tracks_segments(dynamic):
    compiled = torch.compile(layer_norm_orders, dynamic=dynamic, backend=BACKEND)
    # same total atom count, different molecule boundaries, then a different count
    for sizes, seed in (([5, 7], 0), ([4, 8], 1), ([3, 3, 6], 2), ([9], 3)):
        x, seg, parts = _batch(sizes, seed)
        got = compiled(x, seg)
        assert torch.allclose(got, _eager(parts), atol=1e-4), f"segments {sizes} not honoured under compile"


def test_compiled_pooled_without_segments():
    compiled = torch.compile(layer_norm_orders, dynamic=True, backend=BACKEND)
    x, _, _ = _batch([5, 7], 0)
    assert torch.allclose(compiled(x), layer_norm_orders(x), atol=1e-5)
