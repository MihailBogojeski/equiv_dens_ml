"""Reverse vs forward mode autodiff for the density gradient (time, peak memory, with/without training backward).

python benchmarks/bench_grad_mode.py {cpu,cuda}
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import torch
from torch.func import jvp
from equiv_dens.nn.property_output.density import DensityExpansion
from equiv_dens.utils.orbitals import PackedLayout
from equiv_dens.utils.base import to_angstrom

dev = sys.argv[1]
if dev == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.3)
import numpy as np
from ase.data import atomic_numbers
basis = np.load('/workspace/datasets/augccpvqzjkfit_orbital_basis_df.npy', allow_pickle=True).item()
radial = np.load('/workspace/datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy', allow_pickle=True).item()
basis = {atomic_numbers[k]: v for k, v in basis.items() if k in ('H', 'O')}
radial = {atomic_numbers[k]: v for k, v in radial.items() if k in ('H', 'O')}
lay = PackedLayout(basis, radial)
zs = [8, 1, 1]
B, G = 5, 5000
dt = torch.float32
l = lay.get(zs, dev, dt)
torch.manual_seed(0)
packed = {}
for L in range(lay.Lmax + 1):
    S = l['S'][L]
    packed[L] = {'atom': l['atom'][L],
                 'coeff': (torch.randn(B, S, 2 * L + 1, device=dev, dtype=dt) * 0.1).requires_grad_(),
                 'width': l['width0'][L].expand(B, -1, -1).clone().requires_grad_(),
                 'scale': l['scale0'][L].expand(B, -1, -1).clone().requires_grad_()}
pos = torch.tensor([[0., 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]], device=dev, dtype=dt).expand(B, -1, -1) + 0
coords = torch.randn(B, G, 3, device=dev, dtype=dt) * 1.5
Lmax = lay.Lmax


def reverse(c, create):
    c = c.detach().requires_grad_(True)
    d = DensityExpansion.density_at(c, pos, packed, Lmax)
    g, = torch.autograd.grad(d.sum(), c, create_graph=create)
    return d, g


def forward(c, create):
    f = lambda x: DensityExpansion.density_at(x, pos, packed, Lmax)
    gs = []
    for k in range(3):
        t = torch.zeros_like(c)
        t[..., k] = 1
        d, g = jvp(f, (c,), (t,))
        gs.append(g)
    return d, torch.stack(gs, -1)


def run(fn, bwd):
    def step():
        d, g = fn(coords, bwd)
        if bwd:
            (d.pow(2).mean() + g.pow(2).mean()).backward()
    step()
    if dev == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(5):
        t = time.time()
        step()
        if dev == 'cuda':
            torch.cuda.synchronize()
        ts.append(time.time() - t)
    mem = torch.cuda.max_memory_allocated() / 2**20 if dev == 'cuda' else 0
    return '%.1f ms %.0f MB' % (1000 * sorted(ts)[2], mem)


d1, g1 = reverse(coords, False)
d2, g2 = forward(coords, False)
print('max |rev - fwd| / max |g|', ((g1 - g2).abs().max() / g1.abs().max()).item())
for name, fn in (('reverse', reverse), ('forward (3 jvp)', forward)):
    print('%-16s grad only: %s | with param backward: %s' % (name, run(fn, False), run(fn, True)))
