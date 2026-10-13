#!/usr/bin/env python3
"""
Numerical-equivalence reference for refactors (float64, CPU, fixed seed).

  python benchmarks/reference.py capture [--names base parity dgrad norm2 water8]
      Builds the model for each config, runs one train-mode forward + loss + backward on a fixed batch and
      saves state_dict, input batch, outputs and parameter gradients to benchmarks/reference_data/<name>/ref.pt.
  python benchmarks/reference.py compare [--names ...] [--tol 1e-9]
      Builds the model with the current code, loads the saved state_dict (strict for all parameters and
      small buffers; only the large constant CG buffers are not stored), runs the same batch and
      prints max-abs / max-rel (max-abs / max|ref|) differences per quantity. Exit code 1 on any failure.

Quantities: sph_repr (canonical = list of per-L tensors [.., 2L+1, F]; packed tensors [.., (L+1)^2, F] are split
by compare), df_coeffs, density, density_grad (where enabled), energy, forces, loss, and d(loss)/d(param) for all params.
"""
import argparse
import os
import sys

import numpy as np
import torch

import common as C
from equiv_dens.data.custom_samplers import set_up_data_loader

REF_DIR = os.environ.get('EQUIV_REF_DIR', os.path.join(C.ROOT, 'benchmarks/reference_data'))
COMMON = ['--dtype=torch.float64', '--use_gpu=False', '--density_subsamples=1000']
CONFIGS = {  # name -> (dataset, batch size, extra overrides)
    'base': ('h2o_small', 2, []),
    'parity': ('h2o_small', 2, ['--parity_dens=True', '--parity_en=True']),
    'dgrad': ('h2o_small', 2, ['--density_grad=True', '--density_grad_weight=1.0']),
    'norm2': ('h2o_small', 2, ['--normalize=2']),
    'water8': ('water8', 1, []),
}
KEYS = ['sph_repr', 'df_coeffs', 'density', 'density_grad', 'energy', 'forces']


def clone(batch):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def setup(name):
    dataset, bs, extra = CONFIGS[name]
    overrides = C.dataset_overrides(dataset, bs) + COMMON + extra
    torch.manual_seed(0)
    np.random.seed(0)
    args, ds, train_ds, model, error_dict = C.build([C.DEFAULT_CONFIG], overrides)
    model.double().train()
    return overrides, args, train_ds, model, error_dict


def run(model, error_dict, batch):
    """Forward + loss + backward on a copy of `batch`. Returns canonical outputs and parameter grads."""
    data = model.transform_input(clone(batch))
    model.zero_grad()
    pred = model(data)
    loss = error_dict.compute(pred, data)['loss']
    loss.backward()
    out = {}
    for k in KEYS:
        if k in pred:
            v = pred[k]
            out[k] = [t.detach().cpu() for t in v] if isinstance(v, (list, tuple)) else v.detach().cpu()
    out['loss'] = loss.detach().cpu()
    grads = {n: (p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p).cpu())
             for n, p in model.named_parameters()}
    return out, grads


def capture(name):
    overrides, args, train_ds, model, error_dict = setup(name)
    loader = set_up_data_loader(train_ds, CONFIGS[name][1], False, 0.7, False, False)
    batch = next(iter(loader))
    out, grads = run(model, error_dict, batch)
    os.makedirs(f'{REF_DIR}/{name}', exist_ok=True)
    # large constant buffers (Clebsch-Gordan tensors, ~3 GB) are rebuilt by the code and not stored
    state = {k: v for k, v in model.state_dict().items() if v.numel() < 10**6}
    torch.save({'overrides': overrides, 'state_dict': state, 'batch': batch,
                'outputs': out, 'grads': grads}, f'{REF_DIR}/{name}/ref.pt')
    print(f'captured {name}: keys {list(out)} n_params {len(grads)}')


def to_per_L(x, ref):
    """Convert a packed tensor [.., (L+1)^2, F] to a list of per-L tensors shaped like `ref`."""
    if isinstance(x, (list, tuple)):
        return list(x)
    out, s = [], 0
    for L, r in enumerate(ref):
        out.append(x[..., s:s + 2 * L + 1, :r.shape[-1]].reshape(r.shape))
        s += 2 * L + 1
    return out


def diff(a, b):
    a, b = a.detach().cpu().double(), b.double()
    abs_d = (a - b).abs().max().item()
    return abs_d, abs_d / max(b.abs().max().item(), 1e-300)


def compare(name, tol):
    ref = torch.load(f'{REF_DIR}/{name}/ref.pt', weights_only=False)
    overrides, args, train_ds, model, error_dict = setup(name)
    missing, unexpected = model.load_state_dict(ref['state_dict'], strict=False)
    params = dict(model.named_parameters())
    assert not unexpected and not [k for k in missing if k in params], (unexpected, missing)  # strict on all params
    out, grads = run(model, error_dict, ref['batch'])
    rows = []
    for k, r in ref['outputs'].items():
        if k not in out:
            rows.append((k, None, None))
            continue
        if k == 'sph_repr':
            new = to_per_L(out[k], r)
            ds = [diff(n, o) for n, o in zip(new, r)]
            rows.append((k, max(d[0] for d in ds), max(d[1] for d in ds)))
        elif isinstance(r, list):
            ds = [diff(n, o) for n, o in zip(out[k], r)]
            rows.append((k, max(d[0] for d in ds), max(d[1] for d in ds)))
        else:
            rows.append((k, *diff(out[k], r)))
    gd = {n: diff(grads[n], g) for n, g in ref['grads'].items()}
    worst = max(gd, key=lambda n: gd[n][1])
    rows.append(('grads (worst: ' + worst + ')', *gd[worst]))
    print(f'\n== {name} (tol {tol:g} rel) ==')
    print(f"{'quantity':<56}{'max abs':>12}{'max rel':>12}  ")
    ok = True
    for k, a, r in rows:
        if a is None:
            print(f'{k:<56}{"MISSING":>24}  FAIL')
            ok = False
            continue
        good = r <= tol
        ok &= good
        print(f'{k:<56}{a:>12.3e}{r:>12.3e}  {"pass" if good else "FAIL"}')
    return ok


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['capture', 'compare'])
    p.add_argument('--names', nargs='+', default=list(CONFIGS))
    p.add_argument('--tol', type=float, default=1e-9)
    cli = p.parse_args()
    if cli.mode == 'capture':
        for n in cli.names:
            capture(n)
    else:
        oks = [compare(n, cli.tol) for n in cli.names]
        print('\nALL PASS' if all(oks) else '\nSOME FAILED')
        sys.exit(0 if all(oks) else 1)
