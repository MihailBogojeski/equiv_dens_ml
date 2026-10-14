#!/usr/bin/env python3
"""
Compiled vs eager training steps on batches of different sizes: step times, recompiles, max deviation of the outputs
and of the parameter gradients (relative to the max |grad| of each parameter, floored at 1e-10).

  python benchmarks/compile_check.py --dataset h2o_small --batch_sizes 5 3 10 7 5
Unknown flags (--key=value) are passed through as training-arg overrides.
"""
import argparse
import time

import torch
import torch._dynamo

import common as C
from equiv_dens.data.custom_samplers import set_up_data_loader

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='h2o_small')
p.add_argument('--batch_sizes', type=int, nargs='+', default=[5, 3, 10, 7, 5])
cli, overrides = p.parse_known_args()

overrides = C.dataset_overrides(cli.dataset, max(cli.batch_sizes)) + ['--use_gpu=True'] + overrides
args, dataset, train_ds, model, error_dict = C.build([C.DEFAULT_CONFIG], overrides)
model.train()
batches = [model.transform_input(C.to_device(next(iter(set_up_data_loader(train_ds, bs, False, 0.7, True, True))),
                                             torch.device('cuda'))) for bs in cli.batch_sizes]
KEYS = ['energy', 'forces', 'df_coeffs', 'density']


def step(data):
    model.zero_grad()
    pred = model(data)
    loss = error_dict.compute(pred, data)['loss']
    loss.backward()
    torch.cuda.synchronize()
    out = {k: pred[k].detach().clone() for k in KEYS if k in pred}
    out['loss'] = loss.detach()
    out['grads'] = {n: q.grad.clone() if q.grad is not None else torch.zeros_like(q) for n, q in model.named_parameters()}
    return out


eager = [step(b) for b in batches]
model.enable_compile()
for i, (bs, b) in enumerate(zip(cli.batch_sizes, batches)):
    for rep in range(2):
        t = time.perf_counter()
        out = step(b)
        dt = time.perf_counter() - t
        frames = torch._dynamo.utils.counters['stats']['unique_graphs']
        dev = {k: ((out[k] - eager[i][k]).abs().max() / eager[i][k].abs().max()).item() for k in out if k != 'grads'}
        g_dev = {n: ((g - eager[i]['grads'][n]).abs().max() / eager[i]['grads'][n].abs().max().clamp_min(1e-10)).item()
                 for n, g in out['grads'].items()}
        dev['grads'] = max(g_dev.values())
        worst = max(g_dev, key=g_dev.get)
        print(f'bs={bs:3d} rep={rep} {dt * 1e3:10.1f} ms  graphs compiled so far: {frames}  '
              + ' '.join(f'{k}={v:.1e}' for k, v in dev.items()) + f' (worst grad: {worst})', flush=True)
        if dev['grads'] > 1e-3: print(out['grads'][worst].flatten()[:6].tolist(), eager[i]['grads'][worst].flatten()[:6].tolist())
model.eval()
t = time.perf_counter()
model(batches[0])
torch.cuda.synchronize()
print(f'eval forward {(time.perf_counter() - t) * 1e3:.1f} ms, graphs compiled: '
      f"{torch._dynamo.utils.counters['stats']['unique_graphs']}")
