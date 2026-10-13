#!/usr/bin/env python3
"""
Kernel launches and GPU time of one training step (forward + loss + backward) with torch.profiler.

  python benchmarks/profile_step.py --dataset h2o_small --batch_size 5 [--compile=True] [--top 25]
Unknown flags (--key=value) are passed through as training-arg overrides, like bench_model.py.
"""
import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

import common as C
from equiv_dens.data.custom_samplers import set_up_data_loader

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='h2o_small')
p.add_argument('--batch_size', type=int, default=5)
p.add_argument('--warmup', type=int, default=3)
p.add_argument('--top', type=int, default=20)
p.add_argument('--sort', default='device_time_total')
cli, overrides = p.parse_known_args()

overrides = C.dataset_overrides(cli.dataset, cli.batch_size) + ['--use_gpu=True'] + overrides
args, dataset, train_ds, model, error_dict = C.build([C.DEFAULT_CONFIG], overrides)
model.train()
loader = set_up_data_loader(train_ds, cli.batch_size, False, 0.7, True, True)
data = model.transform_input(C.to_device(next(iter(loader)), torch.device('cuda')))


def step():
    model.zero_grad()
    error_dict.compute(model(data), data)['loss'].backward()


t = time.perf_counter()
step()
torch.cuda.synchronize()
first = time.perf_counter() - t
for _ in range(cli.warmup - 1):
    step()
torch.cuda.synchronize()
t = time.perf_counter()
step()
torch.cuda.synchronize()
wall = (time.perf_counter() - t) * 1e3

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    step()
    torch.cuda.synchronize()
kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and 'memcpy' not in e.name.lower()
           and 'memset' not in e.name.lower()]
gpu_ms = sum(e.device_time for e in kernels) / 1e3
print(prof.key_averages().table(sort_by=cli.sort, row_limit=cli.top))
print(f'{cli.dataset} bs={cli.batch_size}: {len(kernels)} kernel launches, {gpu_ms:.1f} ms GPU kernel time, '
      f'{wall:.1f} ms wall (unprofiled step), first step {first:.1f} s')
if args.compile:
    times = torch._dynamo.utils.compile_times(repr='raw')
    print('compile (s):', {k: round(sum(times[k]), 1) for k in ['_compile.compile_inner', 'bytecode_tracing',
                                                              'aot_trace_joint_graph', 'compile_fx_inner'] if k in times})
