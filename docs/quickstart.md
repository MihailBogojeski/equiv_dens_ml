# Quick Start

Get equiv_dens_ml running in a few minutes.

## Installation

```bash
conda create -n equiv_dens_ml python=3.10 pip
conda activate equiv_dens_ml
pip install -e . -r requirements.txt
```

Optional GPU-accelerated DFT: `pip install equiv-dens[gpu-cuda12]` (or `gpu-cuda11` / `gpu-cuda13`).

## First training run

```bash
python run.py train @config/training/h2o_small_all_001.txt @config/training/test_overrides.txt
```

Outputs go to `scratch/test_runs`. Add `--max_steps=20` for a very short test.

## First inference

With a trained model (or a paper model after `git lfs pull`):

```bash
python run.py infer --model path/to/model --trajectory path/to/traj.npy --dpm-intor
```

## Console script

When installed, use `equiv-dens` instead of `python run.py`:

```bash
equiv-dens train @config/training/h2o_small_all_001.txt
equiv-dens md @config/md/nn/polythiophene_2mer.txt
```

See [SETUP.md](SETUP.md) for full setup and [README.md](../README.md) for the complete command reference.
