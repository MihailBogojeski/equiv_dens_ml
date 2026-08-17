# Training

Training workflows and configuration.

## Modes

| Mode | Script | Description |
|------|--------|-------------|
| density | train_dens.py | Density-only training |
| joint | train.py | Joint density, energy, forces |
| multiphase | train_all.py | Density first, then energy/forces with frozen density head |
| energy | train_only_en.py | Energy-only training |

## Quick reference

```bash
python run.py train --mode density @config/training/h2o_small_all_001.txt
python run.py train --mode joint @config/training/h2o_small_all_001.txt
python run.py train --mode multiphase @config/training/h2o_small_all_001.txt
python run.py train --mode energy @config/training/h2o_small_all_001.txt
# QM7x energy phase (after density converged):
python run.py train --mode energy @config/training/qm7x_energy_001.txt
```

## Quick tests

Add `@config/training/test_overrides.txt` or pass `--save_dir=scratch/test_runs --keep_checkpoints=0 --max_steps=20`:

```bash
python run.py train @config/training/h2o_small_all_001.txt @config/training/test_overrides.txt
```

## Configurations

Argument files live in `config/training/`. See [args_reference.md](args_reference.md) for parameter documentation and [MODELS.md](MODELS.md) for model comparison.

## QM7x two-phase workflow

QM7x uses a density-then-energy workflow:

1. **Phase 1 (density):** Train density-only until converged. A pre-trained model is provided in `paper/models/qm7x/2024-04-22_bjOUNzrR/`.
2. **Phase 2 (energy):** Run energy-only training with the density head frozen:

```bash
python run.py train --mode energy @config/training/qm7x_energy_001.txt
```

Run `./scripts/setup_qm7x.sh` first to extract the archive. See the main [README](../README.md#qm7x-molecular-dataset) for prerequisites (density/DFT files, auxiliary basis files).

## Scripts

- **scripts/training/train_all.py** – Multi-phase training
- **scripts/training/train.py** – Joint training
- **scripts/training/train_dens.py** – Density-only
- **scripts/training/train_only_en.py** – Energy-only
