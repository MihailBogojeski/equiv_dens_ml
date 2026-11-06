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

## QM7x two-stage workflow (strict recipe)

QM7x follows the paper's **stage D only** on this machine: Mihail's pre-trained
density checkpoint (`paper/models/qm7x/2024-04-22_bjOUNzrR/`) stands in for the
multiphase density job. Energy/forces training is a separate GPU job with the
density head frozen — the same pattern as
`submit_water_train_energy.sbatch`.

1. **Labels:** ORCA CPU arrays label train/valid (`watch_qm7x_campaign.sh`, short
   walltime, `--resume`). Assemble to `datasets/qm7x_*_dft_augccpvdz.npy`.
2. **Energy stage:** `sbatch scripts/revision/submit_qm7x_energy.sbatch`
   (`--mode energy`, `forces_weight=0.99`, `density_weight=0`).

Validate with `scripts/revision/analysis/validate_training_recipe.py`.

```bash
# Background driver (ORCA -> assemble -> energy):
nohup scripts/revision/qm7x/watch_qm7x_campaign.sh >> logs/qm7x_watchdog.log 2>&1 &

# Or energy only, once labels exist:
sbatch scripts/revision/submit_qm7x_energy.sbatch
```

Legacy one-shot extract: `./scripts/setup_qm7x.sh` (if present). See
[README](../README.md#qm7x-molecular-dataset) for auxiliary basis files.

## JACS revision water campaign (multiphase only)

The revision water models follow the paper recipe in
[`third_party/edense-paper/main_v8.tex`](../third_party/edense-paper/main_v8.tex):
unconstrained density (DF pretrain + AFE), physical-constraint fine-tuning, then
frozen-density energy training. **Do not use joint mode** for MD-capable models.

1. **Density stages** (`submit_water_train.sbatch`, `--mode multiphase`):
   `@config/training/water_multiphase_overlay.txt` sets `energy_weight=0`,
   `density_fine_tuning=True`, and AFE loss (`perc_mae`). Constraint projection
   and positivity activate only in the fine-tuning phase via
   `modify_args_by_phase`.
2. **Energy stage** (`submit_water_train_energy.sbatch`, `--mode energy`):
   loads the density checkpoint from `.density_run_dir` and writes to
   `results/revision/<name>_energy/`.

Validate finished runs with
`scripts/revision/analysis/validate_training_recipe.py`.

## Scripts

- **scripts/training/train_all.py** – Multi-phase training
- **scripts/training/train.py** – Joint training
- **scripts/training/train_dens.py** – Density-only
- **scripts/training/train_only_en.py** – Energy-only
