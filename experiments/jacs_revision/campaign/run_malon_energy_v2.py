#!/usr/bin/env python3
"""Malonaldehyde: finish the density run, then fit and evaluate its energy head.

Replaces scripts/revision/recovery/run_malon_recovery.py (job 18681393, never
started), which ran archived code with two energy-training bugs fixed on
2026-09-28:

- ErrorDict compared (B, 1) predictions with the (N,) malonaldehyde labels,
  i.e. every prediction against every label in the batch;
- the energy head's layer norm pooled statistics over the whole batch.

This driver runs the working tree, which has both fixes. The energy model
also rebuilds the density as it was fine-tuned (integral constraint), which is
the density the letter's ε_ρ describes. Outputs are isolated under
results/revision/malon_energy_v2/; the original density runs are only read.

    .venv/bin/python -B experiments/jacs_revision/campaign/run_malon_energy_v2.py --theory pbe_d4_avdz
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/revision/lib"))
from energy_train_spec import config_stack  # noqa: E402

ORIGINAL = {"pbe_d4_avdz": "2026-08-26_LvzjVEeX", "wb97mv_def2tzvpd": "2026-08-26_p7k7whBB"}
CONSTRAINT_KEY = "density_repr_model.1.integral_constraint.integral_scale"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--theory", choices=ORIGINAL, required=True)
    parser.add_argument("--stage", choices=["density", "energy", "evaluate", "all"], default="all")
    parser.add_argument("--out-name", default="malon_energy_v2", help="folder under results/revision (separate runs must not share one)")
    parser.add_argument("--train-arg", action="append", default=[], metavar="--flag=value",
                        help="extra argument for the energy training command (repeatable), e.g. --train-arg=--energy_weight=0.001")
    args = parser.parse_args()
    import torch

    theory = args.theory
    output = REPO / "results/revision" / args.out_name / ("malonaldehyde_" + theory)
    output.mkdir(parents=True, exist_ok=True)
    python = str(REPO / ".venv/bin/python")
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", WANDB_MODE="offline",
               # Python headers for Triton's gcc-built runtime helper (see train_energy_head.sh)
               CPATH=env.get("CPATH", "/scratch/mh7373/tools/python3.12-include"),
               WANDB_DIR=str(output), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    source = REPO / "results/revision" / ("malonaldehyde_" + theory) / ORIGINAL[theory]
    density = output / "density" / source.name
    if not density.exists():
        staging = density.with_name(density.name + ".copying")
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "checkpoints").mkdir(parents=True)
        for path in [source / "args.txt", *source.glob("best_*.pth")]:
            shutil.copy2(path, staging / path.name)
        for path in [source / "checkpoints/latest_checkpoint.pth", *source.glob("checkpoints/best_*.pth")]:
            shutil.copy2(path, staging / "checkpoints" / path.name)
        staging.rename(density)
        (output / "source.json").write_text(json.dumps({"source_density": str(source)}, indent=2) + "\n")

    def run(command, log):
        with (output / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps(command) + "\n")
        print("starting", log, flush=True)
        with (output / log).open("a") as stream:
            subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    if args.stage in {"density", "all"} and not (output / "density.done").exists():
        configs = ["@config/training/malonaldehyde_pt_001.txt"]
        if theory == "pbe_d4_avdz":
            configs.append("@config/training/malonaldehyde_pbe_orca_overlay.txt")
        configs.append("@config/training/water_multiphase_overlay.txt")
        run([python, "-B", "scripts/training/train_all.py", *configs,
             "--restart=" + str(density), "--num_workers=2", "--wandb_mode=offline"], "density.log")
        (output / "density.done").touch()

    if args.stage in {"energy", "all"} and not (output / "energy.done").exists():
        if not (output / "density.done").exists():
            raise RuntimeError("density stage has not finished; refusing premature energy training")
        best = torch.load(density / f"best_{density.name.split('_')[-1]}.pth", map_location="cpu", weights_only=False)
        density_args = torch.load(density / "checkpoints/latest_checkpoint.pth", map_location="cpu",
                                  weights_only=False)["args"]
        extra = []
        if CONSTRAINT_KEY in best:
            # rebuild the density as it was fine-tuned; direct models also need positive coefficients
            extra = ["--integral_constraint=coeffs_in_coeffs_net",
                     f"--positive_coeffs={not density_args.remove_atom_density}"]
        energy_parent = output / "energy"
        candidates = sorted(energy_parent.glob("*/checkpoints/latest_checkpoint.pth"))
        command = [python, "-B", "scripts/training/train_only_en.py", *config_stack("malonaldehyde", theory),
                   "--save_dir=" + str(energy_parent), "--load_from=" + str(density),
                   "--args_file_name=malon_energy_v2_" + theory, "--num_workers=2", "--wandb_mode=offline",
                   "--append_atom_density=False", "--init_seed=20260928",
                   # no structure-wide layer norm in the energy head (the config stack sets 2): with it the
                   # head is non-local and non-additive (energy_head_locality.py); the paper's final
                   # checkpoints were trained with 0. The normalize_en=2 attempt is in energy_normalize_en2/.
                   "--normalize_en=0", "--train_eval_size=80", *extra, *args.train_arg]
        if candidates:
            command.append("--restart=" + str(candidates[-1].parents[1]))
        run(command, "energy.log")
        (output / "energy.done").touch()

    if args.stage in {"evaluate", "all"} and not (output / "evaluation/summary.json").exists():
        if not (output / "energy.done").exists():
            raise RuntimeError("energy stage has not finished")
        candidates = sorted((output / "energy").glob("*/checkpoints/latest_checkpoint.pth"))
        run([python, "-B", "scripts/revision/analysis/energy_generalization.py",
             "--repo", str(REPO), "--run", str(candidates[-1].parents[1]), "--system", "malonaldehyde",
             "--theory", theory, "--out", str(output / "evaluation"), "--device", "cuda"], "evaluation.log")


if __name__ == "__main__":
    main()
