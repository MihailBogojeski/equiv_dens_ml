#!/usr/bin/env python3
"""Resume an isolated density checkpoint, then fit and evaluate its energy head."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE / "scripts/revision/lib"))
from energy_train_spec import config_stack

ORIGINAL = {"pbe_d4_avdz": "2026-08-26_LvzjVEeX", "wb97mv_def2tzvpd": "2026-08-26_p7k7whBB"}


def config_paths(tokens):
    return ["@" + str(CODE / token[1:]) if token.startswith("@") else token for token in tokens]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--theory", choices=ORIGINAL, required=True)
    parser.add_argument("--stage", choices=["density", "energy", "evaluate", "all"], default="all")
    args = parser.parse_args()
    repo = args.repo.resolve()
    theory = args.theory
    output = repo / "results/revision/jacs_recovery_20260927" / ("malonaldehyde_" + theory)
    output.mkdir(parents=True, exist_ok=True)
    python = str(repo / ".venv/bin/python")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(CODE / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               WANDB_MODE="offline", WANDB_DIR=str(output), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    source = repo / "results/revision" / ("malonaldehyde_" + theory) / ORIGINAL[theory]
    density = output / "density" / source.name
    if not density.exists():
        staging = density.with_name(density.name + ".copying")
        (staging / "checkpoints").mkdir(parents=True, exist_ok=False)
        for path in [source / "args.txt", *source.glob("best_*.pth")]:
            shutil.copy2(path, staging / path.name)
        for path in [source / "checkpoints/latest_checkpoint.pth", *source.glob("checkpoints/best_*.pth")]:
            shutil.copy2(path, staging / "checkpoints" / path.name)
        staging.rename(density)
        (output / "source.json").write_text(json.dumps({"source_density": str(source), "code": str(CODE)}, indent=2))

    def run(command, log):
        with (output / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps(command) + "\n")
        print("starting", log, flush=True)
        with (output / log).open("a") as stream:
            subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    if args.stage in {"density", "all"} and not (output / "density.done").exists():
        configs = ["@config/training/malonaldehyde_pt_001.txt"]
        if theory == "pbe_d4_avdz":
            configs.append("@config/training/malonaldehyde_pbe_orca_overlay.txt")
        configs.append("@config/training/water_multiphase_overlay.txt")
        run([python, "-B", str(CODE / "scripts/training/train_all.py"), *config_paths(configs),
             "--restart=" + str(density), "--num_workers=2", "--wandb_mode=offline"], "density.log")
        (output / "density.done").touch()
    if args.stage in {"energy", "all"} and not (output / "energy.done").exists():
        if not (output / "density.done").exists():
            raise RuntimeError("Density stage has not finished; refusing premature energy training")
        energy_parent = output / "energy"
        candidates = sorted(energy_parent.glob("*/checkpoints/latest_checkpoint.pth"))
        command = [python, "-B", str(CODE / "scripts/training/train_only_en.py"),
                   *config_paths(config_stack("malonaldehyde", theory)), "--save_dir=" + str(energy_parent),
                   "--load_from=" + str(density), "--args_file_name=jacs_malon_energy_" + theory,
                   "--num_workers=2", "--wandb_mode=offline", "--append_atom_density=False", "--init_seed=20260927"]
        # Match the existing water heads: the energy-only workflow uses the
        # learned density coefficients without concatenating a separate SAD.
        if candidates:
            command.append("--restart=" + str(candidates[-1].parents[1]))
        run(command, "energy.log")
        (output / "energy.done").touch()
    if args.stage in {"evaluate", "all"} and not (output / "evaluation/summary.json").exists():
        if not (output / "energy.done").exists():
            raise RuntimeError("Energy stage has not finished")
        candidates = sorted((output / "energy").glob("*/checkpoints/latest_checkpoint.pth"))
        run([python, "-B", str(CODE / "scripts/revision/analysis/energy_generalization.py"),
             "--repo", str(repo), "--run", str(candidates[-1].parents[1]), "--system", "malonaldehyde",
             "--theory", theory, "--out", str(output / "evaluation"), "--device", "cuda"], "evaluation.log")


if __name__ == "__main__":
    main()
