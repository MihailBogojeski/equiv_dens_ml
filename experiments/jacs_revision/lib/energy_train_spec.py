"""Shared energy-stage spec, config stack, and pack-submit gates.

``submit_water_train_energy.sbatch`` (solo and packed) and the water
watchdog must agree on RUN:THEORY[:SEED] → campaign name and argv, or a
packed job trains the wrong labels into the wrong directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from experiments.jacs_revision.lib.paths import repo_root

ENERGY_BASE = "@config/training/water_energy_base.txt"
ENERGY_OVERLAY = {
    "wb97mv_def2tzvpd": "@config/training/water_energy_wb97mv.txt",
    "pbe_d4_avdz": "@config/training/water_energy_pbe.txt",
}
MALON = "@config/training/malonaldehyde_pt_001.txt"
MALON_PBE = "@config/training/malonaldehyde_pbe_orca_overlay.txt"
DIRECT_MP = "@config/training/water_direct_multiphase_overlay.txt"
CURVE = "@config/training/water_learning_curve_ckpt.txt"


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """Split ``RUN:THEORY`` or ``RUN:THEORY:SEED``."""
    parts = spec.split(":")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"spec must be RUN:THEORY[:SEED], got {spec!r}")
    seed = parts[2] if len(parts) > 2 and parts[2] else None
    return parts[0], parts[1], seed


def campaign_name(run: str, theory: str, seed: str | None = None) -> str:
    """Directory stem under ``results/revision/``, matching the energy sbatch."""
    if run == "water":
        name = f"water_{theory}"
    elif run == "malonaldehyde":
        name = f"malonaldehyde_{theory}"
    elif run == "water_direct":
        name = f"water_direct_{theory}"
    elif run.startswith("cutoff_"):
        name = f"water_pbe_orca_cutoff_{run[len('cutoff_'):]}"
    elif run.startswith("water_direct_n") or run.startswith("water_n"):
        name = f"{run}_{theory}"
    elif run == "water_direct_curve":
        name = f"water_direct_{theory}_curve"
    elif run == "water_curve":
        name = f"water_{theory}_curve"
    else:
        raise ValueError(f"unknown RUN {run}")
    if seed:
        name = f"{name}_seed{seed}"
    return name


def config_stack(run: str, theory: str, seed: str | None = None) -> list[str]:
    """``@``-files and extra flags for ``run.py train --mode energy``."""
    if run.startswith("cutoff_") and theory != "pbe_d4_avdz":
        raise ValueError(f"cutoff sweep runs at PBE only, not {theory}")
    if run == "malonaldehyde":
        if theory not in ENERGY_OVERLAY:
            raise ValueError(f"no energy config registered for theory {theory}")
        stack = [MALON, ENERGY_BASE]
        if theory == "pbe_d4_avdz":
            stack.append(MALON_PBE)
        # ENERGY_BASE supplies the head recipe but contains water split sizes.
        # Keep all three disjoint malonaldehyde sets at their actual sizes.
        stack.extend(["--num_train=400", "--num_valid=80", "--num_test=125"])
    else:
        if theory not in ENERGY_OVERLAY:
            raise ValueError(f"no energy config registered for theory {theory}")
        stack = [ENERGY_BASE, ENERGY_OVERLAY[theory]]
        if run == "water":
            pass
        elif run == "water_direct":
            stack.append(DIRECT_MP)
        elif run.startswith("cutoff_"):
            cut = run[len("cutoff_"):]
            stack.append(f"@config/training/water_clusters_cutoff_{cut}.txt")
        elif run.startswith("water_direct_n") or run.startswith("water_n"):
            n = (
                run[len("water_direct_n"):]
                if run.startswith("water_direct_n")
                else run[len("water_n"):]
            )
            stem = (
                f"datasets/revision/water_labels/learning_curves/"
                f"water_train_n{n}_{theory}"
            )
            stack.extend(
                [
                    f"--np_dataset={stem}_base.npy",
                    f"--dens_dataset={stem}.npy",
                    f"--num_train={n}",
                ]
            )
            if run.startswith("water_direct_n"):
                stack.append(DIRECT_MP)
        elif run in {"water_curve", "water_direct_curve"}:
            stack.append(CURVE)
            if run == "water_direct_curve":
                stack.append(DIRECT_MP)
        else:
            raise ValueError(f"unknown RUN {run}")
    if seed:
        stack.append(f"--init_seed={seed}")
    return stack


def spec_is_done(repo: Path, spec: str) -> bool:
    """True when the watchdog should not launch this energy stage again."""
    name = campaign_name(*parse_spec(spec))
    return (Path(repo) / "results" / "revision" / name / ".training_done").is_file()


def density_parent(repo: Path, spec: str) -> Path:
    name = campaign_name(*parse_spec(spec))
    return Path(repo) / "results" / "revision" / name


def energy_save_dir(repo: Path, spec: str) -> Path:
    return density_parent(repo, spec).with_name(
        density_parent(repo, spec).name + "_energy"
    )


def density_run_dir(repo: Path, spec: str) -> Path | None:
    marker = density_parent(repo, spec) / ".density_run_dir"
    if not marker.is_file():
        return None
    path = Path(marker.read_text().strip())
    return path if path.is_dir() else None


def has_atomic_energies(run_dir: Path) -> bool:
    args_file = Path(run_dir) / "args.txt"
    if not args_file.is_file():
        return False
    found = False
    for line in args_file.read_text().splitlines():
        if line.startswith("--atomic_energies="):
            found = True
            value = line.split("=", 1)[1]
            if value in {"", "None"}:
                return False
    return found


def latest_run_dir(save_dir: Path) -> Path | None:
    """Delegate to the shell helper so resume picks the same directory."""
    script = repo_root() / "scripts" / "revision" / "lib" / "latest_run_dir.sh"
    result = subprocess.run(
        ["bash", str(script), str(save_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip()
    return Path(text) if text else None


def choose_resume(repo: Path, spec: str, force_new: bool = False) -> Path | None:
    """Mixed-N water must not resume a head trained on E − mean(E)."""
    if force_new:
        return None
    run, _theory, _seed = parse_spec(spec)
    latest = latest_run_dir(energy_save_dir(repo, spec))
    if latest is None:
        return None
    if run != "malonaldehyde" and not has_atomic_energies(latest):
        return None
    return latest


def build_train_argv(
    spec: str,
    repo: Path,
    *,
    density_run: str,
    resume: str | None = None,
    python: str | None = None,
) -> list[str]:
    """Argv for one energy training process. Never calls sbatch."""
    run, theory, seed = parse_spec(spec)
    name = campaign_name(run, theory, seed)
    repo = Path(repo)
    python = python or str(repo / ".venv" / "bin" / "python")
    argv = [
        python,
        str(repo / "run.py"),
        "train",
        "--mode",
        "energy",
        *config_stack(run, theory, seed),
        f"--args_file_name={name}_energy",
        f"--save_dir={energy_save_dir(repo, spec)}",
        f"--load_from={density_run}",
    ]
    if resume:
        argv.append(f"--restart={resume}")
    return argv


def build_dry_run_argv(spec: str, repo: Path, density_run: str = "DENSITY_RUN") -> list[str]:
    """Same argv as a real launch, with a placeholder density path."""
    return build_train_argv(spec, Path(repo), density_run=density_run)


def mark_training_done(repo: Path, spec: str) -> None:
    parent = density_parent(repo, spec)
    parent.mkdir(parents=True, exist_ok=True)
    (parent / ".training_done").touch()


def is_energy_queue_name(job_name: str) -> bool:
    return job_name == "tre-pack" or job_name.startswith("tre-")


def should_submit_pack(ready_specs: list[str], queued_job_names: list[str]) -> bool:
    """One packed GPU job if any energy spec is ready and none is already queued."""
    if not ready_specs:
        return False
    if any(is_energy_queue_name(name) for name in queued_job_names):
        return False
    return True


WATER_ENERGY_SPECS = ("water:pbe_d4_avdz", "water:wb97mv_def2tzvpd")
MALON_ENERGY_SPECS = (
    "malonaldehyde:pbe_d4_avdz",
    "malonaldehyde:wb97mv_def2tzvpd",
)


def next_energy_pack(
    done_specs: list[str],
    queued_job_names: list[str],
    *,
    malon_density_ready: bool,
    water_specs: list[str] | None = None,
    malon_specs: list[str] | None = None,
) -> list[str]:
    """Specs for the single energy pack to submit now.

    Water heads that lack ``.training_done`` take the pack. Malonaldehyde
    energy is returned only after those water parents are done, its density
    fine-tune is ready, and no ``tre-*`` job is already queued.
    """
    if any(is_energy_queue_name(name) for name in queued_job_names):
        return []
    done = set(done_specs)
    water_left = [spec for spec in (water_specs or WATER_ENERGY_SPECS) if spec not in done]
    if water_left:
        return water_left
    if not malon_density_ready:
        return []
    return [spec for spec in (malon_specs or MALON_ENERGY_SPECS) if spec not in done]
