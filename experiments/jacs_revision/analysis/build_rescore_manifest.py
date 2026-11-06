#!/usr/bin/env python3
"""Write the manifests for the full-density re-score (rescore_full_density.py).

Every evaluation the response letter took from ``csh_evaluate.py`` is listed
here against the same label files and model runs, so the re-score changes one
thing only: the reference density (full SCF instead of the DF projection) and
the quadrature (full level-2 grid instead of 4000 random points).

Checkpoints: ``production`` is ``best_<ID>.pth``, the file ``csh_evaluate``
scored (after the density fine-tuning phase where one exists). ``density`` is
``checkpoints/best_checkpoint_density.pth``, the end of the density phase. The
N_train runs have no fine-tuning phase, so their production file *is* the
density-phase checkpoint. The full-data direct PBE run diverged in fine-tuning
(epsilon ~ 7.6 on every structure), so both of its files are scored and the
direct arm is reported from the density phase.

Usage:
  python experiments/jacs_revision/analysis/build_rescore_manifest.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = REPO / "results" / "revision" / "rescore_fullref"
EVAL = OUT / "eval"
WATER_LABELS = "datasets/revision/water_labels"
MALON_LABELS = "datasets/revision/malonaldehyde_labels"
GEOM_LABELS = "datasets/revision/ood_geom_labels"
TIERS = ["water_id_test", "water_ood_size", "water_ood_order", "water_ood_density"]
THEORIES = ["pbe_d4_avdz", "wb97mv_def2tzvpd"]


def run_dir(name: str) -> str:
    """Density run directory for results/revision/<name>, as the analysis resolves it."""
    root = REPO / "results" / "revision" / name
    marker = root / ".density_run_dir"
    if marker.is_file():
        found = Path(marker.read_text().strip())
    else:
        found = Path(subprocess.run(
            [str(REPO / "scripts/revision/lib/latest_run_dir.sh"), str(root)],
            capture_output=True, text=True, check=True, cwd=REPO,
        ).stdout.strip())
    # Markers hold either absolute or repo-relative paths.
    return str(found.relative_to(REPO)) if found.is_absolute() else str(found)


def model(name: str, run: str, out: Path, label: str, phase: str = "production") -> dict:
    spec = {"name": name, "run_dir": run, "out": str(out.relative_to(REPO)), "label": label, "phase": phase}
    if phase == "density":
        spec["checkpoint"] = f"{run}/checkpoints/best_checkpoint_density.pth"
    return spec


def water_label_set(theory: str, tier: str, models: list[dict], with_df: bool = True) -> dict:
    return {
        "name": f"{tier}_{theory}",
        "np": f"{WATER_LABELS}/{tier}_{theory}_base.npy",
        "dens": f"{WATER_LABELS}/{tier}_{theory}.npy",
        "elements": "1,8",
        "with_df": with_df,
        "models": models,
    }


def pbe_water() -> list[dict]:
    theory = "pbe_d4_avdz"
    sets = []
    for tier in TIERS:
        ms = [
            model("pbe", run_dir("water_pbe_d4_avdz"), EVAL / theory / f"{tier}.json", tier),
            model("pbe_density", run_dir("water_pbe_d4_avdz"), EVAL / f"{theory}_densityphase" / f"{tier}.json", tier, "density"),
            model("direct_pbe", run_dir("water_direct_pbe_d4_avdz"), EVAL / f"direct_{theory}" / f"{tier}.json", tier),
            model("direct_pbe_density", run_dir("water_direct_pbe_d4_avdz"), EVAL / f"direct_{theory}_densityphase" / f"{tier}.json", tier, "density"),
        ]
        if tier in ("water_id_test", "water_ood_size"):
            for cut in (4, 5, 6, 8):
                ms.append(model(f"cutoff_{cut}", run_dir(f"water_pbe_orca_cutoff_{cut}"),
                                EVAL / "cutoff_sweep" / f"cutoff_{cut}_{tier}.json", f"cutoff_{cut}_{tier}"))
        for n in (50, 125, 250, 500):
            ms.append(model(f"ntrain_{n}_sad", run_dir(f"water_n{n}_pbe_d4_avdz"),
                            EVAL / "learning_curves" / f"ntrain_{n}_sad_{tier}.json", tier))
            ms.append(model(f"ntrain_{n}_direct", run_dir(f"water_direct_n{n}_pbe_d4_avdz"),
                            EVAL / "learning_curves" / f"ntrain_{n}_direct_{tier}.json", tier))
        sets.append(water_label_set(theory, tier, ms))
    return sets


def hybrid_water_and_malon() -> list[dict]:
    theory = "wb97mv_def2tzvpd"
    sets = []
    for tier in TIERS:
        ms = [
            model("wb97mv", run_dir("water_wb97mv_def2tzvpd"), EVAL / theory / f"{tier}.json", tier),
            model("wb97mv_density", run_dir("water_wb97mv_def2tzvpd"), EVAL / f"{theory}_densityphase" / f"{tier}.json", tier, "density"),
            model("direct_wb97mv", run_dir("water_direct_wb97mv_def2tzvpd"), EVAL / f"direct_{theory}" / f"{tier}.json", tier),
            model("direct_wb97mv_density", run_dir("water_direct_wb97mv_def2tzvpd"), EVAL / f"direct_{theory}_densityphase" / f"{tier}.json", tier, "density"),
        ]
        for seed in ("seed7", "seed123", "seed20260821"):
            ms.append(model(f"wb97mv_{seed}", run_dir(f"water_wb97mv_def2tzvpd_{seed}"),
                            EVAL / f"{theory}_{seed}" / f"{tier}.json", tier))
        sets.append(water_label_set(theory, tier, ms))
    for th in THEORIES:
        run = run_dir(f"malonaldehyde_{th}")
        label = "malonaldehyde_ood_proton_transfer"
        sets.append({
            "name": f"{label}_{th}",
            "np": f"{MALON_LABELS}/{label}_{th}_base.npy",
            "dens": f"{MALON_LABELS}/{label}_{th}.npy",
            "elements": "1,6,8",
            "with_df": True,
            "models": [model(f"malon_{th}", run, EVAL / th / f"{label}.json", label)],
        })
    return sets


def extreme_modes() -> list[dict]:
    """One-mode displacement scans (R1.2), scored with the model trained at each level."""
    # Published models: architecture from the checkpoint's pickled args, SAD
    # prior and basis files from the MD override config, elements from labels.
    paper = {
        "ethanol": ("paper/models/ethanol/2024-03-22_96w7KyGG", "config/md/nn/ethanol.txt"),
        "ethanethiol": ("paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4", "config/md/nn/ethanethiol.txt"),
        "resorcinol": ("paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1", "config/md/nn/resorcinol.txt"),
        "thiophene_2mer": ("paper/models/polythiophene/2024-03-23_1XDL67zp_ext", "config/md/nn/polythiophene_2mer.txt"),
    }
    revision = {
        ("water_n2", "pbe_d4_avdz"): (run_dir("water_pbe_d4_avdz"), "1,8"),
        ("water_n2", "wb97mv_def2tzvpd"): (run_dir("water_wb97mv_def2tzvpd"), "1,8"),
        ("water_n6", "pbe_d4_avdz"): (run_dir("water_pbe_d4_avdz"), "1,8"),
        ("water_n6", "wb97mv_def2tzvpd"): (run_dir("water_wb97mv_def2tzvpd"), "1,8"),
        ("malonaldehyde", "pbe_d4_avdz"): (run_dir("malonaldehyde_pbe_d4_avdz"), "1,6,8"),
        ("malonaldehyde", "wb97mv_def2tzvpd"): (run_dir("malonaldehyde_wb97mv_def2tzvpd"), "1,6,8"),
    }
    short = {"pbe_d4_avdz": "pbe", "wb97mv_def2tzvpd": "hybrid"}
    sets = []
    for (system, th), (run, elements) in revision.items():
        name = f"{system}_{short[th]}"
        sets.append(_mode_set(system, th, elements, [model(name, run, EVAL / "ood_geom" / f"{name}.json", name)]))
    for system, (run, override) in paper.items():
        # The published models are PBE; score them on the PBE-D4 labels only.
        name = f"{system}_pbe"
        spec = model(name, run, EVAL / "ood_geom" / f"{name}.json", name)
        spec.update(args_source="checkpoint", args_override=override)
        sets.append(_mode_set(system, "pbe_d4_avdz", "auto", [spec]))
    return sets


def _mode_set(system: str, theory: str, elements: str, models: list[dict]) -> dict:
    return {
        "name": f"ood_modes_{system}_{theory}",
        "np": f"{GEOM_LABELS}/ood_modes_{system}_{theory}_base.npy",
        "dens": f"{GEOM_LABELS}/ood_modes_{system}_{theory}.npy",
        "elements": elements,
        "with_df": True,
        "models": models,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sets in (
        ("manifest_pbe_water.json", pbe_water()),
        ("manifest_hybrid_water_malon.json", hybrid_water_and_malon()),
        ("manifest_extreme_modes.json", extreme_modes()),
    ):
        path = OUT / name
        path.write_text(json.dumps(sets, indent=1) + "\n")
        n_models = sum(len(s["models"]) for s in sets)
        print(f"wrote {path.relative_to(REPO)}: {len(sets)} label sets, {n_models} model evaluations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
