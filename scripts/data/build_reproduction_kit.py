#!/usr/bin/env python3
"""Stage, upload and verify the figure-reproduction kit (Hugging Face dataset repo).

The kit holds every file the JACS-revision figures, tables and the step-timing benchmark
read, laid out with the same relative paths as this checkout so that
``scripts/data/fetch_reproduction_kit.sh`` can link it straight into a clone:

    datasets/...          label arrays, basis/prior tables, geometries (self-contained copies)
    results/revision/...  minimal checkpoints of every scored model (best_<ID>.pth, args.txt,
                          latest/best checkpoints where the scored weights are the EMA) and all
                          score / evaluation / timing outputs
    paper/models/...      the three original-paper models used by the timing benchmark
    manifest.json         path, bytes, sha256 and origin of every file
    README.md             layout, download and per-figure commands

Stages:
    build_reproduction_kit.py --stage DIR [--restored DIR] [--dry-run]
    build_reproduction_kit.py --stage DIR --upload [--repo muhammadhasyim/equiv_dens_ml_dataset]
    build_reproduction_kit.py --verify DIR --manifest DIR/manifest.json

Files are hard-linked into --stage when possible (same filesystem), copied otherwise.
Nothing under the checkout is modified. Tokens are never printed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "muhammadhasyim/equiv_dens_ml_dataset"

# --- selection -------------------------------------------------------------------------------

DATASET_DIRS = [
    "datasets/revision/water_labels",
    "datasets/revision/ood_geom_labels",
    "datasets/revision/water_clusters",
    "datasets/revision/malonaldehyde_labels",
    "datasets/revision/ood_modes",
    "datasets/revision/water_ood",
    "datasets/revision/water_ood_rackers",
]
DATASET_FILE_GLOBS = [
    "datasets/revision/sad_*.npy",
    "datasets/revision/atomic_energies_*.npy",
    "datasets/revision/external/*.csv",
    "datasets/revision/external/*.md",
    "datasets/revision/external/*.xyz",
    "datasets/augccpv*.npy",
    "datasets/free_atom_densities_*",
    "datasets/atomization_energy_augccpvdz.npy",
    "datasets/ethanol_train_10.*",
    "datasets/resorcinol_augccpvdz_test.npy",
    "datasets/resorcinol_combo_kmeansidx-1000_train.npy",
    "datasets/thiophene6mer_test_d4.npy",
    "datasets/thiophene12mer_test.xyz",
    "datasets/thiophene12mer_init.npy",
    "datasets/thiophene_all_train_d4.npy",
]
RUN_DIR_GLOBS = [
    "results/revision/water_pbe_d4_avdz/*",
    "results/revision/water_wb97mv_def2tzvpd/*",
    "results/revision/water_direct_pbe_d4_avdz/*",
    "results/revision/water_direct_wb97mv_def2tzvpd/*",
    "results/revision/water_pbe_orca_cutoff_*/*",
    "results/revision/water_n*_pbe_d4_avdz/*",
    "results/revision/water_direct_n*_pbe_d4_avdz/*",
    "results/revision/water_wb97mv_def2tzvpd_seed*/*",
    "results/revision/water_*_energy_v3*/*",
    "results/revision/water_*_energy_cutoff_*/*",
    "results/revision/water_wb97mv_def2tzvpd_cutoff_*/*",
    "results/revision/water_pbe_d4_avdz_energy/*",
    "results/revision/water_direct_pbe_d4_avdz_energy/*",
    "results/revision/malonaldehyde_*/*",
    "results/revision/malon_energy_fw/*/energy/*",
    "paper/models/polythiophene/2024-03-23_1XDL67zp_ext",
    "paper/models/ethanol/2024-04-18_5K9Ne7it_ext",
    "paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1",
]
RESULTS_ROOT = "results/revision"
RESULTS_EXCLUDE_DIRS = {
    "water_orca", "ood_geom_orca", "cutoff_migration", "wb97mv_resume_20261101",
    "wb97mv_gpu_packing_20261102", "energy_training_dashboard", "manuscript_cutoff_update_20261025",
    "logs", "full_density_pilot", "similarity_full", "figures",
}
RESULTS_EXCLUDE_SUBDIRS = {"results/revision/timing_v2/fastmodel_gl067"}  # contaminated by gpu_burn
RESULTS_INCLUDE_FIGURES = "results/revision/figures/response_v2"  # reference outputs of the figure script
RESULTS_SUFFIXES = {".json", ".jsonl", ".md", ".csv", ".npz", ".txt", ".pdf", ".png", ".yaml", ".yml", ".tsv"}
RESULTS_MAX_BYTES = 200 * 2**20

RUN_ID = re.compile(r"^\d{4}-\d{2}-\d{2}_[A-Za-z0-9]{8}")


def run_file_wanted(rel_in_run: str) -> bool:
    """Minimal checkpoint set of one run directory."""
    parts = rel_in_run.split("/")
    if len(parts) == 1:
        name = parts[0]
        return (name.startswith("best_") and name.endswith(".pth")) or name == "args.txt" or name.endswith(".json")
    if len(parts) == 2 and parts[0] == "checkpoints":
        name = parts[1]
        return name == "latest_checkpoint.pth" or (name.startswith("best_checkpoint") and name.endswith(".pth"))
    return False


def walk_files(root: Path, followlinks: bool):
    for folder, dirs, names in os.walk(root, followlinks=followlinks):
        dirs.sort()
        for name in sorted(names):
            yield Path(folder) / name


def select(repo: Path, restored: Path | None):
    """Yield (relative path in kit, source path, category, origin)."""
    seen = set()

    def emit(rel, src, category, origin):
        rel = str(rel)
        if rel in seen or not src.is_file():
            return
        seen.add(rel)
        yield rel, src, category, origin

    for d in DATASET_DIRS:
        root = repo / d
        if not root.exists():
            print(f"warning: missing {d}", file=sys.stderr)
            continue
        for f in walk_files(root, followlinks=True):
            yield from emit(Path(d) / f.relative_to(root), f, "datasets", "checkout")
    for pattern in DATASET_FILE_GLOBS:
        for f in sorted(repo.glob(pattern)):
            if f.is_file():
                yield from emit(f.relative_to(repo), f, "datasets", "checkout")
    for pattern in RUN_DIR_GLOBS:
        for run in sorted(repo.glob(pattern)):
            if not run.is_dir():
                continue
            cat = "paper_models" if pattern.startswith("paper/") else "checkpoints"
            for f in walk_files(run, followlinks=False):
                rel_in_run = str(f.relative_to(run))
                if run_file_wanted(rel_in_run):
                    yield from emit(f.relative_to(repo), f, cat, "checkout")
    results = repo / RESULTS_ROOT
    for folder, dirs, names in os.walk(results, followlinks=False):
        rel_folder = Path(folder).relative_to(repo)
        if rel_folder == Path(RESULTS_ROOT):
            dirs[:] = sorted(d for d in dirs if d not in RESULTS_EXCLUDE_DIRS)
        dirs[:] = sorted(d for d in dirs if str(rel_folder / d) not in RESULTS_EXCLUDE_SUBDIRS)
        for name in sorted(names):
            f = Path(folder) / name
            if f.suffix in RESULTS_SUFFIXES and f.is_file() and not f.is_symlink() and f.stat().st_size <= RESULTS_MAX_BYTES:
                yield from emit(f.relative_to(repo), f, "results", "checkout")
    figs = repo / RESULTS_INCLUDE_FIGURES
    if figs.exists():
        for f in walk_files(figs, followlinks=False):
            if f.suffix in RESULTS_SUFFIXES:
                yield from emit(f.relative_to(repo), f, "results", "checkout")
    if restored is not None:
        for part in sorted(p for p in restored.iterdir() if p.is_dir()):
            for f in walk_files(part, followlinks=False):
                rel = f.relative_to(part)
                parts = rel.parts
                run_idx = next((i for i, p in enumerate(parts) if RUN_ID.match(p)), None)
                if run_idx is None:
                    continue
                run = Path(*parts[: run_idx + 1])
                if run_file_wanted(str(Path(*parts[run_idx + 1:]))):
                    cat = "paper_models" if parts[0] == "paper" else "checkpoints"
                    yield from emit(rel, f, cat, f"archive:{part.name}")


# --- staging -----------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def place(src: Path, dst: Path):
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "kept"
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def build(args):
    repo = args.repo_root
    entries = list(select(repo, args.restored))
    totals = {}
    for rel, src, cat, origin in entries:
        totals.setdefault(cat, [0, 0])
        totals[cat][0] += 1
        totals[cat][1] += src.stat().st_size
    for cat, (n, b) in sorted(totals.items()):
        print(f"{cat:13s} {n:6d} files {b / 2**30:8.2f} GiB")
    print(f"{'total':13s} {sum(v[0] for v in totals.values()):6d} files "
          f"{sum(v[1] for v in totals.values()) / 2**30:8.2f} GiB")
    if args.dry_run:
        if args.list:
            for rel, src, cat, origin in entries:
                print(cat, origin, rel, src.stat().st_size)
        return
    stage = args.stage
    stage.mkdir(parents=True, exist_ok=True)
    actions = {}
    for rel, src, cat, origin in entries:
        how = place(src, stage / rel)
        actions[how] = actions.get(how, 0) + 1
    print("placed:", actions)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        hashes = dict(zip((e[0] for e in entries), pool.map(lambda e: sha256(e[1]), entries)))
    files = [{"path": rel, "bytes": src.stat().st_size, "sha256": hashes[rel], "category": cat, "origin": origin}
             for rel, src, cat, origin in entries]
    manifest = {
        "repo_id": args.repo,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "checkout": str(repo),
        "totals": {cat: {"files": n, "bytes": b} for cat, (n, b) in totals.items()},
        "files": files,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (stage / "README.md").write_text(readme(manifest))
    print("staged", len(files), "files into", stage)


def readme(manifest) -> str:
    t = manifest["totals"]
    rows = "\n".join(f"| {cat} | {v['files']} | {v['bytes'] / 2**30:.2f} |" for cat, v in sorted(t.items()))
    return f"""---
license: other
pretty_name: equiv_dens_ml figure-reproduction kit
---

# equiv_dens_ml figure-reproduction kit

Everything the JACS-revision figures, tables and the step-timing benchmark of
[equiv_dens_ml](https://github.com/MihailBogojeski/equiv_dens_ml) read, laid out with the
same relative paths as the code checkout. Generated {manifest['generated']} by
`scripts/data/build_reproduction_kit.py`; `manifest.json` lists every file with its SHA-256.

| category | files | GiB |
|---|---|---|
{rows}

## Layout

- `datasets/` label arrays (`*_base.npy` and full density labels for the water tiers, extreme normal
  modes and malonaldehyde), SAD priors, atomic energies, basis and free-atom tables, geometries.
- `results/revision/<run family>/<date_ID>/` minimal checkpoints: `best_<ID>.pth` (bare state dict),
  `args.txt`, and `checkpoints/latest_checkpoint.pth` / `best_checkpoint*.pth` where the scored weights
  are the EMA of a continued run (energy cutoff sweep at 100k steps) or where an energy head loads the
  density run (`best_checkpoint_density.pth`).
- `results/revision/...` score and evaluation outputs: `energy_forces_v3/*/{{summary.json,per_frame.csv}}`,
  `rescore_fullref/eval`, `cutoff_100k_evaluation_20261101`, `timing_v2/{{dedicated_gpu,split}}/runs`,
  `overlap_v2/*.npz`, `equilibrium_v2`, `density_functionals*`, `cv`, and `figures/response_v2` (the
  reference outputs of the figure script).
- `paper/models/` the three original-paper models the timing benchmark loads.

## Use

```bash
pip install -U huggingface_hub
scripts/data/fetch_reproduction_kit.sh --repo {manifest['repo_id']}   # downloads and links into the checkout
.venv/bin/python scripts/data/build_reproduction_kit.py --verify /path/to/download --manifest /path/to/download/manifest.json
```

Then, from the checkout:

| figure / table | command |
|---|---|
| water tiers, error vs distance, energy vs distance, malonaldehyde, extreme modes, equilibrium, timing | `python experiments/jacs_revision/analysis/plot_response_figures_v2.py` |
| energy cutoff sweep figure and tables | `python experiments/jacs_revision/analysis/cutoff_sweep_100k_update.py --check-docs` |
| timing tables | `python experiments/jacs_revision/analysis/summarize_step_timing.py results/revision/timing_v2/dedicated_gpu` |
| density tier tables | `python experiments/jacs_revision/analysis/summarize_rescore_fullref.py` |
| re-score an energy head | `python experiments/jacs_revision/analysis/score_energy_forces.py --run results/revision/<family>/<run> --theory <theory> --out <dir>` |
| step timing benchmark (GPU) | `python experiments/jacs_revision/analysis/benchmark_step_timing.py drive --devices cuda --threads 16` |

The raw ORCA label outputs, MD trajectories and full training states are not part of this kit; they
remain in the private research archive (`muhammadhasyim/equiv-dens-research-archive`).
"""


def upload(args):
    from huggingface_hub import HfApi, get_token
    print("token present:", get_token() is not None)
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(args.repo)
    print("repo", args.repo, "private" if info.private else "PUBLIC")
    api.upload_large_folder(repo_id=args.repo, folder_path=str(args.stage), repo_type="dataset",
                            num_workers=args.workers, print_report_every=60)
    info = api.dataset_info(args.repo)
    print("uploaded; revision", info.sha)


def verify(args):
    manifest = json.loads(Path(args.manifest).read_text())
    bad, missing = [], []
    for entry in manifest["files"]:
        p = args.verify / entry["path"]
        if not p.is_file():
            missing.append(entry["path"])
        elif sha256(p) != entry["sha256"]:
            bad.append(entry["path"])
    print(f"checked {len(manifest['files'])} files: {len(missing)} missing, {len(bad)} mismatched")
    for p in missing[:20]:
        print("missing", p)
    for p in bad[:20]:
        print("mismatch", p)
    sys.exit(1 if (bad or missing) else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--stage", type=Path, help="staging directory (hard links into the kit layout)")
    ap.add_argument("--restored", type=Path, help="directory of archive parts restored with archive_to_hf.py")
    ap.add_argument("--repo", default=DEFAULT_REPO_ID)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="with --dry-run: print every selected file")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--verify", type=Path, help="downloaded kit directory to check against --manifest")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    if args.verify:
        if not args.manifest:
            ap.error("--verify needs --manifest")
        verify(args)
    elif args.upload:
        if not args.stage or not (args.stage / "manifest.json").is_file():
            ap.error("--upload needs a staged kit with manifest.json")
        upload(args)
    else:
        if not args.stage and not args.dry_run:
            ap.error("--stage or --dry-run")
        build(args)


if __name__ == "__main__":
    main()
