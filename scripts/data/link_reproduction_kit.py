#!/usr/bin/env python3
"""Link a downloaded figure-reproduction kit into this checkout.

The kit (``scripts/data/build_reproduction_kit.py``, Hub repo
``muhammadhasyim/equiv_dens_ml_dataset``) mirrors the checkout layout: ``datasets/``,
``results/revision/`` and ``paper/models/``. For every entry of those trees that is
missing from the checkout a symlink into the kit is created; existing files and
directories (git-tracked geometries, local runs, scores you produced yourself) are
left untouched and the walk descends into existing directories so that, for example,
a local run directory that only has ``best_<ID>.pth`` gains the kit's
``checkpoints/latest_checkpoint.pth`` and ``args.txt`` next to it.

    python scripts/data/link_reproduction_kit.py --kit-root /path/to/download [--repo-root .] [--dry-run]

Never deletes or overwrites anything. Broken symlinks at a target path are replaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOTS = ("datasets", "results/revision", "paper/models")
SKIP = {"manifest.json", "README.md", ".cache", ".gitattributes"}


def link_missing(src: Path, dst: Path, dry_run: bool, log: list[str]) -> None:
    for child in sorted(src.iterdir()):
        if child.name in SKIP:
            continue
        target = dst / child.name
        if target.is_symlink() and not target.exists():
            log.append(f"replace broken link {target}")
            if not dry_run:
                target.unlink()
        if not target.exists() and not target.is_symlink():
            log.append(f"link {target} -> {child}")
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(child)
        elif child.is_dir() and target.is_dir() and not target.is_symlink():
            link_missing(child, target, dry_run, log)
        elif child.is_file() and target.is_file() and target.stat().st_size != child.stat().st_size:
            log.append(f"keep (differs from kit) {target}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    kit = args.kit_root.resolve()
    repo = args.repo_root.resolve()
    log: list[str] = []
    for root in ROOTS:
        src = kit / root
        if not src.is_dir():
            log.append(f"kit has no {root}/")
            continue
        (repo / root).mkdir(parents=True, exist_ok=True)
        link_missing(src, repo / root, args.dry_run, log)
    links = sum(1 for line in log if line.startswith("link "))
    for line in log:
        if not line.startswith("link "):
            print(line)
    print(f"{'would create' if args.dry_run else 'created'} {links} links from {kit} into {repo}")


if __name__ == "__main__":
    main()
