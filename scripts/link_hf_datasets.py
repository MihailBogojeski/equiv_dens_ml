#!/usr/bin/env python3
"""Symlink a downloaded Hugging Face datasets snapshot into ``datasets/``.

Used by ``scripts/download_hf_datasets.sh`` after ``hf download``. Walks the
downloaded root and, for every entry that is not already tracked by git under
``datasets/``, creates (or refreshes) a symlink at the corresponding
``datasets/`` path pointing back into the download root. Tracked geometry and
manifest files (small ``.xyz``/``.json`` inputs kept in git) are left alone so
mixed directories such as ``datasets/revision/water_clusters/`` end up with
tracked files sitting next to symlinked label arrays.

Purpose and usage:
    Restore the ``datasets/`` layout that training/eval code expects after a
    fresh clone plus ``hf download``, without duplicating any bytes.

Parameters:
    --data-root: Directory that ``hf download`` populated (the Hub repo mirror).
    --repo-root: Root of this git checkout (defaults to the repo containing
        this script).

Side effects:
    Creates symlinks under ``datasets/``. Never deletes or modifies files in
    ``--data-root``. Existing broken symlinks at a target path are replaced;
    existing real files/directories that are *not* git-tracked are left in
    place with a warning (to avoid silently shadowing local work).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def tracked_set(repo_root: Path) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "datasets/"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout
    return set(out.splitlines())


def _link_tree(
    node: Path,
    data_root: Path,
    datasets_root: Path,
    tracked: set[str],
    warnings: list[str],
) -> None:
    rel = node.relative_to(data_root)
    dst = datasets_root / rel
    dst_rel_str = str(Path("datasets") / rel)

    if dst.is_symlink():
        dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(node)
        return

    if node.is_dir():
        has_tracked = any(t == dst_rel_str or t.startswith(dst_rel_str + "/") for t in tracked)
        if not dst.exists() and not has_tracked:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(node)
            return
        if not has_tracked and not dst.exists():
            dst.symlink_to(node)
            return
        # Mixed or pre-existing: recurse so tracked siblings are untouched.
        dst.mkdir(parents=True, exist_ok=True)
        for child in sorted(node.iterdir()):
            _link_tree(child, data_root, datasets_root, tracked, warnings)
        return

    # node is a file
    if dst.exists() and not dst.is_symlink():
        if dst_rel_str in tracked:
            return  # tracked file, never touch
        warnings.append(f"skip (untracked real file already present): {dst_rel_str}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(node)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Directory populated by `hf download`")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Root of this git checkout",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    if not data_root.is_dir():
        print(f"error: --data-root {data_root} does not exist", file=sys.stderr)
        return 1

    tracked = tracked_set(repo_root)
    warnings: list[str] = []
    for child in sorted(data_root.iterdir()):
        if child.name in {".cache", ".gitattributes", "README.md"}:
            continue
        _link_tree(child, data_root, repo_root / "datasets", tracked, warnings)

    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"Linked datasets/ -> {data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
