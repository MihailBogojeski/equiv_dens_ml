#!/usr/bin/env python3
"""Phase A: 10× each existing parent by subsampled Langevin (or ice rigid-body).

Writes a *new* tree (``*_md10x/``) so the finished 3k campaign is untouched.
GFN2-xTB is the default engine; tests and dry runs inject ``--calculator fake``.
Ice cutouts prefer rigid-molecule translation/libration so the oxygen network
stays crystalline; droplets continue a short melt and keep the second half.

Usage:
  python scripts/revision/water/thicken_with_md.py \\
    --water-out datasets/revision/water_clusters_md10x \\
    --ood-out datasets/revision/water_ood_md10x \\
    --malon-out datasets/revision/malonaldehyde_md10x
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/thicken_with_md.py'


import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# tblite/OpenBLAS otherwise spawn a thread team per worker and the 4-way
# GFN2 pool thrashes. One thread per process is the right default here.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)

from experiments.jacs_revision.water.generate_malonaldehyde import proton_transfer_indices  # noqa: E402
from experiments.jacs_revision.water.md_sampling import (  # noqa: E402
    MALON_TRAIN_DELTA_MAX,
    Frame,
    choose_calculator,
    dedup_frames,
    infer_kind,
    protocol_for,
    read_xyz_symbol_frames,
    thicken_one_parent,
    write_xyz_frames,
)

_REPO_ROOT = repo_root()

# Phase A default is intra-parent only: drop snapshots that did not move.
# Global RMSD/CV floors belong to Phase B parent search, where the goal is
# distinct isomers rather than 10 thermal draws of the same lattice.
PARENT_RMSD_CUTOFF = 0.05
RMSD_CUTOFF = {
    "train": 0.12,
    "val": 0.12,
    "id_test": 0.12,
    "ood_size": 0.20,
    "ood_order": 0.25,
    "ood_density": 0.25,
    "ood_proton_transfer": 0.08,
}
CV_FLOOR = {
    "train": 0.04,
    "val": 0.04,
    "id_test": 0.04,
    "ood_size": 0.05,
    "ood_order": 0.08,
    "ood_density": 0.08,
    "ood_proton_transfer": None,
}


def _malon_idx_from_frame(symbols: list[str], coords):
    from ase import Atoms

    return proton_transfer_indices(Atoms(symbols=symbols, positions=coords))


def _thicken_parent_job(payload: dict) -> dict:
    """Module-level worker so ProcessPoolExecutor can pickle the task."""
    symbols = payload["symbols"]
    coords = np.asarray(payload["coords"], dtype=float)
    kind = payload["kind"]
    ice_mode = payload["ice_mode"]
    calc = None
    if not (kind == "ice" and ice_mode == "rigid"):
        calc = choose_calculator(payload["calculator_name"], kind, len(symbols))
    malon_idx = None
    if kind.startswith("malon"):
        try:
            malon_idx = _malon_idx_from_frame(symbols, coords)
        except ValueError:
            malon_idx = None
    try:
        frames, meta = thicken_one_parent(
            symbols,
            coords,
            n_keep=payload["n_keep"],
            n_eq=payload["n_eq"],
            stride=payload["stride"],
            timestep_fs=payload["timestep_fs"],
            temperature_k=payload["temperature_k"],
            calculator=calc,
            kind=kind,
            seed=payload["seed"],
            delta_max=MALON_TRAIN_DELTA_MAX,
            malon_idx=malon_idx,
            ice_mode=ice_mode if kind == "ice" else "md",
        )
    except Exception as exc:
        print(
            f"  parent {payload['parent_index']} failed ({type(exc).__name__}: {exc}); skipping",
            flush=True,
        )
        return {
            "parent_index": payload["parent_index"],
            "parent_comment": payload["comment"],
            "n_atoms": len(symbols),
            "frames": [],
            "meta": {
                "n_accepted": 0,
                "n_rejected_steric": 0,
                "n_rejected_basin": 0,
                "protocol": "failed",
                "temperature_k": payload["temperature_k"],
                "error": f"{type(exc).__name__}: {exc}",
            },
            "kind": kind,
        }
    serial = []
    for frame in frames:
        serial.append(
            {
                "symbols": frame.symbols,
                "coords": frame.coords.tolist(),
                "comment": f"{payload['comment']} | {frame.comment} parent={payload['parent_index']}",
                "meta": {
                    **frame.meta,
                    "parent_index": payload["parent_index"],
                    "parent_comment": payload["comment"],
                    "split": payload["split"],
                },
            }
        )
    return {
        "parent_index": payload["parent_index"],
        "parent_comment": payload["comment"],
        "n_atoms": len(symbols),
        "frames": serial,
        "meta": meta,
        "kind": kind,
    }


def thicken_xyz(
    src: Path,
    dest: Path,
    split: str,
    *,
    n_keep: int,
    calculator_name: str,
    seed: int,
    max_parents: int | None,
    ice_mode: str,
    skip_dedup: bool,
    n_eq_override: int | None = None,
    stride_override: int | None = None,
    workers: int = 1,
    resume: bool = True,
    dedup_mode: str = "parent",
) -> dict:
    if not src.is_file():
        return {"xyz": str(src), "skipped": True, "reason": "missing"}
    parents = read_xyz_symbol_frames(src)
    if max_parents is not None:
        parents = parents[: max(0, max_parents)]

    progress_path = dest.with_suffix(dest.suffix + ".progress.json")
    kept: list[Frame] = []
    records: list[dict] = []
    completed_parents: set[int] = set()
    if resume and dest.is_file() and progress_path.is_file():
        existing = json.loads(progress_path.read_text())
        completed_parents = {int(i) for i in existing.get("completed_parents", [])}
        if not completed_parents and "next_parent" in existing:
            completed_parents = set(range(int(existing["next_parent"])))
        kept = [
            Frame(s, c, comment, {"parent_index": i})
            for i, (s, c, comment) in enumerate(read_xyz_symbol_frames(dest))
        ]
        records = list(existing.get("parents", []))
        print(f"  {split}: resume, {len(completed_parents)} parents done, {len(kept)} frames", flush=True)

    payloads = []
    for parent_i, (symbols, coords, comment) in enumerate(parents):
        if parent_i in completed_parents:
            continue
        kind = infer_kind(split, len(symbols), symbols)
        proto = protocol_for(kind, len(symbols))
        payloads.append(
            {
                "parent_index": parent_i,
                "symbols": symbols,
                "coords": np.asarray(coords, dtype=float).tolist(),
                "comment": comment,
                "kind": kind,
                "n_keep": n_keep,
                "n_eq": n_eq_override if n_eq_override is not None else proto["n_eq"],
                "stride": stride_override if stride_override is not None else proto["stride"],
                "timestep_fs": proto["timestep_fs"],
                "temperature_k": proto["temperature_k"],
                "calculator_name": calculator_name,
                "ice_mode": ice_mode,
                "seed": seed + parent_i * 101,
                "split": split,
            }
        )

    n_rejected_dedup = 0

    def _absorb(result: dict) -> None:
        nonlocal n_rejected_dedup
        frames = [
            Frame(item["symbols"], np.asarray(item["coords"], dtype=float), item["comment"], item["meta"])
            for item in result["frames"]
        ]
        if skip_dedup or dedup_mode == "none":
            accepted, rejected = frames, []
        elif dedup_mode == "parent":
            accepted, rejected = dedup_frames(
                frames,
                rmsd_cutoff=PARENT_RMSD_CUTOFF,
                cv_floor=None,
                existing=None,
            )
        else:
            accepted, rejected = dedup_frames(
                frames,
                rmsd_cutoff=RMSD_CUTOFF.get(split, 0.15),
                cv_floor=CV_FLOOR.get(split),
                existing=kept,
            )
        kept.extend(accepted)
        n_rejected_dedup += len(rejected)
        meta = result["meta"]
        records.append(
            {
                "parent_index": result["parent_index"],
                "parent_comment": result["parent_comment"],
                "n_atoms": result["n_atoms"],
                "kind": result["kind"],
                "accepted": len(accepted),
                "rejected_steric": meta["n_rejected_steric"],
                "rejected_basin": meta["n_rejected_basin"],
                "rejected_dedup": len(rejected),
                "protocol": meta["protocol"],
                "temperature_k": meta["temperature_k"],
            }
        )
        print(
            f"  {split} parent {result['parent_index'] + 1}/{len(parents)}: "
            f"+{len(accepted)} kept (running {len(kept)})",
            flush=True,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_xyz_frames(dest, kept)
        completed_parents.add(int(result["parent_index"]))
        next_parent = 0
        while next_parent in completed_parents:
            next_parent += 1
        progress_path.write_text(
            json.dumps(
                {
                    "next_parent": next_parent,
                    "completed_parents": sorted(completed_parents),
                    "n_frames": len(kept),
                    "parents": records,
                },
                indent=2,
            )
        )

    if workers <= 1:
        for payload in payloads:
            _absorb(_thicken_parent_job(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_thicken_parent_job, payload) for payload in payloads]
            for fut in as_completed(futures):
                _absorb(fut.result())

    return {
        "xyz": str(dest),
        "n_parents": len(parents),
        "n_frames": len(kept),
        "n_rejected_dedup": n_rejected_dedup,
        "parents": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-dir", type=Path, default=_REPO_ROOT / "datasets/revision/water_clusters")
    parser.add_argument("--ood-dir", type=Path, default=_REPO_ROOT / "datasets/revision/water_ood")
    parser.add_argument("--malon-dir", type=Path, default=_REPO_ROOT / "datasets/revision/malonaldehyde")
    parser.add_argument("--water-out", type=Path, default=_REPO_ROOT / "datasets/revision/water_clusters_md10x")
    parser.add_argument("--ood-out", type=Path, default=_REPO_ROOT / "datasets/revision/water_ood_md10x")
    parser.add_argument("--malon-out", type=Path, default=_REPO_ROOT / "datasets/revision/malonaldehyde_md10x")
    parser.add_argument("--n-keep", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--calculator",
        choices=("auto", "gfn2", "tip4p", "fake", "none"),
        default="auto",
        help="auto: GFN2 for small/malon, TIP4P for n≥16 ice/droplets",
    )
    parser.add_argument(
        "--ice-mode",
        choices=("md", "rigid"),
        default="rigid",
        help="rigid keeps the crystalline oxygen network (default); md adds short Langevin",
    )
    parser.add_argument("--max-parents", type=int, default=None, help="smoke: thicken only the first N parents per file")
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument(
        "--dedup",
        choices=("none", "parent", "global"),
        default="parent",
        help="parent: drop unmoved snapshots of one parent (Phase A). global: RMSD/CV vs all kept (Phase B)",
    )
    parser.add_argument("--n-eq", type=int, default=None, help="override equilibration steps for every parent")
    parser.add_argument("--stride", type=int, default=None, help="override snapshot stride (MD steps)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--splits",
        default="train,val,id_test,ood_size,ood_order,ood_density,ood_proton_transfer",
        help="comma-separated split names to process",
    )
    args = parser.parse_args()

    wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
    jobs = []
    for split in ("train", "val", "id_test", "ood_size"):
        if split in wanted:
            jobs.append((args.water_dir / f"{split}.xyz", args.water_out / f"{split}.xyz", split))
    for split in ("ood_order", "ood_density"):
        if split in wanted:
            jobs.append((args.ood_dir / f"{split}.xyz", args.ood_out / f"{split}.xyz", split))
    # Malonaldehyde shares split names with water but lives in its own tree.
    if "train" in wanted:
        jobs.append((args.malon_dir / "train.xyz", args.malon_out / "train.xyz", "train"))
    if "val" in wanted:
        jobs.append((args.malon_dir / "val.xyz", args.malon_out / "val.xyz", "val"))
    if "ood_proton_transfer" in wanted:
        jobs.append(
            (
                args.malon_dir / "ood_proton_transfer.xyz",
                args.malon_out / "ood_proton_transfer.xyz",
                "ood_proton_transfer",
            )
        )

    # The loop above appended malon train/val whenever those names were wanted,
    # which is correct because water and malon live in different output trees.
    manifest = {
        "seed": args.seed,
        "n_keep": args.n_keep,
        "calculator": args.calculator,
        "ice_mode": args.ice_mode,
        "splits": {},
    }
    for src, dest, split in jobs:
        if not src.is_file():
            print(f"skip missing {src}")
            continue
        print(f"thickening {src} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        manifest["splits"][str(dest)] = thicken_xyz(
            src,
            dest,
            split,
            n_keep=args.n_keep,
            calculator_name=args.calculator,
            seed=args.seed,
            max_parents=args.max_parents,
            ice_mode=args.ice_mode,
            skip_dedup=args.skip_dedup,
            dedup_mode=args.dedup,
            n_eq_override=args.n_eq,
            stride_override=args.stride,
            workers=args.workers,
            resume=not args.no_resume,
        )

    for out_dir in (args.water_out, args.ood_out, args.malon_out):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in rec.items() if kk != "parents"} for k, rec in manifest["splits"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
