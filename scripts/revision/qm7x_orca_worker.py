#!/usr/bin/env python3
"""Run one shard of geometries through ORCA and convert to DensNet calc_dicts.

For each closed-shell frame: write .inp, run orca, orca_2json, map MOs to
PySCF order, DF-fit onto the auxiliary basis, append (mol.pack(), calc_dict).

Restart model. Progress is recorded per frame in ``status.jsonl`` and results
are re-saved after every success, so a task killed at the walltime resumes at
the next unfinished frame rather than starting over. Shards are additionally
claimed with an exclusive-create lock, which lets a CPU array and a GPU array
drain the same shard directory concurrently; a claim left behind by a killed
worker goes stale and is reclaimed. Re-running a finished shard is a no-op.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from qm7x_orca_common import (  # noqa: E402
    MAXCORE_MB_DEFAULT,
    NPROCS_DEFAULT,
    is_closed_shell,
    read_shard,
    run_orca_single,
)
from shard_claim import (  # noqa: E402
    DEFAULT_STALE_S,
    acquire_claim,
    atomic_save_npy,
    atomic_write_json,
    touch_claim,
)
from theory_levels import DEFAULT_LEVEL, get_level, level_keys  # noqa: E402

ORCA_BIN_DEFAULT = os.environ.get("ORCA_BIN", "orca")
ORCA2JSON_DEFAULT = os.environ.get("ORCA2JSON_BIN", "orca_2json")


def _which(name: str) -> str:
    return shutil.which(name) or name


def cleanup_orca_scratch(job_dir: Path, keep_out: bool) -> None:
    keep_suffix = {".json", ".npy", ".jsonl"}
    if keep_out:
        keep_suffix.update({".out", ".inp"})
    if not job_dir.exists():
        return
    for path in job_dir.iterdir():
        if path.suffix in keep_suffix or path.name in {"status.jsonl", "results.npy"}:
            continue
        if path.is_file():
            path.unlink()


def process_frame(
    frame: dict,
    job_dir: Path,
    *,
    level,
    orca_bin: str,
    orca2json: str,
    nprocs: int,
    maxcore_mb: int,
    fit_df: bool,
    dry_run: bool,
    keep_orca: bool,
) -> dict:
    index = int(frame["index"])
    z = frame["atom_numbers"]
    xyz = frame["positions"]
    if not is_closed_shell(z):
        return {"index": index, "status": "skipped_open_shell", "n_elec": int(sum(z))}

    if dry_run:
        from qm7x_orca_common import write_orca_inp

        job_dir.mkdir(parents=True, exist_ok=True)
        inp = write_orca_inp(
            job_dir / "job.inp", z, xyz, nprocs=nprocs, maxcore_mb=maxcore_mb, level=level
        )
        return {
            "index": index,
            "status": "dry_run",
            "inp": str(inp),
            "theory": level.key,
            "keywords": level.orca_keywords,
        }

    mol, calc, diag = run_orca_single(
        z,
        xyz,
        job_dir,
        level=level,
        orca_bin=orca_bin,
        orca2json_bin=orca2json,
        nprocs=nprocs,
        maxcore_mb=maxcore_mb,
        fit_df=fit_df,
    )
    if not keep_orca:
        cleanup_orca_scratch(job_dir, keep_out=False)
    return {
        "index": index,
        "status": "ok",
        "energy": float(calc["energy"]),
        "n_atoms": diag["n_atoms"],
        "nao": diag["nao"],
        "naux": diag["naux"],
        "xc": calc["xc"],
        "auxbasis": calc["auxbasis"],
        "theory": level.key,
        "t_scf_s": round(diag["t_scf_s"], 2),
        "t_df_s": round(diag["t_df_s"], 2),
        "mo_orthonormality_error": diag["mo_orthonormality_error"],
        "packed": (mol.pack(), calc),
    }


def load_done_indices(status_path: Path) -> set[int]:
    done: set[int] = set()
    if not status_path.exists():
        return done
    for line in status_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") in {"ok", "skipped_open_shell"}:
            done.add(int(rec["index"]))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--theory", default=DEFAULT_LEVEL, choices=level_keys())
    parser.add_argument("--orca-bin", default=ORCA_BIN_DEFAULT)
    parser.add_argument("--orca2json-bin", default=ORCA2JSON_DEFAULT)
    parser.add_argument("--nprocs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", NPROCS_DEFAULT)))
    parser.add_argument("--maxcore-mb", type=int, default=MAXCORE_MB_DEFAULT)
    parser.add_argument("--no-df", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-orca", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--no-claim", action="store_true", help="skip the shard lock (single-pool runs)")
    parser.add_argument("--claim-stale-s", type=float, default=DEFAULT_STALE_S)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="reopen a finished shard and re-run only the frames that errored",
    )
    parser.add_argument("--ortho-tol", type=float, default=1e-6, help="fail a frame whose AO map looks wrong")
    args = parser.parse_args()

    level = get_level(args.theory)
    shard = read_shard(args.shard)
    shard_id = int(shard["shard_id"])
    split = shard.get("split", "train")
    outdir = Path(args.outdir) if args.outdir else _REPO_ROOT / "results/revision/qm7x_orca" / split
    shard_dir = outdir / f"shard_{shard_id:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    status_path = shard_dir / "status.jsonl"
    results_path = shard_dir / "results.npy"
    done_path = shard_dir / "shard.done"
    claim_path = shard_dir / "shard.claim"

    if done_path.exists():
        if not args.retry_errors:
            print(f"shard {shard_id:04d} already done: {done_path}")
            return 0
        # Errored frames were never added to the done set, so clearing the
        # marker is enough to make the normal resume path pick them up again.
        done_path.unlink(missing_ok=True)
        print(f"shard {shard_id:04d} reopened for error retry")

    holding_claim = False
    if not args.no_claim:
        if not acquire_claim(claim_path, args.claim_stale_s, f"{args.theory}/orca"):
            print(f"shard {shard_id:04d} claimed by another worker; nothing to do")
            return 0
        holding_claim = True

    try:
        results = []
        if args.resume and results_path.exists():
            results = list(np.load(results_path, allow_pickle=True))
        done = load_done_indices(status_path) if args.resume else set()

        orca_bin = _which(args.orca_bin)
        orca2json = _which(args.orca2json_bin)
        if not args.dry_run and shutil.which(Path(orca_bin).name) is None and not Path(orca_bin).exists():
            raise SystemExit(f"ORCA binary not found: {orca_bin}")

        scratch_root = Path(os.environ.get("SLURM_TMPDIR", shard_dir / "scratch"))
        scratch_root.mkdir(parents=True, exist_ok=True)

        n_ok = n_err = 0
        t_start = time.time()
        for frame in shard["frames"]:
            index = int(frame["index"])
            if index in done:
                continue
            job_dir = scratch_root / f"frame_{index:06d}"
            try:
                rec = process_frame(
                    frame,
                    job_dir,
                    level=level,
                    orca_bin=orca_bin,
                    orca2json=orca2json,
                    nprocs=args.nprocs,
                    maxcore_mb=args.maxcore_mb,
                    fit_df=not args.no_df,
                    dry_run=args.dry_run,
                    keep_orca=args.keep_orca,
                )
                ortho = rec.get("mo_orthonormality_error")
                if ortho is not None and ortho > args.ortho_tol:
                    # The AO map is the one silent failure mode here: ORCA's
                    # energy is fine while the density we store is scrambled.
                    rec.pop("packed", None)
                    rec["status"] = "error"
                    rec["error"] = f"AO map broken: |C^T S C - I| = {ortho:.3e}"
            except Exception as exc:
                rec = {"index": index, "status": "error", "error": f"{type(exc).__name__}: {exc}"}

            packed = rec.pop("packed", None)
            with status_path.open("a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")
            if packed is not None:
                results.append({"index": index, "mol": packed[0], "calc": packed[1]})
                atomic_save_npy(results_path, np.array(results, dtype=object))
                n_ok += 1
            elif rec["status"] == "error":
                n_err += 1
            if holding_claim:
                touch_claim(claim_path)
            print(f"shard {shard_id:04d} frame {index} {rec['status']}", flush=True)

        if results:
            atomic_save_npy(results_path, np.array(results, dtype=object))

        atomic_write_json(
            done_path,
            {
                "shard_id": shard_id,
                "split": split,
                "theory": args.theory,
                "n_frames": len(shard["frames"]),
                "n_results": len(results),
                "n_ok_this_pass": n_ok,
                "n_error_this_pass": n_err,
                "elapsed_s": round(time.time() - t_start, 1),
                "host": os.uname().nodename,
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        print(f"done shard {shard_id:04d} results={len(results)} errors={n_err} status={status_path}")
        return 0
    finally:
        if holding_claim:
            claim_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
