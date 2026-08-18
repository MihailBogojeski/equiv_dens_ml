#!/usr/bin/env python3
"""Run one QM7-X shard through ORCA and convert to a DensNet calc_dict.

For each closed-shell frame: write .inp, run orca, orca_2json, map MOs to
PySCF order, DF-fit onto aug-cc-pVQZ-JKfit, append (mol.pack(), calc_dict).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from qm7x_orca_common import (  # noqa: E402
    AUXBASIS,
    MAXCORE_MB_DEFAULT,
    NPROCS_DEFAULT,
    THEORY_KEYWORDS,
    XC,
    build_pyscf_mol,
    calc_dict_from_orca,
    forces_from_gradient,
    is_closed_shell,
    mos_from_orca_json,
    parse_orca_energy,
    parse_orca_engrad,
    parse_orca_gradient,
    read_shard,
    write_orca_inp,
)

ORCA_BIN_DEFAULT = os.environ.get("ORCA_BIN", "orca")
ORCA2JSON_DEFAULT = os.environ.get("ORCA2JSON_BIN", "orca_2json")


def _which(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    return name


def run_cmd(cmd: list[str], cwd: Path, log: Path) -> None:
    with log.open("ab") as fh:
        fh.write(("+ " + " ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def cleanup_orca_scratch(job_dir: Path, keep_out: bool) -> None:
    keep_suffix = {".json", ".npy", ".jsonl"}
    if keep_out:
        keep_suffix.add(".out")
        keep_suffix.add(".inp")
    for path in job_dir.iterdir():
        if path.suffix in keep_suffix or path.name in {"status.jsonl", "results.npy"}:
            continue
        if path.is_file():
            path.unlink()


def process_frame(
    frame: dict,
    job_dir: Path,
    *,
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

    job_dir.mkdir(parents=True, exist_ok=True)
    inp = write_orca_inp(
        job_dir / "job.inp",
        z,
        xyz,
        nprocs=nprocs,
        maxcore_mb=maxcore_mb,
    )
    if dry_run:
        return {
            "index": index,
            "status": "dry_run",
            "inp": str(inp),
            "theory": THEORY_KEYWORDS,
        }

    out_path = job_dir / "job.out"
    run_cmd([orca_bin, str(inp.name)], cwd=job_dir, log=out_path)
    engrad_path = job_dir / "job.engrad"
    if engrad_path.exists():
        energy, gradient = parse_orca_engrad(engrad_path.read_text())
    else:
        out_text = out_path.read_text(errors="replace")
        energy = parse_orca_energy(out_text)
        gradient = parse_orca_gradient(out_text)
    forces = forces_from_gradient(gradient)

    gbw = job_dir / "job.gbw"
    if not gbw.exists():
        raise FileNotFoundError(f"ORCA did not write {gbw}")
    run_cmd([orca2json, str(gbw.name)], cwd=job_dir, log=job_dir / "orca_2json.log")
    json_path = job_dir / "job.json"
    if not json_path.exists():
        # ORCA 6 sometimes writes <basename>.json next to the GBW.
        candidates = list(job_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"orca_2json produced no JSON in {job_dir}")
        json_path = candidates[0]
    data = json.loads(json_path.read_text())
    mol = build_pyscf_mol(z, xyz)
    mo_coeff, mo_occ, _mo_e = mos_from_orca_json(mol, data)
    calc = calc_dict_from_orca(mol, energy, forces, mo_coeff, mo_occ, fit_df=fit_df)
    if not keep_orca:
        cleanup_orca_scratch(job_dir, keep_out=False)
    return {
        "index": index,
        "status": "ok",
        "energy": float(energy),
        "n_atoms": int(len(z)),
        "nao": int(mol.nao),
        "naux": int(calc["df_coeff"].shape[0]) if "df_coeff" in calc else 0,
        "xc": XC,
        "auxbasis": AUXBASIS,
        "packed": (mol.pack(), calc),
    }


def load_done_indices(status_path: Path) -> set[int]:
    done = set()
    if not status_path.exists():
        return done
    for line in status_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") in {"ok", "skipped_open_shell"}:
            done.add(int(rec["index"]))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--orca-bin", default=ORCA_BIN_DEFAULT)
    parser.add_argument("--orca2json-bin", default=ORCA2JSON_DEFAULT)
    parser.add_argument("--nprocs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", NPROCS_DEFAULT)))
    parser.add_argument("--maxcore-mb", type=int, default=MAXCORE_MB_DEFAULT)
    parser.add_argument("--no-df", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-orca", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()

    shard = read_shard(args.shard)
    shard_id = int(shard["shard_id"])
    split = shard.get("split", "train")
    outdir = Path(args.outdir) if args.outdir else _REPO_ROOT / "results/revision/qm7x_orca" / split
    shard_dir = outdir / f"shard_{shard_id:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    status_path = shard_dir / "status.jsonl"
    results_path = shard_dir / "results.npy"

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

    for frame in shard["frames"]:
        index = int(frame["index"])
        if index in done:
            continue
        job_dir = scratch_root / f"frame_{index:06d}"
        try:
            rec = process_frame(
                frame,
                job_dir,
                orca_bin=orca_bin,
                orca2json=orca2json,
                nprocs=args.nprocs,
                maxcore_mb=args.maxcore_mb,
                fit_df=not args.no_df,
                dry_run=args.dry_run,
                keep_orca=args.keep_orca,
            )
        except Exception as exc:
            rec = {"index": index, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        packed = rec.pop("packed", None)
        with status_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if packed is not None:
            results.append({"index": index, "mol": packed[0], "calc": packed[1]})
            np.save(results_path, np.array(results, dtype=object), allow_pickle=True)
        print(f"shard {shard_id:04d} frame {index} {rec['status']}", flush=True)

    if results:
        np.save(results_path, np.array(results, dtype=object), allow_pickle=True)
    print(f"done shard {shard_id:04d} results={len(results)} status={status_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
