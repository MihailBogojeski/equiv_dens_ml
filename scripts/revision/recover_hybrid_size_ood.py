#!/usr/bin/env python3
"""Drain only the missing md40x hybrid n=8/10 shards in bounded CPU arrays.

cpu48 permits 3000 submitted tasks per user: the complete 3095-task array
cannot be submitted even to an empty queue. A small dependent coordinator
submits the next window after the current window leaves the queue.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "water"))
from campaign_status import array_spec, live_task_ids, survey


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    theory = "wb97mv_def2tzvpd"
    shard_dir = repo / "datasets/revision/shards_md40x/water_ood_size_small" / theory
    outdir = repo / "results/revision/water_orca_md40x/water_ood_size_small" / theory
    name = "m40-water_ood_size_small-" + theory
    info = survey(shard_dir, outdir, 86400, live_task_ids())
    # A worker writes shard.done even if individual frames failed. Reopen only
    # incomplete shards; a marker alone is not evidence of accepted labels.
    failed = []
    for marker in outdir.glob("shard_*/shard.done"):
        record = json.loads(marker.read_text())
        if record["n_results"] != record["n_frames"]:
            failed.append(int(marker.parent.name.split("_")[-1]))
    info["outstanding"] = sorted(set(info["outstanding"]) | set(failed))
    info["n_done"] -= len(failed)
    if not info["n_shards"]:
        raise RuntimeError("No prepared shards; refusing an empty campaign")
    queued = subprocess.check_output(["squeue", "-h", "-u", os.environ["USER"],
                                      "-n", name, "-o", "%i"], text=True).strip()
    if queued:
        raise RuntimeError("A recovery array is already queued; refusing duplicate work")
    if not info["outstanding"]:
        print(json.dumps({"n_done": info["n_done"], "n_shards": info["n_shards"],
                          "counts": info["counts"]}))
        if info["n_done"] != info["n_shards"]:
            raise RuntimeError("Unfinished shards are claimed by another job")
        return
    if args.round >= 12:
        raise RuntimeError("Recovery round limit reached; inspect failures before resubmitting")
    qos_jobs = subprocess.check_output(["squeue", "-r", "-h", "-u", os.environ["USER"],
                                        "-o", "%q"], text=True).splitlines()
    capacity = min(2000, 3000 - qos_jobs.count("cpu48") - 4)
    if capacity <= 0:
        raise RuntimeError("cpu48 submission slots exhausted; no work submitted")
    spec = array_spec(info["outstanding"], max_tasks=capacity)
    command = ["sbatch", "--parsable", "--job-name=" + name, "--array=" + spec + "%200",
               "--export=ALL,SHARD_DIR=" + str(shard_dir) + ",OUTDIR=" + str(outdir) +
               ",THEORY=" + theory + ",RETRY_ERRORS=1", str(repo / "scripts/revision/submit_water_orca.sbatch")]
    print(shlex.join(command), flush=True)
    if not args.submit:
        return
    job = subprocess.check_output(command, cwd=repo, text=True).strip().split(";")[0]
    next_command = [sys.executable, "-B", str(Path(__file__).resolve()), "--repo", str(repo),
                    "--round", str(args.round + 1), "--submit"]
    continuation = subprocess.check_output([
        "sbatch", "--parsable", "--job-name=jacs-hybrid-continue", "--dependency=afterany:" + job,
        "--account=torch_pr_283_chemistry", "--partition=cs", "--qos=cpu48",
        "--cpus-per-task=1", "--mem=2G", "--time=00:10:00",
        "--output=logs/jacs_hybrid_continue_%j.out", "--wrap=" + shlex.join(next_command),
    ], cwd=repo, text=True).strip().split(";")[0]
    record = {"utc": datetime.now(timezone.utc).isoformat(), "round": args.round,
              "array_job": job, "array": spec, "continuation_job": continuation,
              "n_done_before": info["n_done"], "n_shards": info["n_shards"]}
    record_dir = repo / "results/revision/jacs_recovery_20260927"
    record_dir.mkdir(parents=True, exist_ok=True)
    with (record_dir / "hybrid_submissions.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
