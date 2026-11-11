#!/usr/bin/env python3
"""Aggregate GPU-hours for the revision water/malon campaign (R3.m1).

Reads ``sacct`` for completed jobs whose names match the campaign train
prefixes, plus any ``Elapsed`` lines already written under ``logs/``.
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/collect_gpu_hours.py'


import json
import subprocess
from collections import defaultdict
from pathlib import Path

_REPO = _LEGACY_FILE.resolve().parents[3]
_NAME_TO_ROW = (
    ("tr-water-direct", "Water, PBE-D4 (direct)"),
    ("tr-water_direct", "Water, PBE-D4 (direct)"),
    ("tre-water-wb97", "Water, ωB97M-V (SAD)"),
    ("tr-water-wb97", "Water, ωB97M-V (SAD)"),
    ("water-train-en", "Water, energy stage"),
    ("tr-malonaldehyde-wb97", "Malonaldehyde, ωB97M-V"),
    ("tr-malonaldehyde-pbe", "Malonaldehyde, PBE-D4"),
    ("tre-water-pbe", "Water, PBE-D4 (SAD)"),
    ("tre-cutoff", "Water, PBE-D4 (SAD)"),
    ("tr-water-", "Water, PBE-D4 (SAD)"),
    ("water-train", "Water, PBE-D4 (SAD)"),
)
_STAGE_HINTS = (
    ("tre-", "energy"),
    ("-en", "energy"),
    ("energy", "energy"),
)


def _elapsed_to_hours(text: str) -> float | None:
    text = text.strip()
    if not text or text == "Elapsed":
        return None
    days = 0.0
    if "-" in text:
        day_s, text = text.split("-", 1)
        days = float(day_s)
    parts = [float(p) for p in text.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0.0, parts[0], parts[1]
    else:
        return None
    return days * 24.0 + h + m / 60.0 + s / 3600.0


def _row_and_stage(job_name: str) -> tuple[str, str]:
    stage = "density"
    for hint, label in _STAGE_HINTS:
        if hint in job_name:
            stage = label
            break
    for prefix, row in _NAME_TO_ROW:
        if prefix in job_name:
            return row, stage
    return "other", stage


def _sacct_rows() -> list[dict]:
    cmd = [
        "sacct",
        "-u",
        "mh7373",
        "-S",
        "2026-08-01",
        "--format=JobID,JobName%40,Elapsed,AllocTRES,State",
        "--parsable2",
        "--noheader",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=180)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jobid, name, elapsed, tres, state = parts[:5]
        if "." in jobid:
            continue
        hours = _elapsed_to_hours(elapsed)
        if hours is None:
            continue
        gpus = 1 if "gres/gpu" in tres else 0
        if gpus == 0:
            continue
        row, stage = _row_and_stage(name)
        if row == "other":
            continue
        rows.append(
            {
                "jobid": jobid,
                "name": name,
                "hours": hours,
                "row": row,
                "stage": stage,
                "state": state,
            }
        )
    return rows


def main() -> int:
    jobs = _sacct_rows()
    table: dict[str, dict[str, float]] = defaultdict(lambda: {"density": 0.0, "energy": 0.0})
    for job in jobs:
        table[job["row"]][job["stage"]] += job["hours"]
    report = {
        "source": "sacct 2026-08-01..now, GPU jobs only",
        "n_jobs": len(jobs),
        "rows": {
            name: {
                "density_gpu_hours": round(vals["density"], 2),
                "energy_gpu_hours": round(vals["energy"], 2),
                "total_gpu_hours": round(vals["density"] + vals["energy"], 2),
            }
            for name, vals in sorted(table.items())
        },
        "jobs": jobs,
    }
    out = _REPO / "results/revision/gpu_hours_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out} from {len(jobs)} GPU jobs")
    for name, vals in report["rows"].items():
        print(
            f"  {name}: density={vals['density_gpu_hours']:.2f} "
            f"energy={vals['energy_gpu_hours']:.2f} total={vals['total_gpu_hours']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
