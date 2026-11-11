#!/usr/bin/env python3
"""Write .density_training_done for runs whose logs already finished density."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/sync_density_training_markers.py'


import subprocess
import sys
from pathlib import Path

_REPO = _LEGACY_FILE.resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "water"))

from experiments.jacs_revision.water.density_log_finished import density_training_complete  # noqa: E402

CAMPAIGN_STEMS = [
    "water_pbe_d4_avdz",
    "water_wb97mv_def2tzvpd",
    "water_wb97mv_def2tzvpd_seed7",
    "water_wb97mv_def2tzvpd_seed123",
    "water_wb97mv_def2tzvpd_seed20260821",
    "water_direct_pbe_d4_avdz",
    "water_direct_wb97mv_def2tzvpd",
    "malonaldehyde_pbe_d4_avdz",
    "malonaldehyde_wb97mv_def2tzvpd",
    "water_pbe_orca_cutoff_4",
    "water_pbe_orca_cutoff_5",
    "water_pbe_orca_cutoff_6",
    "water_pbe_orca_cutoff_8",
]

WAVE1_STEMS = [
    *(f"water_n{n}_pbe_d4_avdz" for n in (50, 125, 250, 500)),
    *(f"water_direct_n{n}_pbe_d4_avdz" for n in (50, 125, 250, 500)),
    "water_pbe_d4_avdz_curve",
    "water_direct_pbe_d4_avdz_curve",
]


def _latest_run_dir(save_dir: Path) -> Path | None:
    script = _REPO / "scripts" / "revision" / "lib" / "latest_run_dir.sh"
    out = subprocess.run(
        [str(script), str(save_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    line = out.stdout.strip()
    return Path(line) if line else None


def sync_stem(stem: str, dry_run: bool = False) -> str:
    save_dir = _REPO / "results" / "revision" / stem
    marker = save_dir / ".density_training_done"
    log = _REPO / "logs" / f"train_{stem}.log"
    if marker.is_file():
        return "already_marked"
    if not density_training_complete(log, stem):
        return "unfinished"
    run_dir = _latest_run_dir(save_dir)
    if run_dir is None:
        return "no_checkpoint"
    if dry_run:
        return f"would_mark ({run_dir.name})"
    save_dir.mkdir(parents=True, exist_ok=True)
    marker.touch()
    (save_dir / ".density_run_dir").write_text(f"{run_dir}\n")
    return f"marked ({run_dir.name})"


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    stems = CAMPAIGN_STEMS + WAVE1_STEMS
    for stem in stems:
        status = sync_stem(stem, dry_run=dry_run)
        print(f"{stem}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
