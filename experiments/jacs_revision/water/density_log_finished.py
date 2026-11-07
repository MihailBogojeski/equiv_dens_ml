#!/usr/bin/env python3
"""Return whether a campaign log finished the density stages we report."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/density_log_finished.py'


import sys
from pathlib import Path

_REPO = _LEGACY_FILE.resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

from experiments.jacs_revision.analysis.train_log_curve import parse_density_curve_result  # noqa: E402


def density_training_complete(log_path: Path, stem: str) -> bool:
    """True when unconstrained density LR-stopped (SAD) or multiphase test ran (direct)."""
    if not log_path.is_file():
        return False
    text = log_path.read_text(errors="replace")
    if "Starting test evaluation" in text and "finished trainer run of phase density_fine_tuning" in text:
        return True
    if "Starting test evaluation" in text and "finished trainer run of phase density" in text:
        # Direct models that skip a separate fine-tune still run the test pass.
        if stem.startswith("water_direct") or stem.startswith("malonaldehyde"):
            return True

    result = parse_density_curve_result(log_path)
    if len(result.points) < 2:
        return False
    if result.stop_reason == "max_steps":
        return True
    if result.stop_reason != "lr":
        return False

    # SAD water, cutoffs, hybrid seeds: letter curves use unconstrained density only.
    if stem.startswith("water_pbe") or stem.startswith("water_wb97mv") or stem.startswith(
        "water_pbe_orca_cutoff"
    ):
        return True
    if stem.startswith("malonaldehyde"):
        return "finished trainer run of phase density_fine_tuning" in text
    return False


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: density_log_finished.py <run_stem>", file=sys.stderr)
        return 2
    stem = args[0]
    log = _REPO / "logs" / f"train_{stem}.log"
    return 0 if density_training_complete(log, stem) else 1


if __name__ == "__main__":
    raise SystemExit(main())
