#!/usr/bin/env python3
"""Validate that a trained checkpoint follows the paper multiphase recipe.

Reads args.txt and optional test-error JSON, failing if the run looks like a
joint-mode fit (energy head co-trained with density) or if post-training metrics
are outside paper-scale thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_args_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("--") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.lstrip("-")] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Timestamped model directory")
    parser.add_argument(
        "--errors-json",
        type=Path,
        default=None,
        help="Optional JSON with energy_mae, forces_mae, density_perc_mae keys",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    args_file = run_dir / "args.txt"
    if not args_file.is_file():
        print(f"missing {args_file}", file=sys.stderr)
        return 1

    cfg = parse_args_file(args_file)
    failures: list[str] = []

    energy_w = float(cfg.get("energy_weight", "0"))
    forces_w = float(cfg.get("forces_weight", "0"))
    density_w = float(cfg.get("density_weight", "0"))
    fine_tune = cfg.get("density_fine_tuning", "False").lower() == "true"

    is_energy_stage = density_w == 0.0 and (energy_w > 0 or forces_w > 0)
    is_density_parent = (run_dir.parent / ".density_training_done").is_file()

    if is_energy_stage:
        if energy_w <= 0 or forces_w <= 0:
            failures.append("energy stage must train both energy and forces")
    elif is_density_parent or fine_tune or density_w > 0:
        if energy_w > 0 or forces_w > 0:
            failures.append(
                "density multiphase job must not co-optimise energy/forces "
                f"(energy_weight={energy_w}, forces_weight={forces_w})"
            )

    if args.errors_json and args.errors_json.is_file():
        metrics = json.loads(args.errors_json.read_text())
        energy_mae = metrics.get("energy_mae") or metrics.get("energy_loss")
        forces_mae = metrics.get("forces_mae") or metrics.get("forces_loss")
        density_afe = metrics.get("density_perc_mae") or metrics.get("density_mae")
        if density_afe is not None and float(density_afe) > 0.002:
            failures.append(f"density AFE {density_afe} > 0.002")
        if is_energy_stage and energy_mae is not None and float(energy_mae) > 10.0:
            failures.append(f"energy MAE {energy_mae} kcal/mol looks like joint-mode training")
        if is_energy_stage and forces_mae is not None and float(forces_mae) > 0.15:
            failures.append(f"forces MAE {forces_mae} > 0.15 kcal/mol/Å")

    log_hint = run_dir.parent.parent / "logs" / f"train_{run_dir.parent.name}.log"
    if log_hint.is_file():
        head = log_hint.read_text(errors="replace")[:8000]
        if re.search(r"Total params is 8681467", head) and "finished trainer run of phase energy" not in head:
            if energy_w > 0 and forces_w > 0 and density_w > 0:
                failures.append("training log shows joint-mode parameter count with all losses active")

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 1

    print(f"OK: {run_dir} passes multiphase recipe checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
