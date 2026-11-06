#!/usr/bin/env python3
"""Summarize g-xTB JSONL single-point campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    energies = [r["energy_hartree"] for r in recs if r.get("energy_hartree") is not None]
    failed = [r["index"] for r in recs if r.get("energy_hartree") is None]
    nonzero_rc = [r["index"] for r in recs if r.get("returncode", 0) != 0]
    return {
        "path": str(path),
        "n_frames": len(recs),
        "n_ok": len(energies),
        "n_failed": len(failed),
        "n_nonzero_returncode": len(nonzero_rc),
        "failed_indices": failed[:20],
        "energy_hartree_mean": sum(energies) / len(energies) if energies else None,
        "energy_hartree_min": min(energies) if energies else None,
        "energy_hartree_max": max(energies) if energies else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("results/revision/gxtb"))
    parser.add_argument("--out", type=Path, default=Path("results/revision/gxtb_summary.json"))
    args = parser.parse_args()
    rows = [summarize(p) for p in sorted(args.dir.glob("*.jsonl"))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
