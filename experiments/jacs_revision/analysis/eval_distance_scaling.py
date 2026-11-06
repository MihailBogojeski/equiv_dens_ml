#!/usr/bin/env python3
"""OMol25 distance-scaling metrics on a water λ scan.

ΔE and ΔF are measured from the lowest-energy short-range point on each
parent scan. Density AFE is joined from a csh_evaluate report when given.

Usage:
  python scripts/revision/analysis/eval_distance_scaling.py \\
    --manifest datasets/revision/ood_scale/manifest.json \\
    --energies results/revision/ood_scale_energies.json \\
    --out results/revision/eval/ood_distance_scale.json
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/eval_distance_scaling.py'


import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR, _SCRIPT_DIR.parent / "water")
from ase import Atoms  # noqa: E402
from experiments.jacs_revision.analysis.error_vs_distance import load_errors  # noqa: E402
from experiments.jacs_revision.water.generate_ood_distance_scale import read_xyz_symbol_frames  # noqa: E402
from tblite.ase import TBLite  # noqa: E402

_REPO_ROOT = repo_root()


def scan_deltas(
    records: list[dict],
    energies: np.ndarray,
    forces: np.ndarray | None = None,
    afe: dict[int, float] | None = None,
) -> list[dict]:
    """Attach ΔE / ΔF to each scan frame relative to its parent's SR minimum."""
    energies = np.asarray(energies, dtype=float)
    if len(energies) != len(records):
        raise ValueError(f"energies ({len(energies)}) != records ({len(records)})")
    by_parent: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_parent.setdefault(str(rec["parent"]), []).append(i)

    rows: list[dict] = []
    for parent, idxs in by_parent.items():
        sr = [i for i in idxs if rec_regime(records[i]) == "SR"]
        if not sr:
            raise ValueError(f"parent {parent} has no short-range points")
        ref = min(sr, key=lambda i: energies[i])
        f_ref = forces[ref] if forces is not None else None
        for i in idxs:
            rec = dict(records[i])
            rec["energy"] = float(energies[i])
            rec["delta_e"] = float(energies[i] - energies[ref])
            rec["is_sr_reference"] = i == ref
            rec["frame_index"] = i
            if forces is not None:
                df = np.asarray(forces[i], dtype=float) - np.asarray(f_ref, dtype=float)
                rec["delta_f_mae"] = float(np.mean(np.abs(df)))
            if afe is not None and i in afe:
                rec["afe"] = float(afe[i])
            rows.append(rec)
    rows.sort(key=lambda r: r["frame_index"])
    return rows


def rec_regime(rec: dict) -> str:
    return str(rec.get("regime", "SR"))


def summarize_sr_lr(rows: list[dict]) -> dict:
    out = {}
    for regime in ("SR", "LR"):
        subset = [r for r in rows if rec_regime(r) == regime]
        if not subset:
            out[regime] = {"n": 0}
            continue
        de = np.array([r["delta_e"] for r in subset], dtype=float)
        payload = {
            "n": len(subset),
            "delta_e_mae": float(np.mean(np.abs(de))),
            "delta_e_median": float(np.median(np.abs(de))),
        }
        if any("delta_f_mae" in r for r in subset):
            df = np.array([r["delta_f_mae"] for r in subset if "delta_f_mae" in r], dtype=float)
            payload["delta_f_mae"] = float(np.mean(df))
        if any("afe" in r for r in subset):
            af = np.array([r["afe"] for r in subset if "afe" in r], dtype=float)
            payload["afe_mean"] = float(np.mean(af))
        out[regime] = payload
    return out


def energies_from_xyz_xtb(xyz_path: Path) -> np.ndarray:
    """GFN2-xTB total energies (eV) for each frame in a multi-frame XYZ."""
    calc = TBLite(method="GFN2-xTB", verbosity=0)
    vals = []
    for symbols, coords, _comment in read_xyz_symbol_frames(xyz_path):
        atoms = Atoms(symbols=symbols, positions=coords)
        atoms.calc = calc
        vals.append(float(atoms.get_potential_energy()))
    return np.asarray(vals, dtype=float)


def energies_from_label_npy(path: Path) -> np.ndarray:
    dens = np.load(path, allow_pickle=True)
    vals = []
    for item in dens:
        calc = item[1]
        vals.append(float(calc["energy"]))
    return np.asarray(vals, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--energies", type=Path, help="JSON list of energies aligned to the manifest")
    parser.add_argument("--dens", type=Path, help="DFT label npy with energy on each calc dict")
    parser.add_argument("--xtb", action="store_true", help="score the manifest XYZ with GFN2-xTB")
    parser.add_argument("--eval-json", type=Path, help="csh_evaluate report for density AFE")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    records = manifest["records"]
    if args.dens is not None:
        energies = energies_from_label_npy(args.dens)
    elif args.energies is not None:
        energies = np.asarray(json.loads(Path(args.energies).read_text()), dtype=float)
    elif args.xtb:
        xyz = Path(manifest["xyz"])
        energies = energies_from_xyz_xtb(xyz)
    else:
        raise SystemExit("pass --energies, --dens, or --xtb")

    afe = load_errors(args.eval_json) if args.eval_json else None
    rows = scan_deltas(records, energies, afe=afe)
    payload = {
        "manifest": str(args.manifest),
        "n_frames": len(rows),
        "records": rows,
        "sr_lr": summarize_sr_lr(rows),
        "dft_label_command": (
            "python scripts/revision/lib/generate_dft_labels.py "
            "--trajectory datasets/revision/ood_scale/ood_distance_scale.xyz "
            "--output-dir datasets/revision/ood_scale --output-prefix water_ood_scale "
            "--xc pbe --d4 --df"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["sr_lr"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
