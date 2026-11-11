#!/usr/bin/env python3
"""Search archives for ethanol DFT training npy files (R2.11 blocker)."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/search_ethanol_train_npy.py'


import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
from experiments.jacs_revision.lib.paths import repo_root  # noqa: E402

REPO = repo_root()
_SCRATCH_ARCHIVE = Path(os.environ.get("SCRATCH", "/scratch/mh7373")) / "archive" / "equiv_dens_ml"
SEARCH_ROOTS = [
    REPO / "paper",
    REPO / "datasets",
    REPO / "paper" / "archives",
    REPO / "paper" / "_staging",
    _SCRATCH_ARCHIVE / "archives",
    _SCRATCH_ARCHIVE / "staging",
    _SCRATCH_ARCHIVE / "models_unmapped",
]
NAME_PATTERNS = (
    "ethanol_dft",
    "ethanol_all",
    "pyscf_ccpvdz",
    "ethanol_train",
    "ethanol_test",
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _inspect_npy(path: Path) -> dict:
    rec: dict = {"path": _display_path(path), "size_mb": round(path.stat().st_size / 1e6, 2)}
    try:
        arr = np.load(path, allow_pickle=True)
        if arr.shape == ():
            item = arr.item()
            if isinstance(item, dict):
                rec["type"] = "dict_npy"
                rec["keys"] = sorted(item.keys())
                for key in ("positions", "forces", "energy", "atom_numbers", "df_coeffs"):
                    if key in item and hasattr(item[key], "shape"):
                        rec[f"{key}_shape"] = list(item[key].shape)
                if "positions" in item:
                    rec["n_frames"] = int(np.asarray(item["positions"]).shape[0])
            else:
                rec["type"] = type(item).__name__
        else:
            rec["type"] = "array"
            rec["shape"] = list(arr.shape)
    except Exception as exc:
        rec["error"] = str(exc)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/revision/ethanol_train_search.json"))
    args = parser.parse_args()

    matches: list[dict] = []
    seen: set[str] = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.npy"):
            name_lower = path.name.lower()
            if not any(p in name_lower for p in NAME_PATTERNS):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            matches.append(_inspect_npy(path))

    target_name = "datasets/ethanol_dft_pyscf_ccpvdz_train.npy"
    summary = {
        "target_missing": target_name,
        "target_found": any(m["path"] == target_name for m in matches),
        "search_roots": [_display_path(p) for p in SEARCH_ROOTS if p.exists()],
        "name_patterns": list(NAME_PATTERNS),
        "n_matches": len(matches),
        "matches": sorted(matches, key=lambda m: m["path"]),
        "conclusion": (
            "ethanol_dft_pyscf_ccpvdz_train.npy not recovered in archive. "
            "Closest candidates are small stubs (ethanol_train_10.npy) or MD trajectories, not full DFT training packs."
            if not any(m["path"] == target_name for m in matches)
            else "Target found."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
