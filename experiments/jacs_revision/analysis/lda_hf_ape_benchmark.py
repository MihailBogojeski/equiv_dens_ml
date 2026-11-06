#!/usr/bin/env python3
"""Single-point LDA/HF vs PBE energy comparison on water ID-test geometries."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/lda_hf_ape_benchmark.py'


import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)
sys.path.insert(0, str(_SCRIPT_DIR.parent / "lib"))

from experiments.jacs_revision.lib.generate_dft_labels import compute_frame  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-npy",
        type=Path,
        default=Path("datasets/revision/water_labels/water_id_test_pbe_d4_avdz_base.npy"),
    )
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/revision/lda_hf_energy_water_id_test.json"),
    )
    args = ap.parse_args()

    blob = np.load(args.base_npy, allow_pickle=True).item()
    n = min(args.limit, len(blob["positions"]))
    records = []
    for i in range(n):
        pos = blob["positions"][i]
        anum = blob["atom_numbers"][i]
        e_ref = float(blob["energy"][i]) if "energy" in blob else None
        _, pbe = compute_frame(pos, anum, use_gpu=False, use_df=True, xc="pbe", use_d4=True)
        _, lda = compute_frame(pos, anum, use_gpu=False, use_df=True, xc="lda", use_d4=False)
        _, hf = compute_frame(pos, anum, use_gpu=False, use_df=True, xc="hf", use_d4=False)
        records.append(
            {
                "index": i,
                "natoms": int(np.sum(anum > 0)),
                "pbe_energy_hartree": pbe["energy"],
                "lda_energy_hartree": lda["energy"],
                "hf_energy_hartree": hf["energy"],
                "ref_energy_hartree": e_ref,
                "lda_minus_pbe_mEh": (lda["energy"] - pbe["energy"]) * 1000,
                "hf_minus_pbe_mEh": (hf["energy"] - pbe["energy"]) * 1000,
            }
        )
        print(f"  [{i + 1}/{n}] done", flush=True)

    lda_d = [r["lda_minus_pbe_mEh"] for r in records]
    hf_d = [r["hf_minus_pbe_mEh"] for r in records]
    summary = {
        "n_frames": n,
        "base_npy": str(args.base_npy),
        "lda_minus_pbe_mEh_mean": float(np.mean(lda_d)),
        "hf_minus_pbe_mEh_mean": float(np.mean(hf_d)),
        "note": "Single-point energies on identical geometries; grid-density APE deferred.",
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
