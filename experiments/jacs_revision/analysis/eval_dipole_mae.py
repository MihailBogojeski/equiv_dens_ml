#!/usr/bin/env python3
"""MAE / RMSE of predicted dipoles vs stored DFT dipole_moment (Debye).

DFT labels and DenSNet ``--dpm_intor`` both use common origin (0,0,0).
Water clusters are charge-neutral, so a COM shift does not change μ.
Pass ``--origin com`` to shift both sides by Q R_com for charged systems.

Does not load a checkpoint: predicted dipoles come from a npy written by
an earlier eval that used ``--dpm_intor``. No second ORCA campaign.

Usage:
  python scripts/revision/analysis/eval_dipole_mae.py \\
    --base-npy datasets/revision/water_labels_md40x/water_id_test_pbe_d4_avdz_base.npy \\
    --pred-npy results/revision/dipole_pred/water_id_test.npy
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/eval_dipole_mae.py'


import argparse
import json
import sys
from pathlib import Path


_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)

import numpy as np

_REPO_ROOT = repo_root()

# CODATA: 1 a.u. dipole = 2.541746473 Debye
AU_TO_DEBYE = 2.541746473


def _as_dict(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.ndim == 0:
        raw = raw.item()
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a dict npy (keys or dipole_moment)")
    return raw


def nuclear_com(atom_numbers: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Z-weighted nuclear centre, matching DipoleMomentCalc's COM convention."""
    z = np.asarray(atom_numbers, dtype=float).reshape(-1)
    xyz = np.asarray(positions, dtype=float).reshape(-1, 3)
    mask = z > 0
    z = z[mask]
    xyz = xyz[mask]
    weight = float(z.sum())
    if weight <= 0.0:
        return np.zeros(3, dtype=float)
    return (z[:, None] * xyz).sum(axis=0) / weight


def net_charge(atom_numbers: np.ndarray, n_electrons: float) -> float:
    z = np.asarray(atom_numbers, dtype=float).reshape(-1)
    return float(z[z > 0].sum() - n_electrons)


def shift_dipole_to_com(
    dipole: np.ndarray,
    atom_numbers: np.ndarray,
    positions: np.ndarray,
    n_electrons: float,
) -> np.ndarray:
    """μ_com = μ_0 − Q R_com. Neutral molecules are unchanged."""
    q = net_charge(atom_numbers, n_electrons)
    com = nuclear_com(atom_numbers, positions)
    return np.asarray(dipole, dtype=float) - q * com


def dipole_mae_debye(
    reference: np.ndarray,
    predicted: np.ndarray,
    already_debye: bool = False,
) -> dict:
    ref = np.asarray(reference, dtype=float).reshape(-1, 3)
    pred = np.asarray(predicted, dtype=float).reshape(-1, 3)
    if ref.shape != pred.shape:
        raise ValueError(f"dipole shape mismatch: ref {ref.shape} pred {pred.shape}")
    scale = 1.0 if already_debye else AU_TO_DEBYE
    delta = (pred - ref) * scale
    norms = np.linalg.norm(delta, axis=1)
    return {
        "n_frames": int(len(norms)),
        "mae_debye": float(norms.mean()) if len(norms) else float("nan"),
        "rmse_debye": float(np.sqrt(np.mean(norms**2))) if len(norms) else float("nan"),
        "units": "Debye",
    }


def _n_electrons_row(atom_numbers: np.ndarray) -> float:
    z = np.asarray(atom_numbers, dtype=float).reshape(-1)
    return float(z[z > 0].sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-npy", type=Path, required=True)
    parser.add_argument("--pred-npy", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--origin", choices=("label", "com"), default="label")
    parser.add_argument("--already-debye", action="store_true")
    args = parser.parse_args(argv)

    base = _as_dict(args.base_npy)
    pred = _as_dict(args.pred_npy)
    if "dipole_moment" not in base:
        raise SystemExit(f"{args.base_npy}: missing dipole_moment")
    if "dipole_moment" not in pred:
        raise SystemExit(f"{args.pred_npy}: missing dipole_moment")
    ref = np.asarray(base["dipole_moment"], dtype=float).reshape(-1, 3)
    pred_mu = np.asarray(pred["dipole_moment"], dtype=float).reshape(-1, 3)
    n = min(len(ref), len(pred_mu))
    ref, pred_mu = ref[:n], pred_mu[:n]
    if args.origin == "com":
        positions = np.asarray(base["positions"], dtype=float)
        numbers = np.asarray(base["atom_numbers"])
        ref_s, pred_s = [], []
        for i in range(n):
            n_e = _n_electrons_row(numbers[i])
            ref_s.append(shift_dipole_to_com(ref[i], numbers[i], positions[i], n_e))
            pred_s.append(shift_dipole_to_com(pred_mu[i], numbers[i], positions[i], n_e))
        ref = np.asarray(ref_s)
        pred_mu = np.asarray(pred_s)
    stats = dipole_mae_debye(ref, pred_mu, already_debye=args.already_debye)
    stats["base_npy"] = str(args.base_npy)
    stats["pred_npy"] = str(args.pred_npy)
    stats["origin"] = args.origin
    print(json.dumps(stats, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
