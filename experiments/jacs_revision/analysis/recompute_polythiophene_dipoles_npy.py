#!/usr/bin/env python3
"""Recompute dipole moments from archived polythiophene .npy trajectories (R3.6).

Pilot scope: 10-mer replica 0, frames [start, end), writing dipole_moment into
an output .npy suitable for ir_vs_length_npy.py.
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/recompute_polythiophene_dipoles_npy.py'


import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)
_REPO_ROOT = repo_root()
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from recompute_polythiophene_dipoles_parallel import (  # noqa: E402
    GLOBAL_RANDOM_SEED,
    load_polythiophene_model,
)


def _dipole_from_calc(calc) -> np.ndarray:
    if "dipole_moment" in calc.results:
        dipole = calc.results["dipole_moment"]
    elif "dipole" in calc.results:
        dipole = calc.results["dipole"]
    else:
        raise KeyError("calculator returned no dipole_moment")
    if isinstance(dipole, torch.Tensor):
        dipole = dipole.detach().cpu().numpy()
    return np.asarray(dipole, dtype=np.float64).reshape(3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy", type=Path, required=True)
    ap.add_argument(
        "--model-path",
        type=Path,
        default=_REPO_ROOT / "paper/models/polythiophene/2024-03-23_1XDL67zp",
    )
    ap.add_argument("--oligomer", default="10mer")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=50_000)
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu-grid-accel", action="store_true")
    args = ap.parse_args()

    data = np.load(args.npy, allow_pickle=True).item()
    positions = data["positions"]
    atom_numbers = data["atom_numbers"]
    if atom_numbers.ndim == 1:
        atom_numbers = np.tile(atom_numbers[None, :], (positions.shape[0], 1))

    end = min(args.end_frame, positions.shape[0])
    indices = list(range(args.start_frame, end, args.stride))
    print(f"processing {len(indices)} frames from {args.npy} [{args.start_frame}:{end})")

    calculator, _ = load_polythiophene_model(
        args.model_path,
        device=args.device,
        enable_grid_caching=False,
        gpu_grid_acceleration=args.gpu_grid_accel,
        oligomer=args.oligomer,
    )

    dipoles = []
    t0 = time.time()
    for k, frame_idx in enumerate(indices):
        pos = positions[frame_idx]
        anum = atom_numbers[frame_idx]
        sym = []
        for z in anum:
            if z == 1:
                sym.append("H")
            elif z == 6:
                sym.append("C")
            elif z == 16:
                sym.append("S")
            else:
                sym.append("X")
        atoms = Atoms(symbols=sym, positions=pos)
        np.random.seed(GLOBAL_RANDOM_SEED)
        torch.manual_seed(GLOBAL_RANDOM_SEED)
        atoms.calc = calculator
        atoms.get_potential_energy()
        dipoles.append(_dipole_from_calc(calculator))
        if (k + 1) % 25 == 0 or k + 1 == len(indices):
            rate = (k + 1) / max(time.time() - t0, 1e-6)
            print(f"  [{k + 1}/{len(indices)}] {rate:.3f} frames/s", flush=True)

    out_data = {
        "positions": positions[args.start_frame:end:args.stride],
        "atom_numbers": atom_numbers[args.start_frame:end:args.stride],
        "dipole_moment": np.asarray(dipoles, dtype=np.float64),
        "time_step_fs": args.dt_fs * args.stride,
        "source_npy": str(args.npy),
        "frame_range": [args.start_frame, end],
        "stride": args.stride,
        "model_path": str(args.model_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out_data, allow_pickle=True)
    print(f"wrote {args.out} ({len(dipoles)} dipoles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
