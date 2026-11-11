#!/usr/bin/env python3
"""IR convergence from archived compressed .npy trajectories (R3.6).

Loads dipole moments from simulation_*_compressed_*.npy or *_dpm*.npy files,
computes DACF-based IR spectra at successive trajectory prefixes, and reports
peak positions in fingerprint (500–1800 cm^-1) and XH-stretch (2800–3800 cm^-1)
windows.
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/ir_vs_length_npy.py'


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
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "md"))

from compute_ir_spectra import acf_to_ir_spectrum, dipole_acf  # noqa: E402


def load_dipoles_from_npy(path: Path) -> tuple[np.ndarray, float]:
    """Return (dipole [n_steps, 3], dt_fs)."""
    data = np.load(path, allow_pickle=True).item()
    if "dipole_moment" not in data:
        raise KeyError(f"{path}: missing dipole_moment key; keys={list(data.keys())}")
    dipole = np.asarray(data["dipole_moment"], dtype=np.float64)
    if dipole.ndim != 2 or dipole.shape[1] != 3:
        raise ValueError(f"{path}: expected dipole shape (n, 3), got {dipole.shape}")
    dt_fs = float(data.get("time_step_fs", 1.0))
    return dipole, dt_fs


def _window_peak(wn: np.ndarray, intens: np.ndarray, lo: float, hi: float) -> float | None:
    mask = (wn >= lo) & (wn <= hi)
    if not np.any(mask):
        return None
    return float(wn[mask][np.argmax(intens[mask])])


def _integrated_area(wn: np.ndarray, intens: np.ndarray, lo: float, hi: float) -> float | None:
    mask = (wn >= lo) & (wn <= hi)
    if not np.any(mask):
        return None
    return float(np.trapezoid(intens[mask], wn[mask]))


def analyze_trajectory(
    dipole: np.ndarray,
    dt_fs: float,
    windows_ps: list[float],
    out_dir: Path,
    label: str,
) -> dict:
    n_steps = dipole.shape[0]
    duration_ps = n_steps * dt_fs / 1000.0
    rows = []
    for wps in windows_ps:
        n = min(n_steps, int(round(wps * 1000.0 / dt_fs)))
        if n < 100:
            continue
        acf = dipole_acf(dipole[:n])
        wn, intens = acf_to_ir_spectrum(acf, dt_fs)
        rec = {
            "window_ps": wps,
            "used_ps": n * dt_fs / 1000.0,
            "peak_fp_cm1": _window_peak(wn, intens, 500, 1800),
            "peak_xh_cm1": _window_peak(wn, intens, 2800, 3800),
            "area_fp": _integrated_area(wn, intens, 500, 1800),
            "area_xh": _integrated_area(wn, intens, 2800, 3800),
        }
        rows.append(rec)
        safe_label = label.replace("/", "_")
        np.savetxt(
            out_dir / f"ir_{safe_label}_{int(wps)}ps.csv",
            np.column_stack([wn, intens]),
            delimiter=",",
            header="wavenumber_cm-1,intensity",
            comments="",
        )

    summary = {
        "label": label,
        "full_duration_ps": duration_ps,
        "dt_fs": dt_fs,
        "n_frames": n_steps,
        "windows": rows,
    }
    if len(rows) >= 2:
        base = next((r for r in rows if abs(r["window_ps"] - 100) < 1e-6), rows[0])
        last = rows[-1]
        if base["peak_fp_cm1"] is not None and last["peak_fp_cm1"] is not None:
            summary["delta_fp_last_minus_100_cm1"] = last["peak_fp_cm1"] - base["peak_fp_cm1"]
        if base["peak_xh_cm1"] is not None and last["peak_xh_cm1"] is not None:
            summary["delta_xh_last_minus_100_cm1"] = last["peak_xh_cm1"] - base["peak_xh_cm1"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", type=Path, action="append", required=True, help="Trajectory npy with dipole_moment")
    parser.add_argument("--label", type=str, action="append", help="Label per --npy (defaults to filename stem)")
    parser.add_argument("--windows", default="25,50,100,200")
    parser.add_argument("--dt-fs", type=float, default=1.0, help="Timestep in fs if not stored in npy")
    parser.add_argument("--out", type=Path, default=Path("results/revision/ir_convergence"))
    args = parser.parse_args()

    if args.label and len(args.label) != len(args.npy):
        parser.error("Provide one --label per --npy")

    windows = [float(x) for x in args.windows.split(",") if x]
    args.out.mkdir(parents=True, exist_ok=True)

    trajectories = []
    for i, npy_path in enumerate(args.npy):
        label = args.label[i] if args.label else npy_path.stem
        dipole, dt_fs = load_dipoles_from_npy(npy_path)
        if dt_fs == 1.0 and args.dt_fs != 1.0:
            dt_fs = args.dt_fs
        traj_summary = analyze_trajectory(dipole, dt_fs, windows, args.out, label)
        traj_summary["npy"] = str(npy_path)
        trajectories.append(traj_summary)

    summary = {"trajectories": trajectories}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
