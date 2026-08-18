#!/usr/bin/env python3
"""IR spectra from successive trajectory prefixes (R3.6).

Windows are in picoseconds. Peak-position drift is reported in the
fingerprint (500–1800 cm^-1) and XH-stretch (2800–3800 cm^-1) windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "md"))

from compute_ir_spectra import acf_to_ir_spectrum, dipole_acf, load_dipoles_from_hdf5  # noqa: E402


def _window_peak(wn, intens, lo, hi):
    mask = (wn >= lo) & (wn <= hi)
    if not np.any(mask):
        return None
    return float(wn[mask][np.argmax(intens[mask])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--windows", default="50,100,200,500")
    parser.add_argument("--out", type=Path, default=Path("results/revision/ir_convergence"))
    args = parser.parse_args()

    dipole, dt_fs = load_dipoles_from_hdf5(args.hdf5)
    if dipole.ndim == 3:
        dipole = dipole[:, 0, :]
    n_steps = dipole.shape[0]
    duration_ps = n_steps * dt_fs / 1000.0
    windows = [float(x) for x in args.windows.split(",") if x]

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for wps in windows:
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
        }
        rows.append(rec)
        np.savetxt(
            args.out / f"ir_{int(wps)}ps.csv",
            np.column_stack([wn, intens]),
            delimiter=",",
            header="wavenumber_cm-1,intensity",
            comments="",
        )

    summary = {
        "hdf5": str(args.hdf5),
        "full_duration_ps": duration_ps,
        "dt_fs": dt_fs,
        "windows": rows,
    }
    if len(rows) >= 2:
        summary["delta_fp_last_minus_100"] = None
        base = next((r for r in rows if abs(r["window_ps"] - 100) < 1e-6), rows[0])
        last = rows[-1]
        if base["peak_fp_cm1"] is not None and last["peak_fp_cm1"] is not None:
            summary["delta_fp_last_minus_ref_cm1"] = last["peak_fp_cm1"] - base["peak_fp_cm1"]
        if base["peak_xh_cm1"] is not None and last["peak_xh_cm1"] is not None:
            summary["delta_xh_last_minus_ref_cm1"] = last["peak_xh_cm1"] - base["peak_xh_cm1"]

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
