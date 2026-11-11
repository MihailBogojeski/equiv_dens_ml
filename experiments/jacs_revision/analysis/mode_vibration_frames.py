#!/usr/bin/env python3
"""Frames of one normal mode vibrating, for render_density_error.py (linear response).

Frame k of --n per period has Q_k = A sin(2 pi k / n). Written per frame:
  <out>/vib_<k>_ref.cube   rho_DFT(Q_k) = rho_0 + Q_k d rho_DFT/dQ  (the cloud)
  <out>/vib_<k>_ref.xyz    nuclei at R_0 + Q_k * mode displacement
  <out>/frames.json        f_k = sin(2 pi k / n) per frame, used to scale the colour ramps

The colour maps stay those of the static analysis (mode_response_cubes.py): at
time t each instantaneous quantity is f(t) times its static map, so the renderer
rescales the ramp by 1/f(t) (and flips it when f < 0) instead of writing new maps.
A is chosen so the largest single-atom displacement is --max-disp (an exaggeration
for visibility; the response itself is the linear one computed at 0.02 A).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mode_response_cubes import BOHR, read_cube, write_like  # same directory


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubes", type=Path, required=True)
    ap.add_argument("--tag", default="pt8_cc_0000")
    ap.add_argument("--geoms", type=Path, required=True, help="geoms.npy from mode_response_setup.py")
    ap.add_argument("--mode-info", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-disp", type=float, default=0.15, help="largest atom displacement (Angstrom)")
    ap.add_argument("--out", type=Path, required=True)
    opt = ap.parse_args()

    info = json.loads(opt.mode_info.read_text())
    g = np.load(opt.geoms, allow_pickle=True).item()
    pos0, z = g["positions"][0], g["atom_numbers"]
    disp = (g["positions"][1] - g["positions"][2]) / (2 * info["dq_amu05_A"])  # R per unit Q
    amp = opt.max_disp / np.linalg.norm(disp, axis=1).max()
    rho0 = read_cube(opt.cubes / f"{opt.tag}_ref.cube")
    resp = read_cube(opt.cubes / f"{opt.tag}_resp_dft.cube")["values"]
    symbols = {1: "H", 6: "C", 8: "O", 16: "S"}
    opt.out.mkdir(parents=True, exist_ok=True)
    fr = []
    for k in range(opt.n):
        f = float(np.sin(2 * np.pi * k / opt.n))
        q = amp * f
        write_like(rho0, opt.out / f"vib_{k:03d}_ref.cube", rho0["values"] + q * resp, f"vib {k} Q={q:.4f}")
        with open(opt.out / f"vib_{k:03d}_ref.xyz", "w") as fh:
            fh.write(f"{len(z)}\nvib {k}\n")
            for zi, p in zip(z, pos0 + q * disp):
                fh.write(f"{symbols[int(zi)]} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        fr.append({"k": k, "f": f, "Q": q})
        print(k, round(f, 3), flush=True)
    (opt.out / "frames.json").write_text(json.dumps({"amplitude_Q": amp, "max_disp_A": opt.max_disp,
                                                      "frequency_cm1": info["frequency_cm1"], "frames": fr}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
