#!/usr/bin/env python3
"""Maps of one IR mode's density response and of per-atom Born-charge errors, as cube files.

Inputs: the cubes of the minimum and the +/- displaced structures written by
density_error_cubes.py on one shared grid (--box), and mode_info.json from
mode_response_setup.py. Outputs in <cubes>/:

  <tag>_resp_dft.cube      d rho_DFT / dQ          (e bohr^-3 per amu^1/2 A), signed
  <tag>_resp_dn.cube       d rho_DenSNet / dQ      same units, signed
  <tag>_resp_err.cube      100 |d rho_DenSNet/dQ - d rho_DFT/dQ| / max|d rho_DFT/dQ on the isosurface|  (percent)
  <tag>_born_long.cube     |Z*_long(DenSNet) - Z*_long(DFT)| of the nearest atoms, Gaussian-blended (e)
  <tag>_mode_dmu.cube      |d mu/dQ contribution of the nearest atoms, DenSNet - DFT| (e amu^-1/2)

The last two paint per-atom quantities onto the density: each grid point takes a
Gaussian-weighted average (sigma --sigma) of the atoms' values, so a whole atom's
region lights up when that atom's response is wrong.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOHR = 0.52917721092


def read_cube(path: Path):
    with open(path) as f:
        lines = f.readlines()
    natm, *origin = lines[2].split()
    natm = int(natm); origin = np.array(origin[:3], float)
    shape, axes = [], []
    for line in lines[3:6]:
        n, *a = line.split(); shape.append(int(n)); axes.append(np.array(a, float))
    atoms = [line.split() for line in lines[6:6 + natm]]
    z = np.array([int(a[0]) for a in atoms]); xyz = np.array([[float(v) for v in a[2:5]] for a in atoms])
    vals = np.array(" ".join(lines[6 + natm:]).split(), float).reshape(shape)
    return {"origin": origin, "spacing": axes[0][0], "shape": tuple(shape), "z": z, "xyz": xyz, "values": vals,
            "header": lines[:6 + natm]}


def write_like(template: dict, path: Path, values: np.ndarray, comment: str) -> None:
    with open(path, "w") as f:
        f.write(f"{comment}\n")
        f.writelines(template["header"][1:])
        flat = values.reshape(-1, values.shape[-1])
        for row in flat:
            for s in range(0, len(row), 6):
                f.write("".join(f" {v:12.5E}" for v in row[s:s + 6]) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cubes", type=Path, required=True)
    ap.add_argument("--label-set", required=True)
    ap.add_argument("--indices", type=int, nargs=3, default=[0, 1, 2], help="minimum, +Q, -Q")
    ap.add_argument("--mode-info", type=Path, required=True)
    ap.add_argument("--level", type=float, default=0.03, help="isosurface used to normalise resp_err")
    ap.add_argument("--sigma", type=float, default=0.6, help="Gaussian width (Angstrom) for painting atom values")
    opt = ap.parse_args()

    info = json.loads(opt.mode_info.read_text())
    dq = info["dq_amu05_A"]
    t0, tp, tm = (f"{opt.label_set}_{i:04d}" for i in opt.indices)
    ref0 = read_cube(opt.cubes / f"{t0}_ref.cube")
    rp, rm = (read_cube(opt.cubes / f"{t}_ref.cube")["values"] for t in (tp, tm))
    pp, pm = (read_cube(opt.cubes / f"{t}_pred.cube")["values"] for t in (tp, tm))
    resp_dft = (rp - rm) / (2 * dq)
    resp_dn = (pp - pm) / (2 * dq)
    rho0 = ref0["values"]
    shell = (rho0 > 0.8 * opt.level) & (rho0 < 1.25 * opt.level)
    scale = np.abs(resp_dft[shell]).max()
    err = 100 * np.abs(resp_dn - resp_dft) / scale

    # per-atom values painted onto the grid
    nx, ny, nz = ref0["shape"]
    axes = [ref0["origin"][i] + ref0["spacing"] * np.arange(n) for i, n in enumerate((nx, ny, nz))]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3) * BOHR  # Angstrom
    xyz = ref0["xyz"] * BOHR
    atoms = info["atoms"]
    born = np.array([abs(a["born_densnet"]["flux_long"] - a["born_dft"]["flux_long"]) for a in atoms])
    mdmu = np.array([np.linalg.norm(np.subtract(a["mode_dmu_densnet"], a["mode_dmu_dft"])) for a in atoms])
    num_b = np.zeros(len(grid)); num_m = np.zeros(len(grid)); den = np.zeros(len(grid))
    for i in range(len(xyz)):
        w = np.exp(-((grid - xyz[i]) ** 2).sum(1) / (2 * opt.sigma ** 2))
        num_b += w * born[i]; num_m += w * mdmu[i]; den += w
    den = np.maximum(den, 1e-30)
    painted_b = (num_b / den).reshape(ref0["shape"])
    painted_m = (num_m / den).reshape(ref0["shape"])

    for name, arr in (("resp_dft", resp_dft), ("resp_dn", resp_dn), ("resp_err", err),
                      ("born_long", painted_b), ("mode_dmu", painted_m)):
        write_like(ref0, opt.cubes / f"{t0}_{name}.cube", arr, f"{t0} {name}")
    summary = {"resp_scale_on_isosurface": float(scale), "resp_err_pct_p95_on_isosurface": float(np.percentile(err[shell], 95)),
               "born_long_err_by_element": {el: float(np.mean([born[i] for i, a in enumerate(atoms) if a["Z"] == z]))
                                            for el, z in (("C", 6), ("S", 16), ("H", 1))},
               "mode_dmu_err_by_element": {el: float(np.mean([mdmu[i] for i, a in enumerate(atoms) if a["Z"] == z]))
                                           for el, z in (("C", 6), ("S", 16), ("H", 1))}}
    (opt.cubes / f"{t0}_mode_maps.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
