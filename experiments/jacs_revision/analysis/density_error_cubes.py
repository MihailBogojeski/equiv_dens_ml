#!/usr/bin/env python3
"""DenSNet and DFT densities on a cubic grid, with local error maps, for visualization.

For each requested structure of a label set (the manifests used by
rescore_full_density.py), this writes Gaussian cube files in the PySCF molecule
frame (bohr):

  <out>/<tag>_ref.cube       rho_DFT, from the stored MO coefficients (full SCF)
  <out>/<tag>_pred.cube      rho_DenSNet (SAD prior + learned correction)
  <out>/<tag>_err.cube       |rho_DenSNet - rho_DFT|
  <out>/<tag>_graderr.cube   |grad rho_DenSNet - grad rho_DFT|
  <out>/<tag>_grad.cube      |grad rho_DFT|
  <out>/<tag>_errrel.cube    100 |rho_DenSNet - rho_DFT| / rho_DFT            (percent)
  <out>/<tag>_graderrrel.cube 100 |grad rho_DenSNet - grad rho_DFT| / |grad rho_DFT| (percent)
  <out>/<tag>.json           grid, geometry, integrated errors

Gradients use the same central differences on both densities, so stencil
errors largely cancel in the difference. Near the nuclear cusps both
densities are steep; the rendering looks at a valence isosurface, far from them.

    .venv/bin/python -B experiments/jacs_revision/analysis/density_error_cubes.py \
        --manifest results/revision/rescore_fullref/manifest_extreme_modes.json \
        --label-set ood_modes_ethanethiol_pbe_d4_avdz --indices 0 1 2 --out results/revision/density_movie/test
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(REPO), str(REPO / "src")]

BOHR = 0.52917721092  # Angstrom


def write_cube(path: Path, mol, origin: np.ndarray, spacing: float, values: np.ndarray, comment: str) -> None:
    nx, ny, nz = values.shape
    coords = mol.atom_coords()  # bohr
    with open(path, "w") as f:
        f.write(f"{comment}\nDenSNet density-error visualization\n")
        f.write(f"{mol.natm:5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        for n, axis in zip((nx, ny, nz), np.eye(3) * spacing):
            f.write(f"{n:5d} {axis[0]:12.6f} {axis[1]:12.6f} {axis[2]:12.6f}\n")
        for z, xyz in zip(mol.atom_charges(), coords):
            f.write(f"{z:5d} {float(z):12.6f} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}\n")
        flat = values.reshape(nx * ny, nz)
        for row in flat:
            for start in range(0, nz, 6):
                f.write("".join(f" {v:12.5E}" for v in row[start:start + 6]) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--label-set", required=True, help="name of the label set in the manifest")
    ap.add_argument("--model", default=None, help="model name within the label set (default: first)")
    ap.add_argument("--indices", type=int, nargs="+", required=True, help="structure indices in the label set")
    ap.add_argument("--spacing", type=float, default=0.2, help="grid spacing (bohr)")
    ap.add_argument("--margin", type=float, default=4.5, help="box margin around the atoms (bohr)")
    ap.add_argument("--box", type=float, nargs=6, default=None,
                    help="fixed box xmin xmax ymin ymax zmin zmax (bohr), for movies with a constant frame")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--out", type=Path, required=True)
    opt = ap.parse_args()

    import torch
    from pyscf.dft import numint

    from experiments.jacs_revision.analysis import rescore_full_density as R

    sets = json.loads(opt.manifest.read_text())
    sets = sets["label_sets"] if isinstance(sets, dict) else sets
    label_set = next(s for s in sets if s["name"] == opt.label_set)
    spec = next(m for m in label_set["models"] if opt.model in (None, m["name"]))
    spec = dict(spec, out=str(opt.out / "_unused.json"))
    entry = R.ModelEntry(spec, label_set, opt.device, 2, Path(tempfile.mkdtemp(prefix="dens_movie_")))
    reference = R.make_dataset(entry.args, label_set["np"], label_set["dens"], label_set["elements"], False, 2,
                               grid_only=True)
    opt.out.mkdir(parents=True, exist_ok=True)

    for idx in opt.indices:
        t0 = time.time()
        ref = reference.get_properties([idx])
        shift = ref.get("pos_shift", 0)
        shift = shift.double().numpy().reshape(-1)[:3] if torch.is_tensor(shift) else np.zeros(3)
        mol = reference.mols[idx]
        if not mol._built:
            mol.build(verbose=0)
        xyz = mol.atom_coords()
        if opt.box:
            lo = np.array(opt.box[0::2]); hi = np.array(opt.box[1::2])
        else:
            lo = xyz.min(0) - opt.margin; hi = xyz.max(0) + opt.margin
        axes = [np.arange(a, b + 1e-9, opt.spacing) for a, b in zip(lo, hi)]
        shape = tuple(len(a) for a in axes)
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)  # bohr, PySCF frame

        # reference: full SCF density from MO coefficients
        mo = np.asarray(reference.coeffs[idx]["mo_coeff"]); occ = np.asarray(reference.coeffs[idx]["mo_occ"])
        if occ.ndim == 1:
            mo, occ = mo[None], occ[None]
        c_occ = np.concatenate([m[:, o > 0] * np.sqrt(o[o > 0]) for m, o in zip(mo, occ)], axis=1)
        rho_ref = np.empty(len(grid))
        for s in range(0, len(grid), 50000):
            ao = numint.eval_ao(mol, grid[s:s + 50000])
            rho_ref[s:s + 50000] = ((ao @ c_occ) ** 2).sum(1)

        # model: same points, in the dataset's (shifted, Angstrom) frame
        props = entry.dataset.get_properties([idx])
        coords = torch.as_tensor(grid * BOHR + shift, dtype=props["batch_positions"].dtype)[None]
        props["coords"] = coords
        props["coord_weights"] = torch.full((1, len(grid)), (opt.spacing * BOHR) ** 3, dtype=coords.dtype)
        props["atom_density"] = entry.dataset.sample_atom_density(props["batch_positions"],
                                                                  props["batch_atom_numbers"], coords)
        rho_pred = R.predict_density(entry.model, props, opt.device, opt.chunk)

        ref3, pred3 = rho_ref.reshape(shape), rho_pred.reshape(shape)
        g_ref = np.stack(np.gradient(ref3, opt.spacing), -1)
        g_pred = np.stack(np.gradient(pred3, opt.spacing), -1)
        err = np.abs(pred3 - ref3)
        graderr = np.linalg.norm(g_pred - g_ref, axis=-1)
        gnorm = np.linalg.norm(g_ref, axis=-1)
        dv = opt.spacing ** 3
        n_ref = float(ref3.sum() * dv)
        tag = f"{label_set['name']}_{idx:04d}"
        # relative maps (percent); only meaningful on a valence isosurface of rho_ref
        errrel = 100 * err / np.maximum(ref3, 1e-8)
        graderrrel = 100 * graderr / np.maximum(gnorm, 1e-8)
        for name, arr in (("ref", ref3), ("pred", pred3), ("err", err), ("graderr", graderr), ("grad", gnorm),
                          ("errrel", errrel), ("graderrrel", graderrrel)):
            write_cube(opt.out / f"{tag}_{name}.cube", mol, lo, opt.spacing, arr, f"{tag} {name}")
        with open(opt.out / f"{tag}_ref.xyz", "w") as f:  # nuclei for the stick model (Angstrom, same frame)
            f.write(f"{mol.natm}\n{tag}\n")
            for i in range(mol.natm):
                x, y, z = xyz[i] * BOHR
                f.write(f"{mol.atom_pure_symbol(i)} {x:.6f} {y:.6f} {z:.6f}\n")
        info = {
            "label_set": label_set["name"], "index": idx, "model": spec["name"], "run_dir": spec["run_dir"],
            "shape": shape, "spacing_bohr": opt.spacing, "origin_bohr": lo.tolist(),
            "n_elec_grid_ref": n_ref, "n_elec_grid_pred": float(pred3.sum() * dv),
            "afe_grid_pct": float(100 * err.sum() * dv / n_ref),
            "seconds": round(time.time() - t0, 1),
        }
        (opt.out / f"{tag}.json").write_text(json.dumps(info, indent=1) + "\n")
        print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
