#!/usr/bin/env python3
"""DFT-label geometries for the JACS revision.

Wraps the paper PySCF pipeline (see scripts/data/generate_polythiophene_dataset.py)
but allows --xc pbe|pbe0 and optional D4 / density fitting.

Usage:
  python scripts/revision/generate_dft_labels.py \\
    --trajectory datasets/revision/water_clusters/train.xyz \\
    --output-prefix water_train --xc pbe --d4 --df
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ase
import numpy as np
from pyscf import df, dft, gto
from pyscf.scf import hf

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "data"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from generate_polythiophene_dataset import (  # noqa: E402
    _is_precomputed_frame,
    _to_numpy,
    _try_import_gpu4pyscf,
    load_trajectory,
)

hf.MUTE_CHKFILE = True

BASIS = "augccpvdz"
AUXBASIS = "augccpvqzjkfit"


def compute_frame(positions, atom_numbers, use_gpu, use_df, xc, use_d4):
    atom = [(int(anum), pos) for anum, pos in zip(atom_numbers, positions) if anum > 0]
    nelec = sum(gto.charge(anum) for anum, _ in atom)
    if nelec % 2 != 0:
        raise ValueError(f"Odd electron count ({nelec}); closed-shell only.")
    mol = gto.M(atom=atom, basis=BASIS)
    mol.build()

    mf = dft.RKS(mol)
    mf.chkfile = None
    mf.xc = xc
    mf.max_cycle = 1000
    if use_gpu and _try_import_gpu4pyscf():
        mf = mf.to_gpu()
    mf.kernel()
    if use_gpu:
        mf = mf.to_cpu()

    if use_d4:
        import dftd4.pyscf as d4disp

        d4mf = d4disp.energy(mf).run()
        gradients = d4mf.nuc_grad_method().kernel()
        energy = float(d4mf.e_tot)
        mo_coeff = _to_numpy(d4mf.mo_coeff)
        mo_occ = _to_numpy(d4mf.mo_occ)
        dm1 = d4mf.make_rdm1(d4mf.mo_coeff, d4mf.mo_occ)
    else:
        gradients = mf.nuc_grad_method().kernel()
        energy = float(mf.e_tot)
        mo_coeff = _to_numpy(mf.mo_coeff)
        mo_occ = _to_numpy(mf.mo_occ)
        dm1 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)

    calc_dict = {
        "mo_coeff": mo_coeff,
        "mo_occ": mo_occ,
        "energy": energy,
        "forces": _to_numpy(-gradients / ase.units.Bohr),
        "xc": xc,
    }
    if use_df:
        import scipy

        dm1 = _to_numpy(dm1)
        auxmol = df.addons.make_auxmol(mol, AUXBASIS)
        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
        ints_2c2e = auxmol.intor("int2c2e")
        nao = mol.nao
        naux = auxmol.nao
        df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao * nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        calc_dict["df_coeff"] = np.einsum("Pij,ij->P", df_coef, dm1)
        calc_dict["auxbasis"] = AUXBASIS

    # Analytic dipole from the density.
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        dip_ints = mol.intor("int1e_r", comp=3)
    calc_dict["dipole"] = np.einsum("xij,ji->x", dip_ints, _to_numpy(dm1))
    return (mol.pack(), calc_dict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", "-t", required=True)
    parser.add_argument("--output-prefix", "-o", default=None)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--xc", default="pbe", choices=("pbe", "pbe0", "b3lyp"))
    parser.add_argument("--d4", action="store_true")
    parser.add_argument("--df", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many frames after stride")
    parser.add_argument("--end-index", type=int, default=0, help="Exclusive end after stride; 0 = all")
    parser.add_argument("--resume", action="store_true", help="Skip frames already in the output npy")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or Path(args.trajectory).stem
    dens_path = out_dir / f"{prefix}_pyscf_augccpvdz_{args.xc}.npy"
    npy_path = out_dir / f"{prefix}_npy.npy"

    use_gpu = (not args.no_gpu) and _try_import_gpu4pyscf()
    frames = load_trajectory(args.trajectory, stride=args.stride)
    if args.start_index:
        frames = frames[args.start_index :]
    if args.end_index:
        frames = frames[: max(0, args.end_index - args.start_index)]
    if args.max_frames:
        frames = frames[: args.max_frames]
    if not args.quiet:
        print(f"frames={len(frames)} xc={args.xc} d4={args.d4} df={args.df} gpu={use_gpu}")

    if frames and _is_precomputed_frame(frames[0]):
        results = list(frames)
    else:
        results = []
        if args.resume and dens_path.exists():
            results = list(np.load(dens_path, allow_pickle=True))
            if not args.quiet:
                print(f"resume: {len(results)} frames already in {dens_path}")
            frames = frames[len(results) :]
        offset = len(results)
        for i, (pos, anum) in enumerate(frames):
            results.append(compute_frame(pos, anum, use_gpu, args.df, args.xc, args.d4))
            if not args.quiet:
                print(
                    f"  [{offset + i + 1}/{offset + len(frames)}] "
                    f"E={results[-1][1]['energy']:.6f}",
                    flush=True,
                )
            if (offset + i + 1) % args.save_interval == 0:
                np.save(dens_path, np.array(results, dtype=object), allow_pickle=True)

    np.save(dens_path, np.array(results, dtype=object), allow_pickle=True)

    geo = {"positions": [], "atom_numbers": None, "energy": [], "forces": []}
    for pos, anum in (
        frames if not (frames and _is_precomputed_frame(frames[0])) else []
    ) or []:
        geo["positions"].append(pos)
        geo["atom_numbers"] = anum if geo["atom_numbers"] is None else geo["atom_numbers"]
    if not geo["positions"]:
        from generate_polythiophene_dataset import load_trajectory as _lt  # already imported

        raw = _lt(args.trajectory, stride=args.stride)
        if args.max_frames:
            raw = raw[: args.max_frames]
        if raw and not _is_precomputed_frame(raw[0]):
            geo["positions"] = [p for p, _ in raw]
            geo["atom_numbers"] = raw[0][1]

    if geo["positions"]:
        geo["positions"] = np.asarray(geo["positions"])
        geo["atom_numbers"] = np.asarray(geo["atom_numbers"])
        geo["energy"] = np.array([r[1]["energy"] for r in results])
        geo["forces"] = np.array([r[1]["forces"] for r in results])
        if "dipole" in results[0][1]:
            geo["dipole_moment"] = np.array([r[1]["dipole"] for r in results])
        np.save(npy_path, geo, allow_pickle=True)

    print(f"wrote {dens_path}")
    if geo["positions"]:
        print(f"wrote {npy_path}")


if __name__ == "__main__":
    main()
