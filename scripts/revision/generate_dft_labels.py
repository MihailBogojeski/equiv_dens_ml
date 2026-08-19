#!/usr/bin/env python3
"""DFT-label geometries for the JACS revision.

Wraps the paper PySCF pipeline (see scripts/data/generate_polythiophene_dataset.py)
with optional D4 and density fitting, on CPU or GPU.

The level of theory can be given either as a `theory_levels` entry (`--theory
wb97mv_def2tzvpd`, which is what the campaign uses and what the ORCA path also
reads) or as the original loose `--xc`/`--basis` pair. The default remains
`--xc pbe` at aug-cc-pVDZ so existing output files keep their names and the
partially finished runs on disk can still be resumed.

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
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "data"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from generate_polythiophene_dataset import (  # noqa: E402
    _is_precomputed_frame,
    _to_numpy,
    _try_import_gpu4pyscf,
    load_trajectory,
)
from pyscf_labeler import density_fit_coeffs  # noqa: E402
from theory_levels import basis_tag, get_level, level_keys  # noqa: E402

hf.MUTE_CHKFILE = True

BASIS = "augccpvdz"
AUXBASIS = "augccpvqzjkfit"


def compute_frame(positions, atom_numbers, use_gpu, use_df, xc, use_d4, basis=BASIS, nlc_grid_level=1):
    atom = [(int(anum), pos) for anum, pos in zip(atom_numbers, positions) if anum > 0]
    nelec = sum(gto.charge(anum) for anum, _ in atom)
    if nelec % 2 != 0:
        raise ValueError(f"Odd electron count ({nelec}); closed-shell only.")
    mol = gto.M(atom=atom, basis=basis)
    mol.build()

    mf = dft.RKS(mol)
    mf.chkfile = None
    mf.xc = xc
    mf.max_cycle = 1000
    if mf._numint.libxc.is_nlc(xc):
        # wB97M-V and friends carry VV10; pin the NLC grid so the cost of a
        # frame does not depend on which PySCF default the machine happens to have.
        mf.nlcgrids.level = nlc_grid_level
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
        dm1 = _to_numpy(dm1)
        calc_dict["df_coeff"] = density_fit_coeffs(mol, dm1, AUXBASIS)
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
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument(
        "--theory",
        default=None,
        choices=level_keys(),
        help="theory_levels entry; overrides --xc/--basis/--d4 and is what the campaign uses",
    )
    parser.add_argument("--nlc-grid-level", type=int, default=1)
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

    if args.theory:
        level = get_level(args.theory)
        if "pyscf" not in level.engines:
            raise SystemExit(
                f"theory {args.theory} is restricted to {level.engines}; "
                "its labels must not be produced by PySCF (see theory_levels.py)"
            )
        args.xc, args.basis, args.d4 = level.pyscf_xc, level.pyscf_basis, level.d4

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or Path(args.trajectory).stem
    dens_path = out_dir / f"{prefix}_pyscf_{basis_tag(args.basis)}_{args.xc}.npy"
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
        print(
            f"frames={len(frames)} xc={args.xc} basis={args.basis} "
            f"d4={args.d4} df={args.df} gpu={use_gpu} -> {dens_path.name}"
        )

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
            results.append(
                compute_frame(
                    pos,
                    anum,
                    use_gpu,
                    args.df,
                    args.xc,
                    args.d4,
                    basis=args.basis,
                    nlc_grid_level=args.nlc_grid_level,
                )
            )
            if not args.quiet:
                print(
                    f"  [{offset + i + 1}/{offset + len(frames)}] "
                    f"E={results[-1][1]['energy']:.6f}",
                    flush=True,
                )
            if (offset + i + 1) % args.save_interval == 0:
                np.save(dens_path, np.array(results, dtype=object), allow_pickle=True)

    np.save(dens_path, np.array(results, dtype=object), allow_pickle=True)

    raw = load_trajectory(args.trajectory, stride=args.stride)
    if args.start_index:
        raw = raw[args.start_index :]
    if args.end_index:
        raw = raw[: max(0, args.end_index - args.start_index)]
    if args.max_frames:
        raw = raw[: args.max_frames]
    raw = raw[: len(results)]
    wrote_npy = False
    if raw and not _is_precomputed_frame(raw[0]):
        geo = {
            "positions": np.asarray([p for p, _ in raw], dtype=object),
            "atom_numbers": np.asarray([a for _, a in raw], dtype=object),
            "energy": np.array([r[1]["energy"] for r in results]),
            "forces": np.asarray([r[1]["forces"] for r in results], dtype=object),
        }
        if "dipole" in results[0][1]:
            geo["dipole_moment"] = np.array([r[1]["dipole"] for r in results])
        np.save(npy_path, geo, allow_pickle=True)
        wrote_npy = True

    print(f"wrote {dens_path}")
    if wrote_npy:
        print(f"wrote {npy_path}")


if __name__ == "__main__":
    main()
