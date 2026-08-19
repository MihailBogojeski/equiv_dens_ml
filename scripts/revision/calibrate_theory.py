#!/usr/bin/env python3
"""Measure cost and cross-code agreement before committing to a large campaign.

Two things have to be settled before thousands of array tasks are submitted:

1. What each frame costs, per engine and per cluster size, so ``frames-per-shard``
   and ``--mem`` follow a measurement rather than a guess.
2. Whether ORCA and PySCF actually produce the same density. They only share a
   split if they do. def2-TZVPD is the first basis in this repo with f functions
   on oxygen, so the ``FLIP_ABS_M`` solid-harmonic phase rule in
   qm7x_orca_common is exercised here for the first time.

Usage:
  python scripts/revision/calibrate_theory.py \\
    --sizes 2,3,6 --theories pbe_d4_avdz,wb97mv_def2tzvpd \\
    --engines orca,pyscf --out results/revision/calibration/calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import df

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from qm7x_orca_common import run_orca_single  # noqa: E402
from pyscf_labeler import gpu_available, label_frame  # noqa: E402
from theory_levels import get_level, level_keys  # noqa: E402

WATER_DIR = _REPO_ROOT / "datasets/revision/water_clusters"


def read_xyz_frames(path: Path) -> list[tuple[list[int], list[list[float]]]]:
    """Multi-frame XYZ -> [(atomic numbers, positions)]."""
    from ase.data import atomic_numbers

    lines = Path(path).read_text().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        nat = int(lines[i].strip())
        z, xyz = [], []
        for row in lines[i + 2 : i + 2 + nat]:
            parts = row.split()
            sym = parts[0]
            z.append(atomic_numbers[sym[0].upper() + sym[1:].lower()] if not sym.isdigit() else int(sym))
            xyz.append([float(v) for v in parts[1:4]])
        frames.append((z, xyz))
        i += 2 + nat
    return frames


def water_frame(n: int) -> tuple[list[int], list[list[float]]]:
    """One representative (H2O)_n frame from the generated minima."""
    candidates = sorted((WATER_DIR / "minima").glob(f"n{n}_*.xyz"))
    if not candidates:
        raise SystemExit(f"no minimum geometry for n={n} under {WATER_DIR / 'minima'}")
    return read_xyz_frames(candidates[0])[0]


def peak_rss_mb() -> float:
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(self_rss, child_rss) / 1024.0


def compare(calc_a: dict, calc_b: dict, j2c: np.ndarray | None = None) -> dict:
    """Agreement between two calc_dicts for the same geometry and theory.

    Two numbers are reported for the density because they answer different
    questions. The plain L2 over fit coefficients is a *coefficient*-space
    distance, and aug-cc-pVQZ-JKFIT is deliberately over-complete, so it has
    near-null directions in which the coefficients can move a long way while the
    density they represent barely changes. The Coulomb-metric distance

        sqrt( dc^T J dc / c^T J c ),   J_PQ = (P|Q),

    is the self-interaction energy of the density *difference*, so it is blind to
    those directions and is the quantity the paper reports. Judging the codes on
    the coefficient norm alone would reject label sets whose densities are in
    fact identical to within the SCF convergence.
    """
    out = {"d_energy_hartree": abs(float(calc_a["energy"]) - float(calc_b["energy"]))}
    if "df_coeff" in calc_a and "df_coeff" in calc_b:
        a = np.asarray(calc_a["df_coeff"], dtype=float)
        b = np.asarray(calc_b["df_coeff"], dtype=float)
        if a.shape == b.shape:
            denom = np.linalg.norm(a)
            out["df_rel_l2"] = float(np.linalg.norm(a - b) / denom) if denom else float("inf")
            out["df_max_abs"] = float(np.abs(a - b).max())
            if j2c is not None and j2c.shape[0] == a.shape[0]:
                delta = a - b
                num = float(delta @ j2c @ delta)
                ref = float(a @ j2c @ a)
                out["density_coulomb_rel"] = float(np.sqrt(max(num, 0.0) / ref)) if ref > 0 else float("inf")
        else:
            out["df_rel_l2"] = float("inf")
            out["df_shape_mismatch"] = [list(a.shape), list(b.shape)]
    if "dipole" in calc_a and "dipole" in calc_b:
        out["d_dipole_max"] = float(
            np.abs(np.asarray(calc_a["dipole"]) - np.asarray(calc_b["dipole"])).max()
        )
    if "forces" in calc_a and "forces" in calc_b:
        fa = np.asarray(calc_a["forces"], dtype=float)
        fb = np.asarray(calc_b["forces"], dtype=float)
        if fa.shape == fb.shape:
            out["d_force_max"] = float(np.abs(fa - fb).max())
    return out


def recommend(records: list[dict], walltime_s: float, safety: float) -> dict:
    """Per (theory, engine, size) frames-per-shard and memory from measured cost."""
    budget = walltime_s * safety
    rec: dict[str, dict] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        key = f"{r['theory']}|{r['engine']}"
        entry = rec.setdefault(key, {"sizes": {}})
        total = r["t_total_s"]
        entry["sizes"][str(r["n_water"])] = {
            "t_total_s": round(total, 2),
            "nao": r["nao"],
            "naux": r["naux"],
            "peak_rss_mb": round(r["peak_rss_mb"], 1),
            "frames_per_shard": max(1, int(budget // total)) if total > 0 else 1,
            "mem_request_gb": max(8, int(np.ceil(r["peak_rss_mb"] / 1024.0 * 2.0))),
        }
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="2,3,6", help="comma-separated (H2O)_n sizes")
    parser.add_argument("--xyz", type=Path, default=None, help="use frames from this XYZ instead")
    parser.add_argument("--xyz-frames", default="0", help="frame indices when --xyz is given")
    parser.add_argument("--theories", default="pbe_d4_avdz,wb97mv_def2tzvpd")
    parser.add_argument("--engines", default="orca,pyscf")
    parser.add_argument("--nprocs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    parser.add_argument("--maxcore-mb", type=int, default=4000)
    parser.add_argument("--gpu", action="store_true", help="run the PySCF engine on the GPU")
    parser.add_argument("--reference-orca", action="store_true", help="also run ORCA without RI approximations")
    parser.add_argument("--no-forces", action="store_true")
    parser.add_argument("--walltime-s", type=float, default=4 * 3600.0)
    parser.add_argument("--safety", type=float, default=0.7, help="fraction of walltime to fill")
    parser.add_argument("--tol-energy", type=float, default=1e-5)
    parser.add_argument("--tol-df-rel-l2", type=float, default=1e-4)
    parser.add_argument(
        "--tol-density-rel",
        type=float,
        default=1e-4,
        help="gate on the Coulomb-metric density difference, not the coefficient norm",
    )
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "results/revision/calibration/calibration.json")
    args = parser.parse_args()

    theories = [t.strip() for t in args.theories.split(",") if t.strip()]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [t for t in theories if t not in level_keys()]
    if unknown:
        raise SystemExit(f"unknown theories {unknown}; known: {level_keys()}")

    if args.xyz:
        idx = [int(v) for v in args.xyz_frames.split(",") if v.strip()]
        all_frames = read_xyz_frames(args.xyz)
        geometries = [(f"frame{i}", all_frames[i]) for i in idx]
    else:
        sizes = [int(v) for v in args.sizes.split(",") if v.strip()]
        geometries = [(str(n), water_frame(n)) for n in sizes]

    scratch_root = Path(args.scratch or os.environ.get("SLURM_TMPDIR", args.out.parent / "scratch"))
    scratch_root.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    orca_bin = shutil.which("orca") or "orca"
    orca2json = shutil.which("orca_2json") or "orca_2json"

    records: list[dict] = []
    calcs: dict[tuple[str, str, str], dict] = {}
    mols: dict[tuple[str, str], object] = {}

    for theory_key in theories:
        theory = get_level(theory_key)
        for tag, (z, xyz) in geometries:
            n_water = sum(1 for a in z if a == 8) or len(z)
            for engine in engines:
                variants = [False, True] if (engine == "orca" and args.reference_orca) else [False]
                for is_ref in variants:
                    name = f"{engine}{'-ref' if is_ref else ''}"
                    job_dir = scratch_root / f"{theory_key}_{tag}_{name}"
                    t0 = time.perf_counter()
                    try:
                        if engine == "orca":
                            _mol, calc, diag = run_orca_single(
                                z, xyz, job_dir,
                                level=theory,
                                orca_bin=orca_bin,
                                orca2json_bin=orca2json,
                                nprocs=args.nprocs,
                                maxcore_mb=args.maxcore_mb,
                                fit_df=True,
                                reference=is_ref,
                            )
                        else:
                            _mol, calc, diag = label_frame(
                                z, xyz, theory,
                                use_gpu=args.gpu,
                                fit_df=True,
                                with_forces=not args.no_forces,
                            )
                        status, error = "ok", None
                    except Exception as exc:
                        calc, diag = {}, {}
                        status, error = "error", f"{type(exc).__name__}: {exc}"
                    total = time.perf_counter() - t0

                    rec = {
                        "theory": theory_key,
                        "theory_label": theory.label,
                        "geometry": tag,
                        "n_water": n_water,
                        "n_atoms": len(z),
                        "engine": name,
                        "status": status,
                        "t_total_s": total,
                        "peak_rss_mb": peak_rss_mb(),
                        "nao": diag.get("nao", 0),
                        "naux": diag.get("naux", 0),
                        **{k: v for k, v in diag.items() if k not in {"nao", "naux", "theory", "engine"}},
                    }
                    if error:
                        rec["error"] = error
                    else:
                        rec["energy"] = float(calc["energy"])
                        calcs[(theory_key, tag, name)] = calc
                        mols.setdefault((theory_key, tag), _mol)
                    records.append(rec)
                    print(
                        f"{theory_key:18s} n={n_water:<3d} {name:10s} {status:5s} "
                        f"{total:8.1f}s nao={rec['nao']:<5d} naux={rec['naux']:<5d}"
                        + (f"  ortho={diag['mo_orthonormality_error']:.2e}" if "mo_orthonormality_error" in diag else "")
                        + (f"  {error}" if error else ""),
                        flush=True,
                    )

    comparisons = []
    for theory_key in theories:
        theory = get_level(theory_key)
        for tag, _ in geometries:
            a = calcs.get((theory_key, tag, "orca"))
            b = calcs.get((theory_key, tag, "pyscf"))
            pairs = [("orca_vs_pyscf", a, b)]
            ref = calcs.get((theory_key, tag, "orca-ref"))
            if ref is not None:
                pairs.append(("orca_vs_orca_ref", a, ref))
                pairs.append(("orca_ref_vs_pyscf", ref, b))

            j2c = None
            mol = mols.get((theory_key, tag))
            if mol is not None:
                j2c = df.addons.make_auxmol(mol, theory.auxbasis).intor("int2c2e")

            for pair_name, ca, cb in pairs:
                if ca is None or cb is None:
                    continue
                cmp = compare(ca, cb, j2c=j2c)
                cmp.update({"theory": theory_key, "geometry": tag, "pair": pair_name})
                cmp["passes_gate"] = bool(
                    cmp["d_energy_hartree"] <= args.tol_energy
                    and cmp.get("density_coulomb_rel", cmp.get("df_rel_l2", float("inf")))
                    <= args.tol_density_rel
                )
                comparisons.append(cmp)
                print(
                    f"  cmp {theory_key:18s} {tag:>4s} {pair_name:18s} "
                    f"dE={cmp['d_energy_hartree']:.3e} dfL2={cmp.get('df_rel_l2', float('nan')):.3e} "
                    f"coul={cmp.get('density_coulomb_rel', float('nan')):.3e} "
                    f"{'PASS' if cmp['passes_gate'] else 'FAIL'}",
                    flush=True,
                )

    ortho = [r["mo_orthonormality_error"] for r in records if "mo_orthonormality_error" in r]
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": os.uname().nodename,
        "nprocs": args.nprocs,
        "gpu_requested": args.gpu,
        "gpu_available": gpu_available(),
        "tolerances": {
            "energy_hartree": args.tol_energy,
            "df_rel_l2": args.tol_df_rel_l2,
            "density_coulomb_rel": args.tol_density_rel,
        },
        "records": records,
        "comparisons": comparisons,
        "max_mo_orthonormality_error": max(ortho) if ortho else None,
        "recommendations": recommend(records, args.walltime_s, args.safety),
    }
    args.out.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {args.out}")

    failures = [c for c in comparisons if not c["passes_gate"]]
    if failures:
        print(f"WARNING: {len(failures)}/{len(comparisons)} cross-code comparisons outside tolerance")
    if ortho and max(ortho) > 1e-6:
        print(f"WARNING: AO permutation suspect, max |C^T S C - I| = {max(ortho):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
