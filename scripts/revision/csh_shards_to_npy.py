#!/usr/bin/env python3
"""Package density-fitted CSH shards into the NPY pair the DenSNet loader expects.

`AtomsDensityData` reads two files: a geometry dict and an object array of
(mol.pack(), calc_dict) pairs. Producing exactly that format means the training
code needs no modification.

`mo_coeff` and `mo_occ` are read unconditionally by the loader, but the real MO
coefficients would be nao^2 per structure - tens of megabytes each at these
system sizes - so placeholders are stored instead. They are never dereferenced
when the reference density comes from the fitted coefficients
(`--projected_density=True`), and any run that turns that off will fail loudly
rather than train on zeros.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omol_csh_common import BASIS, ECP  # noqa: E402

_KEYERROR_RE = re.compile(r"^KeyError:\s*(-?\d+)$")


def classify_failure(error: str) -> str:
    """Bucket a recorded failure's `error` string by cause.

    `keyerror` failures are the ones `--retry-failures` against a wider AO
    table can fix; `drift` failures are the `--n-elec-tol` rejects, whose
    rate rising with molecule size is the expected, documented consequence
    of using an absolute (not relative) tolerance.
    """
    error = str(error).strip()
    if _KEYERROR_RE.match(error):
        return "keyerror"
    if error.startswith("n_elec drift"):
        return "drift"
    return "other"


def load_shards(outdir: Path):
    records, failures = [], []
    for path in sorted(outdir.glob("shard_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            records.extend(list(data["records"]))
            failures.extend(list(data["failures"]))
    return records, failures


def check_completeness(
    outdir: Path,
    manifest_entries,
    shard_size: int,
    failures,
    allow_incomplete: bool,
) -> list[str]:
    """Return warning strings; raise SystemExit if incomplete and not `allow_incomplete`.

    Refuses (unless overridden) when any of:
      - a manifest is given and fewer `.done` markers exist than
        `ceil(n_manifest_entries / shard_size)`,
      - packaging would otherwise proceed while shards still have `.claim`
        files (a first pass or retry is still in flight for them), or
      - residual `failures` still include a `KeyError` that looks retryable
        (i.e. a fresh `--retry-failures` pass might still recover it).
    """
    warnings = []
    if manifest_entries is not None:
        n_expected = math.ceil(len(manifest_entries) / shard_size)
        n_done = len(list(outdir.glob("shard_*.done")))
        if n_done < n_expected:
            warnings.append(
                f"only {n_done}/{n_expected} shards have a .done marker "
                f"(manifest has {len(manifest_entries)} entries at shard-size {shard_size})"
            )
    in_flight = sorted(p.stem for p in outdir.glob("shard_*.claim"))
    if in_flight:
        warnings.append(f"{len(in_flight)} shard(s) still claimed (in flight): {in_flight}")
    n_keyerror = sum(1 for f in failures if classify_failure(f.get("error", "")) == "keyerror")
    if n_keyerror:
        warnings.append(
            f"{n_keyerror} residual failure(s) are KeyErrors that look retryable "
            "(re-run csh_shard_to_df.py --retry-failures against a wider AO table)"
        )
    if warnings and not allow_incomplete:
        raise SystemExit(
            "REFUSING to package, dataset looks incomplete:\n  - "
            + "\n  - ".join(warnings)
            + "\nPass --allow-incomplete to package a mid-flight snapshot anyway."
        )
    return warnings


def build_pair(records, auxbasis: str, with_mo: bool):
    positions = np.empty(len(records), dtype=object)
    numbers = np.empty(len(records), dtype=object)
    dens = []
    for i, rec in enumerate(records):
        positions[i] = np.asarray(rec["positions"], dtype=np.float64)
        numbers[i] = np.asarray(rec["atom_numbers"], dtype=np.int64)

        atom = [
            (int(z), tuple(float(x) for x in xyz))
            for z, xyz in zip(rec["atom_numbers"], rec["positions"])
        ]
        mol = gto.Mole()
        mol.atom = atom
        mol.basis = BASIS
        mol.ecp = ECP
        mol.charge = int(rec["charge"])
        mol.spin = 0
        mol.unit = "Angstrom"
        mol.verbose = 0
        mol.build()

        calc = {
            "df_coeff": np.asarray(rec["df_coeff"], dtype=np.float64),
            "auxbasis": auxbasis,
            "xc": "wb97m-v",
            "energy": 0.0,
            "n_elec": int(rec["n_elec"]),
            "charge": int(rec["charge"]),
        }
        if with_mo:
            calc["mo_coeff"] = np.asarray(rec["mo_coeff"], dtype=np.float64)
            calc["mo_occ"] = np.asarray(rec["mo_occ"], dtype=np.float64)
        else:
            calc["mo_coeff"] = np.zeros((1, 1), dtype=np.float64)
            calc["mo_occ"] = np.zeros((1,), dtype=np.float64)
        dens.append((mol.pack(), calc))

    geo = {"positions": positions, "atom_numbers": numbers}
    return geo, np.array(dens, dtype=object)


def failure_size_stats(failures, manifest_entries):
    """natoms summary for a subset of `failures`, joined against `manifest_entries` by path.

    Documents (without changing) the size bias of the absolute `--n-elec-tol`
    cutoff: larger molecules accumulate more numerical drift in the Fock ->
    density -> DF-coefficient pipeline, so they are disproportionately
    represented among `drift` rejects.
    """
    if not manifest_entries:
        return None
    path_to_natoms = {e["path"]: e["natoms"] for e in manifest_entries}
    sizes = [path_to_natoms[f["path"]] for f in failures if f.get("path") in path_to_natoms]
    if not sizes:
        return None
    arr = np.array(sizes)
    return {
        "n": int(arr.size),
        "natoms_median": float(np.median(arr)),
        "natoms_mean": float(arr.mean()),
        "natoms_max": int(arr.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--auxbasis", default="augccpvqzjkfit")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--val-prefix", default=None)
    parser.add_argument("--max-atoms", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-mo", action="store_true")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest the shards were built from; enables the .done-count completeness "
        "check and per-cause rejected-size stats.",
    )
    parser.add_argument("--shard-size", type=int, default=64, help="Must match csh_shard_to_df.py's --shard-size.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Package a mid-flight snapshot even if the .done count looks short or "
        "retryable KeyError failures remain.",
    )
    args = parser.parse_args()

    manifest_entries = None
    if args.manifest:
        manifest_entries = json.loads(Path(args.manifest).read_text())["entries"]

    records, failures = load_shards(Path(args.shards))
    print(f"loaded {len(records)} records, {len(failures)} failures from {args.shards}")

    completeness_warnings = check_completeness(
        Path(args.shards), manifest_entries, args.shard_size, failures, args.allow_incomplete
    )
    for w in completeness_warnings:
        print(f"WARNING: {w}")

    failure_causes = Counter(classify_failure(f.get("error", "")) for f in failures)
    print(
        f"failure causes: keyerror={failure_causes['keyerror']} "
        f"drift={failure_causes['drift']} other={failure_causes['other']}"
    )

    if args.max_atoms:
        records = [r for r in records if len(r["atom_numbers"]) <= args.max_atoms]
        print(f"after natoms<={args.max_atoms}: {len(records)}")
    if not records:
        raise SystemExit("no records")

    natoms = np.array([len(r["atom_numbers"]) for r in records])
    print(f"natoms med/mean/max: {np.median(natoms):.0f} {natoms.mean():.0f} {natoms.max()}")
    drift = np.array([abs(r["n_elec_err"]) for r in records])
    print(f"electron-count drift: max {drift.max():.2e}, mean {drift.mean():.2e}")

    drift_size_stats = None
    if manifest_entries is not None:
        drift_failures = [f for f in failures if classify_failure(f.get("error", "")) == "drift"]
        drift_size_stats = failure_size_stats(drift_failures, manifest_entries)
        if drift_size_stats:
            print(
                f"n_elec-tol rejects (drift) natoms median/mean/max: "
                f"{drift_size_stats['natoms_median']:.0f} {drift_size_stats['natoms_mean']:.0f} "
                f"{drift_size_stats['natoms_max']} (n={drift_size_stats['n']}); "
                f"absolute tolerance so larger structures are disproportionately rejected"
            )

    val_records = []
    if args.val_fraction > 0:
        # Stratify by size so the validation set spans the same range of system
        # sizes as training rather than concentrating in one tier.
        order = np.argsort(natoms)
        stride = max(int(round(1.0 / args.val_fraction)), 2)
        val_idx = set(order[::stride].tolist())
        val_records = [records[i] for i in sorted(val_idx)]
        records = [r for i, r in enumerate(records) if i not in val_idx]
        print(f"split: {len(records)} train / {len(val_records)} val")

    for tag, recs, prefix in (
        ("train", records, args.out_prefix),
        ("val", val_records, args.val_prefix or f"{args.out_prefix}_val"),
    ):
        if not recs:
            continue
        geo, dens = build_pair(recs, args.auxbasis, args.with_mo)
        geo_path = Path(f"{prefix}_npy.npy")
        dens_path = Path(f"{prefix}_pyscf_{args.auxbasis}_wb97mv.npy")
        geo_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(geo_path, geo, allow_pickle=True)
        np.save(dens_path, dens, allow_pickle=True)
        print(f"{tag}: {len(recs)} -> {geo_path} , {dens_path}")

    summary = {
        "n_records": len(records),
        "n_val": len(val_records),
        "n_failures": len(failures),
        "auxbasis": args.auxbasis,
        "max_electron_drift": float(drift.max()),
        "failure_causes": {
            "n_keyerror": failure_causes["keyerror"],
            "n_drift": failure_causes["drift"],
            "n_other": failure_causes["other"],
        },
        "completeness_warnings": completeness_warnings,
    }
    if drift_size_stats:
        summary["drift_rejected_size_stats"] = drift_size_stats
    Path(f"{args.out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
