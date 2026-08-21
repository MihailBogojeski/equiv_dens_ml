#!/usr/bin/env python3
"""Compare DenSNet accuracy across levels of theory, tier by tier (R3.1, R3.3).

Reviewer 3 asked whether the architecture is limited to GGA densities. The
campaign answers that by labelling *the same geometries* at PBE-D4/aug-cc-pVDZ
and at wB97M-V/def2-TZVPD and training with identical settings, so a difference
between the two columns is a property of the reference data rather than of the
model.

Two things that sound like details and are not:

Frames are paired on ``source_index`` before anything is subtracted. The tiers
are shard-assembled per theory and a shard that failed at one level and not the
other leaves the two evaluations covering different frames; comparing bare
means would then fold that difference into the theory comparison. The paired
mean difference and its standard error are computed over the intersection only,
and the report says how many frames that was.

The spread across ensemble members (``--theory name:dirA,dirB,dirC``) is
reported as a standard deviation over runs that differ only in ``--init_seed``.
That is the R3.3 uncertainty. It is deliberately not pooled with the per-frame
scatter, which is a different quantity: one says how much the fit moves when
retrained, the other how much the error varies across structures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TIERS = [
    ("water_id_test", "in-distribution test, n=2-6"),
    ("water_ood_size", "size OOD, n=8,10,12"),
    ("water_ood_order", "orientational OOD, ice cutouts"),
    ("water_ood_density", "density OOD, droplets"),
    ("malonaldehyde_ood_proton_transfer", "proton-transfer scan"),
]


def load_eval(path: Path):
    """Return {source_index or index: afe} plus the stored summary."""
    if not path.is_file():
        return None, None
    blob = json.loads(path.read_text())
    records = blob.get("records", [])
    per_frame = {}
    for rec in records:
        key = rec.get("source_index", rec.get("index"))
        if key is not None:
            per_frame[int(key)] = float(rec["afe"])
    return per_frame, blob.get("summary", {})


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def collect(theory_dirs, tier):
    """Per-member frame maps for one tier, skipping members not yet evaluated."""
    members = []
    for d in theory_dirs:
        per_frame, summary = load_eval(Path(d) / f"{tier}.json")
        if per_frame:
            members.append((per_frame, summary))
    return members


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theory", action="append", required=True, metavar="NAME:DIR[,DIR...]",
                    help="label and one or more eval directories; several directories are "
                         "treated as ensemble members differing only in --init_seed")
    ap.add_argument("--out", type=Path, default=Path("results/revision/pbe_vs_hybrid.json"))
    ap.add_argument("--markdown", type=Path, default=None,
                    help="also write a table ready to paste into the response letter")
    cli = ap.parse_args()

    theories = []
    for spec in cli.theory:
        name, _, dirs = spec.partition(":")
        if not dirs:
            ap.error(f"--theory {spec!r} needs NAME:DIR")
        theories.append((name, [d for d in dirs.split(",") if d]))

    report = {
        "theories": {n: d for n, d in theories},
        "afe_definition": "integral |rho_pred - rho_ref| dV / N_elec, per structure",
        "tiers": {},
    }

    for tier, description in TIERS:
        entry = {"description": description, "per_theory": {}}
        frames_by_theory = {}
        for name, dirs in theories:
            members = collect(dirs, tier)
            if not members:
                continue
            # Ensemble member means, then mean and spread over members.
            member_means = [mean(list(pf.values())) for pf, _ in members]
            entry["per_theory"][name] = {
                "n_members": len(members),
                "n_structures": len(members[0][0]),
                "afe_mean": mean(member_means),
                "afe_std_over_seeds": stdev(member_means),
                "afe_median": members[0][1].get("afe_median"),
                "afe_p90": members[0][1].get("afe_p90"),
            }
            # For pairing use the first member of each theory: the paired
            # difference is about the reference data, not about seed noise.
            frames_by_theory[name] = members[0][0]

        if len(frames_by_theory) == 2:
            (na, fa), (nb, fb) = frames_by_theory.items()
            shared = sorted(set(fa) & set(fb))
            if shared:
                diffs = [fb[i] - fa[i] for i in shared]
                m = mean(diffs)
                sd = stdev(diffs)
                entry["paired"] = {
                    "baseline": na,
                    "comparison": nb,
                    "n_paired": len(shared),
                    "n_only_in_baseline": len(set(fa) - set(fb)),
                    "n_only_in_comparison": len(set(fb) - set(fa)),
                    "mean_afe_difference": m,
                    "stderr_of_difference": sd / math.sqrt(len(shared)) if shared else float("nan"),
                    "ratio_comparison_over_baseline": (
                        mean([fb[i] for i in shared]) / mean([fa[i] for i in shared])
                        if mean([fa[i] for i in shared]) else float("nan")
                    ),
                }
        if entry["per_theory"]:
            report["tiers"][tier] = entry

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {cli.out}")

    names = [n for n, _ in theories]
    header = f"{'tier':<36}" + "".join(f"{n:>26}" for n in names) + f"{'paired diff':>18}"
    print()
    print(header)
    print("-" * len(header))
    lines = []
    for tier, entry in report["tiers"].items():
        row = f"{tier:<36}"
        for n in names:
            pt = entry["per_theory"].get(n)
            if not pt:
                row += f"{'-':>26}"
            elif pt["n_members"] > 1:
                row += f"{pt['afe_mean']:>18.5f} +/- {pt['afe_std_over_seeds']:.5f}"
            else:
                row += f"{pt['afe_mean']:>26.5f}"
        paired = entry.get("paired")
        row += f"{paired['mean_afe_difference']:>18.5f}" if paired else f"{'-':>18}"
        print(row)
        lines.append((tier, entry))

    if cli.markdown:
        md = ["| tier | " + " | ".join(names) + " | paired difference | n paired |",
              "| --- | " + " | ".join("---" for _ in names) + " | --- | --- |"]
        for tier, entry in lines:
            cells = []
            for n in names:
                pt = entry["per_theory"].get(n)
                if not pt:
                    cells.append("-")
                elif pt["n_members"] > 1:
                    cells.append(f"{pt['afe_mean']:.5f} ± {pt['afe_std_over_seeds']:.5f}")
                else:
                    cells.append(f"{pt['afe_mean']:.5f}")
            paired = entry.get("paired")
            cells.append(f"{paired['mean_afe_difference']:+.5f}" if paired else "-")
            cells.append(str(paired["n_paired"]) if paired else "-")
            md.append(f"| {tier} | " + " | ".join(cells) + " |")
        cli.markdown.parent.mkdir(parents=True, exist_ok=True)
        cli.markdown.write_text("\n".join(md) + "\n")
        print(f"\nwrote {cli.markdown}")

    if not report["tiers"]:
        print("\nno evaluations found yet; run scripts/revision/run_ood_analysis.sh first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
