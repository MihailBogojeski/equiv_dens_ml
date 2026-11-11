#!/usr/bin/env python3
"""Merge per-element AO maps from several probes into one canonical table.

Each probe solves the stored-to-PySCF signed permutation against a full
precision ORCA reference for one molecule. The map is a property of the
element, so elements appearing in more than one probe are a free consistency
check: if two independently solved molecules disagree for a shared element, the
solve is wrong and the table must not be written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", nargs="+", help="probe JSON files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    table: dict[str, dict] = {}
    sources: dict[str, list[str]] = {}
    conflicts = []

    for path in args.probes:
        data = json.loads(Path(path).read_text())
        entry = data["entry"]
        if not data.get("consistent", False):
            print(f"WARNING: {path} reports inconsistent per-element maps")
        for z, rec in data["per_element"].items():
            if z in table:
                if table[z]["perm"] != rec["perm"] or table[z]["signs"] != rec["signs"]:
                    conflicts.append((z, sources[z][0], entry))
                else:
                    sources[z].append(entry)
            else:
                table[z] = {"perm": rec["perm"], "signs": rec["signs"]}
                sources[z] = [entry]

    print(f"elements resolved: {sorted(int(z) for z in table)}")
    for z in sorted(table, key=int):
        n_flip = sum(1 for s in table[z]["signs"] if s < 0)
        print(
            f"  Z={int(z):>2}  nao={len(table[z]['perm']):>3}  flips={n_flip:>2}  "
            f"from {len(sources[z])} molecule(s)"
        )

    if conflicts:
        print("\nCONFLICTS (same element, different map):")
        for z, a, b in conflicts:
            print(f"  Z={z}: {a} vs {b}")
        raise SystemExit("refusing to write a table with conflicting elements")

    cross = [z for z in table if len(sources[z]) > 1]
    print(f"\ncross-validated elements (seen in >1 molecule): {sorted(int(z) for z in cross)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "description": (
                    "Signed permutation mapping stored OMol_CSH AO order to PySCF "
                    "def2-TZVPD order. perm[i] is the stored slot holding the AO "
                    "PySCF places at i; signs[i] is its phase."
                ),
                "basis": "def2-tzvpd",
                "per_element": table,
                "sources": sources,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
