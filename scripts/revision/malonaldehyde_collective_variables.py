#!/usr/bin/env python3
"""Collective variables for the malonaldehyde proton-transfer split.

water_collective_variables.py was run on these frames first and the numbers it
produced are meaningless: it counts every oxygen as a water, so a malonaldehyde
molecule comes back as a two-water cluster with an undefined tetrahedral order.

The reason malonaldehyde is in this study at all is that it supplies a *physical*
axis along which to report error, rather than an abstract descriptor distance:

    delta = r(O_donor-H) - r(H-O_acceptor)

is negative in one enol basin, zero at the transition state and positive in the
other. Training covers one basin only, so delta doubles as the distance from the
training distribution, and the headline error-versus-distance curve gets an
x-axis a chemist can read.

Two supporting variables come along because they are what the density actually
does: r(O-O) sets how far the proton has to go, and the C-O bond-length
alternation tracks the pi-system reorganising from enol to keto, which is the
density change the model has to predict and has never been trained on.

Usage:
  python scripts/revision/malonaldehyde_collective_variables.py \\
    --xyz datasets/revision/malonaldehyde/ood_proton_transfer.xyz \\
    --manifest datasets/revision/malonaldehyde/manifest.json \\
    --label malon_ood_pt --out results/revision/cv/malon_ood_pt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from water_collective_variables import read_xyz_frames  # noqa: E402

CV_NAMES = ("delta_pt", "abs_delta_pt", "r_oo", "co_alternation", "r_donor_h", "r_acceptor_h")


def carbon_bonded_to(z: np.ndarray, xyz: np.ndarray, oxygen: int) -> int:
    """Index of the carbon nearest `oxygen`, i.e. the one it is bonded to."""
    carbons = np.flatnonzero(z == 6)
    if carbons.size == 0:
        return -1
    d = np.linalg.norm(xyz[carbons] - xyz[oxygen], axis=1)
    return int(carbons[int(np.argmin(d))])


def frame_cvs(z: np.ndarray, xyz: np.ndarray, donor: int, hydrogen: int, acceptor: int) -> dict:
    r_dh = float(np.linalg.norm(xyz[hydrogen] - xyz[donor]))
    r_ah = float(np.linalg.norm(xyz[hydrogen] - xyz[acceptor]))
    c_donor = carbon_bonded_to(z, xyz, donor)
    c_acceptor = carbon_bonded_to(z, xyz, acceptor)
    alternation = float("nan")
    if c_donor >= 0 and c_acceptor >= 0:
        alternation = float(
            np.linalg.norm(xyz[c_donor] - xyz[donor]) - np.linalg.norm(xyz[c_acceptor] - xyz[acceptor])
        )
    delta = r_dh - r_ah
    return {
        "delta_pt": delta,
        # The training basin sits at negative delta, so the *unsigned* value is
        # what grows monotonically as a frame leaves it; the signed one is kept
        # because it is the coordinate the scan was run along.
        "abs_delta_pt": abs(delta),
        "r_oo": float(np.linalg.norm(xyz[acceptor] - xyz[donor])),
        "co_alternation": alternation,
        "r_donor_h": r_dh,
        "r_acceptor_h": r_ah,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--donor", type=int, default=None)
    parser.add_argument("--hydrogen", type=int, default=None)
    parser.add_argument("--acceptor", type=int, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    donor, hydrogen, acceptor = args.donor, args.hydrogen, args.acceptor
    if args.manifest is not None and None in (donor, hydrogen, acceptor):
        idx = json.loads(args.manifest.read_text())["indices"]
        donor = idx["donor"] if donor is None else donor
        hydrogen = idx["hydrogen"] if hydrogen is None else hydrogen
        acceptor = idx["acceptor"] if acceptor is None else acceptor
    if None in (donor, hydrogen, acceptor):
        raise SystemExit("give --manifest or all of --donor/--hydrogen/--acceptor")

    frames = read_xyz_frames(args.xyz)
    rows = []
    for i, (z, xyz, comment) in enumerate(frames):
        z = np.asarray(z)
        if z[donor] != 8 or z[acceptor] != 8 or z[hydrogen] != 1:
            raise SystemExit(
                f"frame {i}: indices {donor}/{hydrogen}/{acceptor} are not O/H/O "
                f"but {z[donor]}/{z[hydrogen]}/{z[acceptor]}; atom order is not what "
                "the manifest recorded"
            )
        row = frame_cvs(z, np.asarray(xyz, dtype=float), donor, hydrogen, acceptor)
        row["index"] = i
        row["comment"] = comment
        rows.append(row)

    summary = {}
    for name in CV_NAMES:
        values = np.array([r[name] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            summary[name] = {
                "mean": float(finite.mean()),
                "std": float(finite.std()),
                "min": float(finite.min()),
                "max": float(finite.max()),
            }

    payload = {
        "label": args.label or args.xyz.stem,
        "xyz": str(args.xyz),
        "indices": {"donor": donor, "hydrogen": hydrogen, "acceptor": acceptor},
        "n_frames": len(rows),
        "summary": summary,
        "frames": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"{payload['label']}: {len(rows)} frames -> {args.out}")
    for name, entry in summary.items():
        print(f"  {name:16s} mean {entry['mean']:+.3f}  range [{entry['min']:+.3f}, {entry['max']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
