#!/usr/bin/env python3
"""Hydrogen-bond geometry histograms for water-cluster size extrapolation (R1.1)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_xyz(path: Path):
    text = path.read_text().strip().splitlines()
    frames = []
    i = 0
    while i < len(text):
        n = int(text[i].split()[0])
        atoms = []
        for line in text[i + 2 : i + 2 + n]:
            p = line.split()
            atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
        frames.append(atoms)
        i += 2 + n
    return frames


def _dist(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))


def pair_hist(values, lo, hi, nbins):
    width = (hi - lo) / nbins
    counts = [0] * nbins
    for v in values:
        if lo <= v < hi:
            counts[int((v - lo) / width)] += 1
    return {"lo": lo, "hi": hi, "nbins": nbins, "counts": counts, "n": len(values)}


def collect(frames):
    oo = []
    oh = []
    for atoms in frames:
        oxy = [p[1:] for p in atoms if p[0] == "O"]
        hyd = [p[1:] for p in atoms if p[0] == "H"]
        for i, a in enumerate(oxy):
            for b in oxy[i + 1 :]:
                d = _dist(a, b)
                if d < 4.0:
                    oo.append(d)
            for h in hyd:
                d = _dist(a, h)
                if 1.3 < d < 2.5:
                    oh.append(d)
    return oo, oh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--pred-xyz", type=Path, default=None, help="Optional predicted XYZ")
    parser.add_argument("--out", type=Path, default=Path("results/revision/water_hbond_metrics.json"))
    args = parser.parse_args()

    ref_oo, ref_oh = collect(read_xyz(args.ref))
    summary = {
        "ref": str(args.ref),
        "ref_OO": pair_hist(ref_oo, 2.2, 3.6, 28),
        "ref_O_H": pair_hist(ref_oh, 1.3, 2.5, 24),
        "ref_mean_OO": sum(ref_oo) / len(ref_oo) if ref_oo else None,
        "ref_mean_O_H": sum(ref_oh) / len(ref_oh) if ref_oh else None,
    }
    if args.pred_xyz:
        pred_oo, pred_oh = collect(read_xyz(args.pred_xyz))
        summary["pred"] = str(args.pred_xyz)
        summary["pred_mean_OO"] = sum(pred_oo) / len(pred_oo) if pred_oo else None
        summary["pred_mean_O_H"] = sum(pred_oh) / len(pred_oh) if pred_oh else None
        summary["delta_mean_OO"] = (
            summary["pred_mean_OO"] - summary["ref_mean_OO"]
            if summary["pred_mean_OO"] is not None and summary["ref_mean_OO"] is not None
            else None
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if "hist" not in k and k not in ("ref_OO", "ref_O_H")}, indent=2))


if __name__ == "__main__":
    main()
