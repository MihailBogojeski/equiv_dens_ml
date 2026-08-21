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


def _normalised(hist):
    total = sum(hist["counts"])
    return [c / total for c in hist["counts"]] if total else [0.0] * hist["nbins"]


def _bin_centres(hist):
    width = (hist["hi"] - hist["lo"]) / hist["nbins"]
    return [hist["lo"] + (i + 0.5) * width for i in range(hist["nbins"])]


def peak_position(hist):
    """Distance at the tallest bin, which is the number a reader compares."""
    if not any(hist["counts"]):
        return None
    centres = _bin_centres(hist)
    return centres[max(range(hist["nbins"]), key=lambda i: hist["counts"][i])]


def compare_hists(ref, pred):
    """Agreement between two histograms sharing one binning.

    Overlap and Jensen-Shannon are both reported because they fail differently.
    Overlap is the shared area, so it is easy to state in a sentence but is
    insensitive to *where* the missing mass went. Jensen-Shannon is bounded and
    penalises mass that moved to a distant bin, which is exactly the failure
    mode that matters here: a model that keeps the right number of hydrogen
    bonds but puts them at the wrong O-O distance.
    """
    p, q = _normalised(ref), _normalised(pred)
    overlap = sum(min(a, b) for a, b in zip(p, q))
    js = 0.0
    for a, b in zip(p, q):
        m = 0.5 * (a + b)
        if a > 0:
            js += 0.5 * a * math.log(a / m, 2)
        if b > 0:
            js += 0.5 * b * math.log(b / m, 2)
    ref_peak, pred_peak = peak_position(ref), peak_position(pred)
    return {
        "overlap": overlap,
        "jensen_shannon_divergence_bits": js,
        "ref_peak": ref_peak,
        "pred_peak": pred_peak,
        "peak_shift": (pred_peak - ref_peak) if (ref_peak is not None and pred_peak is not None) else None,
    }


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
        ref_frames = read_xyz(args.ref)
        pred_frames = read_xyz(args.pred_xyz)
        if len(pred_frames) != len(ref_frames):
            # Not fatal -- a relaxation can drop frames that failed to converge
            # -- but the histograms are then over different ensembles, so it has
            # to appear in the output rather than only on someone's terminal.
            print(
                f"warning: {len(pred_frames)} predicted frames against "
                f"{len(ref_frames)} reference frames; histograms cover different sets"
            )
        pred_oo, pred_oh = collect(pred_frames)
        summary["pred"] = str(args.pred_xyz)
        summary["n_frames_ref"] = len(ref_frames)
        summary["n_frames_pred"] = len(pred_frames)
        # Same binning as the reference, so the two are directly comparable and
        # can be plotted on one axis. Emitting only the means, as before, threw
        # away the shape -- and the shape of the O-O distribution is what R1.1
        # actually asks about.
        summary["pred_OO"] = pair_hist(pred_oo, 2.2, 3.6, 28)
        summary["pred_O_H"] = pair_hist(pred_oh, 1.3, 2.5, 24)
        summary["pred_mean_OO"] = sum(pred_oo) / len(pred_oo) if pred_oo else None
        summary["pred_mean_O_H"] = sum(pred_oh) / len(pred_oh) if pred_oh else None
        summary["delta_mean_OO"] = (
            summary["pred_mean_OO"] - summary["ref_mean_OO"]
            if summary["pred_mean_OO"] is not None and summary["ref_mean_OO"] is not None
            else None
        )
        summary["delta_mean_O_H"] = (
            summary["pred_mean_O_H"] - summary["ref_mean_O_H"]
            if summary["pred_mean_O_H"] is not None and summary["ref_mean_O_H"] is not None
            else None
        )
        summary["agreement_OO"] = compare_hists(summary["ref_OO"], summary["pred_OO"])
        summary["agreement_O_H"] = compare_hists(summary["ref_O_H"], summary["pred_O_H"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    skip = {"ref_OO", "ref_O_H", "pred_OO", "pred_O_H"}
    print(json.dumps({k: v for k, v in summary.items() if k not in skip}, indent=2))


if __name__ == "__main__":
    main()
