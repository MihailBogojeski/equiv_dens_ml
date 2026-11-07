#!/usr/bin/env python3
"""Training-similarity coordinates for the frames scored on the energy-training review page.

The same coordinates as the density-error analysis, so energy and density curves share an axis:

  * cv_dist (water): nearest-neighbour distance to the closest training frame in standardised
    collective-variable space (ood_overlap_report.DEFAULT_CVS, restricted to the columns that are
    finite for every training frame), in units of the training set's own extent, the median
    pairwise distance within it ("training widths"). Exactly error_vs_distance.py's x-axis, with
    frames joined on source_index.
  * rmsd: Hungarian RMSD (rotation- and permutation-minimised, element-aware; train_test_overlap.py)
    to the nearest training structure of the same composition. Defined for malonaldehyde and for the
    water tiers whose cluster size occurs in training (n = 2-6); NaN for larger clusters.
  * delta (malonaldehyde): the signed proton-transfer coordinate delta_pt of the project's CV files,
    error_vs_distance.py's physical axis.

A frame that is itself a training frame is measured against its nearest *other* training frame.
Writes similarity.csv and similarity.json next to per_frame.csv.

    .venv/bin/python -B experiments/jacs_revision/analysis/energy_similarity.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(Path(__file__).resolve().parent)]
OUT = REPO / "results/revision/energy_training_dashboard"
CV = REPO / "results/revision/cv"
WATER_CV = {"val": "water_val", "id_test": "water_id_test", "ood_size": "water_ood_size", "ood_order": "ood_order", "ood_density": "ood_density"}
MALON_CV = {"train": "malon_train_pt", "val": "malon_val_pt", "ood_proton_transfer": "malon_ood_proton_transfer_pt"}
LIMIT = {"water": 40, "malon": 125}
THEORIES = ("pbe_d4_avdz", "wb97mv_def2tzvpd")


def labels(name, split, theory):
    return np.load(REPO / f"datasets/revision/{name}_labels/{name}_{split}_{theory}_base.npy", allow_pickle=True).item()


def structures(data, idx, elements):
    from train_test_overlap import Structure

    out = []
    for i in idx:
        z = np.asarray(data["atom_numbers"][i]).astype(int)
        keep = z > 0
        out.append(Structure(z[keep], np.asarray(data["positions"][i], float)[keep], elements))
    return out


def rmsd_to_train(queries, q_comp, refs, r_comp, same_objects=False):
    """Nearest-training Hungarian RMSD per query, composition by composition; NaN without a match."""
    from train_test_overlap import nearest_rmsd

    out = np.full(len(queries), np.nan)
    for comp in sorted(set(q_comp)):
        qi = [i for i, c in enumerate(q_comp) if c == comp]
        ri = [j for j, c in enumerate(r_comp) if c == comp]
        if not ri:
            continue
        d, _ = nearest_rmsd([queries[i] for i in qi], [refs[j] for j in ri], exclude_self=same_objects)
        out[qi] = d
    return out


def water(theory):
    from experiments.jacs_revision.analysis.error_vs_distance import train_extent_scale
    from experiments.jacs_revision.analysis.ood_overlap_report import DEFAULT_CVS, load_cvs, nearest_neighbour_distances, train_internal_scale

    names = tuple(DEFAULT_CVS)
    train_cv, _ = load_cvs(CV / "water_train.json", names)
    dense = [i for i in range(len(names)) if np.isfinite(train_cv[:, i]).all()]
    train_dense = train_cv[:, dense]
    unit, spacing = train_extent_scale(train_dense), train_internal_scale(train_dense)
    train = labels("water", "train", theory)
    elements = [1, 8]
    refs = structures(train, range(len(train["energy"])), elements)
    r_comp = [len(s.z) for s in refs]
    base_rmsd = rmsd_to_train(refs[::10], r_comp[::10], refs, r_comp, same_objects=True)
    rows = []
    for split, cvname in WATER_CV.items():
        data = labels("water", split, theory)
        n = min(LIMIT["water"], len(data["energy"]))
        payload = json.loads((CV / f"{cvname}.json").read_text())
        by_index = {int(f["index"]): f for f in payload["frames"]}
        src = np.asarray(data["source_index"]).astype(int)[:n]
        cvm = np.array([[float(by_index[int(s)].get(names[c], np.nan)) if int(s) in by_index else np.nan for c in dense] for s in src])
        ok = np.isfinite(cvm).all(1)
        dist = np.full(n, np.nan)
        if ok.any():
            dist[ok] = nearest_neighbour_distances(train_dense, cvm[ok]) / unit
        q = structures(data, range(n), elements)
        rm = rmsd_to_train(q, [len(s.z) for s in q], refs, r_comp)
        for i in range(n):
            rows.append({"split": split, "index": i, "cv_dist": float(dist[i]), "rmsd": float(rm[i]), "delta": float("nan")})
        print(f"water {theory:18s} {split:12s} n={n:3d} CV distance median {np.nanmedian(dist):6.3f} training widths "
              f"({np.nanmedian(dist) * unit / spacing:6.1f} spacings); Hungarian RMSD median "
              f"{np.nanmedian(rm) if np.isfinite(rm).any() else float('nan'):.3f} A ({int(np.isfinite(rm).sum())} with a same-size training cluster)", flush=True)
    info = {"cv_dist": {"definition": "nearest-training distance in standardised CV space / median pairwise distance within the training set",
                        "cvs_used": [names[i] for i in dense], "train_extent": unit, "train_spacing_in_widths": spacing / unit, "n_train": int(len(train_dense))},
            "rmsd": {"definition": "Hungarian RMSD to the nearest same-size training cluster (A)",
                     "train_to_train_median": float(np.nanmedian(base_rmsd)), "train_to_train_p90": float(np.nanpercentile(base_rmsd, 90))}}
    return rows, info


def malon(theory):
    train = labels("malonaldehyde", "train", theory)
    elements = [1, 6, 8]
    refs = structures(train, range(len(train["energy"])), elements)
    r_comp = [0] * len(refs)
    nb = min(LIMIT["malon"], len(refs))  # the plotted training sample doubles as the train-to-train baseline
    base = rmsd_to_train(refs[:nb], r_comp[:nb], refs, r_comp, same_objects=True)
    tr_delta = np.array([f["delta_pt"] for f in json.loads((CV / "malon_train_pt.json").read_text())["frames"]], float)
    rows = []
    for split, cvname in MALON_CV.items():
        data = train if split == "train" else labels("malonaldehyde", split, theory)
        n = min(LIMIT["malon"], len(data["energy"]))
        by_index = {int(f["index"]): f for f in json.loads((CV / f"{cvname}.json").read_text())["frames"]}
        src = np.asarray(data["source_index"]).astype(int)[:n]
        delta = np.array([float(by_index[int(s)]["delta_pt"]) if int(s) in by_index else np.nan for s in src])
        if split == "train":
            rm = base[:n]
        else:
            q = structures(data, range(n), elements)
            rm = rmsd_to_train(q, [0] * n, refs, r_comp)
        for i in range(n):
            rows.append({"split": split, "index": i, "cv_dist": float("nan"), "rmsd": float(rm[i]), "delta": float(delta[i])})
        print(f"malon {theory:18s} {split:20s} n={n:3d} Hungarian RMSD median {np.nanmedian(rm):.3f} A; delta range "
              f"{np.nanmin(delta):+.2f} to {np.nanmax(delta):+.2f} A", flush=True)
    info = {"rmsd": {"definition": "Hungarian RMSD to the nearest training structure (A)", "train_to_train_median": float(np.nanmedian(base)),
                     "train_to_train_p90": float(np.nanpercentile(base, 90)), "n_train": len(refs)},
            "delta": {"definition": "signed proton-transfer coordinate delta_pt (A)", "train_min": float(tr_delta.min()), "train_max": float(tr_delta.max()),
                      "train_abs_p99": float(np.percentile(np.abs(tr_delta), 99))}}
    return rows, info


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--water-limit", type=int, default=LIMIT["water"], help="frames per water split (default: the dashboard's 40)")
    ap.add_argument("--out-root", type=Path, help="write <out-root>/<mol>/similarity.{csv,json} once (geometries are the same "
                                                  "at both levels of theory) instead of the dashboard's per-theory folders")
    opt = ap.parse_args()
    LIMIT["water"] = opt.water_limit
    for mol, fn in (("malon", malon), ("water", water)):
        for theory in (THEORIES[:1] if opt.out_root else THEORIES):
            rows, info = fn(theory)
            out = opt.out_root / mol if opt.out_root else OUT / f"score_{mol}_{theory}"
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "similarity.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["split", "index", "cv_dist", "rmsd", "delta"])
                w.writeheader()
                w.writerows(rows)
            (out / "similarity.json").write_text(json.dumps(info, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
