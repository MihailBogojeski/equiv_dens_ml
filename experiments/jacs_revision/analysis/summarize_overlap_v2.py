#!/usr/bin/env python3
"""Table and figure for the R1.5 train/test overlap (train_test_overlap.py outputs).

Writes results/revision/overlap_v2/overlap_summary.{json,md} and
results/revision/figures/response/overlap_v2.{pdf,png}: for each system the
nearest-training RMSD of test frames next to that of training frames, and the
same for SOAP distances.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DIR = REPO / "results" / "revision" / "overlap_v2"
FIG = REPO / "results" / "revision" / "figures" / "response"
ORDER = [
    ("ethanol_md17_seed42", "Ethanol (MD17 split)"),
    ("ethanethiol", "Ethanethiol"),
    ("resorcinol", "Resorcinol"),
    ("thiophene_4mer", "Thiophene 4-mer"),
    ("thiophene_6mer", "Thiophene 6-mer"),
    ("thiophene_12mer", "Thiophene 12-mer"),
    ("water_id", "Water, ID test"),
    ("water_ood_size", "Water, n = 8-12"),
]
OLD = {  # values in the draft letter (element-blind matching before alignment)
    "resorcinol": 0.093, "ethanethiol": 0.600, "thiophene_4mer": 1.15, "thiophene_6mer": 0.53,
}


def main() -> int:
    rows = []
    for stem, label in ORDER:
        rep = json.loads((DIR / f"{stem}.json").read_text())
        arr = np.load(DIR / f"{stem}.npz")
        row = {"system": label, "key": stem, "n_train": rep["n_train"], "n_test": rep["n_test"],
               "soap_test_median": rep["soap_test_to_train"]["median"],
               "soap_train_median": rep["soap_train_to_train"]["median"],
               "soap_ratio": rep["soap_median_ratio"], "note": rep.get("note", "")}
        if rep.get("rmsd_test_to_train"):
            row.update(rmsd_test_median=rep["rmsd_test_to_train"]["median"],
                       rmsd_test_p90=rep["rmsd_test_to_train"]["p90"],
                       rmsd_test_frac_below_0p2=rep["rmsd_test_to_train"]["frac_below_0.2"],
                       rmsd_train_median=rep["rmsd_train_to_train"]["median"],
                       rmsd_ratio=rep["rmsd_median_ratio"], ks_p=rep["rmsd_ks"]["pvalue"])
        if stem.startswith("ethanol_md17"):
            seeds = [json.loads((DIR / f"ethanol_md17_seed{s}.json").read_text()) for s in (42, 1, 2)]
            row["rmsd_ratio_seeds"] = [s["rmsd_median_ratio"] for s in seeds]
        row["old_letter_median"] = OLD.get(stem)
        row["_arrays"] = {k: arr[k] for k in arr.files}
        rows.append(row)

    lines = [
        "| System | n train / test | NN RMSD test→train, median (Å) | train→train (Å) | ratio | test < 0.2 Å | SOAP ratio | draft letter (Å) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "rmsd_test_median" in r:
            rm = (f"{r['rmsd_test_median']:.3f} | {r['rmsd_train_median']:.3f} | {r['rmsd_ratio']:.2f} | "
                  f"{100 * r['rmsd_test_frac_below_0p2']:.0f}%")
        else:
            rm = "— | — | — | —"
        old = f"{r['old_letter_median']:.3f}" if r["old_letter_median"] is not None else "—"
        lines.append(f"| {r['system']} | {r['n_train']} / {r['n_test']} | {rm} | {r['soap_ratio']:.2f} | {old} |")
    table = "\n".join(lines)
    (DIR / "overlap_summary.md").write_text(
        "Nearest-training distances (train_test_overlap.py). Ratio = median(test→train) / "
        "median(train→train); ~1 means the test set sits inside the training distribution. "
        "RMSD: element-aware, permutation- and rotation-minimised, same composition only. "
        "SOAP: structure-averaged, L2-normalised, Euclidean.\n\n" + table + "\n"
    )
    (DIR / "overlap_summary.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "_arrays"} for r in rows], indent=1) + "\n")
    print(table)

    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), constrained_layout=True)
    names = [r["system"] for r in rows]
    y = np.arange(len(rows))
    for ax, key_t, key_b, xlabel in (
        (axes[0], "rmsd_test", "rmsd_base", "nearest-training RMSD (Å)"),
        (axes[1], "soap_test", "soap_base", "nearest-training SOAP distance"),
    ):
        for i, r in enumerate(rows):
            arr = r["_arrays"]
            if key_t not in arr:
                continue
            for vals, dy, color, lab in ((arr[key_b], -0.17, "#8a8a8a", "train → train"),
                                         (arr[key_t], 0.17, "#1f5fa8", "test → train")):
                q = np.percentile(vals, [10, 25, 50, 75, 90])
                ax.plot([q[0], q[4]], [i + dy] * 2, color=color, lw=1)
                ax.plot([q[1], q[3]], [i + dy] * 2, color=color, lw=5, solid_capstyle="butt",
                        label=lab if i == 0 else None)
                ax.plot(q[2], i + dy, "o", color="white", mec=color, ms=4)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ticks = [0.05, 0.1, 0.2, 0.5, 1.0] if key_t == "rmsd_test" else [0.003, 0.01, 0.03, 0.1, 0.3]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].set_title("Test frames against the training set (bars: 25-75%, whiskers: 10-90%)", fontsize=9)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"overlap_v2.{ext}", dpi=200)
    print(f"wrote {DIR / 'overlap_summary.md'} and {FIG / 'overlap_v2.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
