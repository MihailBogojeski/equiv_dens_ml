#!/usr/bin/env python3
"""Generate figures bundled with the JACS response letter."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/plot_response_letter_figures.py'


import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, NullFormatter

_HERE = _LEGACY_FILE.resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from experiments.jacs_revision.analysis.mpl_classic_tex import apply as apply_classic_tex, response_figure_path, save_response_figure
from experiments.jacs_revision.analysis.render_dataset_structures import (
    MALON_TRAIN_DELTA_MAX,
    malon_region,
    render_gallery,
)
from experiments.jacs_revision.analysis.learning_curve_records import series_for_figure
from experiments.jacs_revision.analysis.error_vs_distance import EPSILON_RHO, EPSILON_RHO_PERCENT, style_epsilon_rho_ticks
from experiments.jacs_revision.analysis.plot_ethanethiol_ir_compare import compare_spectra
from experiments.jacs_revision.analysis.score_extreme_modes import make_figure as make_extreme_mode_error_figure
from experiments.jacs_revision.analysis.timing_policy import gpu_speedup_is_clear
from experiments.jacs_revision.analysis.train_log_curve import parse_density_curve, parse_density_curve_result

apply_classic_tex()

_REPO = _LEGACY_FILE.resolve().parents[3]
OUT = _REPO / "results" / "revision" / "figures" / "response"
OUT.mkdir(parents=True, exist_ok=True)

HARTREE_TO_KCAL = 627.509474

WATER_CV = {
    "Train": "cv/water_train.json",
    "ID test": "cv/water_id_test.json",
    "Size OOD": "cv/water_ood_size.json",
    "Ice": "cv/ood_order.json",
    "Droplets": "cv/ood_density.json",
    "Rackers/Rakshit": "cv/rackers.json",
}

# n_water is discrete (2–6, 8/10/12, 16/20/24) and looks empty as a
# density histogram. q3 is pooled per molecule with ≥3 O neighbours.
CV_HIST_PANELS = (
    ("q3_per_molecule", r"$q_3$"),
    ("n_hbond", r"$n_{\mathrm{HB}}$"),
    ("local_density", r"$\rho_{\mathrm{loc}}$ / g cm$^{-3}$"),
    ("mean_nn_oo", r"$\bar r_{\mathrm{OO}}$ / $\mathrm{\AA}$"),
)
CV_HIST_BINS = 40

# Default is the finished 3k assemble. Point WATER_LABEL_ROOT at
# datasets/revision/water_labels_md40x once those npy exist.
_WATER_LABEL_ROOT = os.environ.get("WATER_LABEL_ROOT", "datasets/revision/water_labels")

WATER_ENERGY = {
    "Train PBE": f"{_WATER_LABEL_ROOT}/water_train_pbe_d4_avdz_base.npy",
    "ID PBE": f"{_WATER_LABEL_ROOT}/water_id_test_pbe_d4_avdz_base.npy",
    "Size OOD PBE": f"{_WATER_LABEL_ROOT}/water_ood_size_pbe_d4_avdz_base.npy",
    "Train $\\omega$B97M-V": f"{_WATER_LABEL_ROOT}/water_train_wb97mv_def2tzvpd_base.npy",
    "ID $\\omega$B97M-V": f"{_WATER_LABEL_ROOT}/water_id_test_wb97mv_def2tzvpd_base.npy",
    "Size OOD $\\omega$B97M-V": f"{_WATER_LABEL_ROOT}/water_ood_size_wb97mv_def2tzvpd_base.npy",
}

# Absolute DFT energies sit near -47900 kcal/mol per water. Shift by this
# round constant so tick labels stay small integers and the ~45 kcal/mol
# functional offset remains visible on a shared axis.
HIST_FILL_ALPHA = 0.28
ENERGY_HIST_BINS = 24
ENERGY_OFFSET_KCAL = 47900.0
ENERGY_XLABEL = r"$E + 47{,}900$ / kcal mol$^{-1}$"
HIST_YLABEL = "Probability density"
ENERGY_HIST_COLORS = {
    "Train": "#4c72b0",
    "ID test": "#55a868",
    "Size OOD": "#c44e52",
}
ENERGY_HIST_PANELS = (
    (
        "PBE-D4",
        (
            ("Train", "Train PBE"),
            ("ID test", "ID PBE"),
            ("Size OOD", "Size OOD PBE"),
        ),
    ),
    (
        r"$\omega$B97M-V",
        (
            ("Train", "Train $\\omega$B97M-V"),
            ("ID test", "ID $\\omega$B97M-V"),
            ("Size OOD", "Size OOD $\\omega$B97M-V"),
        ),
    ),
)

LEARNING_RUNS = [
    ("water_pbe_d4_avdz", "Water PBE"),
    ("water_wb97mv_def2tzvpd", "Water $\\omega$B97M-V"),
    ("water_wb97mv_def2tzvpd_seed7", "Water hybrid s7"),
    ("water_wb97mv_def2tzvpd_seed123", "Water hybrid s123"),
    ("water_wb97mv_def2tzvpd_seed20260821", "Water hybrid s20260821"),
    ("water_direct_pbe_d4_avdz", "Direct PBE"),
    ("water_direct_wb97mv_def2tzvpd", "Direct $\\omega$B97M-V"),
    ("malonaldehyde_pbe_d4_avdz", "Malon PBE"),
    ("malonaldehyde_wb97mv_def2tzvpd", "Malon $\\omega$B97M-V"),
    ("water_pbe_orca_cutoff_4", r"Cutoff $4\,\mathrm{\AA}$"),
    ("water_pbe_orca_cutoff_5", r"Cutoff $5\,\mathrm{\AA}$"),
    ("water_pbe_orca_cutoff_6", r"Cutoff $6\,\mathrm{\AA}$"),
    ("water_pbe_orca_cutoff_8", r"Cutoff $8\,\mathrm{\AA}$"),
]

# Wave 1 data-efficiency / keep-ckpt runs. N=1250 SAD/direct use the
# campaign logs above; the _curve stems are the keep-checkpoint copies.
NTRAIN_LOG_RUNS = [
    ("water_n50_pbe_d4_avdz", 50, "sad"),
    ("water_direct_n50_pbe_d4_avdz", 50, "direct"),
    ("water_n125_pbe_d4_avdz", 125, "sad"),
    ("water_direct_n125_pbe_d4_avdz", 125, "direct"),
    ("water_n250_pbe_d4_avdz", 250, "sad"),
    ("water_direct_n250_pbe_d4_avdz", 250, "direct"),
    ("water_n500_pbe_d4_avdz", 500, "sad"),
    ("water_direct_n500_pbe_d4_avdz", 500, "direct"),
    ("water_pbe_d4_avdz", 1250, "sad"),
    ("water_direct_pbe_d4_avdz", 1250, "direct"),
]

# Subset sizes only; full N=1250 reuses campaign logs above (no _curve duplicates).
NTRAIN_VAL_RUNS = [row for row in NTRAIN_LOG_RUNS if row[2] in ("sad", "direct")]

_HYBRID_SEED_STEMS = [
    "water_wb97mv_def2tzvpd_seed7",
    "water_wb97mv_def2tzvpd_seed123",
    "water_wb97mv_def2tzvpd_seed20260821",
]


def _load(path: Path):
    return json.loads(path.read_text())


def plot_ir_convergence():
    eth = _load(_REPO / "results/revision/ir_convergence_ethanol/summary.json")
    res = _load(_REPO / "results/revision/ir_convergence_resorcinol/summary.json")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    eth_w = eth["trajectories"][0]["windows"]
    ax.plot(
        [w["window_ps"] for w in eth_w],
        [w["peak_fp_cm1"] for w in eth_w],
        "o-",
        label="Ethanol (rep 0)",
        color="#1f77b4",
    )

    agg = res["aggregate"]["windows"]
    xs = [w["window_ps"] for w in agg]
    ys = [w["peak_fp_mean"] for w in agg]
    yerr = [w["peak_fp_std"] for w in agg]
    ax.errorbar(xs, ys, yerr=yerr, fmt="s-", capsize=4, label="Resorcinol (4 replicas)", color="#d62728")

    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"$\tilde\nu$ / cm$^{-1}$")
    ax.legend()
    fig.tight_layout()
    save_response_figure(fig, OUT / "ir_convergence_fp_peak.png")


def plot_ethanethiol_ir_mlips() -> Path | None:
    """Render the Reviewer 3 ethanethiol comparison once all runs exist."""
    # The 2026-09-27 recovery rerun (1 fs dipole sampling, 300 K). The older
    # results/revision/ethanethiol_ir_mlips runs sampled dipoles every 10 fs
    # (S-H/C-H stretches alias below 1667 cm^-1) and ran hot; do not use them.
    root = _REPO / "results/revision/jacs_recovery_20260927/ethanethiol_ir"
    paths = {
        "DenSNet": root / "densnet/ir_50ps.csv",
        "MACE-OFF + DenSNet dipole": root / "maceoff/ir_50ps.csv",
        "MACE-POLAR-1-M": root / "macepolar/ir_50ps.csv",
        "SO3LR": root / "so3lr/ir_50ps.csv",
        "NIST": _REPO / "datasets/revision/external/ethanethiol_nist_gas_ir.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        print(f"ethanethiol IR comparison skipped: {len(missing)} spectra missing")
        return None
    spectra = {}
    for name, path in paths.items():
        values = np.loadtxt(path, delimiter=",", skiprows=1)
        spectra[name] = (values[:, 0], values[:, 1])
    destination = OUT / "ethanethiol_ir_mlips.png"
    compare_spectra(
        spectra,
        figure_path=destination,
        metrics_path=root / "metrics.json",
    )
    return destination


def plot_hbond_ref():
    hb = _load(_REPO / "results/revision/water_hbond_ood_size.json")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, key, xlabel in (
        (axes[0], "ref_OO", r"$r_{\mathrm{OO}}$ / $\mathrm{\AA}$"),
        (axes[1], "ref_O_H", r"$r_{\mathrm{OH}}$ / $\mathrm{\AA}$"),
    ):
        h = hb[key]
        lo, hi, nb = h["lo"], h["hi"], h["nbins"]
        width = (hi - lo) / nb
        centres = [lo + (i + 0.5) * width for i in range(nb)]
        counts = np.array(h["counts"], dtype=float)
        counts /= counts.sum() if counts.sum() else 1.0
        ax.bar(centres, counts, width=width * 0.95, color="#4c72b0", edgecolor="none")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Fraction of contacts")
    fig.tight_layout()
    save_response_figure(fig, OUT / "hbond_ref_ood_size.png")


def plot_pbe_vs_hybrid_bars():
    data = _load(_REPO / "results/revision/pbe_vs_hybrid.json")
    tiers = [
        ("water_id_test", "ID test"),
        ("water_ood_size", "Size OOD"),
        ("water_ood_order", "Ice cutouts"),
        ("water_ood_density", "Droplets"),
        ("malonaldehyde_ood_proton_transfer", "Malon PT"),
    ]
    labels, pbe, hyb = [], [], []
    for key, short in tiers:
        t = data["tiers"][key]["per_theory"]
        labels.append(short)
        pbe.append(100 * t["pbe_d4_avdz"]["afe_mean"])
        hyb.append(100 * t["wb97mv_def2tzvpd"]["afe_mean"])

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - w / 2, pbe, w, label="PBE-D4", color="#aec7e8")
    ax.bar(x + w / 2, hyb, w, label="$\\omega$B97M-V", color="#ff9896")
    ax.set_ylabel(EPSILON_RHO_PERCENT)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    save_response_figure(fig, OUT / "pbe_vs_hybrid_bars.png")


def _cv_path(rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path
    return _REPO / "results" / "revision" / path


def _cv_frames(rel: str) -> list[dict]:
    return _load(_cv_path(rel))["frames"]


def iter_water_cv_splits():
    """Yield (label, frames) for CV JSON files that exist."""
    for label, rel in WATER_CV.items():
        path = _cv_path(rel)
        if path.is_file():
            yield label, _load(path)["frames"]


def _finite(frames: list[dict], key: str) -> np.ndarray:
    vals = []
    for frame in frames:
        value = frame.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            vals.append(number)
    return np.asarray(vals, dtype=float)


def cv_series(frames: list[dict], key: str) -> np.ndarray:
    """Scalar CV, or all per-molecule q3 samples when key is q3_per_molecule."""
    if key == "q3_per_molecule":
        chunks = [
            np.asarray(row.get(key) or [], dtype=float)
            for row in frames
        ]
        chunks = [chunk[np.isfinite(chunk)] for chunk in chunks if chunk.size]
        if not chunks:
            return np.asarray([], dtype=float)
        return np.concatenate(chunks)
    return _finite(frames, key)


def shared_hist_bins(
    *arrays: np.ndarray,
    n_bins: int = CV_HIST_BINS,
    pad_frac: float = 0.04,
) -> np.ndarray:
    """Common bin edges spanning every series, with a small pad and no empty wings."""
    finite = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            finite.append(values)
    if not finite:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo = min(float(values.min()) for values in finite)
    hi = max(float(values.max()) for values in finite)
    if hi <= lo:
        hi = lo + 1.0
    pad = pad_frac * (hi - lo) + 1e-9
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def shift_energy_kcal(
    values: np.ndarray,
    offset: float = ENERGY_OFFSET_KCAL,
) -> np.ndarray:
    """Move per-water energies onto a small integer axis."""
    return np.asarray(values, dtype=float) + offset


def draw_faded_histogram(ax, values, bins, color, label) -> None:
    """Filled histogram used by the response-letter CV and energy figures."""
    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=HIST_FILL_ALPHA,
        linewidth=0,
        color=color,
        label=label,
    )
    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        color=color,
    )


def style_energy_hist_axis(ax) -> None:
    ax.set_xlabel(ENERGY_XLABEL)
    ax.set_ylabel(HIST_YLABEL)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))


def _energy_per_water(rel: str) -> np.ndarray:
    payload = np.load(_REPO / rel, allow_pickle=True).item()
    energy = np.asarray(payload["energy"], dtype=float)
    n_water = np.sum(np.asarray(payload["atom_numbers"]) > 0, axis=1) / 3.0
    return energy * HARTREE_TO_KCAL / n_water


def plot_dataset_cv_histograms():
    colors = {
        "Train": "#4c72b0",
        "ID test": "#55a868",
        "Size OOD": "#c44e52",
        "Ice": "#8172b3",
        "Droplets": "#ccb974",
        "Rackers/Rakshit": "#8c564b",
    }
    loaded = dict(iter_water_cv_splits())
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6))
    for ax, (key, title) in zip(axes.ravel(), CV_HIST_PANELS):
        series = {label: cv_series(frames, key) for label, frames in loaded.items()}
        bins = shared_hist_bins(*series.values(), n_bins=CV_HIST_BINS)
        for label, values in series.items():
            if values.size == 0:
                continue
            draw_faded_histogram(ax, values, bins, colors[label], label)
        ax.set_xlabel(title)
    for ax in axes[:, 0]:
        ax.set_ylabel(HIST_YLABEL)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    dest = response_figure_path(OUT / "dataset_water_cv_histograms.png")
    save_response_figure(fig, dest)
    return dest


def plot_dataset_energy_histograms():
    panels = []
    all_shifted = []
    for title, series in ENERGY_HIST_PANELS:
        loaded = {}
        for label, key in series:
            path = WATER_ENERGY[key]
            if not (_REPO / path).is_file():
                continue
            shifted = shift_energy_kcal(_energy_per_water(path))
            loaded[label] = shifted
            all_shifted.append(shifted)
        panels.append((title, loaded))
    bins = shared_hist_bins(*all_shifted, n_bins=ENERGY_HIST_BINS)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.6, 4.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, (title, loaded) in zip(axes, panels):
        for label, values in loaded.items():
            draw_faded_histogram(ax, values, bins, ENERGY_HIST_COLORS[label], label)
        ax.set_title(title)
        style_energy_hist_axis(ax)
        ax.set_xlim(float(bins[0]), float(bins[-1]))
    axes[1].set_ylabel("")
    axes[0].legend(loc="upper right")
    dest = response_figure_path(OUT / "dataset_water_energy_histograms.png")
    save_response_figure(fig, dest)
    return dest


def plot_malon_histograms():
    train = _cv_frames("cv/malon_train_pt.json")
    ood = _cv_frames("cv/malon_ood_proton_transfer_pt.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, key, xlabel in (
        (axes[0], "r_oo", r"$r_{\mathrm{OO}}$ / $\mathrm{\AA}$"),
        (axes[1], "delta_pt", r"$\delta$ / $\mathrm{\AA}$"),
    ):
        ax.hist(_finite(train, key), bins=20, density=True, histtype="step",
                linewidth=1.6, label="Train (enol)", color="#4c72b0")
        ax.hist(_finite(ood, key), bins=20, density=True, histtype="step",
                linewidth=1.6, label="PT scan", color="#c44e52")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(HIST_YLABEL)
    axes[1].axvspan(-2.0, MALON_TRAIN_DELTA_MAX, color="#4c72b0", alpha=0.12, zorder=0)
    axes[1].axvline(MALON_TRAIN_DELTA_MAX, color="#4c72b0", ls="--", lw=1.0)
    axes[1].axvline(0.0, color="0.35", ls=":", lw=1.0)
    axes[1].set_xlim(-1.6, 0.9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save_response_figure(fig, OUT / "dataset_malon_histograms.png")


def malon_error_points(cv_frames: list[dict], eval_records: list[dict]) -> list[dict]:
    errors: dict[int, float] = {}
    for record in eval_records:
        key = record.get("source_index", record["index"])
        errors[int(key)] = float(record["afe"])
    points = []
    for row in cv_frames:
        idx = int(row["index"])
        if idx not in errors:
            continue
        delta = float(row["delta_pt"])
        points.append({"delta": delta, "afe": errors[idx], "region": malon_region(delta)})
    return points


def plot_malon_error_vs_delta():
    cv = _cv_frames("cv/malon_ood_proton_transfer_pt.json")
    train = _finite(_cv_frames("cv/malon_train_pt.json"), "delta_pt")
    eval_path = _REPO / "results/revision/eval/wb97mv_def2tzvpd/malonaldehyde_ood_proton_transfer.json"
    records = _load(eval_path)["records"]
    points = malon_error_points(cv, records)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axvspan(float(train.min()) - 0.05, MALON_TRAIN_DELTA_MAX, color="#4c72b0", alpha=0.12, zorder=0)
    ax.axvline(MALON_TRAIN_DELTA_MAX, color="#4c72b0", ls="--", lw=1.1)
    ax.axvline(0.0, color="0.35", ls=":", lw=1.0)
    ax.hist(train, bins=24, density=True, histtype="stepfilled", color="#4c72b0", alpha=0.25, label=r"Train $\delta$")
    in_train = [p for p in points if p["region"] == "training"]
    held = [p for p in points if p["region"] == "held-out"]
    ax2 = ax.twinx()
    if in_train:
        ax2.scatter(
            [p["delta"] for p in in_train],
            [p["afe"] for p in in_train],
            s=18,
            color="#4c72b0",
            label="Scan, still in train well",
            zorder=3,
        )
    ax2.scatter(
        [p["delta"] for p in held],
        [p["afe"] for p in held],
        s=18,
        color="#c44e52",
        label="Scan, held-out",
        zorder=3,
    )
    ax.set_xlabel(r"$\delta$ / $\mathrm{\AA}$")
    ax.set_ylabel(HIST_YLABEL)
    style_epsilon_rho_ticks(ax2)
    ax2.yaxis.get_offset_text().set_size(12)
    ax.set_xlim(-1.6, 0.9)
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc="upper left")
    fig.tight_layout()
    save_response_figure(fig, OUT / "malon_error_vs_delta.png")


def plot_timing_bars():
    cpu = _load(_REPO / "results/revision/figure2_timing.json")["methods"]
    gpu = _load(_REPO / "results/revision/figure2_timing_gpu.json")["methods"]
    densnet_cpu = _load(_REPO / "results/revision/figure2_timing_densnet.json")["methods"]
    cpu["densnet"] = densnet_cpu["densnet"]
    order = [
        ("aimnet2", "AIMNet2"),
        ("so3lr", "SO3LR"),
        ("gfn2xtb", "GFN2-xTB"),
        ("gxtb", "g-xTB"),
        ("dftb", "DFTB/3ob"),
        ("maceoff", "MACE-OFF"),
        ("macepolar", "MACE-POLAR-1-M"),
        ("densnet", "DenSNet"),
    ]
    labels, cpu_s, gpu_s = [], [], []
    for key, label in order:
        if key not in cpu:
            continue
        labels.append(label)
        cpu_mean = cpu[key]["mean_s"]
        cpu_s.append(cpu_mean)
        gpu_mean = gpu.get(key, {}).get("mean_s")
        gpu_s.append(
            gpu_mean
            if gpu_mean is not None and gpu_speedup_is_clear(cpu_mean, gpu_mean)
            else np.nan
        )

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - 0.18, cpu_s, 0.36, label="CPU", color="#aec7e8")
    ax.bar(x + 0.18, gpu_s, 0.36, label="GPU", color="#ff9896")
    ax.set_yscale("log")
    ax.set_ylabel(r"$t$ / s")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    save_response_figure(fig, OUT / "calculations_timing_bars.png")


def _log_path(stem: str) -> Path:
    return _REPO / "logs" / f"train_{stem}.log"


def _plot_stitched_curve(
    ax: plt.Axes,
    stem: str,
    label: str,
    color,
    *,
    linewidth: float = 0.9,
    mark_lr_stop: bool = True,
) -> bool:
    log = _log_path(stem)
    if not log.is_file():
        return False
    result = parse_density_curve_result(log)
    if len(result.points) < 2:
        return False
    steps = [p["step"] for p in result.points]
    # training logs store the fraction; the letter quotes ε_ρ in percent
    valid = [100 * p.get("valid_perc_mae", p["valid_mae"]) for p in result.points]
    ax.plot(steps, valid, label=label, color=color, linewidth=linewidth)
    if mark_lr_stop and result.stop_reason == "lr":
        ax.scatter(
            [steps[-1]],
            [valid[-1]],
            color=color,
            s=16,
            zorder=5,
            edgecolors="0.15",
            linewidths=0.5,
        )
    return True


# Drawn at the width of one journal column so that text is 7-9 pt when placed at
# natural size (\includegraphics[width=3.4in]).
LC_FIGSIZE = (3.4, 2.45)
LC_RC = {"font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
         "legend.fontsize": 7.5, "lines.linewidth": 1.1}


def _style_density_step_ax(ax: plt.Axes) -> None:
    ax.set_ylabel(r"Validation $\varepsilon_\rho$ / %")
    ax.set_xlabel("Training step")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 3.0)  # the step-0 spike runs off the top
    ax.set_yticks([0.03, 0.1, 0.3, 1, 3], ["0.03", "0.1", "0.3", "1", "3"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(left=0)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    ax.legend(loc="upper right", handlelength=1.6)


def _save_density_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    save_response_figure(fig, OUT / filename)


def _hybrid_seed_band(ax: plt.Axes) -> None:
    curves = []
    for stem in _HYBRID_SEED_STEMS:
        log = _log_path(stem)
        if not log.is_file():
            continue
        result = parse_density_curve_result(log)
        if len(result.points) < 2:
            continue
        curves.append(result.points)
    if not curves:
        return
    all_steps = sorted({p["step"] for c in curves for p in c})
    stack = []
    for c in curves:
        by_step = {p["step"]: 100 * p.get("valid_perc_mae", p["valid_mae"]) for p in c}
        stack.append([by_step.get(s, np.nan) for s in all_steps])
    arr = np.asarray(stack, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    color = "#ff7f0e"
    ax.fill_between(
        all_steps,
        np.maximum(mean - std, 1e-6),
        mean + std,
        color=color,
        alpha=0.25,
        label=r"$\omega$B97M-V seeds (mean $\pm$ std)",
    )
    ax.plot(all_steps, mean, color=color, linewidth=1.0)


def plot_learning_curves_pbe_vs_hybrid() -> None:
    """R3.1 + R3.3: PBE vs hybrid SAD water; seeds as a band."""
    fig, ax = plt.subplots(figsize=LC_FIGSIZE)
    _plot_stitched_curve(ax, "water_pbe_d4_avdz", "Water PBE-D4 (SAD)", "#1f77b4")
    _hybrid_seed_band(ax)
    _plot_stitched_curve(
        ax,
        "water_wb97mv_def2tzvpd",
        r"Water $\omega$B97M-V (SAD)",
        "#ff7f0e",
        linewidth=0.7,
        mark_lr_stop=False,
    )
    _style_density_step_ax(ax)
    _save_density_figure(fig, "learning_curves_pbe_vs_hybrid.png")


def plot_learning_curves_sad_vs_direct() -> None:
    """R3.4: delta vs direct at PBE-D4."""
    fig, ax = plt.subplots(figsize=LC_FIGSIZE)
    _plot_stitched_curve(ax, "water_pbe_d4_avdz", "SAD ($\\rho_{\\mathrm{SAD}}+\\Delta\\rho$)", "#1f77b4")
    _plot_stitched_curve(ax, "water_direct_pbe_d4_avdz", "Direct ($\\rho_{\\mathrm{ML}}$)", "#d62728")
    _style_density_step_ax(ax)
    _save_density_figure(fig, "learning_curves_sad_vs_direct.png")


def plot_learning_curves_cutoff() -> None:
    """R3.5: neighbour-cutoff sweep on PBE water (SAD)."""
    fig, ax = plt.subplots(figsize=LC_FIGSIZE)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
    for color, cut in zip(cmap, ("4", "5", "6", "8")):
        _plot_stitched_curve(
            ax,
            f"water_pbe_orca_cutoff_{cut}",
            rf"${cut}\,\mathrm{{\AA}}$",
            color,
        )
    _style_density_step_ax(ax)
    _save_density_figure(fig, "learning_curves_cutoff.png")


def plot_learning_curves_malonaldehyde() -> None:
    """R1.1 SI: malonaldehyde proton-transfer training curves."""
    fig, ax = plt.subplots(figsize=LC_FIGSIZE)
    _plot_stitched_curve(ax, "malonaldehyde_pbe_d4_avdz", "Malon PBE-D4", "#1f77b4")
    _plot_stitched_curve(
        ax,
        "malonaldehyde_wb97mv_def2tzvpd",
        r"Malon $\omega$B97M-V",
        "#ff7f0e",
    )
    _style_density_step_ax(ax)
    _save_density_figure(fig, "learning_curves_malonaldehyde.png")


def plot_learning_curves_dashboard() -> None:
    """Internal overview (not cited in the response letter)."""
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 9.4), sharex=False)
    groups = (
        (axes[0], LEARNING_RUNS[:5], "Water SAD density (campaign)"),
        (axes[1], LEARNING_RUNS[5:], "Direct, malonaldehyde, cutoff sweep"),
        (
            axes[2],
            [
                (stem, f"{'SAD' if arm == 'sad' else 'Direct'} N={n}")
                for stem, n, arm in NTRAIN_VAL_RUNS
            ],
            "Wave 1 $N_{\\mathrm{train}}$",
        ),
    )
    cmap = plt.cm.tab10
    for ax, runs, title in groups:
        for i, (stem, label) in enumerate(runs):
            _plot_stitched_curve(ax, stem, label, cmap(i % 10), mark_lr_stop=False)
        _style_density_step_ax(ax)
    save_response_figure(fig, OUT / "learning_curves_density.png")


def plot_reviewer_learning_curves() -> None:
    with plt.rc_context(LC_RC):
        plot_learning_curves_pbe_vs_hybrid()
        plot_learning_curves_sad_vs_direct()
        plot_learning_curves_cutoff()
        plot_learning_curves_malonaldehyde()
    plot_learning_curves_dashboard()


def plot_ntrain_val_from_logs(out_path: Path | None = None) -> Path | None:
    """N_train curve from stitched validation perc. MAE until AFE scoring exists."""
    eval_json = _REPO / "results/revision/eval/learning_curves_ntrain.json"
    if eval_json.is_file():
        return plot_ntrain_learning_curves(eval_json, out_path)
    series = {"sad": [], "direct": []}
    for stem, n_train, arm in NTRAIN_VAL_RUNS:
        log = _log_path(stem)
        if not log.is_file():
            continue
        curve = parse_density_curve(log)
        if not curve:
            continue
        last = curve[-1]
        series[arm].append((n_train, last.get("valid_perc_mae", last["valid_mae"]), last["step"]))
    if not any(series.values()):
        print("n_train val-MAE figure skipped: no subset logs")
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for arm, color, label in (
        ("sad", "#1f77b4", "SAD"),
        ("direct", "#d62728", "Direct"),
    ):
        pts = sorted(series[arm])
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            marker="o",
            linewidth=1.6,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\mathrm{train}}$")
    ax.set_ylabel(r"$\mathrm{MAE}_\rho$")
    ax.legend()
    fig.tight_layout()
    dest = response_figure_path(out_path or (OUT / "learning_curves_ntrain.png"))
    save_response_figure(fig, dest)
    return dest


_SPLIT_STYLE = {
    "water_id_test": ("-", "ID test"),
    "water_ood_size": ("--", "Size OOD"),
    "water_ood_order": ("-.", "Ice"),
    "water_ood_density": (":", "Droplets"),
}
_ARM_COLOR = {"sad": "#1f77b4", "direct": "#d62728"}
_ARM_LABEL = {"sad": "SAD", "direct": "Direct"}


def _plot_curve_series(
    records: list,
    x_key: str,
    xlabel: str,
    dest: Path,
    log_x: bool = False,
) -> Path:
    series = series_for_figure(records, x_key=x_key)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for (arm, split), points in sorted(series.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ls, split_label = _SPLIT_STYLE.get(split, ("-", split))
        ax.plot(
            xs,
            ys,
            linestyle=ls,
            color=_ARM_COLOR.get(arm, "0.3"),
            marker="o",
            linewidth=1.6,
            label=f"{_ARM_LABEL.get(arm, arm)} {split_label}",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathrm{MAE}_\rho$")
    ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.legend(ncol=2)
    fig.tight_layout()
    dest = response_figure_path(dest)
    save_response_figure(fig, dest)
    return dest


def plot_ntrain_learning_curves(
    eval_json: Path | None = None,
    out_path: Path | None = None,
) -> Path | None:
    """AFE versus N_train on the reviewer water splits."""
    path = eval_json or (_REPO / "results/revision/eval/learning_curves_ntrain.json")
    if not path.is_file():
        print(f"n_train learning-curve figure skipped: {path} missing")
        return None
    payload = _load(path)
    dest = out_path or (OUT / "learning_curves_ntrain.png")
    return _plot_curve_series(
        payload["records"],
        "n_train",
        r"$N_{\mathrm{train}}$",
        dest,
        log_x=True,
    )


def plot_ood_step_learning_curves(
    eval_json: Path | None = None,
    out_path: Path | None = None,
) -> Path | None:
    """AFE versus optimisation step on the reviewer water splits."""
    path = eval_json or (_REPO / "results/revision/eval/learning_curves_step.json")
    if not path.is_file():
        print(f"OOD-vs-step learning-curve figure skipped: {path} missing")
        return None
    payload = _load(path)
    dest = out_path or (OUT / "learning_curves_ood_step.png")
    return _plot_curve_series(
        payload["records"],
        "step",
        r"$N_{\mathrm{step}}$",
        dest,
    )


def plot_distance_scale(eval_json: Path | None = None, out_path: Path | None = None) -> Path | None:
    """ΔE vs min O-O for the rigid-monomer water scan. Marks the 6 A cutoff."""
    path = eval_json or (_REPO / "results/revision/eval/ood_distance_scale.json")
    if not path.is_file():
        print(f"distance-scale figure skipped: {path} missing")
        return None
    payload = _load(path)
    rec = payload["records"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for regime, color in (("SR", "#1f77b4"), ("LR", "#d62728")):
        rows = [r for r in rec if r.get("regime") == regime]
        if not rows:
            continue
        ax.scatter(
            [r["r_oo_min"] for r in rows],
            [r["delta_e"] for r in rows],
            s=18,
            alpha=0.75,
            color=color,
            label=regime,
        )
    ax.axvline(6.0, color="0.35", linestyle="--", linewidth=1.0, label=r"$6\,\mathrm{\AA}$ cutoff")
    ax.set_xlabel(r"$\min r_{\mathrm{OO}}$ / $\mathrm{\AA}$")
    ax.set_ylabel(r"$\Delta E$")
    ax.legend()
    fig.tight_layout()
    dest = response_figure_path(out_path or (OUT / "ood_distance_scale.png"))
    save_response_figure(fig, dest)
    return dest


def plot_extreme_modes(manifest_path: Path | None = None, out_path: Path | None = None) -> Path | None:
    """|q_k| and RMSD for one-mode extreme-amplitude frames."""
    path = manifest_path or (_REPO / "datasets/revision/ood_modes/water_n2/manifest.json")
    if not path.is_file():
        print(f"extreme-mode figure skipped: {path} missing")
        return None
    payload = _load(path)
    rec = payload["records"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter(
        [r["rmsd"] for r in rec],
        [abs(r["q_k"]) for r in rec],
        s=80,
        alpha=0.7,
        color="#2ca02c",
    )
    ax.set_xlabel(r"$\mathrm{RMSD}$ / $\mathrm{\AA}$")
    ax.set_ylabel(r"$|q_k|$ / amu$^{1/2}$ $\mathrm{\AA}$")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    dest = response_figure_path(out_path or (OUT / "ood_extreme_modes.png"))
    save_response_figure(fig, dest)
    return dest


def plot_extreme_mode_error(score_path: Path | None = None, out_path: Path | None = None) -> Path | None:
    """AFE versus |q_k| and RMSD from score_extreme_modes.py."""
    path = score_path or (_REPO / "results/revision/eval/ood_extreme_modes.json")
    if not path.is_file():
        print(f"extreme-mode error figure skipped: {path} missing")
        return None
    dest = response_figure_path(out_path or (OUT / "ood_extreme_mode_error.png"))
    make_extreme_mode_error_figure(_load(path)["systems"], dest)
    return dest


def main() -> int:
    plot_ir_convergence()
    plot_ethanethiol_ir_mlips()
    plot_hbond_ref()
    plot_pbe_vs_hybrid_bars()
    plot_timing_bars()
    plot_dataset_cv_histograms()
    plot_dataset_energy_histograms()
    plot_malon_histograms()
    plot_malon_error_vs_delta()
    plot_reviewer_learning_curves()
    plot_ntrain_val_from_logs()
    plot_ood_step_learning_curves()
    plot_distance_scale()
    plot_extreme_modes()
    plot_extreme_mode_error()
    try:
        thio = _HERE.parent / "thiophene"
        if str(thio) not in sys.path:
            sys.path.insert(0, str(thio))
        from experiments.jacs_revision.thiophene.plot_thiophene_mh_figures import plot_available

        plt.style.use("default")
        plt.rcdefaults()
        plot_available(OUT)
    except Exception as exc:
        print(f"thiophene MH figures skipped: {exc}")
    apply_classic_tex()
    try:
        render_gallery(_REPO, OUT)
    except Exception as exc:
        print(f"structure gallery skipped: {exc}")
    print(f"wrote figures under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
