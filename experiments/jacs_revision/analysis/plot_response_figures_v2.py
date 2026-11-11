#!/usr/bin/env python3
"""Response-letter figures rebuilt from the 2026-09-28 recalculations.

Vector PDFs for the letter go to results/revision/figures/response_v2/ (same
file names where a figure replaces one already in the letter), PNG previews to
results/revision/figures/response_v2/png/. Style is the letter's (classic
matplotlib, Computer Modern mathtext; mpl_classic_tex).

Density errors are the full-density re-score (rescore_full_density.py). For
each model and tier the evaluation used is, in order of preference: the
fine-tuned checkpoint rebuilt with its charge constraint; the density-phase
checkpoint; the older fine-tuned evaluation. The choice is written to
provenance.json, so re-running this script after the pending evaluations
finish gives the final figures.

Usage:
  python experiments/jacs_revision/analysis/plot_response_figures_v2.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _p in (str(REPO), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.jacs_revision.analysis.error_vs_distance import (  # noqa: E402
    TIER_COLORS,
    draw_saturation_fit,
    fit_series_saturation,
    series_from_eval,
    style_epsilon_rho_ticks,
)
from experiments.jacs_revision.analysis.mpl_classic_tex import apply as apply_classic_tex  # noqa: E402

EVAL = REPO / "results/revision/rescore_fullref/eval"
# Complete evaluations from before the constraint fix; used while the rescore of
# the same file is still being written (partial files are subsets, not estimates).
EVAL_OLD = REPO / "results/revision/rescore_fullref/eval_before_ft_constraint"
SELECTED = REPO / "results/revision/rescore_fullref/selected"
CV = REPO / "results/revision/cv"
OUT = REPO / "results/revision/figures/response_v2"
PNG = OUT / "png"
THEORIES = {"pbe_d4_avdz": "PBE-D4", "wb97mv_def2tzvpd": r"$\omega$B97M-V"}
TIERS = [("water_id_test", "ID test"), ("water_ood_size", "Size OOD"),
         ("water_ood_order", "Ice cutouts"), ("water_ood_density", "Droplets")]
MALON = "malonaldehyde_ood_proton_transfer"
SEEDS = ["_seed7", "_seed123", "_seed20260821"]
PROVENANCE: dict[str, str] = {}

apply_classic_tex()
# The letter shrinks these figures to page width, so the fonts are set larger
# than the house style to print at >= 9 pt.
FONT_SCALE = 1.35
matplotlib.rcParams.update({k: matplotlib.rcParams[k] * FONT_SCALE for k in
                            ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
matplotlib.rcParams["axes.titlesize"] = matplotlib.rcParams["axes.labelsize"]


def fs(size: float) -> float:
    """Explicit font size, scaled like the rcParams above."""
    return size * FONT_SCALE


PCT = r"$\varepsilon_\rho$ / \%" if matplotlib.rcParams["text.usetex"] else r"$\varepsilon_\rho$ / %"
_FT_CACHE: dict[str, bool] = {}


def checkpoint_fine_tuned(path: str | None) -> bool:
    """True if the checkpoint went through density_fine_tuning (has the charge constraint)."""
    if not path:
        return False
    if path not in _FT_CACHE:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        _FT_CACHE[path] = any("integral_constraint" in k for k in state)
    return _FT_CACHE[path]


# Where each PDF is used besides OUT: the response letter (v2) and, for the
# figures moved into the SI, the revised paper. Copied on every save so the
# documents never carry a stale version.
LETTER_FIGS = REPO / "docs/response-to-jacs-review/figures/v2/response"
PAPER_FIGS = REPO / "docs/equiv-ml-paper/figures/revision"
COPY_TO = {name: [LETTER_FIGS] for name in (
    "error_vs_distance_water_pbe_hybrid", "pbe_vs_hybrid_bars", "malon_error_vs_delta", "ood_extreme_mode_error",
    "learning_curves_ntrain_fullref", "cutoff_sweep_fullref", "density_functional_errors",
    "calculations_timing_scan", "densnet_step_breakdown", "equilibrium_frequencies")}
COPY_TO["error_vs_distance_water_pbe_hybrid"].append(LETTER_FIGS.parent)  # the letter resolves figures/v2/ first
for _name in ("ood_extreme_mode_error", "equilibrium_frequencies", "densnet_step_breakdown", "cutoff_sweep_fullref"):
    COPY_TO[_name].append(PAPER_FIGS)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(PNG / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / (name + '.pdf')}")
    for dest in COPY_TO.get(name, []):
        if dest.is_dir():
            shutil.copy2(OUT / f"{name}.pdf", dest / f"{name}.pdf")


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def complete(payload) -> bool:
    return payload is not None and payload.get("provenance", {}).get("complete", True) is not False


def pick(model_dir: str, tier: str) -> tuple[dict | None, str]:
    """Best available evaluation of a production model on one tier (see module doc)."""
    ft = load(EVAL / model_dir / f"{tier}.json")
    if complete(ft) and ft["provenance"].get("fine_tuned_rebuilt_with_constraint"):
        return ft, f"{model_dir}/{tier} (fine-tuned, constraint applied)"
    if complete(ft) and not checkpoint_fine_tuned(ft["provenance"].get("checkpoint")):
        return ft, f"{model_dir}/{tier} (production; no fine-tuning phase)"
    dp = load(EVAL / f"{model_dir}_densityphase" / f"{tier}.json")
    if complete(dp):
        return dp, f"{model_dir}_densityphase/{tier} (density-phase checkpoint)"
    if complete(ft):
        return ft, f"{model_dir}/{tier} (fine-tuned, evaluated before the constraint fix)"
    old = load(EVAL_OLD / model_dir / f"{tier}.json")
    if complete(old):
        return old, f"{model_dir}/{tier} (rescore in progress; earlier complete evaluation)"
    return None, "missing"


def afe(payload, key="afe"):
    v = np.array([r[key] for r in payload["records"] if key in r], dtype=float)
    return v[np.isfinite(v)]


# ----------------------------------------------------------------------------- shared: SAD prior, DF floor, log % axis
# Every density-error figure shows, next to the Δ model: the direct model (red,
# open or dashed), the density-fitting floor (dashed) and the error of the SAD
# prior itself (dotted). All on a log ε_ρ axis in percent, since the SAD prior
# (~10%) sits two orders of magnitude above the models (~0.1%).
DFUNC = REPO / "results/revision/density_functionals"
DELTA_COLOR, DIRECT_COLOR, SAD_COLOR, DF_COLOR = "#1f5fa8", "#c44e52", "#7f7f7f", "0.25"
PCT_TICKS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
# The SAD prior (~10%) sits two orders of magnitude above every model and squeezes the model range into the
# bottom of a log axis, so it is left out of the density figures except the extreme-mode one, where the
# errors approach it. Its values are still reported in the tables.
SHOW_SAD = False


def log_pct_axis(ax, lo: float = 0.03, hi: float = 30.0) -> None:
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [t for t in PCT_TICKS if lo <= t <= hi]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def dfunc(label_set: str):
    """score_density_functionals.py output for one label set (may still be growing)."""
    payload = load(DFUNC / f"{label_set}.json")
    if payload:
        PROVENANCE[f"density_functionals/{label_set}"] = f"{len(payload['frames'])} frames"
    return payload


DFUNC_SAD_CPU = REPO / "results/revision/density_functionals_sad_cpu"  # --sad-only --cpu-reference runs


def sad_by_index(label_set: str) -> dict[int, float]:
    """SAD-prior ε_ρ (%) per structure index of a label set (GPU run, else the CPU SAD-only run)."""
    out = {}
    for payload in (load(DFUNC_SAD_CPU / f"{label_set}.json"), dfunc(label_set)):
        if payload:
            out.update({int(f["index"]): next(iter(f["sad"].values()))["afe_pct"]
                        for f in payload["frames"] if f.get("sad")})
    if out:
        PROVENANCE[f"sad/{label_set}"] = f"{len(out)} frames"
    return out


def sad_mean(label_set: str) -> float | None:
    vals = list(sad_by_index(label_set).values())
    return float(np.mean(vals)) if vals else None


def dw_series(payload_by_tier: dict[str, dict], tag: str) -> list:
    """(tier, d/w, ε_ρ in %) for the water tiers of one model, via error_vs_distance."""
    sel = SELECTED / tag
    sel.mkdir(parents=True, exist_ok=True)
    for tier, payload in payload_by_tier.items():
        (sel / f"{tier}.json").write_text(json.dumps(payload))
    return [(t, d, 100 * e) for t, d, e in series_from_eval(CV / "water_train.json", sel, CV)]


# ----------------------------------------------------------------------------- F1: error vs d/w
def fig_error_vs_distance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)
    for ax, (theory, title) in zip(axes, THEORIES.items()):
        delta, direct, floors, sads = {}, {}, [], []
        for tier, _ in TIERS:
            p, note = pick(theory, tier)
            PROVENANCE[f"error_vs_distance/{theory}/{tier}"] = note
            if p is not None:
                delta[tier] = p
                floors.append(100 * afe(p, "df_floor").mean())
            q, note = pick(f"direct_{theory}", tier)
            PROVENANCE[f"error_vs_distance/direct_{theory}/{tier}"] = note
            if q is not None:
                direct[tier] = q
            s = sad_mean(f"{tier}_{theory}")
            if s is not None:
                sads.append(s)
        names = {"id_test": "ID test", "ood_size": "Size OOD", "ood_order": "Ice", "ood_density": "Droplets"}
        series = dw_series(delta, theory)
        for tier, dist, err in series:
            ax.scatter(dist, err, s=16, alpha=0.55, color=TIER_COLORS[tier], label=names[tier], zorder=3)
        for tier, dist, err in dw_series(direct, f"direct_{theory}"):
            ax.scatter(dist, err, s=16, alpha=0.5, facecolors="none", edgecolors=TIER_COLORS[tier], lw=0.8, zorder=3)
        draw_saturation_fit(ax, series)
        if floors:
            ax.axhline(float(np.mean(floors)), color=DF_COLOR, ls="--", lw=1.3, zorder=2, label="DF floor")
        if sads and SHOW_SAD:
            ax.axhline(float(np.mean(sads)), color=SAD_COLOR, ls=":", lw=1.6, zorder=2, label="SAD prior")
        ax.set_xlabel(r"$d/w$")
        ax.set_xlim(0, 8)
        log_pct_axis(ax, 0.03, 30 if SHOW_SAD else 1)
        ax.set_title(title, fontsize=fs(14))
    axes[0].set_ylabel(PCT)
    from matplotlib.lines import Line2D

    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], ls="", marker="o", mfc="none", mec="0.3", label="Direct ML (open)")]
    labels += ["Direct ML (open)"]
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=fs(10), scatterpoints=1, numpoints=1,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.2)
    save(fig, "error_vs_distance_water_pbe_hybrid")


# ----------------------------------------------------------------------------- F2: tiers, Δ vs direct, PBE vs hybrid
# The four bars of every tier group: (theory, model prefix, legend label, face, edge colour).
ARMS = [("pbe_d4_avdz", "", "PBE-D4, $\\Delta$", "#aec7e8", DELTA_COLOR),
        ("pbe_d4_avdz", "direct_", "PBE-D4, direct", "white", DELTA_COLOR),
        ("wb97mv_def2tzvpd", "", r"$\omega$B97M-V, $\Delta$", "#ff9896", DIRECT_COLOR),
        ("wb97mv_def2tzvpd", "direct_", r"$\omega$B97M-V, direct", "white", DIRECT_COLOR)]


def density_tier_bars(ax, groups, w: float = 0.19) -> dict:
    """ε_ρ bars per tier group and arm (ωB97M-V Δ: mean ± s.d. over seeds), DF floor and SAD prior per theory.

    Returns {(tier, prefix + theory): (value %, s.d. %, n models)}."""
    arms = ARMS
    table = {}
    for gi, (tier, label) in enumerate(groups):
        for ai, (theory, prefix, name, face, edge) in enumerate(arms):
            p, note = pick(f"{prefix}{theory}", tier)
            PROVENANCE[f"tier_bars/{prefix}{theory}/{tier}"] = note
            if p is None:
                continue
            members = [afe(p).mean()]
            if theory.startswith("wb97mv") and not prefix:
                for s in SEEDS:
                    ps, note_s = pick(f"{theory}{s}", tier)
                    PROVENANCE[f"tier_bars/{theory}{s}/{tier}"] = note_s
                    if ps is not None:
                        members.append(afe(ps).mean())
            x = gi + (ai - 1.5) * w
            val, err = 100 * np.mean(members), (100 * np.std(members, ddof=1) if len(members) > 1 else 0.0)
            ax.bar(x, val, w, yerr=err if err else None, capsize=2, color=face, edgecolor=edge, lw=1.3,
                   hatch="//" if prefix else None, label=name if gi == 0 else None)
            table[(tier, prefix + theory)] = (val, err, len(members))
        for ti, theory in enumerate(THEORIES):
            x0 = gi + (2 * ti - 1.5) * w - w / 2
            p, _ = pick(theory, tier)
            if p is not None:
                ax.plot([x0, x0 + 2 * w], [100 * afe(p, "df_floor").mean()] * 2, color=DF_COLOR, lw=1.8, ls="--",
                        label="DF floor" if gi == 0 and ti == 0 else None)
            s = sad_mean(f"{tier}_{theory}") if SHOW_SAD else None
            if s is not None:
                ax.plot([x0, x0 + 2 * w], [s, s], color=SAD_COLOR, lw=2.0, ls=":",
                        label="SAD prior" if gi == 0 and ti == 0 else None)
    ax.set_ylabel(PCT)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    log_pct_axis(ax, 0.03, 30 if SHOW_SAD else 1)
    return table


def fig_pbe_vs_hybrid() -> None:
    groups = TIERS + [(MALON, "Malon PT")]
    arms = ARMS
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    table = density_tier_bars(ax, groups)
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=fs(9.5), handlelength=1.8, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "pbe_vs_hybrid_bars")
    lines = ["| Tier | " + " | ".join(a[2].replace("$\\Delta$", "Δ").replace(r"$\omega$", "ω") for a in arms) + " |",
             "|---" * (len(arms) + 1) + "|"]
    for tier, label in groups:
        cells = []
        for theory, prefix, *_ in arms:
            v = table.get((tier, prefix + theory))
            cells.append("—" if v is None else (f"{v[0]:.3f} ± {v[1]:.3f} (n={v[2]})" if v[2] > 1 else f"{v[0]:.3f}"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (OUT / "tier_bars_table.md").write_text("ε_ρ (%) vs full SCF\n\n" + "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- F3: malonaldehyde
def fig_malon() -> None:
    cv = load(CV / "malon_ood_proton_transfer_pt.json")["frames"]
    delta = {int(f.get("index", i)): float(f["delta_pt"]) for i, f in enumerate(cv)}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, (theory, title) in zip(axes, THEORIES.items()):
        payload, note = pick(theory, MALON)
        PROVENANCE[f"malon/{theory}"] = note
        if payload is None:
            continue
        recs = [r for r in payload["records"] if int(r.get("source_index", -1)) in delta and np.isfinite(r["afe"])]
        x = np.array([delta[int(r["source_index"])] for r in recs])
        order = np.argsort(x)
        ax.axvspan(-1.6, -0.40, color="#4c72b0", alpha=0.12, zorder=0)
        ax.axvline(-0.40, color="#4c72b0", ls="--", lw=1.0)
        ax.axvline(0.0, color="0.35", ls=":", lw=1.0)
        y = 100 * np.array([r["afe"] for r in recs])
        well = x <= -0.40
        ax.scatter(x[well], y[well], s=18, color=DELTA_COLOR, alpha=0.8, label=r"$\Delta$, in training well", zorder=3)
        ax.scatter(x[~well], y[~well], s=18, color="#55a868", alpha=0.8, label=r"$\Delta$, held out", zorder=3)
        direct, note_d = pick(f"direct_{theory}", MALON)
        PROVENANCE[f"malon/direct_{theory}"] = note_d
        if direct is not None:
            dr = [r for r in direct["records"] if int(r.get("source_index", -1)) in delta and np.isfinite(r["afe"])]
            ax.scatter([delta[int(r["source_index"])] for r in dr], [100 * r["afe"] for r in dr], s=18,
                       facecolors="none", edgecolors=DIRECT_COLOR, lw=0.9, label="Direct ML", zorder=3)
        df = 100 * np.array([r["df_floor"] for r in recs])
        ax.plot(x[order], df[order], color=DF_COLOR, ls="--", lw=1.2, label="DF floor")
        sad = sad_by_index(f"{MALON}_{theory}") if SHOW_SAD else {}
        if sad:
            pts = sorted((delta[int(r["source_index"])], sad[int(r["index"])]) for r in recs if int(r["index"]) in sad)
            ax.plot(*np.array(pts).T, color=SAD_COLOR, ls=":", lw=1.6, label="SAD prior")
        ax.set_xlabel(r"$\delta$ / $\mathrm{\AA}$")
        ax.set_xlim(-1.5, 0.85)
        log_pct_axis(ax, 0.03, 30 if SHOW_SAD else 1)
        ax.set_title(title, fontsize=fs(13))
    axes[0].set_ylabel(PCT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=fs(9.5), scatterpoints=1,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
    save(fig, "malon_error_vs_delta")


# ----------------------------------------------------------------------------- F4: extreme modes
def fig_extreme_modes() -> None:
    panels = [("water_n2", "Water dimer"), ("water_n6", "Water hexamer"), ("malonaldehyde", "Malonaldehyde"),
              ("ethanol", "Ethanol"), ("ethanethiol", "Ethanethiol"), ("resorcinol", "Resorcinol"),
              ("thiophene_2mer", "Bithiophene")]
    theories = (("pbe", "pbe_d4_avdz", DELTA_COLOR), ("hybrid", "wb97mv_def2tzvpd", DIRECT_COLOR))
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.8), sharex=True)
    axes = axes.ravel()

    def rung_medians(pairs):
        x, y = np.array(pairs).T
        rungs = np.unique(np.round(x, 1))
        return rungs, np.array([np.median(y[np.round(x, 1) == r]) for r in rungs])

    for ax, (system, title) in zip(axes, panels):
        manifest = load(REPO / "datasets/revision/ood_modes" / system / "manifest.json")
        if manifest is None:
            continue
        rmsd = [r["rmsd"] for r in manifest["records"]]
        for tag, theory, color in theories:
            name = f"{system}_{tag}"
            payload = load(REPO / "results/revision/rescore_fullref/eval_filtered/ood_geom" / f"{name}.json")
            if payload is None:
                continue
            recs = [r for r in payload["records"] if "source_index" in r and np.isfinite(r["afe"])]
            pts = [(rmsd[r["source_index"]], 100 * r["afe"]) for r in recs]
            if not pts:
                continue
            x, y = np.array(pts).T
            ax.scatter(x, y, s=8, facecolors="none", edgecolors=color, alpha=0.35, lw=0.6)
            ax.plot(*rung_medians(pts), color=color, lw=2.0)
            direct_name = f"{system}_direct_{tag}"
            direct = load(EVAL / "ood_geom" / f"{direct_name}.json")
            PROVENANCE[f"extreme/{direct_name}"] = "present" if direct else "missing"
            if direct:
                dp = [(rmsd[r["source_index"]], 100 * r["afe"]) for r in direct["records"]
                      if "source_index" in r and np.isfinite(r["afe"]) and r["afe"] < 10]
                if dp:
                    ax.plot(*rung_medians(dp), color=color, lw=1.6, ls="-.")
            index_to_source = {int(r["index"]): r["source_index"] for r in payload["records"] if "source_index" in r}
            sad = sad_by_index(f"ood_modes_{system}_{theory}")
            sp = [(rmsd[index_to_source[i]], v) for i, v in sad.items() if i in index_to_source]
            if sp:
                ax.plot(*rung_medians(sp), color=color, lw=1.2, ls=":")
            floor = afe(load(EVAL / "ood_geom" / f"{name}.json"), "df_floor").mean()
            ax.axhline(100 * floor, color=color, ls="--", lw=0.9)
        ax.set_xscale("log")
        log_pct_axis(ax, 0.03, 300)
        ax.set_xlim(0.35, 14)
        ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(title, fontsize=fs(12))
    axes[-1].axis("off")
    for ax in axes[3:7]:
        ax.set_xlabel(r"RMSD / $\mathrm{\AA}$")
        ax.tick_params(labelbottom=True)
    axes[0].set_ylabel(PCT)
    axes[4].set_ylabel(PCT)
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([], [], color=DELTA_COLOR, lw=2.0, label="PBE-D4"),
        Line2D([], [], color=DIRECT_COLOR, lw=2.0, label=r"$\omega$B97M-V"),
        Line2D([], [], color="0.2", lw=2.0, label=r"$\Delta$ model, median"),
        Line2D([], [], color="0.2", lw=1.6, ls="-.", label="Direct ML, median"),
        Line2D([], [], color="0.2", lw=1.2, ls=":", label="SAD prior, median"),
        Line2D([], [], color="0.2", lw=0.9, ls="--", label="DF floor"),
        Line2D([], [], ls="", marker="o", mfc="none", mec="0.4", label=r"$\Delta$, single frames"),
    ]
    axes[-1].legend(handles=legend_items, loc="center left", fontsize=fs(9.5), numpoints=1)
    fig.tight_layout()
    save(fig, "ood_extreme_mode_error")


# ----------------------------------------------------------------------------- F5: N_train and cutoff
def fig_ntrain() -> None:
    sizes = [50, 125, 250, 500, 1250]

    def n_points(tier):
        return sum(complete(load(EVAL / "learning_curves" / f"ntrain_{n}_sad_{tier}.json")) for n in sizes[:-1])

    tiers = [(t, l) for t, l in TIERS if n_points(t) >= 3]
    ncols = 2 if len(tiers) > 2 else len(tiers)
    nrows = -(-len(tiers) // ncols)
    fig, grid = plt.subplots(nrows, ncols, figsize=(8.6, 3.5 * nrows + 0.6), sharey=True, sharex=True, squeeze=False)
    axes = grid.ravel()
    for ax, (tier, label) in zip(axes, tiers):
        for arm, color, name, full in (("sad", DELTA_COLOR, r"$\Delta$ model", "pbe_d4_avdz_densityphase"),
                                       ("direct", DIRECT_COLOR, "Direct ML", "direct_pbe_d4_avdz_densityphase")):
            xs, ys = [], []
            for n in sizes:
                path = (EVAL / full / f"{tier}.json") if n == 1250 else (EVAL / "learning_curves" / f"ntrain_{n}_{arm}_{tier}.json")
                payload = load(path)
                if complete(payload):
                    xs.append(n)
                    ys.append(100 * afe(payload).mean())
            if xs:
                ax.plot(xs, ys, "o-", color=color, lw=1.8, ms=5, label=name)
        floor = load(EVAL / "pbe_d4_avdz_densityphase" / f"{tier}.json")
        if floor:
            ax.axhline(100 * afe(floor, "df_floor").mean(), color=DF_COLOR, ls="--", lw=1.2, label="DF floor")
        s = sad_mean(f"{tier}_pbe_d4_avdz") if SHOW_SAD else None
        if s is not None:
            ax.axhline(s, color=SAD_COLOR, ls=":", lw=1.6, label="SAD prior")
        ax.set_xscale("log")
        log_pct_axis(ax, 0.03, 30 if SHOW_SAD else 3)
        ax.set_xlim(35, 1800)
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(label, fontsize=fs(12))
    for ax in grid[-1]:
        ax.set_xlabel(r"$N_{\mathrm{train}}$")
    for ax in grid[:, 0]:
        ax.set_ylabel(PCT)
    for ax in axes[len(tiers):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=fs(10), bbox_to_anchor=(0.5, 0.0))
    save(fig, "learning_curves_ntrain_fullref")


CUTOFF_HEADS = [4, 5, 6]  # energy heads on the cutoff-sweep density networks (score_energy_forces.py)
# CUTOFF_ENERGY_SUFFIX names the sweep score set: "" = the 40k-step heads, "_100k" = the heads continued to 100k steps
# (results/revision/energy_forces_v3/water_pbe_d4_avdz_cutoff_<c><suffix>/summary.json). CUTOFF_STEPS labels them.
CUTOFF_ENERGY_SUFFIX = __import__("os").environ.get("CUTOFF_ENERGY_SUFFIX", "")
CUTOFF_STEPS = __import__("os").environ.get("CUTOFF_STEPS", "40k" if not CUTOFF_ENERGY_SUFFIX else CUTOFF_ENERGY_SUFFIX.strip("_"))


def fig_cutoff() -> None:
    """(a) density error vs neighbor cutoff; (b, c) energy per water and force error of the energy heads trained on
    the same density networks (40k steps; the 100k-step 6 A production head as an open marker)."""
    cuts = [4, 5, 6, 8]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9))
    ax = axes[0]
    for tier, label, color in (("water_id_test", "ID test", TIER_COLORS["id_test"]),
                               ("water_ood_size", "Size OOD", TIER_COLORS["ood_size"])):
        ys = []
        for c in cuts:
            payload = load(EVAL / "cutoff_sweep" / f"cutoff_{c}_{tier}.json")
            if not complete(payload):
                payload = load(EVAL_OLD / "cutoff_sweep" / f"cutoff_{c}_{tier}.json")
            PROVENANCE[f"cutoff/{c}/{tier}"] = ("constraint applied" if payload and payload["provenance"].get("fine_tuned_rebuilt_with_constraint")
                                                else "evaluated before the constraint fix")
            ys.append(100 * afe(payload).mean() if complete(payload) else np.nan)
        ax.plot(cuts, ys, "o-", color=color, lw=2.0, ms=6, label=f"{label}, $\\Delta$ model")
        floor = load(EVAL / "pbe_d4_avdz_densityphase" / f"{tier}.json")
        if floor:
            ax.axhline(100 * afe(floor, "df_floor").mean(), color=color, ls="--", lw=1.1, label=f"{label}, DF floor")
        s = sad_mean(f"{tier}_pbe_d4_avdz") if SHOW_SAD else None
        if s is not None:
            ax.axhline(s, color=color, ls=":", lw=1.5, label=f"{label}, SAD prior")
    ax.set_ylabel(PCT)
    log_pct_axis(ax, 0.03, 30 if SHOW_SAD else 1)
    prod = load(ENERGY / f"water_pbe_d4_avdz{ENERGY_SUFFIX}" / "summary.json")
    for ax, key, ylabel, (lo, hi) in ((axes[1], "energy_mae_per_water_kcal_mol", E_WATER, (0.01, 3.0)),
                                      (axes[2], "force_mae_kcal_mol_A", F_MAE, (0.1, 4.0))):
        for tier, color in (("id_test", TIER_COLORS["id_test"]), ("ood_size", TIER_COLORS["ood_size"])):
            ys = []
            for c in CUTOFF_HEADS:
                summ = load(ENERGY / f"water_pbe_d4_avdz_cutoff_{c}{CUTOFF_ENERGY_SUFFIX}" / "summary.json")
                PROVENANCE[f"cutoff_energy/{c}"] = f"scored ({CUTOFF_STEPS} steps)" if summ else "missing"
                ys.append(summ["by_split"][tier][key] if summ else np.nan)
            ax.plot(CUTOFF_HEADS, ys, "o-", color=color, lw=2.0, ms=6)
            if CUTOFF_ENERGY_SUFFIX:  # the 40k-step heads these runs were continued from, for the budget dependence
                ys40 = [(load(ENERGY / f"water_pbe_d4_avdz_cutoff_{c}" / "summary.json") or {}).get("by_split", {}).get(tier, {}).get(key, np.nan)
                        for c in CUTOFF_HEADS]
                ax.plot(CUTOFF_HEADS, ys40, "s:", color=color, lw=1.2, ms=4, alpha=0.6)
            if prod:
                ax.plot([6], [prod["by_split"][tier][key]], "o", mfc="white", mec=color, ms=8, mew=1.5)
        ax.set_ylabel(ylabel)
        log_axis(ax, lo, hi)
    for ax, letter in zip(axes, "abc"):
        ax.set_xlim(3.6, 8.4)
        ax.set_xticks(cuts)
        ax.set_xlabel(r"Cutoff $r_{\mathrm{c}}$ / $\mathrm{\AA}$")
        ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top", fontsize=fs(12))
    from matplotlib.lines import Line2D

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], ls="", marker="o", mfc="white", mec="0.3", ms=8, mew=1.5))
    labels.append(r"6 $\mathrm{\AA}$ production energy head (100k steps)")
    handles.append(Line2D([], [], ls="-", marker="o", color="0.3", ms=6, lw=2.0))
    labels.append(f"sweep energy heads ({CUTOFF_STEPS} steps)")
    if CUTOFF_ENERGY_SUFFIX:
        handles.append(Line2D([], [], ls=":", marker="s", color="0.3", ms=4, lw=1.2, alpha=0.6))
        labels.append("same heads at 40k steps")
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=fs(8.5), bbox_to_anchor=(0.5, 0.0), numpoints=1)
    save(fig, "cutoff_sweep_fullref")


# ----------------------------------------------------------------------------- F5b: LDA exchange and von Weizsäcker
def fig_density_functionals() -> None:
    """Mean |error| (%) of E_x^LDA and T_vW per tier for Δ, direct, DF and SAD, both functionals."""
    groups = TIERS + [(MALON, "Malon PT")]
    models = {"pbe_d4_avdz": ("pbe", "direct_pbe", "malon_pbe_d4_avdz", "malon_direct_pbe_d4_avdz"),
              "wb97mv_def2tzvpd": ("wb97mv", "direct_wb97mv", "malon_wb97mv_def2tzvpd", "malon_direct_wb97mv_def2tzvpd")}
    arms = [("delta", r"$\Delta$ model", DELTA_COLOR, None), ("direct", "Direct ML", DIRECT_COLOR, "//"),
            ("df", "DF projection", "0.55", None), ("sad", "SAD prior", "0.85", "..")]
    quantities = [("ex_lda_err_pct", r"$|\Delta E_x^{\mathrm{LDA}}|$ / %"), ("tvw_err_pct", r"$|\Delta T_{\mathrm{vW}}|$ / %")]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), sharex=True)
    rows = []
    w = 0.2
    for ci, (theory, title) in enumerate(THEORIES.items()):
        d_name, r_name, md_name, mr_name = models[theory]
        for gi, (tier, label) in enumerate(groups):
            # the SAD-only CPU pass stands in for DF and SAD until the GPU pass (models + ∇ρ) lands
            payload = dfunc(f"{tier}_{theory}") or load(DFUNC_SAD_CPU / f"{tier}_{theory}.json")
            if not payload:
                continue
            frames = payload["frames"]
            is_malon = tier == MALON
            picks = {"delta": [f["models"].get(md_name if is_malon else d_name) for f in frames],
                     "direct": [f["models"].get(mr_name if is_malon else r_name) for f in frames],
                     "df": [f.get("df") for f in frames],
                     "sad": [next(iter(f["sad"].values())) if f.get("sad") else None for f in frames]}
            for ri, (key, ylabel) in enumerate(quantities):
                ax = axes[ri, ci]
                for ai, (arm, name, color, hatch) in enumerate(arms):
                    vals = [abs(r[key]) for r in picks[arm] if r and key in r]
                    if not vals:
                        continue
                    ax.bar(gi + (ai - 1.5) * w, np.mean(vals), w, color=color if arm != "direct" else "white",
                           edgecolor=color if arm != "sad" else "0.4", hatch=hatch, lw=1.2,
                           label=name if gi == 0 else None)
                    rows.append((theory, label, key, arm, float(np.mean(vals)),
                                 float(np.mean([abs(r[key.replace("_pct", "_kcal")]) for r in picks[arm]
                                                if r and key.replace("_pct", "_kcal") in r])), len(vals)))
    for ri, (key, ylabel) in enumerate(quantities):
        for ci, (theory, title) in enumerate(THEORIES.items()):
            ax = axes[ri, ci]
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xticks(range(len(groups)), [g[1].replace(" cutouts", "") for g in groups], fontsize=fs(9))
            ax.set_xlim(-0.6, len(groups) - 0.4)
            if ri == 0:
                ax.set_title(title, fontsize=fs(13))
            if ci == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=fs(10), bbox_to_anchor=(0.5, 0.0))
    save(fig, "density_functional_errors")
    lines = ["| Theory | Tier | Quantity | Density | mean abs error (%) | mean abs error (kcal/mol) | n |",
             "|---|---|---|---|---|---|---|"]
    for theory, label, key, arm, pct, kcal, n in rows:
        lines.append(f"| {theory} | {label} | {key.split('_err')[0]} | {arm} | {pct:.4g} | {kcal:.4g} | {n} |")
    (OUT / "density_functional_errors_table.md").write_text("\n".join(lines) + "\n")

# ----------------------------------------------------------------------------- F6: overlap
def fig_overlap() -> None:
    order = [("ethanol_md17_seed42", "Ethanol (MD17)"), ("ethanethiol", "Ethanethiol"), ("resorcinol", "Resorcinol"),
             ("thiophene_4mer", "Thiophene 4-mer"), ("thiophene_6mer", "Thiophene 6-mer"),
             ("thiophene_12mer", "Thiophene 12-mer"), ("water_id", "Water, ID test"), ("water_ood_size", r"Water, $n=8$-$12$")]
    d = REPO / "results/revision/overlap_v2"
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), sharey=True)
    for ax, key, xlabel, ticks in ((axes[0], "rmsd", r"Nearest-training RMSD / $\mathrm{\AA}$", [0.05, 0.1, 0.2, 0.5, 1]),
                                   (axes[1], "soap", "Nearest-training SOAP distance", [0.003, 0.01, 0.03, 0.1, 0.3])):
        for i, (stem, _) in enumerate(order):
            arr = np.load(d / f"{stem}.npz")
            if f"{key}_test" not in arr.files:
                ax.text(0.03, i, "No training structure of this size", fontsize=fs(9), color="0.4", va="center",
                        transform=ax.get_yaxis_transform())
                continue
            for vals, dy, color, lab in ((arr[f"{key}_base"], -0.17, "0.45", r"Train $\rightarrow$ train"),
                                         (arr[f"{key}_test"], 0.17, "#1f5fa8", r"Test $\rightarrow$ train")):
                q = np.percentile(vals, [10, 25, 50, 75, 90])
                ax.plot([q[0], q[4]], [i + dy] * 2, color=color, lw=1.0)
                ax.plot([q[1], q[3]], [i + dy] * 2, color=color, lw=5, solid_capstyle="butt", label=lab if i == 0 else None)
                ax.plot(q[2], i + dy, "o", color="white", mec=color, ms=4.5)
        ax.set_xscale("log")
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(range(len(order)), [lab for _, lab in order])
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=fs(10), bbox_to_anchor=(0.6, 1.0))
    save(fig, "train_test_overlap")


# ----------------------------------------------------------------------------- F7: timing
TIMING = REPO / "results/revision/timing_v2/dedicated_gpu/runs"
# Stage-resolved DenSNet timings with the force backward timed separately
# (benchmark_step_timing.py densnet_parts, 2026-09-28 split run).
TIMING_SPLIT = REPO / "results/revision/timing_v2/split/runs"
SYS_ORDER = ["ethanol", "resorcinol", "water_n6", "water_n12", "thiophene_6mer", "thiophene_12mer"]
SPLIT_ORDER = ["ethanol", "resorcinol", "water_n6", "water_n12", "thiophene_6mer", "water_n24", "thiophene_12mer",
               "water_n30", "water_4x30"]
PRETTY = {"ethanol": "Ethanol", "resorcinol": "Resorcinol", "water_n6": r"(H$_2$O)$_6$",
          "water_n12": r"(H$_2$O)$_{12}$", "thiophene_6mer": "Thiophene 6-mer", "thiophene_12mer": "Thiophene 12-mer",
          "water_n24": r"(H$_2$O)$_{24}$ ice", "water_n30": r"(H$_2$O)$_{30}$", "water_4x30": r"4$\times$(H$_2$O)$_{30}$"}


def _runs(root: Path = TIMING):
    runs = {}
    for f in root.glob("*.json"):
        r = json.loads(f.read_text())
        runs[(r["system"], r["method"], r["device"])] = r
    return runs


def fig_timing() -> None:
    runs = _runs()
    natoms = {s: next(r["n_atoms"] for (sy, _, _), r in runs.items() if sy == s and "n_atoms" in r) for s in SYS_ORDER}
    styles = [("densnet", dict(color="black", lw=2.6, marker="o", label="DenSNet, energy + forces")),
              ("densnet_dipole", dict(color="black", lw=1.3, ls="--", marker="o", mfc="white", label="DenSNet, energy + forces + dipole")),
              ("so3lr", dict(color="#2ca02c", lw=1.3, marker="o", label="SO3LR")),
              ("maceoff", dict(color="#d62728", lw=1.3, marker="o", label="MACE-OFF")),
              ("macepolar", dict(color="#9467bd", lw=1.3, marker="o", label="MACE-POLAR-1-M")),
              ("aimnet2", dict(color="#8c564b", lw=1.3, marker="o", label="AIMNet2")),
              ("gfn2xtb", dict(color="#7f7f7f", lw=1.3, marker="^", label="GFN2-xTB")),
              ("gxtb", dict(color="#bcbd22", lw=1.3, marker="^", label="g-xTB")),
              ("dftb", dict(color="#17becf", lw=1.3, marker="v", label="DFTB3/3ob")),
              ("dft_gpu", dict(color="#1f77b4", lw=2.2, marker="D", label="DFT (GPU4PySCF)")),
              ("aimd_paper", dict(color="#1f77b4", lw=1.0, ls=":", marker="x", label="Paper's AIMD calculator"))]
    have_cpu = all((s, "densnet", "cpu") in runs for s in SYS_ORDER)  # partial CPU scans are not shown
    fig, axes = plt.subplots(1, 2 if have_cpu else 1, figsize=(12.0 if have_cpu else 9.0, 6.0), sharey=True, squeeze=False)
    for ax, device in zip(axes[0], ("cuda", "cpu")):
        for m, st in styles:
            pts = sorted((natoms[s], runs[(s, m, device)]) for s in SYS_ORDER if (s, m, device) in runs and "median_s" in runs[(s, m, device)])
            if not pts:
                continue
            x = np.array([p[0] for p in pts])
            y = np.array([p[1]["median_s"] for p in pts])
            err = [y - np.array([p[1]["p25_s"] for p in pts]), np.array([p[1]["p75_s"] for p in pts]) - y]
            ax.errorbar(x, y, yerr=err, ms=5, capsize=2, **st)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([9, 14, 18, 36, 44, 86], ["9", "14", "18", "36", "44", "86"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(7.5, 105)
        ax.set_ylim(4e-3, 3e3)
        ax.set_xlabel("Number of atoms")
        ax.set_title("GPU (A100)" if device == "cuda" else "CPU (16 cores)", fontsize=fs(13))
    axes[0][0].set_ylabel(r"Wall time per step / s")
    # GPU-only methods (DFT, AIMD) appear only in the first panel: collect every panel's entries
    seen = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=4, fontsize=fs(9.5), bbox_to_anchor=(0.5, 0.0))
    save(fig, "calculations_timing_scan")


def fig_timing_parts() -> None:
    """Stacked per-stage DenSNet cost vs system size, GPU and CPU (Δ models)."""
    runs = _runs(TIMING_SPLIT) or _runs()
    stages = [("density_network", "Density network (forward)", DELTA_COLOR),
              ("energy_forward", "Energy network (forward)", "#2ca02c"),
              ("force_backward", "Forces (backward through both networks)", "#98df8a"),
              ("energy_forces", "Energy + forces (not split)", "#2ca02c"),
              ("dipole", "Dipole", "#d62728"),
              ("other", "Neighbor list, SAD lookup, output", "#c7c7c7")]
    devices = [d for d in ("cuda", "cpu") if any(k[1] == "densnet_parts" and k[2] == d for k in runs)]
    fig, axes = plt.subplots(1, len(devices), figsize=(6.2 * len(devices), 4.8), squeeze=False)
    table = ["| System | atoms | device | " + " | ".join(s[0] for s in stages) + " | sum (ms) |", "|---" * (len(stages) + 4) + "|"]
    for ax, device in zip(axes[0], devices):
        rows, labels = [], []
        for s in SPLIT_ORDER:
            r = runs.get((s, "densnet_parts", device))
            if not r or "parts" not in r:
                continue
            p = {k: v["median_s"] for k, v in r["parts"].items()}
            p["other"] = sum(p.pop(k, 0.0) for k in ("inputs", "sad_prior", "output"))
            rows.append(p)
            labels.append(f"{PRETTY.get(s, s)} ({r['n_atoms']})")
            table.append(f"| {s} | {r['n_atoms']} | {device} | " + " | ".join(
                f"{1e3 * p[k]:.1f}" if k in p else "—" for k, _, _ in stages) + f" | {1e3 * sum(p.values()):.0f} |")
        y = np.arange(len(rows))
        left = np.zeros(len(rows))
        for key, name, color in stages:
            vals = np.array([row.get(key, 0.0) for row in rows]) * 1e3
            if not vals.any():
                continue
            ax.barh(y, vals, left=left, color=color, edgecolor="white", lw=0.5, label=name)
            left += vals
        for yi, lab in zip(y, labels):
            if lab.startswith(("Ethanol", "Resorcinol")):
                ax.text(left[yi] * 1.02, yi, "grid dipole", va="center", fontsize=fs(8.5), color="#d62728")
        ax.set_yticks(y, labels, fontsize=fs(9.5))
        ax.invert_yaxis()
        ax.set_xlim(0, max(left) * 1.18 if len(left) else 1)
        ax.set_xlabel(f"{'GPU (A100)' if device == 'cuda' else 'CPU (16 cores)'} time per step / ms")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=fs(9.5), bbox_to_anchor=(0.5, 0.0))
    save(fig, "densnet_step_breakdown")
    (OUT / "densnet_step_breakdown_table.md").write_text("Median ms per stage\n\n" + "\n".join(table) + "\n")

# ----------------------------------------------------------------------------- F8: equilibria and spectra
def fig_equilibrium() -> None:
    d = REPO / "results/revision/equilibrium_v2"
    reports = {m: load(d / m / "report.json") for m in ("ethanol", "ethanethiol", "resorcinol")}
    reports = {m: r for m, r in reports.items() if r and r.get("conformers")}
    fig = plt.figure(figsize=(10.4, 5.0))
    gs = fig.add_gridspec(len(reports), 2, width_ratios=[1.0, 1.45])
    ax0 = fig.add_subplot(gs[:, 0])
    markers = {"ethanol": "o", "ethanethiol": "s", "resorcinol": "^"}
    colors = plt.cm.tab10(np.arange(10))
    k = 0
    for mol, rep in reports.items():
        for conf, e in rep["conformers"].items():
            f = e["frequencies"]
            fd = np.array(f["dft_cm1"])
            fm = np.array(f["densnet_cm1_paired"])
            ax0.plot(fd, fm - fd, markers[mol], ms=5.5, mfc="none", mec=colors[k % 10], mew=1.3, ls="",
                     label=f"{mol.capitalize()}, {conf.replace('_', '-')}")
            k += 1
    ax0.axhline(0, color="0.4", lw=0.8)
    ax0.set_xlabel(r"DFT harmonic frequency / cm$^{-1}$")
    ax0.set_ylabel(r"$\nu_{\mathrm{DenSNet}}-\nu_{\mathrm{DFT}}$ / cm$^{-1}$")
    ax0.set_xlim(0, 3900)
    ax0.set_xticks([0, 1000, 2000, 3000])
    eq_handles, eq_labels = ax0.get_legend_handles_labels()
    grid = np.linspace(0, 4000, 4001)

    def lorentz(c, h, fwhm=12.0):
        g = fwhm / 2
        return (np.asarray(h)[:, None] * g ** 2 / ((grid[None] - np.asarray(c)[:, None]) ** 2 + g ** 2)).sum(0)

    for row, (mol, rep) in enumerate(reports.items()):
        ax = fig.add_subplot(gs[row, 1])
        conf = next(iter(rep["conformers"]))
        f = rep["conformers"][conf]["frequencies"]
        sd = lorentz(f["dft_cm1"], f["dft_ir_km_mol"])
        sm = lorentz(f["densnet_cm1_paired"], f["densnet_ir_km_mol_paired"])
        norm = max(sd.max(), sm.max())
        ax.plot(grid, sd / norm, color="black", lw=1.3, label="DFT")
        ax.plot(grid, sm / norm, color="#1f5fa8", lw=1.1, ls="--", label="DenSNet")
        ax.set_yticks([])
        ax.set_ylim(0, 1.7)
        ax.set_xlim(400, 3900)
        ax.set_ylim(0, 1.15)
        ax.text(0.38, 0.93, f"{mol.capitalize()}, {conf.replace('_', '-')}", transform=ax.transAxes, fontsize=fs(10), va="top")
        if row == 0:
            ax.legend(fontsize=fs(9), loc="upper right")
        if row == len(reports) - 1:
            ax.set_xlabel(r"Wavenumber / cm$^{-1}$")
        else:
            ax.set_xticklabels([])
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(eq_handles, eq_labels, loc="lower center", ncol=len(eq_labels), fontsize=fs(9), numpoints=1,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.3)
    save(fig, "equilibrium_frequencies")


# ----------------------------------------------------------------------------- E1-E5: energy and force errors
# The energy-head counterparts of F1-F4, on the same axes: water on d/w (same CVs, same training-extent unit,
# same source_index join), malonaldehyde on δ, extreme modes on the displacement RMSD, and the tier bars.
# Scores come from score_energy_forces.py / score_malon_energy.py (results/revision/energy_forces_v3/<name>/).
# ENERGY_SUFFIX names the score set: "_interim" while the heads are still training, "" for the final heads.
ENERGY = REPO / "results/revision/energy_forces_v3"
# "_final": the 100k-step v3 heads of 17 Oct; "_cluster" / "_interim": earlier, superseded score sets.
ENERGY_SUFFIX = __import__("os").environ.get("ENERGY_SUFFIX", "_final")
WATER_SPLITS = {"id_test": "water_id_test", "ood_size": "water_ood_size", "ood_order": "water_ood_order",
                "ood_density": "water_ood_density"}
# ε_E = |E_pred - E_ref| per structure, ε_F = force MAE per Cartesian component: the energy counterparts of ε_ρ.
E_WATER = r"$\varepsilon_E/N_\mathrm{w}$ / kcal mol$^{-1}$"
E_MOL = r"$\varepsilon_E$ / kcal mol$^{-1}$"
F_MAE = r"$\varepsilon_F$ / kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$"
_LABEL_CACHE: dict[str, dict] = {}


def energy_rows(name: str) -> list[dict] | None:
    """per_frame.csv of one energy score set, numeric columns as floats; None if absent."""
    import csv

    path = ENERGY / f"{name}{ENERGY_SUFFIX}" / "per_frame.csv"
    if not path.is_file():
        PROVENANCE[f"energy/{name}"] = "missing"
        return None
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k != "split":
                r[k] = float(v)
    summ = load(path.parent / "summary.json") or {}
    PROVENANCE[f"energy/{name}"] = f"{len(rows)} frames; {summ.get('provenance', {}).get('checkpoint', '')}"
    return rows


def water_source_index(split: str, theory: str, index: int) -> int:
    """Frame number in the geometry file (what the CV files are numbered over) of a labels-file row."""
    key = f"{split}_{theory}"
    if key not in _LABEL_CACHE:
        _LABEL_CACHE[key] = np.load(REPO / f"datasets/revision/water_labels/water_{split}_{theory}_base.npy",
                                    allow_pickle=True).item()
    data = _LABEL_CACHE[key]
    return int(data["source_index"][index]) if "source_index" in data else int(index)


def water_energy_dw(rows: list[dict], theory: str, value, tag: str) -> list:
    """(tier, d/w, value) for water energy rows: the d/w of fig_error_vs_distance, joined on source_index."""
    sel = SELECTED / f"energy_{tag}"
    sel.mkdir(parents=True, exist_ok=True)
    for split, tier in WATER_SPLITS.items():
        recs = [{"index": int(r["index"]), "source_index": int(r.get("source_index", water_source_index(split, theory, int(r["index"])))),
                 "afe": float(value(r))} for r in rows if r["split"] == split]
        if recs:
            (sel / f"{tier}.json").write_text(json.dumps({"records": recs}))
    return series_from_eval(CV / "water_train.json", sel, CV)


def log_axis(ax, lo: float, hi: float) -> None:
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [t for t in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1e3, 1e4, 1e5, 1e6) if lo <= t <= hi]
    if hi / lo > 1e5:  # wide ranges: decades only
        ticks = [t for t in ticks if np.isclose(np.log10(t), round(np.log10(t)))]
    ax.set_yticks(ticks, [f"{t:g}" if t < 1e4 else f"$10^{{{int(round(np.log10(t)))}}}$" for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def lin_axis(ax, values, q: float = 99.0, pad: float = 1.15) -> int:
    """Linear y from 0 to the q-th percentile of the plotted values (padded); returns the number of points above it."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    top = float(np.percentile(v, q)) * pad if v.size else 1.0
    ax.set_ylim(0, top)
    return int((v > top).sum())


ENERGY_FITS: dict[str, dict] = {}


def binned_median(x, y, nbins: int = 8):
    """Median of y in equal-count bins of x (the trend line drawn over the single frames)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 8:
        return np.array([]), np.array([])
    order = np.argsort(x)
    groups = np.array_split(order, min(nbins, x.size // 4))
    return np.array([np.median(x[g]) for g in groups]), np.array([np.median(y[g]) for g in groups])


def fig_energy_vs_distance() -> None:
    """E1: water energy (per water) and force errors against d/w, Δ filled and direct open, as F1."""
    names = {"id_test": "ID test", "ood_size": "Size OOD", "ood_order": "Ice", "ood_density": "Droplets"}
    quantities = [("E", lambda r: abs(r["energy_error_kcal_mol"]) / r["n_water"], E_WATER),
                  ("F", lambda r: r["force_mae_kcal_mol_A"], F_MAE)]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4), sharex=True, sharey="row")
    plotted = {0: [], 1: []}
    for col, (theory, title) in enumerate(THEORIES.items()):
        delta, direct = energy_rows(f"water_{theory}"), energy_rows(f"water_direct_{theory}")
        for row, (qname, value, ylabel) in enumerate(quantities):
            ax = axes[row, col]
            if delta:
                series = water_energy_dw(delta, theory, value, f"{theory}_{row}")
                for tier, dist, val in series:
                    ax.scatter(dist, val, s=16, alpha=0.55, color=TIER_COLORS[tier], label=names[tier], zorder=3)
                    plotted[row] += list(val)
                params = draw_saturation_fit(ax, series)  # the same ε0 + σ²(1 - e^{-x/ℓ}) as Response Fig. (density)
                if params and params["length_scale"] >= 11.9:  # ℓ ran to the end of the sweep: no saturation found
                    ax.lines[-1].set_linestyle("--")
                    ax.text(0.97, 0.04, r"$\ell$ at fit bound", transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=fs(9))
                if params:
                    ENERGY_FITS[f"{qname}/{theory}"] = params
                    PROVENANCE[f"energy_fit/{qname}/{theory}"] = {k: round(float(v), 4) for k, v in params.items()}
            if direct:
                series_d = water_energy_dw(direct, theory, value, f"direct_{theory}_{row}")
                for tier, dist, val in series_d:
                    ax.scatter(dist, val, s=16, alpha=0.5, facecolors="none", edgecolors=TIER_COLORS[tier], lw=0.8, zorder=3)
                    plotted[row] += list(val)
                params_d = fit_series_saturation(series_d) if series_d else None
                if params_d:
                    ENERGY_FITS[f"{qname}/direct_{theory}"] = params_d
                    PROVENANCE[f"energy_fit/{qname}/direct_{theory}"] = {k: round(float(v), 4) for k, v in params_d.items()}
            ax.set_xlim(0, 8)
            if row == 0:
                ax.set_title(title, fontsize=fs(14))
            else:
                ax.set_xlabel(r"$d/w$")
            if col == 0:
                ax.set_ylabel(ylabel)
    for row in (0, 1):
        clipped = lin_axis(axes[row, 0], plotted[row])
        PROVENANCE[f"energy_vs_distance/clipped_row{row}"] = clipped
    (OUT / "energy_saturation_fits.json").write_text(json.dumps(
        {k: {kk: float(vv) for kk, vv in v.items()} for k, v in ENERGY_FITS.items()}, indent=1) + "\n")
    from matplotlib.lines import Line2D

    handles, labels = axes[1, 0].get_legend_handles_labels()  # force panel: its fit line is always solid
    handles += [Line2D([], [], ls="", marker="o", mfc="none", mec="0.3")]
    labels += ["Direct ML (open)"]
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=fs(10), scatterpoints=1, numpoints=1,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=2.0)
    save(fig, "energy_force_vs_distance_water")


def fig_energy_vs_coordination() -> None:
    """E5: water force and energy errors against the number of waters inside the cutoff of each water."""
    from experiments.jacs_revision.analysis.energy_training_dashboard import water_neighbours

    names = {"id_test": "ID test", "ood_size": "Size OOD", "ood_order": "Ice", "ood_density": "Droplets"}
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True, sharey="row")
    for col, (theory, title) in enumerate(THEORIES.items()):
        for prefix, filled in (("", True), ("direct_", False)):
            rows = energy_rows(f"water_{prefix}{theory}")
            if not rows:
                continue
            nb = np.array([water_neighbours(r["split"], theory, int(r["index"]), 6.0)[0] for r in rows])
            for row, value in enumerate((lambda r: abs(r["energy_error_kcal_mol"]) / r["n_water"],
                                         lambda r: r["force_mae_kcal_mol_A"])):
                ax = axes[row, col]
                y = np.array([value(r) for r in rows])
                for split, tier in WATER_SPLITS.items():
                    m = np.array([r["split"] == split for r in rows])
                    if not m.any():
                        continue
                    t = tier.replace("water_", "")
                    kw = dict(color=TIER_COLORS[t], alpha=0.55) if filled else dict(facecolors="none", edgecolors=TIER_COLORS[t], alpha=0.5, lw=0.8)
                    ax.scatter(nb[m], y[m], s=16, zorder=3, label=names[t] if filled else None, **kw)
                if filled:
                    bx, by = binned_median(nb, y)
                    ax.plot(bx, by, color="0.12", lw=2.6, zorder=4, label=r"$\Delta$, binned median")
    for col, (theory, title) in enumerate(THEORIES.items()):
        for row, ylabel in enumerate((E_WATER, F_MAE)):
            ax = axes[row, col]
            ax.axvline(5.0, color="0.35", ls="--", lw=1.1, zorder=2, label="training maximum" if row == col == 0 else None)
            if col == 0:
                lin_axis(ax, [y for c in ax.collections for y in c.get_offsets()[:, 1]] +
                         [y for c in axes[row, 1].collections for y in c.get_offsets()[:, 1]])
            if row == 0:
                ax.set_title(title, fontsize=fs(14))
            else:
                ax.set_xlabel(r"waters within $r_c$, per water")
            if col == 0:
                ax.set_ylabel(ylabel)
    from matplotlib.lines import Line2D

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], ls="", marker="o", mfc="none", mec="0.3")]
    labels += ["Direct ML (open)"]
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=fs(9.5), scatterpoints=1, numpoints=1,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
    save(fig, "energy_force_vs_coordination_water")


def fig_malon_energy() -> None:
    """E2: malonaldehyde energy and force errors along δ, as F3 (energy heads exist for the Δ models only)."""
    cv = load(CV / "malon_ood_proton_transfer_pt.json")["frames"]
    delta = {int(f.get("index", i)): float(f["delta_pt"]) for i, f in enumerate(cv)}
    # independent linear y scales: the ωB97M-V transition-state errors are several times the PBE-D4 ones
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True)
    for col, (theory, title) in enumerate(THEORIES.items()):
        rows = energy_rows(f"malon_{theory}")
        rows = [r for r in rows or [] if r["split"] == "ood_proton_transfer" and int(r["source_index"]) in delta]
        x = np.array([delta[int(r["source_index"])] for r in rows])
        for row, (value, ylabel, lim) in enumerate(((lambda r: abs(r["energy_error_kcal_mol"]), E_MOL, None),
                                                    (lambda r: r["force_mae_kcal_mol_A"], F_MAE, None))):
            ax = axes[row, col]
            ax.axvspan(-1.6, -0.40, color="#4c72b0", alpha=0.12, zorder=0)
            ax.axvline(-0.40, color="#4c72b0", ls="--", lw=1.0)
            ax.axvline(0.0, color="0.35", ls=":", lw=1.0)
            if rows:
                y = np.array([value(r) for r in rows])
                well = x <= -0.40
                ax.scatter(x[well], y[well], s=18, color=DELTA_COLOR, alpha=0.8, label=r"$\Delta$, in training well", zorder=3)
                ax.scatter(x[~well], y[~well], s=18, color="#55a868", alpha=0.8, label=r"$\Delta$, held out", zorder=3)
            ax.set_xlim(-1.5, 0.85)
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(title, fontsize=fs(13))
            else:
                ax.set_xlabel(r"$\delta$ / $\mathrm{\AA}$")
            if col == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=fs(9.5), scatterpoints=1, bbox_to_anchor=(0.5, 0.0))
    save(fig, "malon_energy_force_vs_delta")


def fig_extreme_mode_energy() -> None:
    """E3: energy and force errors of the extreme-mode frames against the displacement RMSD, as F4.

    Only systems with an energy head at the label's level of theory: the water dimer and hexamer (water heads)
    and malonaldehyde. Energies are per structure; water heads remove the atomic reference energies, the
    malonaldehyde heads the constant offset measured on their training sample."""
    panels = [("water_n2", "Water dimer"), ("water_n6", "Water hexamer"), ("malonaldehyde", "Malonaldehyde")]
    theories = (("pbe_d4_avdz", DELTA_COLOR), ("wb97mv_def2tzvpd", DIRECT_COLOR))
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4), sharex=True)

    def rung_medians(x, y):  # a NaN prediction (diverged head) counts as an infinite error, not as missing
        y = np.where(np.isfinite(y), y, np.inf)
        rungs = np.unique(np.round(x, 1))
        return rungs, np.array([np.median(y[np.round(x, 1) == r]) for r in rungs])

    scale = {}  # (row, col) -> finite Δ / reference medians of rungs with >= 10 frames: the linear y range

    def track(row, col, x, y):
        rungs, med = rung_medians(x, y)
        scale.setdefault((row, col), []).extend(
            float(m) for r, m in zip(rungs, med) if (np.round(x, 1) == r).sum() >= 10 and np.isfinite(m))

    for col, (system, title) in enumerate(panels):
        manifest = load(REPO / "datasets/revision/ood_modes" / system / "manifest.json")
        rmsd = [r["rmsd"] for r in manifest["records"]] if manifest else []
        for theory, color in theories:
            variants = ([("", f"modes_water_{theory}"), ("direct_", f"modes_water_direct_{theory}")] if system.startswith("water")
                        else [("", f"malon_{theory}")])
            for prefix, name in variants:
                rows = [r for r in energy_rows(name) or [] if r["split"] == f"ood_modes_{system}"]
                if not rows or not rmsd:
                    continue
                x = np.array([rmsd[int(r["source_index"])] for r in rows])
                for row, value in enumerate((lambda r: abs(r["energy_error_kcal_mol"]), lambda r: r["force_mae_kcal_mol_A"])):
                    ax = axes[row, col]
                    y = np.array([value(r) for r in rows])
                    if not prefix:
                        track(row, col, x, y)
                        ax.scatter(x, y, s=8, facecolors="none", edgecolors=color, alpha=0.35, lw=0.6)
                        ax.plot(*rung_medians(x, y), color=color, lw=2.0)
                        if row == 0:  # deformation energy: the error of predicting the lowest frame's energy for all
                            ref = np.array([r.get("energy_ref_rel_kcal_mol", r.get("energy_ref_kcal_mol")) for r in rows])
                            ax.plot(*rung_medians(x, ref - ref.min()), color=color, lw=1.2, ls=":")
                            track(row, col, x, ref - ref.min())
                    else:
                        ax.plot(*rung_medians(x, y), color=color, lw=1.6, ls="-.")
        for row, ylabel in enumerate((E_MOL, F_MAE)):
            ax = axes[row, col]
            ax.set_xscale("log")
            med = scale.get((row, col), [])
            ax.set_ylim(0, 1.2 * max(med) if med else 1.0)  # sparse rungs (< 10 frames) and single outliers can lie above
            ax.set_xlim(0.35, 14)
            ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if row == 0:
                ax.set_title(title, fontsize=fs(12))
            else:
                ax.set_xlabel(r"RMSD / $\mathrm{\AA}$")
            if col == 0:
                ax.set_ylabel(ylabel)
    from matplotlib.lines import Line2D

    legend_items = [Line2D([], [], color=DELTA_COLOR, lw=2.0, label="PBE-D4"),
                    Line2D([], [], color=DIRECT_COLOR, lw=2.0, label=r"$\omega$B97M-V"),
                    Line2D([], [], color="0.2", lw=2.0, label=r"$\Delta$ model, median"),
                    Line2D([], [], color="0.2", lw=1.6, ls="-.", label="Direct ML, median"),
                    Line2D([], [], color="0.2", lw=1.2, ls=":", label=r"$E_\mathrm{ref}-E_\mathrm{ref,min}$, median"),
                    Line2D([], [], ls="", marker="o", mfc="none", mec="0.4", label=r"$\Delta$, single frames")]
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(handles=legend_items, loc="lower center", ncol=3, fontsize=fs(9.5), numpoints=1, bbox_to_anchor=(0.5, 0.0))
    save(fig, "ood_extreme_mode_energy_force")


def pooled_force_mae(rows: list[dict]) -> float:
    """Force MAE over every Cartesian component of a tier (summary.json's convention), not a mean of per-frame MAEs."""
    if all("force_abs_sum" in r and "n_atoms" in r for r in rows):
        return float(sum(r["force_abs_sum"] for r in rows) / sum(3 * r["n_atoms"] for r in rows))
    return float(np.mean([r["force_mae_kcal_mol_A"] for r in rows]))


def energy_tier_values(split: str, theory: str, prefix: str) -> tuple[float, float, float, int] | None:
    """(|ΔE| per water, the same after removing the tier's mean signed error, pooled force MAE, n) for one water model."""
    rows = [r for r in energy_rows(f"water_{prefix}{theory}") or [] if r["split"] == split]
    if not rows:
        return None
    per_water = np.array([r["energy_error_kcal_mol"] / r["n_water"] for r in rows])
    return (float(np.abs(per_water).mean()), float(np.abs(per_water - per_water.mean()).mean()),
            pooled_force_mae(rows), len(rows))


def fig_water_tiers_main() -> None:
    """Main-text water figure: ε_ρ, ε_E per water and ε_F per tier, Δ vs direct, PBE-D4 vs ωB97M-V, water only.

    Same bars as pbe_vs_hybrid_bars and energy_force_tier_bars without the malonaldehyde group, all on log axes so
    the in-distribution energies (~0.01 kcal/mol per water) stay visible. The black tick on each energy bar is the
    error after removing the tier's mean signed error, which is most of the out-of-distribution energy error."""
    w = 0.19
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    density_tier_bars(axes[0], TIERS, w)
    splits = list(WATER_SPLITS)
    table = {}
    for gi, split in enumerate(splits):
        for ai, (theory, prefix, name, face, edge) in enumerate(ARMS):
            v = energy_tier_values(split, theory, prefix)
            if v is None:
                continue
            table[(split, prefix + theory)] = v
            x = gi + (ai - 1.5) * w
            for ax, val in ((axes[1], v[0]), (axes[2], v[2])):
                ax.bar(x, val, w, color=face, edgecolor=edge, lw=1.3, hatch="//" if prefix else None)
            axes[1].plot([x - w / 2, x + w / 2], [v[1]] * 2, color="black", lw=2.0, solid_capstyle="butt",
                         label="bias removed" if gi == 0 and ai == 0 else None, zorder=4)
    for ax, ylabel, (lo, hi) in ((axes[1], E_WATER, (0.003, 3.0)), (axes[2], F_MAE, (0.1, 4.0))):
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, len(TIERS) - 0.4)
        log_axis(ax, lo, hi)
    for ax in axes:  # three panels side by side: tier names on two lines
        ax.set_xticks(range(len(TIERS)), ["ID\ntest", "Size\nOOD", "Ice\ncutouts", "Droplets"])
    for ax, letter in zip(axes, "abc"):
        ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top", fontsize=fs(12))
    handles, labels = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.legend(handles + h1, labels + l1, loc="lower center", ncol=7, fontsize=fs(9.5), handlelength=1.8,
               columnspacing=1.1, bbox_to_anchor=(0.5, 0.0))
    save(fig, "water_tiers_density_energy_force")
    lines = ["| Tier | " + " | ".join(a[2].replace("$\\Delta$", "Δ").replace(r"$\omega$", "ω") for a in ARMS) + " |",
             "|---" * (len(ARMS) + 1) + "|"]
    for split, (_, label) in zip(splits, TIERS):
        cells = []
        for theory, prefix, *_ in ARMS:
            v = table.get((split, prefix + theory))
            cells.append("—" if v is None else f"{v[0]:.3f} ({v[1]:.3f}) / {v[2]:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (OUT / "water_energy_force_table.md").write_text(
        "|ΔE| per water (bias removed), kcal/mol / force MAE over all Cartesian components, kcal/mol/Å; score set "
        f"'{ENERGY_SUFFIX or 'final'}'\n\n" + "\n".join(lines) + "\n")


def fig_energy_tier_bars() -> None:
    """E4: energy (per water / per molecule) and force MAE per tier, Δ vs direct, PBE vs ωB97M-V, as F2."""
    groups = [("id_test", "ID\ntest"), ("ood_size", "Size\nOOD"), ("ood_order", "Ice"), ("ood_density", "Droplets"),
              ("ood_proton_transfer", "Malon.\nPT")]
    arms = [("pbe_d4_avdz", "", "PBE-D4, $\\Delta$", "#aec7e8", DELTA_COLOR),
            ("pbe_d4_avdz", "direct_", "PBE-D4, direct", "white", DELTA_COLOR),
            ("wb97mv_def2tzvpd", "", r"$\omega$B97M-V, $\Delta$", "#ff9896", DIRECT_COLOR),
            ("wb97mv_def2tzvpd", "direct_", r"$\omega$B97M-V, direct", "white", DIRECT_COLOR)]
    w = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    table = {}
    for gi, (split, label) in enumerate(groups):
        for ai, (theory, prefix, name, face, edge) in enumerate(arms):
            rows = energy_rows(f"malon_{theory}") if split == "ood_proton_transfer" and not prefix else \
                (None if split == "ood_proton_transfer" else energy_rows(f"water_{prefix}{theory}"))
            rows = [r for r in rows or [] if r["split"] == split]
            if not rows:
                continue
            e = float(np.mean([abs(r["energy_error_kcal_mol"]) / r.get("n_water", 1.0) for r in rows]))
            f = pooled_force_mae(rows)
            table[(split, prefix + theory)] = (e, f, len(rows))
            for ax, val in zip(axes, (e, f)):
                ax.bar(gi + (ai - 1.5) * w, val, w, color=face, edgecolor=edge, lw=1.3, hatch="//" if prefix else None,
                       label=name if gi == 0 else None)
    for ax, ylabel in zip(axes, (r"$\varepsilon_E/N_\mathrm{mol}$ / kcal mol$^{-1}$", F_MAE)):
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(groups)), [g[1] for g in groups])
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=fs(9.5), bbox_to_anchor=(0.5, 0.0))
    save(fig, "energy_force_tier_bars")
    lines = ["| Tier | " + " | ".join(a[2].replace("$\\Delta$", "Δ").replace(r"$\omega$", "ω") for a in arms) + " |",
             "|---" * (len(arms) + 1) + "|"]
    for split, label in groups:
        cells = []
        for theory, prefix, *_ in arms:
            v = table.get((split, prefix + theory))
            cells.append("—" if v is None else f"{v[0]:.3f} / {v[1]:.2f} (n={v[2]})")
        lines.append(f"| {label.replace(chr(10), ' ')} | " + " | ".join(cells) + " |")
    (OUT / "energy_force_tier_table.md").write_text(
        "|ΔE| per water (malonaldehyde: per molecule), kcal/mol / force MAE, kcal/mol/Å; score set "
        f"'{ENERGY_SUFFIX or 'final'}'\n\n" + "\n".join(lines) + "\n")


for _name in ("energy_force_vs_distance_water", "energy_force_vs_coordination_water", "malon_energy_force_vs_delta",
              "ood_extreme_mode_energy_force", "energy_force_tier_bars"):
    COPY_TO[_name] = [LETTER_FIGS]
# The main-text water figures of the revised paper (Section 2.5).
PAPER_WATER = REPO / "docs/equiv-ml-paper/figures/water"
COPY_TO["water_tiers_density_energy_force"] = [PAPER_WATER, LETTER_FIGS]
COPY_TO["error_vs_distance_water_pbe_hybrid"].append(PAPER_WATER)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for fn in (fig_error_vs_distance, fig_pbe_vs_hybrid, fig_malon, fig_extreme_modes, fig_ntrain, fig_cutoff,
               fig_density_functionals, fig_overlap, fig_timing, fig_timing_parts, fig_equilibrium,
               fig_energy_vs_distance, fig_energy_vs_coordination, fig_malon_energy, fig_extreme_mode_energy,
               fig_energy_tier_bars, fig_water_tiers_main):
        if only and fn.__name__ not in only:
            continue
        try:
            fn()
        except Exception as exc:  # keep going; report which figure failed
            print(f"{fn.__name__} failed: {exc!r}")
    prov = {**(load(OUT / "provenance.json") or {}), **PROVENANCE} if only else PROVENANCE  # a subset run keeps the rest
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
