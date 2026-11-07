#!/usr/bin/env python3
"""Dipole magnitude statistics vs polythiophene oligomer size (R2.5, for_kieron §8).

Scans paper/trajectories/polythiophene/dipole/ for *_dpm.npy files and reports
mean |mu|, std, and dipole vector component statistics grouped by mer count.
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/dipole_vs_mer.py'


import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.jacs_revision.analysis.mpl_classic_tex import apply as apply_classic_tex, save_response_figure


_MER_RE = re.compile(r"(\d+)mer")


def _mer_from_name(name: str) -> int | None:
    m = _MER_RE.search(name)
    return int(m.group(1)) if m else None


def _load_dipoles(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True).item()
    return np.asarray(data["dipole_moment"], dtype=np.float64)


def analyze_dir(dipole_dir: Path) -> dict:
    by_mer: dict[int, list[dict]] = {}
    for path in sorted(dipole_dir.glob("*_dpm.npy")):
        if "_dpm_intor" in path.name:
            continue
        mer = _mer_from_name(path.name)
        if mer is None:
            continue
        dipole = _load_dipoles(path)
        magnitudes = np.linalg.norm(dipole, axis=1)
        rec = {
            "file": path.name,
            "n_frames": int(dipole.shape[0]),
            "mu_mean_debye": float(magnitudes.mean()),
            "mu_std_debye": float(magnitudes.std()),
            "mu_min_debye": float(magnitudes.min()),
            "mu_max_debye": float(magnitudes.max()),
            "mu_per_mer_debye": float(magnitudes.mean() / mer),
            "mu_over_sqrt_n_debye": float(magnitudes.mean() / np.sqrt(mer)),
        }
        by_mer.setdefault(mer, []).append(rec)

    summary_mers = []
    for mer in sorted(by_mer):
        replicas = by_mer[mer]
        means = [r["mu_mean_debye"] for r in replicas]
        summary_mers.append(
            {
                "mer": mer,
                "n_replicas": len(replicas),
                "mu_mean_debye": float(np.mean(means)),
                "mu_std_across_replicas_debye": float(np.std(means)),
                "mu_per_mer_debye": float(np.mean(means) / mer),
                "mu_over_sqrt_n_debye": float(np.mean(means) / np.sqrt(mer)),
                "replicas": replicas,
            }
        )

    return {"dipole_dir": str(dipole_dir), "by_mer": summary_mers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dipole-dir",
        type=Path,
        default=Path("paper/trajectories/polythiophene/dipole"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/revision/dipole_vs_mer"))
    args = parser.parse_args()

    summary = analyze_dir(args.dipole_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    apply_classic_tex()
    mers = [m["mer"] for m in summary["by_mer"]]
    mu_means = [m["mu_mean_debye"] for m in summary["by_mer"]]
    mu_errs = [m["mu_std_across_replicas_debye"] for m in summary["by_mer"]]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.errorbar(mers, mu_means, yerr=mu_errs, fmt="o-", capsize=3)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$|\boldsymbol{\mu}|$ / D")
    fig.tight_layout()
    save_response_figure(fig, args.out / "dipole_vs_mer.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
