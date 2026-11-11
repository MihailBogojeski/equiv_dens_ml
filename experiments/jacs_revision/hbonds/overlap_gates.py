"""Train/test overlap gates for the H-bond new-system campaign."""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/hbonds/overlap_gates.py'


import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from experiments.jacs_revision.hbonds.collective_variables import (  # noqa: E402
    compute_frame_cvs,
    cv_names_for_system,
    delta_coordinate,
    proton_transfer_indices,
)
from experiments.jacs_revision.hbonds.common import nearest_train_rmsds  # noqa: E402
from experiments.jacs_revision.hbonds.xyz_io import read_xyz  # noqa: E402

OOD_RMSD_MEDIAN_MIN_A = 0.2
OOD_RMSD_NONE_BELOW_A = 0.2
PARACETAMOL_TRAIN_GAP_A = 0.25
PARACETAMOL_OOD_MEDIAN_MIN_A = 0.4
BC_OOD_MAX = 0.15
ACAC_TRAIN_DELTA_MAX = -0.40


@dataclass
class GateResult:
    passed: bool
    checks: dict[str, object]
    message: str


def bhattacharyya_1d(a: np.ndarray, b: np.ndarray, grid_points: int = 512) -> float:
    from scipy.stats import gaussian_kde

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    if np.std(a) < 1e-12 and np.std(b) < 1e-12:
        return 1.0 if abs(float(np.mean(a) - np.mean(b))) < 1e-9 else 0.0
    spread = max(float(np.std(a)), float(np.std(b)))
    rng = np.random.default_rng(0)
    if np.std(a) < 1e-12:
        a = a + rng.normal(0.0, 1e-3 * spread, size=len(a))
    if np.std(b) < 1e-12:
        b = b + rng.normal(0.0, 1e-3 * spread, size=len(b))
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    pad = 0.1 * (hi - lo) + 1e-9
    grid = np.linspace(lo - pad, hi + pad, grid_points)
    pa = gaussian_kde(a)(grid)
    pb = gaussian_kde(b)(grid)
    step = grid[1] - grid[0]
    pa /= max(pa.sum() * step, 1e-300)
    pb /= max(pb.sum() * step, 1e-300)
    return float(np.sqrt(pa * pb).sum() * step)


def load_cv_matrix(
    system: str,
    frames: list[tuple[list[str], np.ndarray]],
    *,
    pt_indices: dict[str, int] | None = None,
) -> np.ndarray:
    names = cv_names_for_system(system)
    rows = []
    for symbols, pos in frames:
        cvs = compute_frame_cvs(system, symbols, pos, pt_indices=pt_indices)
        rows.append([float(cvs[n]) for n in names])
    return np.asarray(rows, dtype=float)


def check_rmsd_gate(
    train_frames: list[tuple[list[str], np.ndarray]],
    ood_frames: list[tuple[list[str], np.ndarray]],
    *,
    median_min: float = OOD_RMSD_MEDIAN_MIN_A,
    none_below: float = OOD_RMSD_NONE_BELOW_A,
    heavy_only: bool = True,
) -> GateResult:
    rmsds = nearest_train_rmsds(train_frames, ood_frames, heavy_only=heavy_only)
    median = float(np.median(rmsds))
    frac_below = float(np.mean(rmsds < none_below))
    passed = median > median_min and frac_below == 0.0
    checks = {"median_min_rmsd_A": median, "frac_below_threshold": frac_below}
    msg = f"OOD RMSD median={median:.3f} A, frac<{none_below} A={frac_below:.3f}"
    return GateResult(passed, checks, msg)


def check_bc_gate(
    system: str,
    train_frames: list[tuple[list[str], np.ndarray]],
    ood_frames: list[tuple[list[str], np.ndarray]],
    *,
    bc_max: float = BC_OOD_MAX,
    pt_indices: dict[str, int] | None = None,
) -> GateResult:
    train_cv = load_cv_matrix(system, train_frames, pt_indices=pt_indices)
    ood_cv = load_cv_matrix(system, ood_frames, pt_indices=pt_indices)
    bcs = []
    for col in range(train_cv.shape[1]):
        bcs.append(bhattacharyya_1d(train_cv[:, col], ood_cv[:, col]))
    worst = float(np.nanmax(bcs)) if bcs else float("nan")
    passed = bool(np.isfinite(worst) and worst < bc_max)
    checks = {"bhattacharyya_per_cv": bcs, "worst_bc": worst}
    msg = f"worst BC={worst:.3f} (threshold {bc_max})"
    return GateResult(passed, checks, msg)


def check_nma_stretch(train_frames, ood_frames) -> GateResult:
    train_r = [compute_frame_cvs("nma_dimer", s, p)["r_ho"] for s, p in train_frames]
    ood_r = [compute_frame_cvs("nma_dimer", s, p)["r_ho"] for s, p in ood_frames]
    passed = float(np.mean(ood_r)) > float(np.mean(train_r))
    checks = {"train_mean_r_ho": float(np.mean(train_r)), "ood_mean_r_ho": float(np.mean(ood_r))}
    return GateResult(passed, checks, f"stretch mean r(HO): train={checks['train_mean_r_ho']:.3f}, ood={checks['ood_mean_r_ho']:.3f}")


def check_meoh_donor_gate(train_frames) -> GateResult:
    flags = [compute_frame_cvs("meoh_water", s, p)["donor_is_meoh"] for s, p in train_frames]
    passed = all(f > 0.5 for f in flags)
    return GateResult(passed, {"all_meoh_donor": passed}, "train is isomer A (MeOH donor)")


def check_acac_basin(train_frames, ood_frames, idx: dict[str, int] | None = None) -> GateResult:
    if idx is None:
        idx = proton_transfer_indices(train_frames[0][0], train_frames[0][1])
    train_d = [delta_coordinate(p, idx) for _, p in train_frames]
    ood_d = [delta_coordinate(p, idx) for _, p in ood_frames]
    passed = max(train_d) <= ACAC_TRAIN_DELTA_MAX and max(ood_d) > 0.0
    checks = {"train_delta_max": max(train_d), "ood_max_delta": max(ood_d)}
    return GateResult(passed, checks, f"train delta max={checks['train_delta_max']:+.3f}, ood max={checks['ood_max_delta']:+.3f}")


def check_paracetamol_gaps(train_frames, val_frames, id_frames, ood_frames) -> GateResult:
    def min_rmsd(test):
        return float(np.min(nearest_train_rmsds(train_frames, test))) if test else float("inf")

    val_min = min_rmsd(val_frames)
    id_min = min_rmsd(id_frames)
    ood_rmsds = nearest_train_rmsds(train_frames, ood_frames)
    passed = (
        val_min >= PARACETAMOL_TRAIN_GAP_A
        and id_min >= PARACETAMOL_TRAIN_GAP_A
        and float(np.median(ood_rmsds)) >= PARACETAMOL_OOD_MEDIAN_MIN_A
    )
    checks = {
        "val_min_rmsd_A": val_min,
        "id_min_rmsd_A": id_min,
        "ood_median_rmsd_A": float(np.median(ood_rmsds)),
    }
    return GateResult(passed, checks, json.dumps(checks))


def check_pgg_families(manifest: dict) -> GateResult:
    train_f = set(manifest.get("train_family_ids", []))
    ood_f = set(manifest.get("ood_family_ids", []))
    passed = bool(train_f and ood_f and train_f.isdisjoint(ood_f))
    checks = {"train_families": sorted(train_f), "ood_families": sorted(ood_f)}
    return GateResult(passed, checks, f"train families {checks['train_families']} vs OOD {checks['ood_families']}")


def check_crambin_fragments(manifest: dict, ood_frames) -> GateResult:
    frags = manifest.get("fragments", {})
    passed = True
    checks: dict[str, object] = {}
    for name, info in frags.items():
        natm = int(info["natoms"])
        ok = 40 <= natm <= 70
        checks[f"{name}_natoms"] = natm
        passed = passed and ok
    train_regions = set(manifest.get("train_region_ids", []))
    ood_regions = set(manifest.get("ood_region_ids", []))
    passed = passed and train_regions.isdisjoint(ood_regions)
    checks["train_regions"] = sorted(train_regions)
    checks["ood_regions"] = sorted(ood_regions)
    checks["ood_frames"] = len(ood_frames)
    checks["full_protein_train"] = manifest.get("full_protein_train_written", False)
    passed = passed and not checks["full_protein_train"]
    return GateResult(passed, checks, "crambin fragment proxy constraints")


def run_system_gates(system_dir: Path, system: str) -> dict:
    """Run all gates for one system directory."""
    manifest = json.loads((system_dir / "manifest.json").read_text())
    train = read_xyz(system_dir / "train.xyz")
    val = read_xyz(system_dir / "val.xyz") if (system_dir / "val.xyz").exists() else []
    id_test = read_xyz(system_dir / "id_test.xyz") if (system_dir / "id_test.xyz").exists() else []

    ood_name = manifest.get("ood_split", "ood")
    ood_path = system_dir / f"{ood_name}.xyz"
    if not ood_path.exists():
        for candidate in ("ood_proton_transfer.xyz", "ood_stretch.xyz", "ood_isomer_b.xyz", "ood_highT.xyz", "ood_conformer.xyz", "ood_region.xyz"):
            if (system_dir / candidate).exists():
                ood_path = system_dir / candidate
                break
    ood = read_xyz(ood_path)

    results: dict[str, object] = {"system": system, "checks": {}, "passed": True}
    gate_list: list[GateResult] = []

    if system == "nma_dimer":
        gate_list = [
            check_nma_stretch(train, ood),
            check_rmsd_gate(train, ood),
            check_bc_gate(system, train, ood),
        ]
    elif system == "meoh_water":
        gate_list = [
            check_meoh_donor_gate(train),
            check_rmsd_gate(train, ood, heavy_only=False),
            check_bc_gate(system, train, ood),
        ]
    elif system == "acac":
        pt_indices = manifest.get("indices")
        gate_list = [
            check_acac_basin(train, ood, pt_indices),
            check_rmsd_gate(train, ood, heavy_only=False),
            check_bc_gate(system, train, ood, pt_indices=pt_indices),
        ]
    elif system == "paracetamol":
        gate_list = [check_paracetamol_gaps(train, val, id_test, ood)]
    elif system == "pgg":
        gate_list = [check_pgg_families(manifest)]
    elif system == "crambin_frags":
        gate_list = [check_crambin_fragments(manifest, ood)]
    else:
        raise ValueError(system)

    for gate in gate_list:
        results["checks"][gate.message.split()[0] if gate.message else "gate"] = {
            "passed": gate.passed,
            "details": gate.checks,
            "message": gate.message,
        }
        if not gate.passed:
            results["passed"] = False

    out = system_dir / "gate_report.json"
    out.write_text(json.dumps(results, indent=2))
    return results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-dir", type=Path, required=True)
    parser.add_argument("--system", required=True)
    args = parser.parse_args()
    report = run_system_gates(args.system_dir, args.system)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
