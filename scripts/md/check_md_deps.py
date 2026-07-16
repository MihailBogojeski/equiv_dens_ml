#!/usr/bin/env python3
"""
Pre-flight checks for ML-MD and MACE-OFF thiophene scaling runs.

Verifies equiv_dens_ml layout, required files, and optionally runs minimal
test runs. Run from equiv_dens_ml directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ML-MD dependencies (relative to equiv_dens_ml)
ML_MD_FILES = [
    "paper/models/polythiophene/2024-03-23_1XDL67zp_ext/checkpoints/latest_checkpoint.pth",
    "datasets/thiophene_all_train_d4.npy",
    "datasets/augccpvqzjkfit_orbital_basis_df.npy",
    "datasets/augccpvqzjkfit_radial_coeffs_df.npy",
    "datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy",
    "datasets/atomization_energy_augccpvdz.npy",
]

ML_MD_CONFIG = "config/md/nn/polythiophene_1mer.txt"
ML_MD_INIT = "datasets/thiophene1mer_init.npy"


def check_ml_md_deps(root: Path) -> tuple[bool, list[str]]:
    """Check ML-MD dependencies. Returns (all_ok, list of missing paths)."""
    missing = []
    for rel in ML_MD_FILES:
        p = root / rel
        if not p.exists():
            missing.append(str(rel))
    return len(missing) == 0, missing


def check_mace_off_deps(root: Path) -> tuple[bool, str | None]:
    """Check MACE-OFF: mace-torch import. Returns (ok, error_message)."""
    try:
        from mace.calculators import mace_off  # noqa: F401
        return True, None
    except ImportError as e:
        return False, f"mace-torch not installed: {e}"


def run_minimal_ml_md(root: Path) -> tuple[bool, str]:
    """Run minimal ML-MD (20 steps). Returns (ok, output_or_error)."""
    init_path = root / ML_MD_INIT
    if not init_path.exists():
        return False, f"Init file not found: {ML_MD_INIT}. Run generate_thiophene_nmer_inits.py first."

    cmd = [
        sys.executable,
        "run.py",
        "md",
        f"@{root / ML_MD_CONFIG}",
        "--np_dataset_test",
        ML_MD_INIT,
        "--md_steps",
        "20",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            env={**__import__("os").environ, "WANDB_MODE": "disabled"},
        )
        if result.returncode != 0:
            return False, result.stderr or result.stdout or f"exit {result.returncode}"
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Timeout (120s)"
    except Exception as e:
        return False, str(e)


def run_minimal_mace_off(root: Path) -> tuple[bool, str]:
    """Run minimal MACE-OFF (10 steps). Returns (ok, output_or_error)."""
    init_path = root / ML_MD_INIT
    if not init_path.exists():
        return False, f"Init file not found: {ML_MD_INIT}. Run generate_thiophene_nmer_inits.py first."

    script = root / "scripts" / "md" / "mace_off_md_run.py"
    cmd = [
        sys.executable,
        str(script),
        "--structure",
        str(init_path),
        "--output",
        str(root / "scratch" / "test_mace.traj"),
        "--steps",
        "10",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return False, result.stderr or result.stdout or f"exit {result.returncode}"
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Timeout (60s)"
    except Exception as e:
        return False, str(e)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-flight checks for ML-MD and MACE-OFF thiophene runs."
    )
    parser.add_argument(
        "--ml-md",
        action="store_true",
        help="Check ML-MD dependencies only",
    )
    parser.add_argument(
        "--mace-off",
        action="store_true",
        help="Check MACE-OFF dependencies only",
    )
    parser.add_argument(
        "--test-ml-md",
        action="store_true",
        help="Run minimal ML-MD test (20 steps)",
    )
    parser.add_argument(
        "--test-mace-off",
        action="store_true",
        help="Run minimal MACE-OFF test (10 steps)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT,
        help="equiv_dens_ml root (default: auto-detect)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not (root / "run.py").exists():
        print(f"Error: {root} does not look like equiv_dens_ml (no run.py)", file=sys.stderr)
        return 1

    all_ok = True
    check_anything = args.ml_md or args.mace_off or args.test_ml_md or args.test_mace_off

    if not check_anything:
        args.ml_md = True
        args.mace_off = True

    if args.ml_md:
        ok, missing = check_ml_md_deps(root)
        if ok:
            print("ML-MD: all required files present")
        else:
            print("ML-MD: MISSING files:", file=sys.stderr)
            for m in missing:
                print(f"  - {m}", file=sys.stderr)
            all_ok = False

    if args.mace_off:
        ok, err = check_mace_off_deps(root)
        if ok:
            print("MACE-OFF: mace-torch available")
        else:
            print(f"MACE-OFF: {err}", file=sys.stderr)
            all_ok = False

    if args.test_ml_md:
        print("Running minimal ML-MD test (20 steps)...")
        ok, out = run_minimal_ml_md(root)
        if ok:
            print("ML-MD test: PASSED")
        else:
            print("ML-MD test: FAILED", file=sys.stderr)
            if out:
                print(out[:2000], file=sys.stderr)
                if len(out) > 2000:
                    print("...", file=sys.stderr)
            all_ok = False

    if args.test_mace_off:
        print("Running minimal MACE-OFF test (10 steps)...")
        ok, out = run_minimal_mace_off(root)
        if ok:
            print("MACE-OFF test: PASSED")
        else:
            print("MACE-OFF test: FAILED", file=sys.stderr)
            if out:
                print(out[:2000], file=sys.stderr)
                if len(out) > 2000:
                    print("...", file=sys.stderr)
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
