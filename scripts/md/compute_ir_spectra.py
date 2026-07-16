#!/usr/bin/env python3
"""
Compute IR spectra from ML-MD HDF5 trajectories via dipole autocorrelation function.

Loads dipole moments from HDF5, computes C(t) = <δμ(0)·δμ(t)>, takes FFT,
and outputs intensity vs wavenumber (cm^-1). Saves per-replica spectra and
optionally averaged spectra.

Usage:
  python scripts/md/compute_ir_spectra.py --md-dir results/polythiophene_100ps_10rep_1fs/md_logs/2024-03-23_1XDL67zp_ext
  python scripts/md/compute_ir_spectra.py --md-dir scratch/md_logs/2024-03-23_1XDL67zp_ext --out-dir results/ir_spectra_new
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Constants
SPEED_OF_LIGHT_CM_S = 2.99792458e10  # cm/s
DEBYE_TO_ESU_CM = 1e-18  # Debye to e*cm for intensity scaling


def load_dipoles_from_hdf5(hdf5_path: Path, skip_initial: int = 0) -> tuple[np.ndarray, float]:
    """
    Load dipole moments and time step from HDF5. Returns (dipole [n_steps, 3], time_step_fs).
    """
    with h5py.File(hdf5_path, "r") as f:
        if "properties" not in f:
            raise ValueError("No 'properties' group")
        props = f["properties"]
        pos = json.loads(props.attrs["positions"])
        if "dipole_moment" not in pos:
            raise ValueError("dipole_moment not in properties")
        start, stop = pos["dipole_moment"]
        raw = props[skip_initial:, :, start:stop]
        dipole = np.asarray(raw).reshape(raw.shape[0], -1, 3)
        time_step_fs = 0.5
        if "molecules" in f and "time_step" in f["molecules"].attrs:
            time_step_fs = float(f["molecules"].attrs["time_step"]) * 1000
    return dipole, time_step_fs


def dipole_acf(dipole: np.ndarray) -> np.ndarray:
    """Compute normalized dipole fluctuation autocorrelation C(t) = <δμ(0)·δμ(t)> / <δμ·δμ>."""
    # dipole: (n_steps, n_replicas, 3) or (n_steps, 3)
    if dipole.ndim == 3:
        dipole = dipole[:, 0, :]  # use first replica
    mu = dipole - dipole.mean(axis=0)
    n = len(mu)
    # C(t) = sum over t' of mu(t')·mu(t'+t) / (n-t)
    acf = np.zeros(n)
    for lag in range(n):
        if lag == 0:
            acf[lag] = np.sum(mu * mu) / n
        else:
            acf[lag] = np.sum(mu[:-lag] * mu[lag:]) / (n - lag)
    # Normalize
    if acf[0] > 0:
        acf /= acf[0]
    return acf


def acf_to_ir_spectrum(acf: np.ndarray, dt_fs: float, apply_window: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    FFT of ACF to get IR intensity vs wavenumber.
    Returns (wavenumber_cm1, intensity).
    """
    n = len(acf)
    if apply_window:
        window = np.hanning(n)
        acf = acf * window
    # Real FFT (symmetric ACF)
    spec = np.fft.rfft(acf).real
    freqs_hz = np.fft.rfftfreq(n, d=dt_fs * 1e-15)
    wavenumber = freqs_hz / SPEED_OF_LIGHT_CM_S
    # IR intensity ~ omega^2 * |FT(C)|, use wavenumber^2 for proportionality
    intensity = (wavenumber ** 2) * np.maximum(spec, 0)
    return wavenumber, intensity


def discover_hdf5_files(md_dir: Path) -> list[tuple[int, int, Path]]:
    """
    Find HDF5 files matching simulation_scaling_{n}mer_100ps_rep{r}.hdf5 or similar.
    Returns list of (n_mer, replica_idx, path).
    """
    found = []
    for p in md_dir.glob("simulation_*.hdf5"):
        m = re.search(r"scaling_(\d+)mer.*rep(\d+)", p.stem)
        if m:
            found.append((int(m.group(1)), int(m.group(2)), p))
        else:
            m2 = re.search(r"scaling_(\d+)mer", p.stem)
            if m2 and "rep" not in p.stem:
                found.append((int(m2.group(1)), 0, p))
    return sorted(found, key=lambda x: (x[0], x[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute IR spectra from ML-MD HDF5 trajectories")
    parser.add_argument(
        "--md-dir",
        type=Path,
        default=Path("results/polythiophene_100ps_10rep_1fs/md_logs/2024-03-23_1XDL67zp_ext"),
        help="Directory containing simulation HDF5 files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: md_dir/../ir_spectra)",
    )
    parser.add_argument("--skip-initial", type=int, default=0, help="Skip first N steps (equilibration)")
    parser.add_argument("--max-wavenumber", type=float, default=4000, help="Max wavenumber (cm^-1)")
    parser.add_argument("--plot", action="store_true", help="Save PNG plots")
    parser.add_argument("--no-window", action="store_true", help="Disable Hann window on ACF")
    args = parser.parse_args()

    md_dir = args.md_dir.resolve()
    if not md_dir.exists():
        print(f"Error: {md_dir} does not exist")
        sys.exit(1)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = md_dir.parent / "ir_spectra"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_hdf5_files(md_dir)
    if not files:
        print(f"No HDF5 files found in {md_dir}")
        sys.exit(1)

    print(f"Found {len(files)} trajectory files")
    all_spectra = {}  # n_mer -> list of (wavenumber, intensity) or averaged

    for n_mer, rep, hdf5_path in files:
        try:
            dipole, dt_fs = load_dipoles_from_hdf5(hdf5_path, args.skip_initial)
        except Exception as e:
            print(f"  Skip {hdf5_path.name}: {e}")
            continue

        # Handle multiple replicas in same file
        n_replicas = dipole.shape[1]
        for r in range(n_replicas):
            dip_r = dipole[:, r : r + 1, :]
            acf = dipole_acf(dip_r)
            wavenumber, intensity = acf_to_ir_spectrum(acf, dt_fs, apply_window=not args.no_window)

            # Trim to positive wavenumbers
            mask = (wavenumber >= 0) & (wavenumber <= args.max_wavenumber)
            wavenumber = wavenumber[mask]
            intensity = intensity[mask]

            rep_key = rep if n_replicas == 1 else rep * n_replicas + r
            rep_dir = out_dir / f"{n_mer}mer" / f"replica_{rep_key}"
            rep_dir.mkdir(parents=True, exist_ok=True)

            # Save CSV
            csv_path = rep_dir / "ir_spectrum.csv"
            np.savetxt(
                csv_path,
                np.column_stack([wavenumber, intensity]),
                header="frequency_cm,intensity",
                comments="",
                delimiter=",",
            )
            print(f"  {n_mer}mer rep{rep_key}: {csv_path}")

            # Save npz
            np.savez_compressed(
                rep_dir / "ir_spectrum.npz",
                frequency_cm=wavenumber,
                intensity=intensity,
                n_steps=acf.shape[0],
                time_step_fs=dt_fs,
            )

            if n_mer not in all_spectra:
                all_spectra[n_mer] = []
            all_spectra[n_mer].append((wavenumber, intensity))

            if args.plot:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.fill_between(wavenumber, intensity, alpha=0.5)
                ax.plot(wavenumber, intensity, linewidth=1)
                ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
                ax.set_ylabel("Intensity (arb. units)")
                ax.set_title(f"Polythiophene {n_mer}-mer, replica {rep_key}")
                ax.set_xlim(0, args.max_wavenumber)
                ax.set_ylim(bottom=0)
                fig.tight_layout()
                fig.savefig(rep_dir / "ir_spectrum.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

    # Averaged spectrum per n-mer
    for n_mer, spectra in all_spectra.items():
        if len(spectra) < 2:
            continue
        # Interpolate to common grid (use finest grid)
        w_max = max(s[0].max() for s in spectra)
        w_min = max(s[0].min() for s in spectra)
        n_pts = min(len(s[0]) for s in spectra)
        w_common = np.linspace(w_min, w_max, n_pts)
        ints_interp = np.array([np.interp(w_common, s[0], s[1]) for s in spectra])
        int_mean = ints_interp.mean(axis=0)
        int_std = ints_interp.std(axis=0)

        avg_dir = out_dir / f"{n_mer}mer" / "averaged"
        avg_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            avg_dir / "ir_spectrum.csv",
            np.column_stack([w_common, int_mean, int_std]),
            header="frequency_cm,intensity_mean,intensity_std",
            comments="",
            delimiter=",",
        )
        np.savez_compressed(
            avg_dir / "ir_spectrum.npz",
            frequency_cm=w_common,
            intensity_mean=int_mean,
            intensity_std=int_std,
            n_replicas=len(spectra),
        )
        print(f"  {n_mer}mer averaged ({len(spectra)} replicas): {avg_dir / 'ir_spectrum.csv'}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
