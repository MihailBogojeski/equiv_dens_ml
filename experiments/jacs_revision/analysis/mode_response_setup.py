#!/usr/bin/env python3
"""Geometries and per-atom Born-charge errors for visualizing one IR band's density response.

From a Tier H harmonic analysis (results/internal/polythiophene/tierH/<n>mer_mo_coeffs/harmonic.npz
and dft_min.xyz), this
  * picks the strongest DFT mode in a frequency window (default: the C=C band, 1350-1550 cm-1),
  * writes the minimum and the +/- displacements along that mode (max atom displacement --step)
    as a small trajectory for label_trajectory_frames.py,
  * writes per-atom Born tensors Z*_i = d(mu)/d(R_i) for DFT and DenSNet, split into the static
    charge (trace/3) and the longitudinal and transverse charge flux (traceless part along and
    across the chain's principal axis), plus the mode's per-atom contribution to d(mu)/dQ.

    .venv/bin/python experiments/jacs_revision/analysis/mode_response_setup.py \
        --tierh results/internal/polythiophene/tierH/8mer_mo_coeffs --out results/revision/density_movie/pt8_cc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.data import atomic_masses
from ase.io import read

CM1_PER_SQRT_EV_A2_AMU = 521.47091  # sqrt(eV / (A^2 amu)) in cm-1


def modes(hess: np.ndarray, masses: np.ndarray, pos: np.ndarray):
    """Frequencies (cm-1) and mass-weighted eigenvectors, rigid-body motions projected out."""
    n = len(masses)
    m3 = np.repeat(masses, 3)
    sq = np.sqrt(m3)
    com = (masses[:, None] * pos).sum(0) / masses.sum()
    x = pos - com
    rb = []
    for a in range(3):
        t = np.zeros((n, 3)); t[:, a] = 1; rb.append((t.ravel() * sq))
    for a in range(3):
        axis = np.zeros(3); axis[a] = 1
        rb.append((np.cross(axis, x).ravel() * sq))
    q, _ = np.linalg.qr(np.array(rb).T)
    proj = np.eye(3 * n) - q @ q.T
    hmw = hess / np.outer(sq, sq)
    w, v = np.linalg.eigh(proj @ hmw @ proj)
    freq = np.sign(w) * np.sqrt(np.abs(w)) * CM1_PER_SQRT_EV_A2_AMU
    return freq, v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tierh", type=Path, required=True)
    ap.add_argument("--window", type=float, nargs=2, default=[1350.0, 1550.0])
    ap.add_argument("--step", type=float, default=0.02, help="largest single-atom displacement (Angstrom)")
    ap.add_argument("--out", type=Path, required=True)
    opt = ap.parse_args()

    atoms = read(opt.tierh / "dft_min.xyz")
    z, pos = atoms.get_atomic_numbers(), atoms.get_positions()
    masses = atomic_masses[z]
    h = np.load(opt.tierh / "harmonic.npz")
    freq, vec = modes(h["dft_hessian"], masses, pos)
    ir = h["dft_ir"]
    # harmonic.npz stores intensities per mode in its own (ascending) order; match by frequency
    order = np.argsort(h["dft_freqs"])
    fsel = (freq > opt.window[0]) & (freq < opt.window[1])
    dmu = {k: h[f"{k}_dmu"].reshape(len(z), 3, 3) for k in ("dft", "densnet")}  # [atom, R-dir, mu-comp]

    def dmu_dq(k, kind):
        disp = (vec[:, k] / np.sqrt(np.repeat(masses, 3))).reshape(-1, 3)
        per_atom = np.einsum("ib,iba->ia", disp, dmu[kind])  # contribution of each atom to d(mu)/dQ
        return per_atom

    cand = np.where(fsel)[0]
    strength = [np.linalg.norm(dmu_dq(k, "dft").sum(0)) ** 2 for k in cand]
    k = int(cand[int(np.argmax(strength))])
    disp = (vec[:, k] / np.sqrt(np.repeat(masses, 3))).reshape(-1, 3)
    scale = opt.step / np.abs(np.linalg.norm(disp, axis=1)).max()
    dq = scale  # R(Q) = R0 + Q * disp; Q in amu^1/2 A
    geoms = np.stack([pos, pos + dq * disp, pos - dq * disp])

    # chain axis: principal axis of the heavy atoms
    heavy = pos[z > 1] - pos[z > 1].mean(0)
    u = np.linalg.svd(heavy, full_matrices=False)[2][0]

    def born(kind):
        out = []
        for zi in dmu[kind]:
            zt = zi.T  # [mu-comp, R-dir] = d mu_a / d R_b
            q = np.trace(zt) / 3
            flux = zt - q * np.eye(3)
            lon = float(u @ flux @ u)
            tr = float(np.linalg.norm(flux - lon * np.outer(u, u)))
            out.append({"static": float(q), "flux_long": lon, "flux_trans": tr})
        return out

    b_dft, b_dn = born("dft"), born("densnet")
    pa_dft, pa_dn = dmu_dq(k, "dft"), dmu_dq(k, "densnet")
    opt.out.mkdir(parents=True, exist_ok=True)
    np.save(opt.out / "geoms.npy", {"positions": geoms, "atom_numbers": z}, allow_pickle=True)
    info = {
        "tierh": str(opt.tierh), "mode_index": k, "frequency_cm1": float(freq[k]),
        "dq_amu05_A": float(dq), "max_atom_step_A": opt.step, "chain_axis": u.tolist(),
        "dmu_dq_dft": pa_dft.sum(0).tolist(), "dmu_dq_densnet": pa_dn.sum(0).tolist(),
        "intensity_ratio_densnet_over_dft": float(np.linalg.norm(pa_dn.sum(0)) ** 2 / np.linalg.norm(pa_dft.sum(0)) ** 2),
        "atoms": [{"Z": int(z[i]), "born_dft": b_dft[i], "born_densnet": b_dn[i],
                   "mode_dmu_dft": pa_dft[i].tolist(), "mode_dmu_densnet": pa_dn[i].tolist()} for i in range(len(z))],
    }
    (opt.out / "mode_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k2: info[k2] for k2 in ("mode_index", "frequency_cm1", "dq_amu05_A", "intensity_ratio_densnet_over_dft")}))
    for el, name in ((6, "C"), (16, "S"), (1, "H")):
        idx = [i for i in range(len(z)) if z[i] == el]
        a = np.mean([b_dft[i]["flux_long"] for i in idx]); b = np.mean([b_dn[i]["flux_long"] for i in idx])
        print(f"{name}: mean longitudinal flux DFT {a:+.3f}  DenSNet {b:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
