#!/usr/bin/env python3
"""Same-frame force comparison of the two ethanol checkpoints (provenance audit of 96w7KyGG).

Question: does 2024-03-22_96w7KyGG agree poorly with DFT forces because of the checkpoint, or because of how the
current code loads it (flag defaults it predates), or because of the reference basis? Only existing data are used:
the reference forces are the stored PBE/aug-cc-pVDZ labels of the 2026 relabel
(datasets/ethanol_dft_pyscf_ccpvdz_train.npy; despite its name, 900 published ML-MD frames at aug-cc-pVDZ, see
docs/audits/ethanol_checkpoint_provenance_2026-10-20.md). No DFT is run.

Discrimination (Pearson r over all Cartesian force components; scale-free, so the missing output normalization
does not enter):
  * 5K9Ne7it_ext vs aug-cc-pVDZ labels -> size of the cc-pVDZ/aug-cc-pVDZ basis effect for a working checkpoint;
  * 96w7KyGG vs the same labels -> if far below the 5K9Ne7it value, the basis does not explain it;
  * 96w7KyGG vs 5K9Ne7it on the same frames -> model-vs-model agreement, independent of any reference;
  * 96w7KyGG under flag overrides (--set key=value) -> whether a default it predates changes r.

Each run writes per-frame forces and input hashes, so variants are compared on identical frames.

    python experiments/jacs_revision/analysis/ethanol_loader_comparison.py --checkpoint 96w7KyGG \
        --frames 20 --threads 2 --out results/revision/ethanol_loader_comparison/96w7KyGG_current.json
    ... --set density_coeffs=False ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REFERENCE = REPO / "datasets/ethanol_dft_pyscf_ccpvdz_train.npy"
REFERENCE_SHA256 = "d06a6775ed8da02896c4cf8fa79c0127df39fc3fcf0ed113d43259bd34071bf6"
CHECKPOINTS = {
    "96w7KyGG": {"run": "paper/models/ethanol/2024-03-22_96w7KyGG",
                 "latest_sha256": "debed70f1b9adbe63d934b451252b6b907e6469b01a0592235bc4938c5f3e388",
                 # the published MD settings (config/md/nn/ethanol.txt)
                 "atom_dens_path": "datasets/free_atom_densities_augccpvdz_augccpvqzjkfit.npy",
                 "atom_dens_type": "spline"},
    "5K9Ne7it": {"run": "paper/models/ethanol/2024-04-18_5K9Ne7it_ext",
                 "latest_sha256": "b50a31282e2d117acbc97ef15f6376ca80b5ec80da135d8907583038c028a716",
                 # the validated R2.11 settings (config/md/nn/ethanol_ext_paths.txt)
                 "atom_dens_path": "datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy",
                 "atom_dens_type": "mo_coeffs"},
}
HARTREE_EV = 27.211386245988


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_frames(n_total: int, n: int, seed: int) -> np.ndarray:
    """Deterministic frame indices, sorted, identical for every variant."""
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


def force_stats(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Pearson r, through-origin slope and MAE over all Cartesian components."""
    p, r = np.asarray(pred, float).ravel(), np.asarray(ref, float).ravel()
    slope = float(p @ r / (r @ r))
    return {"pearson_r": float(np.corrcoef(p, r)[0, 1]), "slope_through_origin": slope,
            "mae_raw": float(np.abs(p - r).mean()), "mae_after_slope": float(np.abs(p / slope - r).mean()),
            "n_components": int(p.size)}


def load_reference(frames: int, seed: int):
    data = np.load(REFERENCE, allow_pickle=True).item()
    idx = select_frames(len(data["positions"]), frames, seed)
    pos = [np.asarray(data["positions"][i], float) for i in idx]
    z = [np.asarray(data["atom_numbers"][i], int) for i in idx]
    # stored PySCF forces are Hartree/Angstrom (labeler convention); convert to eV/Angstrom
    f = [np.asarray(data["forces"][i], float) * HARTREE_EV for i in idx]
    return idx, z, pos, f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--set", action="append", default=[], help="override a loaded arg, key=value (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="check inputs and hashes only; no model is built")
    opt = ap.parse_args()

    spec = CHECKPOINTS[opt.checkpoint]
    ref_hash = sha256(REFERENCE)
    if ref_hash != REFERENCE_SHA256:
        raise SystemExit(f"reference file changed: {ref_hash}")
    ckpt_hash = sha256(REPO / spec["run"] / "checkpoints/latest_checkpoint.pth")
    if ckpt_hash != spec["latest_sha256"]:
        raise SystemExit(f"checkpoint changed: {ckpt_hash}")
    idx, z, pos, ref = load_reference(opt.frames, opt.seed)
    report = {"checkpoint": spec["run"], "latest_sha256": ckpt_hash, "reference": str(REFERENCE.relative_to(REPO)),
              "reference_sha256": ref_hash, "reference_level": "PBE/aug-cc-pVDZ (2026 relabel of ML-MD frames)",
              "frame_indices": idx.tolist(), "seed": opt.seed, "overrides": opt.set,
              "atom_dens_path": spec["atom_dens_path"], "atom_dens_type": spec["atom_dens_type"]}
    if opt.dry_run:
        opt.out.parent.mkdir(parents=True, exist_ok=True)
        opt.out.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report, indent=1))
        return 0

    import torch
    from ase import Atoms

    torch.set_num_threads(opt.threads)
    sys.path[:0] = [str(REPO), str(REPO / "src")]
    from equiv_dens.md.dft_network_calculator import load_densnet_calculator

    # Overrides go through args_file, which the loader reads first. Pickled checkpoint args then overwrite every
    # key except the data/prior paths and the weight-inferred switches, so prior settings always apply; a --set key
    # that the pickled args also hold is reported as overridden-back (see report["set_effective"]).
    import tempfile

    lines = [ln for ln in (REPO / spec["run"] / "args.txt").read_text().splitlines()
             if not ln.startswith(("--atom_dens_path", "--atom_dens_type"))]
    lines += [f"--atom_dens_path={REPO / spec['atom_dens_path']}", f"--atom_dens_type={spec['atom_dens_type']}"]
    lines += [f"--{item}" for item in opt.set]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        args_file = tmp.name
    calc = load_densnet_calculator(REPO / spec["run"], args_file=args_file, use_gpu=False, repo=REPO, pyscf_grid=False)
    args = getattr(calc, "args", None)
    report["set_effective"] = {item.split("=", 1)[0]: (str(getattr(args, item.split("=", 1)[0], "<unknown>"))
                                                       if args else "<calculator exposes no args>")
                               for item in opt.set}
    pred = []
    for zi, ri in zip(z, pos):
        atoms = Atoms(numbers=zi, positions=ri)
        atoms.calc = calc
        pred.append(np.asarray(atoms.get_forces(), float))
    report["per_frame_forces_eV_A"] = [p.tolist() for p in pred]
    report["stats_vs_reference"] = force_stats(np.array(pred), np.array(ref))
    report["loaded_args_sample"] = ({k: getattr(args, k, None) for k in
                                     ("cutoff", "num_en_basis_functions", "L0_start", "append_atom_density",
                                      "density_coeffs", "normalize_en", "atom_dens_type")} if args else None)
    opt.out.parent.mkdir(parents=True, exist_ok=True)
    opt.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["stats_vs_reference"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
