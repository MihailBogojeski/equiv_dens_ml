#!/usr/bin/env python3
"""Is a DenSNet energy head local and additive? Two non-interacting clusters as the test.

A model whose energy is a sum of local atomic terms gives, for clusters A and B placed far beyond the
cutoff, E(A+B) = E(A) + E(B) and forces on A that do not depend on B at all. layer_norm in the interaction
blocks normalizes each feature over (atoms, m, channels) of the whole structure, so every atom's features,
and its energy, depend on the statistics of every other atom; then neither identity holds. Two copies of the
same cluster leave those statistics unchanged and so cannot reveal it; two different clusters do.

    .venv/bin/python -B experiments/jacs_revision/analysis/energy_head_locality.py \
        --run results/revision/water_pbe_d4_avdz_energy_ft/<run> --theory pbe_d4_avdz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--theory", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--out", type=Path)
    opt = ap.parse_args()

    import torch

    from equiv_dens.md.dft_network_calculator import geometry_to_model_inputs, load_densnet_calculator
    from equiv_dens.utils import orbitals

    run = opt.run.resolve()
    meta = torch.load(run / "checkpoints/latest_checkpoint.pth", map_location="cpu", weights_only=False)
    args = meta["args"]
    calc = load_densnet_calculator(run, np_dataset=str(REPO / args.np_dataset), repo=REPO, use_gpu=opt.device == "cuda")
    model = calc.model
    model.eval()
    model.no_force_props = [p for p in model.no_force_props if p == "energy"]
    device = "cuda" if calc.use_gpu else "cpu"
    norm_flags = {k: getattr(args, k) for k in vars(args) if "normal" in k.lower()}

    def predict(Z, P):
        Z = np.asarray(Z, dtype=np.int64)[None]
        inputs = geometry_to_model_inputs(Z, torch.tensor(np.asarray(P)[None], dtype=torch.float32, device=device),
                                          calc.cutoff, calc.use_gpu)
        if calc.remove_atom_density and calc.atom_dens is not None and calc.atom_dens_type in {"mo_coeffs", "df_coeffs"}:
            key = "atom_" + calc.atom_dens_type
            inputs[key], inputs[key + "_occ"], inputs[key + "_basis"] = orbitals.join_atom_coeffs(
                torch.as_tensor(Z, dtype=torch.long), calc.atom_dens, calc.atom_dens_type)
            for k in (key, key + "_occ", key + "_basis"):
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(device)
        res = model(inputs)
        return float(res["energy"].detach().cpu().reshape(-1)[0]), res["forces"].detach().cpu().numpy().reshape(-1, 3)

    data = np.load(REPO / f"datasets/revision/water_labels/water_id_test_{opt.theory}_base.npy", allow_pickle=True).item()
    nw = (data["atom_numbers"] == 8).sum(1)

    def cluster(i):
        keep = data["atom_numbers"][i] > 0
        return data["atom_numbers"][i][keep], np.asarray(data["positions"][i], float)[keep]

    pairs = [(int(np.flatnonzero(nw == a)[0]), int(np.flatnonzero(nw == b)[0])) for a, b in ((2, 6), (3, 5), (4, 4))]
    pairs[2] = (int(np.flatnonzero(nw == 4)[0]), int(np.flatnonzero(nw == 4)[1]))
    rows = []
    for ia, ib in pairs:
        za, pa = cluster(ia)
        zb, pb = cluster(ib)
        ea, fa = predict(za, pa)
        eb, _ = predict(zb, pb)
        for sep in (12.0, 30.0, 100.0):
            shift = np.array([sep + np.ptp(pa[:, 0]) + np.ptp(pb[:, 0]), 0, 0]) - pb.mean(0) + pa.mean(0)
            z = np.concatenate([za, zb])
            p = np.vstack([pa, pb + shift])
            e, f = predict(z, p)
            # locality: move one atom of B and watch the forces on A
            p2 = p.copy()
            p2[len(za)] += [0.05, 0, 0]
            _, f2 = predict(z, p2)
            rows.append({"A": f"{nw[ia]} waters", "B": f"{nw[ib]} waters", "separation_A": sep,
                         "E_A_plus_E_B": ea + eb, "E_AB": e, "nonadditivity_kcal_mol": e - (ea + eb),
                         "max_force_change_on_A_vs_A_alone": float(np.abs(f[:len(za)] - fa).max()),
                         "max_force_change_on_A_when_B_atom_moves": float(np.abs(f2[:len(za)] - f[:len(za)]).max())})
    # duplicate of one cluster: same statistics, so additive even with structure-wide normalization
    za, pa = cluster(pairs[0][1])
    ea, _ = predict(za, pa)
    e2, _ = predict(np.concatenate([za, za]), np.vstack([pa, pa + [100.0, 0, 0]]))
    # rotation: E invariant, F equivariant (F(R x) = R F(x))
    rot = []
    rng = np.random.default_rng(0)
    for i in sorted({i for pair in pairs for i in pair}):
        z, p = cluster(i)
        e0, f0 = predict(z, p)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.linalg.det(q))
        c = p.mean(0)
        e1, f1 = predict(z, (p - c) @ q.T + c)
        rot.append({"n_water": int(nw[i]), "energy_change_kcal_mol": e1 - e0,
                    "max_force_equivariance_error": float(np.abs(f1 - f0 @ q.T).max())})
    out = {"run": str(run), "theory": opt.theory, "cutoff_A": float(calc.cutoff), "normalization_args": norm_flags,
           "pairs": rows, "duplicate_check": {"E_2A_minus_2E_A_kcal_mol": e2 - 2 * ea}, "rotation": rot}
    print(json.dumps(out, indent=1, default=str))
    if opt.out:
        opt.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
