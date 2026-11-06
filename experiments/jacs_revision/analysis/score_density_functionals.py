#!/usr/bin/env python3
"""Density-derived errors of model, SAD-prior and DF densities against the full SCF density.

For each structure in a label set (same manifests as rescore_full_density.py) this
reports, for the reference, the density-fitting projection, the SAD prior of the
Δ models and every selected model:

- ε_ρ = ∫|ρ − ρ_ref| / N_e (reported in %),
- the LDA exchange energy E_x = −C_x ∫ρ^{4/3} (ρ clipped at 0), and
- the von Weizsäcker kinetic energy T_vW = ⅛ ∫|∇ρ|²/ρ,

and their errors against the reference, in kcal/mol and in % of the reference
value. E_x probes the local density scale; T_vW probes density gradients, i.e.
shell structure and the cusp region. Both are evaluated on the same full level-2
Becke grid as ε_ρ.

Gradients: the reference and DF gradients come from AO derivatives (per Bohr);
the model and SAD gradients from the density head and the free-atom expansion.
Run --fd-check first: it compares every analytic gradient with central finite
differences of the same density and prints the ratio, which must be 1.

    .venv/bin/python -B experiments/jacs_revision/analysis/score_density_functionals.py \
        --manifest results/revision/rescore_fullref/manifest_pbe_water.json \
        --only water_id_test_pbe_d4_avdz --models pbe,direct_pbe --fd-check 400
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _p in (str(REPO), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.jacs_revision.analysis.rescore_full_density import (  # noqa: E402
    BOHR,
    ModelEntry,
    df_density as df_density_cpu,
    make_dataset,
    point_keys,
    sha256,
)

C_X = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)  # LDA exchange: E_x = -C_x ∫ rho^{4/3}
HARTREE_KCAL = 627.5094740631
RHO_MIN = 1e-10  # vW integrand only where the density itself is positive
REF_MIN = 1e-8   # ... and where the reference has density at all (excludes noise in empty space)


# ----------------------------------------------------------------------------- densities
def _ao_deriv1(mol, coords_bohr):
    """AO values and first derivatives on the GPU, as (4, ngrid, nao)."""
    import cupy as cp
    from gpu4pyscf.dft import numint as gnum

    ao = gnum.eval_ao(mol, cp.asarray(coords_bohr), deriv=1)
    if ao.shape[-1] != mol.nao and ao.shape[1] == mol.nao:
        ao = ao.transpose(0, 2, 1)
    if ao.shape != (4, len(coords_bohr), mol.nao):
        raise RuntimeError(f"unexpected AO layout {ao.shape} for nao={mol.nao}, ngrid={len(coords_bohr)}")
    return ao


def reference_density(dataset, idx, coords_bohr, chunk, want_grad):
    """ρ_SCF and ∇ρ_SCF (per Bohr) from the MO coefficients."""
    import cupy as cp

    mol = dataset.mols[idx]
    if not mol._built:
        mol.build(verbose=0)
    mo = np.asarray(dataset.coeffs[idx]["mo_coeff"])
    occ = np.asarray(dataset.coeffs[idx]["mo_occ"])
    if occ.ndim == 1:
        mo, occ = mo[None], occ[None]
    c_occ = cp.asarray(np.concatenate([m[:, o > 0] * np.sqrt(o[o > 0]) for m, o in zip(mo, occ)], axis=1))
    rho = np.empty(len(coords_bohr))
    grad = np.empty((len(coords_bohr), 3)) if want_grad else None
    for start in range(0, len(coords_bohr), chunk):
        stop = min(start + chunk, len(coords_bohr))
        if want_grad:
            ao = _ao_deriv1(mol, coords_bohr[start:stop])
            phi = ao[0] @ c_occ
            rho[start:stop] = cp.asnumpy((phi ** 2).sum(axis=1))
            for k in range(3):
                grad[start:stop, k] = cp.asnumpy(2.0 * (phi * (ao[k + 1] @ c_occ)).sum(axis=1))
        else:
            from gpu4pyscf.dft import numint as gnum

            phi = gnum.eval_ao(mol, cp.asarray(coords_bohr[start:stop])) @ c_occ
            rho[start:stop] = cp.asnumpy((phi ** 2).sum(axis=1))
        del phi
    return rho, grad


def df_density(dataset, idx, coords_bohr, chunk, want_grad):
    """ρ_DF and ∇ρ_DF (per Bohr) from the stored density-fitting coefficients."""
    import cupy as cp

    mol = dataset.mols[idx].copy()
    mol.basis = dataset.density_fitting["auxbasis"]
    mol.build(verbose=0)
    c = cp.asarray(np.concatenate([x[1] for x in dataset.density_fitting["df_coeffs"][idx]]))
    rho = np.empty(len(coords_bohr))
    grad = np.empty((len(coords_bohr), 3)) if want_grad else None
    for start in range(0, len(coords_bohr), chunk):
        stop = min(start + chunk, len(coords_bohr))
        if want_grad:
            ao = _ao_deriv1(mol, coords_bohr[start:stop])
            rho[start:stop] = cp.asnumpy(ao[0] @ c)
            for k in range(3):
                grad[start:stop, k] = cp.asnumpy(ao[k + 1] @ c)
        else:
            from gpu4pyscf.dft import numint as gnum

            rho[start:stop] = cp.asnumpy(gnum.eval_ao(mol, cp.asarray(coords_bohr[start:stop])) @ c)
    return rho, grad


def sad_density(entry, props, coords, want_grad):
    """SAD prior of this model on the grid: (ρ, ∇ρ as returned, tensor for the model input)."""
    out = entry.dataset.sample_atom_density(props["batch_positions"], props["batch_atom_numbers"], coords,
                                            density_grad=want_grad)
    if want_grad:
        return out[..., 0], out[..., 1:]
    return out, None


def model_density(entry, props, device, chunk, want_grad):
    """Model ρ and ∇ρ on every grid point (chunked). props carries coords and the SAD prior."""
    head = entry.model.property_models["density"]
    saved = head.density_grad
    head.density_grad = want_grad
    n_points = props["coords"].shape[1]
    keys = point_keys(props, n_points)
    rho = np.empty(n_points)
    grad = np.empty((n_points, 3)) if want_grad else None
    try:
        for start in range(0, n_points, chunk):
            stop = min(start + chunk, n_points)
            sub = {}
            for key, value in props.items():
                if key in ("density", "density_grad"):
                    continue
                if key in keys:
                    value = value[:, start:stop]
                sub[key] = value.to(device) if torch.is_tensor(value) else value
            result = entry.model(sub)
            rho[start:stop] = result["density"][0].detach().double().cpu().numpy()
            if want_grad:
                grad[start:stop] = result["density_grad"][0].detach().double().cpu().numpy().reshape(-1, 3)
            del result, sub
    finally:
        head.density_grad = saved
    return rho, grad


# ----------------------------------------------------------------------------- functionals
def functionals(rho, grad, weights, rho_ref):
    """N, LDA exchange and vW kinetic energy (Hartree) of one density on the grid."""
    out = {"n": float(weights @ rho), "ex_lda_ha": float(-C_X * (weights @ np.clip(rho, 0.0, None) ** (4.0 / 3.0)))}
    if grad is not None:
        mask = (rho > RHO_MIN) & (rho_ref > REF_MIN)
        out["tvw_ha"] = float(0.125 * np.sum(weights[mask] * np.sum(grad[mask] ** 2, axis=1) / rho[mask]))
    return out


def compare(f, f_ref, rho, rho_ref, weights, n_ref):
    """Errors of one density against the reference: ε_ρ in %, energies in kcal/mol and %."""
    out = dict(f)
    out["afe_pct"] = float(100.0 * (weights @ np.abs(rho - rho_ref)) / n_ref)
    for key in ("ex_lda_ha", "tvw_ha"):
        if key in f and key in f_ref:
            err = f[key] - f_ref[key]
            name = key[:-3]
            out[f"{name}_err_kcal"] = float(err * HARTREE_KCAL)
            out[f"{name}_err_pct"] = float(100.0 * err / abs(f_ref[key]))
    return out


# ----------------------------------------------------------------------------- checks
def fd_check(entries, sad_entries, reference, idx, props_ref, device, chunk, ref_chunk, n_points, h_bohr=1e-3):
    """Analytic ∇ρ against central differences of ρ, x direction, on n_points grid points."""
    coords = props_ref["coords"]
    shift = props_ref.get("pos_shift", 0)
    rng = np.random.default_rng(0)
    rho_ref_all, _ = reference_density(reference, idx, ((coords - (shift.to(coords) if torch.is_tensor(shift) else shift))[0]
                                                        .double().numpy() / BOHR), ref_chunk, False)
    # points where the density is appreciable but not at a nucleus
    candidates = np.where((rho_ref_all > 1e-3) & (rho_ref_all < 10.0))[0]
    pick = torch.as_tensor(rng.choice(candidates, size=min(n_points, len(candidates)), replace=False))
    sub = coords[:, pick]
    h_ang = h_bohr * BOHR
    plus, minus = sub.clone(), sub.clone()
    plus[..., 0] += h_ang
    minus[..., 0] -= h_ang
    frame = lambda c: (c - (shift.to(c) if torch.is_tensor(shift) else shift))[0].double().numpy() / BOHR
    report = {}

    def ratio(analytic_x, rp, rm):
        fd = (rp - rm) / (2.0 * h_bohr)
        return float(np.sum(analytic_x * fd) / np.sum(fd * fd)), float(np.max(np.abs(analytic_x - fd)) / np.max(np.abs(fd)))

    r0, g0 = reference_density(reference, idx, frame(sub), ref_chunk, True)
    rp, _ = reference_density(reference, idx, frame(plus), ref_chunk, False)
    rm, _ = reference_density(reference, idx, frame(minus), ref_chunk, False)
    report["reference"] = ratio(g0[:, 0], rp, rm)
    r0, g0 = df_density(reference, idx, frame(sub), ref_chunk, True)
    rp, _ = df_density(reference, idx, frame(plus), ref_chunk, False)
    rm, _ = df_density(reference, idx, frame(minus), ref_chunk, False)
    report["df"] = ratio(g0[:, 0], rp, rm)
    for entry in sad_entries:
        props = entry.dataset.get_properties([idx])
        _, g = sad_density(entry, props, sub, True)
        rp, _ = sad_density(entry, props, plus, False)
        rm, _ = sad_density(entry, props, minus, False)
        report[f"sad:{Path(str(entry.args.atom_dens_path)).name}"] = ratio(
            g[0, :, 0].double().numpy(), rp[0].double().numpy(), rm[0].double().numpy())
    for entry in entries:
        props = entry.dataset.get_properties([idx])
        vals = {}
        for tag, c in (("0", sub), ("p", plus), ("m", minus)):
            p = dict(props)
            p["coords"] = c
            p["coord_weights"] = torch.ones_like(c[..., 0])
            want = tag == "0"
            dens, dgrad = sad_density(entry, props, c, want)
            p["atom_density"] = dens
            if want:
                p["atom_density_grad"] = dgrad
            vals[tag] = model_density(entry, p, device, chunk, want)
        report[f"model:{entry.spec['name']}"] = ratio(vals["0"][1][:, 0], vals["p"][0], vals["m"][0])
    return report


# ----------------------------------------------------------------------------- driver
def score_label_set(label_set, models, out_dir, device, chunk, ref_chunk, grid_level, limit, want_grad, fd_points,
                    sad_only=False, cpu_reference=False):
    t0 = time.time()
    tmp_root = Path(tempfile.mkdtemp(prefix="dfunc_"))
    specs = [dict(s, out=str(tmp_root / f"{s['name']}.json")) for s in label_set["models"]
             if models is None or s["name"] in models]
    if not specs:
        print(f"== {label_set['name']}: no selected models, skipped", flush=True)
        return
    print(f"== {label_set['name']}: {[s['name'] for s in specs]}", flush=True)
    entries = [ModelEntry(spec, label_set, device, grid_level, tmp_root) for spec in specs]
    # one SAD prior per (file, type) among the Δ models
    sad_entries, seen = [], set()
    for entry in entries:
        key = (str(entry.args.atom_dens_path), str(entry.args.atom_dens_type))
        if getattr(entry.args, "remove_atom_density", False) and key not in seen:
            seen.add(key)
            sad_entries.append(entry)
    # CPU reference: PySCF on the grid (no gradients); GPU: AO values/derivatives with gpu4pyscf
    reference = make_dataset(entries[0].args, label_set["np"], label_set["dens"], label_set["elements"],
                             cpu_reference, grid_level, grid_only=not cpu_reference)
    out_path = out_dir / f"{label_set['name']}.json"
    frames = {}
    if out_path.is_file():
        frames = {int(f["index"]): f for f in json.loads(out_path.read_text()).get("frames", [])}
    n_total = len(reference)
    indices = list(range(n_total if not limit else min(limit, n_total)))
    model_names = [e.spec["name"] for e in entries]

    if fd_points:
        props_ref = reference.get_properties([indices[0]])
        report = fd_check(entries, sad_entries, reference, indices[0], props_ref, device, chunk, ref_chunk, fd_points)
        print("finite-difference check, x derivative (ratio analytic/FD, max rel dev):", flush=True)
        for key, (r, dev) in report.items():
            print(f"  {key:40s} ratio {r:.5f}   max rel dev {dev:.2e}", flush=True)
        return report

    for count, idx in enumerate(indices, start=1):
        done = frames.get(idx)
        if done and (sad_only or all(m in done["models"] for m in model_names)):
            continue
        ref = reference.get_properties([idx])
        coords = ref["coords"]
        weights = ref["coord_weights"][0].double().numpy()
        shift = ref.get("pos_shift", 0)
        coords_bohr = (coords - (shift.to(coords) if torch.is_tensor(shift) else shift))[0].double().numpy() / BOHR
        if cpu_reference:
            rho_ref, grad_ref = ref["density"][0].double().numpy(), None
        else:
            rho_ref, grad_ref = reference_density(reference, idx, coords_bohr, ref_chunk, want_grad)
        n_ref = float(weights @ rho_ref)
        f_ref = functionals(rho_ref, grad_ref, weights, rho_ref)
        frame = {"index": int(idx), "natoms": int((ref["atom_numbers"][0] > 0).sum()),
                 "n_elec": int(ref["atom_numbers"][0].sum()), "n_grid": int(len(weights)), "ref": f_ref,
                 "sad": {}, "models": {}}
        if cpu_reference:
            rho_df, grad_df = df_density_cpu(reference, idx, coords_bohr, chunk), None
        else:
            rho_df, grad_df = df_density(reference, idx, coords_bohr, ref_chunk, want_grad)
        frame["df"] = compare(functionals(rho_df, grad_df, weights, rho_ref), f_ref, rho_df, rho_ref, weights, n_ref)
        priors = {}
        for entry in entries:
            props = entry.dataset.get_properties([idx])
            if not torch.allclose(props["batch_positions"], ref["batch_positions"], atol=1e-5):
                raise RuntimeError(f"{entry.spec['name']}: structure {idx} is centred differently from the reference")
            key = (str(entry.args.atom_dens_path), str(entry.args.atom_dens_type))
            if key not in priors:
                priors[key] = sad_density(entry, props, coords, want_grad)
            dens, dgrad = priors[key]
            if entry in sad_entries:
                rho_sad = dens[0].double().numpy()
                g_sad = None if dgrad is None else dgrad[0].double().numpy()
                name = Path(key[0]).name
                frame["sad"][name] = compare(functionals(rho_sad, g_sad, weights, rho_ref), f_ref, rho_sad, rho_ref,
                                             weights, n_ref)
                frame["sad"][name]["prior_type"] = key[1]
            if sad_only:
                continue
            props["coords"] = coords
            props["coord_weights"] = ref["coord_weights"]
            props["atom_density"] = dens
            if want_grad:
                props["atom_density_grad"] = dgrad
            rho_m, grad_m = model_density(entry, props, device, chunk, want_grad)
            frame["models"][entry.spec["name"]] = compare(functionals(rho_m, grad_m, weights, rho_ref), f_ref,
                                                          rho_m, rho_ref, weights, n_ref)
        frames[idx] = frame
        if device.startswith("cuda"):
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()
        if count % 10 == 0 or count == len(indices):
            write(out_path, label_set, entries, frames)
            m0 = frame["models"].get(model_names[0]) or next(iter(frame["sad"].values()), frame["df"])
            print(f"  [{count}/{len(indices)}] {(time.time() - t0) / count:.1f}s/structure; "
                  f"{model_names[0] if frame['models'] else 'SAD'}: ε {m0['afe_pct']:.3f}%, "
                  f"ΔEx {m0.get('ex_lda_err_kcal', float('nan')):+.2f}, "
                  f"ΔTvW {m0.get('tvw_err_kcal', float('nan')):+.2f} kcal/mol", flush=True)
    write(out_path, label_set, entries, frames)


def summary(frames):
    def stats(records):
        out = {"n": len(records)}
        for key in ("afe_pct", "ex_lda_err_kcal", "ex_lda_err_pct", "tvw_err_kcal", "tvw_err_pct"):
            vals = np.array([r[key] for r in records if key in r])
            if len(vals):
                name = key if key == "afe_pct" else f"{key}_abs"
                out[f"{name}_mean"] = float(np.mean(np.abs(vals)) if key != "afe_pct" else np.mean(vals))
                if key != "afe_pct":
                    out[f"{key}_mean_signed"] = float(np.mean(vals))
        return out

    groups = {"df": [f["df"] for f in frames]}
    for f in frames:
        for name, rec in f["sad"].items():
            groups.setdefault(f"sad:{name}", []).append(rec)
        for name, rec in f["models"].items():
            groups.setdefault(f"model:{name}", []).append(rec)
    return {k: stats(v) for k, v in groups.items()}


def write(out_path, label_set, entries, frames):
    ordered = [frames[k] for k in sorted(frames)]
    payload = {
        "label_set": label_set["name"], "np": label_set["np"], "dens": label_set["dens"],
        "units": "afe in %; energies in kcal/mol; *_pct relative to the reference value",
        "definitions": {"ex_lda": "-C_x ∫ max(ρ,0)^(4/3), C_x = 0.7386",
                        "tvw": f"(1/8) ∫ |∇ρ|²/ρ over points with ρ > {RHO_MIN} and ρ_ref > {REF_MIN}"},
        "models": {e.spec["name"]: {"run_dir": e.spec["run_dir"], "checkpoint": str(e.checkpoint),
                                    "sha256_1MiB": sha256(e.checkpoint), "fine_tuned_constraint": e.fine_tuned,
                                    "remove_atom_density": getattr(e.args, "remove_atom_density", None)}
                   for e in entries},
        "summary": summary(ordered),
        "frames": ordered,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    tmp.replace(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--only", action="append", default=None, help="label-set names (repeatable)")
    ap.add_argument("--models", default=None, help="comma-separated model names from the manifest")
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/revision/density_functionals")
    ap.add_argument("--no-grad", action="store_true", help="skip gradients (no vW); ε_ρ and LDA only")
    ap.add_argument("--sad-only", action="store_true", help="reference, DF and SAD prior only (no model predictions)")
    ap.add_argument("--cpu-reference", action="store_true", help="PySCF reference and DF on the CPU (implies --no-grad)")
    ap.add_argument("--fd-check", type=int, default=0, help="finite-difference gradient check on N points, then stop")
    ap.add_argument("--chunk", type=int, default=60000)
    ap.add_argument("--ref-chunk", type=int, default=20000)
    ap.add_argument("--grid-level", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = ap.parse_args()
    models = set(cli.models.split(",")) if cli.models else None
    for manifest in cli.manifest:
        for label_set in json.loads(manifest.read_text()):
            if cli.only and label_set["name"] not in set(cli.only):
                continue
            score_label_set(label_set, models, cli.out_dir, cli.device, cli.chunk, cli.ref_chunk, cli.grid_level,
                            cli.limit, not (cli.no_grad or cli.cpu_reference), cli.fd_check,
                            sad_only=cli.sad_only, cpu_reference=cli.cpu_reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
