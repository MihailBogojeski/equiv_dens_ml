#!/usr/bin/env python3
"""Collect everything worth tracking about the energy-head trainings into one JSON for the review page.

Energy heads only (the density networks are frozen and not covered). Sources:
  * water v3 runs: offline W&B files (history every 100 steps, validation every 1000, system stats);
  * malonaldehyde v2 runs: energy.log (train every 100 steps, validation every 500);
  * checkpoints (best / latest / last numbered): per-block weight norms of the energy head, its
    update between snapshots, a weight histogram;
  * per-frame scores of the current best checkpoint (score_energy_forces.py, score_malon_energy.py).

Training-log "train" metrics are in units of the training-set force standard deviation; they are
converted to kcal/mol (energy) and kcal/mol/A (forces) with sigma computed from the training labels.
Validation metrics are already physical.

    .venv/bin/python -B experiments/jacs_revision/analysis/energy_training_dashboard.py
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "results/revision/energy_training_dashboard"
HARTREE_KCAL, BOHR_A = 627.509474, 0.529177210903

RUNS = [
    dict(id="water_pbe", group="local", status="stopped 6 Oct; superseded by the SLURM copy", label="Water · PBE-D4/aVDZ", molecule="water", theory="pbe_d4_avdz",
         root="results/revision/water_pbe_d4_avdz_energy_v3", wandb="water_pbe_d4_avdz_energy_v3"),
    dict(id="water_wb97mv", group="local", status="stopped 6 Oct; superseded by the SLURM copy", label="Water · ωB97M-V/def2-TZVPD", molecule="water", theory="wb97mv_def2tzvpd",
         root="results/revision/water_wb97mv_def2tzvpd_energy_v3", wandb="water_wb97mv_def2tzvpd_energy_v3"),
    dict(id="malon_pbe", group="local", status="stopped 6 Oct; superseded by the SLURM copy", label="Malonaldehyde · PBE-D4/aVDZ", molecule="malonaldehyde", theory="pbe_d4_avdz",
         root="results/revision/malon_energy_v2/malonaldehyde_pbe_d4_avdz/energy",
         log="results/revision/malon_energy_v2/malonaldehyde_pbe_d4_avdz/energy.log"),
    dict(id="malon_wb97mv", group="local", status="stopped 6 Oct; superseded by the SLURM copy", label="Malonaldehyde · ωB97M-V/def2-TZVPD", molecule="malonaldehyde", theory="wb97mv_def2tzvpd",
         root="results/revision/malon_energy_v2/malonaldehyde_wb97mv_def2tzvpd/energy",
         log="results/revision/malon_energy_v2/malonaldehyde_wb97mv_def2tzvpd/energy.log"),
]

# Cluster (SLURM) copies: fresh energy heads trained independently of the local runs above, in their own folders.
# group: local = the original runs; slurm = the same recipe on the cluster; direct = heads on the direct density networks.
for th, lab in (("pbe_d4_avdz", "PBE-D4/aVDZ"), ("wb97mv_def2tzvpd", "ωB97M-V/def2-TZVPD")):
    short_id = "pbe" if th == "pbe_d4_avdz" else "wb97mv"
    RUNS.append(dict(id=f"water_{short_id}_slurm", group="slurm", status="finished 17 Oct at 100k steps after the 13 Oct resubmission (job 19748921); scored with the precision-aware single-structure gate", label=f"Water · {lab} · SLURM", molecule="water", theory=th,
                     root=f"results/revision/water_{th}_energy_v3_slurm", wandb=f"water_{th}_energy_v3"))
    RUNS.append(dict(id=f"malon_{short_id}_slurm", group="slurm", status="finished 8 Oct: learning rate reached the 1e-5 floor (job 19170630)", label=f"Malonaldehyde · {lab} · SLURM", molecule="malonaldehyde", theory=th,
                     root=f"results/revision/malon_energy_v2_slurm/malonaldehyde_{th}/energy",
                     log=f"results/revision/malon_energy_v2_slurm/malonaldehyde_{th}/energy.log"))
    RUNS.append(dict(id=f"water_{short_id}_direct", group="direct", status="timed out at the 48 h limit on 10 Oct at 75k steps; resumed 16 Oct (job 19748931)", label=f"Water · {lab} · direct density", molecule="water", theory=th,
                     root=f"results/revision/water_direct_{th}_energy_v3", wandb=f"water_direct_{th}_energy_v3"))
# Energy-head cutoff sweep (submit_energy_cutoff_sweep.sbatch): fresh PBE-D4 heads at 4/5/6 A on the density networks of the
# density cutoff sweep, 40k steps each at equal budget. Roots without a checkpointed run yet are skipped by main().
for c in (4, 5, 6):
    RUNS.append(dict(id=f"water_pbe_cutoff{c}", group="cutoff", label=f"Water · PBE-D4/aVDZ · cutoff {c} Å", molecule="water",
                     theory="pbe_d4_avdz", root=f"results/revision/water_pbe_d4_avdz_energy_cutoff_{c}", wandb="water_pbe_d4_avdz_energy_v3"))


def rnd(x, n=5):
    return [None if (v is None or not np.isfinite(v)) else float(f"{v:.{n}g}") for v in x]


def force_sigma(args):
    """Training-set force standard deviation in kcal/mol/A (the unit of the logged train metrics)."""
    d = np.load(REPO / args.np_dataset, allow_pickle=True).item()
    # The trainer takes the std over the padded force array (padding zeros included): 67.6 for
    # water PBE rather than the 82.8 of the real atoms only. Match it, since it is the log's unit.
    f = np.asarray(d["forces"], float)
    conv = HARTREE_KCAL if args.energy_unit_in == "hartree" else 1.0
    conv /= BOHR_A if args.distance_unit_in == "bohr" else 1.0
    return float(f.std() * conv)


def wandb_series(name, run_id):
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    rows, stats = [], []
    for d in sorted(glob.glob(str(REPO / f"wandb/offline-run-*{name}*{run_id}"))):
        ds = datastore.DataStore()
        ds.open_for_scan(glob.glob(d + "/*.wandb")[0])
        sess = []
        while True:
            try:
                data = ds.scan_data()
            except Exception:
                break
            if data is None:
                break
            r = pb.Record()
            r.ParseFromString(data)
            kind = r.WhichOneof("record_type")
            if kind == "history":
                sess.append({(it.key or "/".join(it.nested_key)): json.loads(it.value_json) for it in r.history.item})
            elif kind == "stats":
                s = {it.key: json.loads(it.value_json) for it in r.stats.item}
                s["_t"] = r.stats.timestamp.seconds
                stats.append(s)
        rows.append(sess)
    return rows, stats


def parse_malon_log(path):
    """Step blocks of train_only_en.py; a restart repeats steps, the last occurrence wins."""
    text = Path(path).read_text(errors="replace").splitlines()
    out, cur, wall = {}, None, 0.0
    num = r"([-+0-9.eE]+|nan|inf)"
    for line in text:
        m = re.match(r"^(\d+)/(\d+) epoch:\s*(\d+)", line)
        if m:
            cur = {"step": int(m.group(1)), "epoch": int(m.group(3))}
            out[cur["step"]] = cur
            continue
        if cur is None:
            continue
        if "train loss:" in line:
            g = lambda k: (lambda mm: float(mm.group(1)) if mm else None)(re.search(k + r":\s*" + num, line))
            cur.setdefault("train_loss", g("train loss"))
            cur.setdefault("train_e", g("train energy_mae"))
            cur.setdefault("train_f", g("train forces_mae"))
            cur.setdefault("train_e_rmse", g("train energy_rmse"))
            cur.setdefault("train_f_rmse", g("train forces_rmse"))
            v = g("valid 0 mae")
            v1 = g("valid 1 mae")  # --train_eval_size subset, scored like validation
            if "valid_e" not in cur:
                cur["valid_e"] = v
                cur["train_eval_e"] = v1
                cur["best_loss"] = g("best loss")
            else:
                cur["valid_f"] = v
                cur["train_eval_f"] = v1
                cur["lr"] = g("lr")
        m = re.match(r"^time elapsed:\s*" + num, line)
        if m and cur is not None and "dt" not in cur:
            cur["dt"] = float(m.group(1))
    return [out[k] for k in sorted(out)]


def weight_summary(run_dir):
    import torch

    ck = run_dir / "checkpoints"
    files = {"latest": ck / "latest_checkpoint.pth", "best": ck / "best_checkpoint.pth"}
    numbered = sorted(ck.glob("checkpoint_0*.pth"))
    if numbered:
        files["previous"] = numbered[-1]
    sd, meta = {}, {}
    for k, f in files.items():
        c = torch.load(f, map_location="cpu", weights_only=False)
        sd[k] = {n: v.float() for n, v in c["model_state_dict"].items() if n.startswith("property_models.energy.") and v.is_floating_point()}
        meta[k] = {"step": int(c["step"]), "epoch": int(c["epoch"]), "valid": c["valid_errors"][0] if c.get("valid_errors") else None,
                   "file_mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))}
        if k == "latest":
            args, sched = c["args"], c["schedulers_state_dict"]
            best_errors = c.get("best_errors")
    def group(name):
        t = name.split(".")[2:]
        g = t[0]
        m = re.match(r"(.*?)(_\d+)+$", g)  # radial_scale_filters_3 -> radial_scale_filters
        g = m.group(1) if m else g
        if len(t) > 1 and t[0] in ("module", "modules", "blocks", "layers") and t[1].isdigit():
            g = f"block {t[1]}" + (f" · {t[2]}" if len(t) > 2 else "")
        return g
    groups = {}
    for n in sd["latest"]:
        groups.setdefault(group(n), []).append(n)
    rows = []
    for g, names in groups.items():
        sq = lambda key: float(sum((sd[key][n] ** 2).sum() for n in names if n in sd[key])) ** 0.5
        nl = sq("latest")
        row = {"group": g, "n_tensors": len(names), "n_params": int(sum(sd["latest"][n].numel() for n in names)),
               "norm_latest": nl, "norm_best": sq("best"),
               "rms_latest": float((sum((sd["latest"][n] ** 2).sum() for n in names) / max(1, sum(sd["latest"][n].numel() for n in names))) ** 0.5)}
        if "previous" in sd:
            d2 = float(sum(((sd["latest"][n] - sd["previous"][n]) ** 2).sum() for n in names if n in sd["previous"]))
            row["rel_update_since_previous"] = d2 ** 0.5 / nl if nl else None
        d2 = float(sum(((sd["latest"][n] - sd["best"][n]) ** 2).sum() for n in names if n in sd["best"]))
        row["rel_distance_to_best"] = d2 ** 0.5 / nl if nl else None
        rows.append(row)
    rows.sort(key=lambda r: -r["n_params"])
    allw = torch.cat([v.flatten() for v in sd["latest"].values()]).numpy()
    lim = float(np.percentile(np.abs(allw), 99.5)) or 1.0
    hist, edges = np.histogram(np.clip(allw, -lim, lim), bins=60, range=(-lim, lim))
    return ({"groups": rows, "snapshots": meta, "n_params": int(allw.size), "norm_total": float(np.linalg.norm(allw)),
             "hist": {"counts": hist.tolist(), "lo": -lim, "hi": lim}, "abs_max": float(np.abs(allw).max()),
             "frac_tiny": float((np.abs(allw) < 1e-6).mean())}, args, sched, best_errors)


_NB_CACHE = {}


def water_neighbours(split, theory, index, cutoff):
    """Mean number of other waters (O–O) and of other atoms within the cutoff, per water / per atom, for one frame."""
    if split not in _NB_CACHE:
        _NB_CACHE[split] = np.load(REPO / f"datasets/revision/water_labels/water_{split}_{theory}_base.npy", allow_pickle=True).item()
    d = _NB_CACHE[split]
    z = d["atom_numbers"][index]
    keep = z > 0
    x = np.asarray(d["positions"][index], float)[keep]
    dist = np.linalg.norm(x[:, None] - x[None], axis=-1)
    o = z[keep] == 8
    return float(((dist[o][:, o] < cutoff).sum(1) - 1).mean()), float(((dist < cutoff).sum(1) - 1).mean())


def training_neighbour_range(theory, cutoff):
    """Per-frame neighbour counts over the whole training set: the range the energy head was trained on."""
    d = np.load(REPO / f"datasets/revision/water_labels/water_train_{theory}_base.npy", allow_pickle=True).item()
    _NB_CACHE["train"] = d
    vals = np.array([water_neighbours("train", theory, i, cutoff) for i in range(len(d["atom_numbers"]))])
    return {"cutoff_A": cutoff, "n_train": len(vals), "waters_max": float(vals[:, 0].max()), "atoms_max": float(vals[:, 1].max()),
            "waters_p95": float(np.percentile(vals[:, 0], 95)), "atoms_p95": float(np.percentile(vals[:, 1], 95))}


_XYZ_COMMENTS: dict[str, list[str]] = {}
WATER_XYZ = {"train": "water_clusters/train.xyz", "val": "water_clusters/val.xyz", "id_test": "water_clusters/id_test.xyz",
             "ood_size": "water_clusters/ood_size.xyz", "ood_order": "water_ood/ood_order.xyz", "ood_density": "water_ood/ood_density.xyz"}


def xyz_comments(rel):
    """Comment line of every frame of an xyz file under datasets/revision (frame i = source_index i)."""
    if rel not in _XYZ_COMMENTS:
        lines, out, i = (REPO / "datasets/revision" / rel).read_text().splitlines(), [], 0
        while i < len(lines):
            n = int(lines[i].split()[0])
            out.append(lines[i + 1].strip())
            i += n + 2
        _XYZ_COMMENTS[rel] = out
    return _XYZ_COMMENTS[rel]


def frame_parent(molecule, split, i):
    """(parent structure, finer detail) of frame i of a split: the minimum a jittered cluster was drawn from, the ice
    cutout, the melted droplet snapshot or the malonaldehyde scan point. Rules from generate_water_cluster_geoms.py
    (blocks of one size; n=6 motifs cycle cyclic/prism/book/cage), generate_ood_water.py and generate_malonaldehyde.py."""
    if molecule == "water":
        if split in ("train", "val", "id_test"):
            bs = 250 if split == "train" else 50
            n, k = 2 + i // bs, i % bs
            motif = {2: "dimer", 3: "cyclic", 4: "cyclic", 5: "cyclic"}.get(n) or ("cyclic", "prism", "book", "cage")[k % 4]
            return f"(H2O){n} {motif}", f"jitter sigma {(0.04, 0.06, 0.08)[k % 3]:.2f} A"
        if split == "ood_size":
            return ("(H2O)8 cube", "(H2O)10 pentagonal prism", "(H2O)12 hexagonal prism")[i // 100], f"jitter sigma {(0.05, 0.07, 0.10)[(i % 100) % 3]:.2f} A"
        c = xyz_comments(WATER_XYZ[split])[i]
        f = dict(t.split("=", 1) for t in c.split() if "=" in t)
        if split == "ood_order":
            return f"{c.split()[0].replace('ice', 'ice ')} cutout, n={f['n']}", f"T={f['T']} rep={f['rep']}"
        kind = "evaporation" if c.startswith("droplet-evaporation") else f"rho={f.get('rho')}"
        return f"droplet n={f['n']}, melt snapshot {f['rep']}", kind
    if split == "ood_proton_transfer":
        c = xyz_comments("malonaldehyde/ood_proton_transfer.xyz")[i]
        return f"scan point {i // 5}", ("relaxed" if i % 5 == 0 else "jittered") + " · " + c.split()[-1]
    return {"train": "training sample", "val": "validation"}.get(split, split), xyz_comments(f"malonaldehyde/{split}.xyz")[i].split()[-1] if split in ("train", "val") else ""


def frame_structure(molecule, split, i, theory):
    """Elements and coordinates (A) of one scored frame, padding removed."""
    path = (f"datasets/revision/water_labels/water_{split}_{theory}_base.npy" if molecule == "water"
            else f"datasets/revision/malonaldehyde_labels/malonaldehyde_{split}_{theory}_base.npy")
    key = "S" + path
    if key not in _NB_CACHE:
        _NB_CACHE[key] = np.load(REPO / path, allow_pickle=True).item()
    d = _NB_CACHE[key]
    z = np.asarray(d["atom_numbers"][i])
    keep = z > 0
    return {"Z": z[keep].astype(int).tolist(), "R": np.round(np.asarray(d["positions"][i], float)[keep], 3).tolist()}


FULL = REPO / "results/revision/energy_forces_v3"  # full-set scores of the cluster heads (submit_energy_score_cluster.sbatch)


def score_dir(run):
    """Per-frame scores of a run's best checkpoint. Cluster and direct runs use the full-set scores when they exist
    (energy_forces_v3/<water|water_direct|malon>_<theory>_cluster); otherwise the dashboard's own first-N subsets in
    score_<mol>_<theory>[_slurm|_direct]."""
    group = run.get("group", "local")
    if group in ("slurm", "direct"):
        name = (f"water_{'direct_' if group == 'direct' else ''}{run['theory']}" if run["molecule"] == "water"
                else f"malon_{run['theory']}")
        full = FULL / f"{name}_cluster"
        if (FULL / f"{name}_final" / "per_frame.csv").exists():  # finished heads, scored after training ended (README_final.md)
            full = FULL / f"{name}_final"
        if (full / "per_frame.csv").exists():
            return full
    suffix = {"local": ""}.get(group, "_" + group)
    return OUT / f"score_{'water' if run['molecule'] == 'water' else 'malon'}_{run['theory']}{suffix}"


def parent_family(molecule, parent):
    """Coarse colour group of a parent: the minimum's motif, the ice polymorph, the droplet size, the scan."""
    if molecule != "water":
        return "scan" if parent.startswith("scan") else parent
    if parent.startswith("ice"):
        return parent.split(" cutout")[0]
    if parent.startswith("droplet"):
        return parent.split(",")[0]
    return parent


def read_scores(run, cutoff=6.0):
    d = score_dir(run)
    if not (d / "per_frame.csv").exists():
        return None
    local = OUT / f"score_{'water' if run['molecule'] == 'water' else 'malon'}_{run['theory']}"
    full_sim = FULL / "similarity_full" / ("water" if run["molecule"] == "water" else "malon")
    if d.parent == FULL and (full_sim / "similarity.csv").exists():
        d_sim = full_sim
    elif not (d / "similarity.csv").exists() and (local / "similarity.csv").exists():
        d_sim = local  # same frames (first N of each split), so the local run's similarity coordinates apply
    else:
        d_sim = d
    rows = [r for r in csv.DictReader(open(d / "per_frame.csv")) if not r["split"].startswith("ood_modes")]
    summ = json.loads((d / "summary.json").read_text())
    num = lambda k: rnd([float(r[k]) for r in rows], 6)
    out = {"split": [r["split"] for r in rows], "summary": {k: v for k, v in summ["by_split"].items() if not k.startswith("ood_modes")}, "provenance": summ.get("provenance", {}),
           "scored_at": time.strftime("%Y-%m-%d %H:%M", time.localtime((d / "per_frame.csv").stat().st_mtime))}
    if (d_sim / "similarity.csv").exists():  # CV-space distance, Hungarian RMSD, delta_pt (energy_similarity.py)
        sim = {(r["split"], int(r["index"])): r for r in csv.DictReader(open(d_sim / "similarity.csv"))}
        if sim and "cv_dist" in next(iter(sim.values())):
            col = lambda k: rnd([float(sim[(r["split"], int(r["index"]))][k]) if (r["split"], int(r["index"])) in sim else np.nan for r in rows], 5)
            out["sim_cv"], out["sim_rmsd"], out["delta"] = col("cv_dist"), col("rmsd"), col("delta")
            out["sim_info"] = json.loads((d_sim / "similarity.json").read_text())
    pd = [frame_parent(run["molecule"], r["split"], int(r["index"])) for r in rows]
    out.update(index=[int(r["index"]) for r in rows], parent=[p[0] for p in pd], detail=[p[1] for p in pd],
               family=[parent_family(run["molecule"], p[0]) for p in pd], coverage="full set" if d.parent == FULL else "first frames of each split")
    if run["molecule"] == "water":
        out.update(e_pred=num("energy_pred_kcal_mol"), e_ref=num("energy_ref_kcal_mol"), e_err=num("energy_error_kcal_mol"),
                   n_water=[int(r["n_water"]) for r in rows], f_mae=num("force_mae_kcal_mol_A"), by_size=summ.get("by_size", []))
        nb = np.array([water_neighbours(r["split"], run["theory"], int(r["index"]), cutoff) for r in rows])
        out.update(nb_water=rnd(nb[:, 0], 3), nb_atoms=rnd(nb[:, 1], 3), nb_info=training_neighbour_range(run["theory"], cutoff))
    else:
        out.update(e_pred=num("energy_pred_rel_kcal_mol"), e_ref=num("energy_ref_rel_kcal_mol"), e_err=num("energy_error_kcal_mol"),
                   f_mae=num("force_mae_kcal_mol_A"), q=num("q_pt_A"), d_oo=num("d_OO_A"), f_H=num("f_mae_H"), f_C=num("f_mae_C"),
                   f_O=num("f_mae_O"), f_Ht=num("f_mae_transfer_H"), offset=summ.get("offset_train_median_kcal_mol"))
        if out.get("delta") and any(v is not None for v in out["delta"]):
            out["q"] = out["delta"]  # one proton-transfer coordinate on the page: the project's delta_pt
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"generated": time.strftime("%Y-%m-%d %H:%M %Z"), "runs": []}
    for run in RUNS:
        run_dirs = sorted(d for d in glob.glob(str(REPO / run["root"] / "20*")) if any(Path(d).glob("checkpoints/*.pth")))
        if not run_dirs:
            print(f"{run['id']:14s} no run with a checkpoint under {run['root']} yet; skipped")
            continue
        run_dir = Path(run_dirs[-1])
        weights, args, sched, best_errors = weight_summary(run_dir)
        sigma = force_sigma(args)
        s0 = sched[0] if isinstance(sched, (list, tuple)) else list(sched.values())[0] if isinstance(sched, dict) and sched and isinstance(list(sched.values())[0], dict) else sched
        entry = {k: run.get(k) for k in ("id", "label", "molecule", "theory", "group", "status")}
        entry.update(run_dir=str(run_dir.relative_to(REPO)), run_id=run_dir.name.split("_")[-1], sigma_force_kcal_mol_A=sigma, weights=weights,
                     config={k: getattr(args, k, None) for k in ("max_steps", "num_train", "num_valid", "num_test", "batch_size", "valid_batch_size", "learning_rate",
                                                                "decay_factor", "decay_patience", "stop_at_learning_rate", "energy_weight", "forces_weight",
                                                                "ema_start_epoch", "normalize_en", "cutoff", "validation_interval", "summary_interval",
                                                                "checkpoint_interval", "np_dataset", "atomic_energies", "integral_constraint", "seed")},
                     scheduler={k: (s0.get(k) if isinstance(s0, dict) else None) for k in ("best", "num_bad_epochs", "_last_lr", "patience", "factor")},
                     best_errors=best_errors)
        S = {k: [] for k in ("step", "t", "train_loss", "train_e", "train_f", "valid_e", "valid_f", "valid_loss", "train_eval_e", "train_eval_f", "lr", "grad_norm", "sec_per_step", "hours")}
        if run["molecule"] == "water":
            sessions, stats = wandb_series(run["wandb"], entry["run_id"])
            step0, hours = 0, 0.0
            for sess in sessions:
                for i, r in enumerate(sess):
                    step = step0 + 100 * i
                    S["step"].append(step); S["t"].append(r["_timestamp"])
                    S["train_loss"].append(r.get("loss_train", np.nan) * sigma); S["train_e"].append(r.get("energy_mae_train", np.nan) * sigma)
                    S["train_f"].append(r.get("forces_mae_train", np.nan) * sigma)
                    S["valid_e"].append(r.get("energy_mae_valid")); S["valid_f"].append(r.get("forces_mae_valid")); S["valid_loss"].append(r.get("loss_valid"))
                    S["train_eval_e"].append(r.get("energy_mae_train_eval")); S["train_eval_f"].append(r.get("forces_mae_train_eval"))
                    S["lr"].append(r.get("learning rate")); S["grad_norm"].append(r.get("gradient_norm"))
                    dt = (r["_timestamp"] - sess[i - 1]["_timestamp"]) / 100 if i else np.nan
                    S["sec_per_step"].append(dt)
                    hours += 0 if i == 0 else (r["_timestamp"] - sess[i - 1]["_timestamp"]) / 3600
                    S["hours"].append(hours)
                # A restart resumes from a 1000-step checkpoint. The W&B history can end a few rows short of
                # it (session 1 ended at 2800 and resumed from 3000) or a few past it (7100 -> 7000), so the
                # nearest multiple of 1000 is used. Step numbers after a restart are therefore approximate.
                step0 = int(round((step0 + 100 * (len(sess) - 1)) / 1000.0)) * 1000 if sess else step0
            entry["step_note"] = "W&B rows are logged every 100 steps; each restart is assumed to resume from the nearest 1000-step checkpoint, so water step numbers are approximate."
            pick = lambda pat: next((k for s in stats[:5] for k in s if re.fullmatch(pat, k)), None)
            keys = {"gpu_util": pick(r"gpu\.\d+\.gpu"), "gpu_mem_pct": pick(r"gpu\.\d+\.memoryAllocated"), "proc_gpu_util": pick(r"gpu\.process\.\d+\.gpu"),
                    "cpu_pct": pick(r"cpu"), "proc_cpu_threads": pick(r"proc\.cpu\.threads"), "proc_rss_mb": pick(r"proc\.memory\.rssMB")}
            stride = max(1, len(stats) // 400)
            entry["system"] = {"t": [s["_t"] for s in stats[::stride]],
                               **{name: rnd([s.get(k, np.nan) if k else np.nan for s in stats[::stride]], 4) for name, k in keys.items()}}
        else:
            rows = parse_malon_log(REPO / run["log"])
            hours = 0.0
            for r in rows:
                S["step"].append(r["step"]); S["t"].append(None)
                S["train_loss"].append((r.get("train_loss") or np.nan) * sigma); S["train_e"].append((r.get("train_e") or np.nan) * sigma)
                S["train_f"].append((r.get("train_f") or np.nan) * sigma)
                S["valid_e"].append(r.get("valid_e")); S["valid_f"].append(r.get("valid_f"))
                S["train_eval_e"].append(r.get("train_eval_e")); S["train_eval_f"].append(r.get("train_eval_f"))
                S["valid_loss"].append((r.get("valid_e") or np.nan) + (r.get("valid_f") or np.nan))
                S["lr"].append(r.get("lr")); S["grad_norm"].append(None)
                dt = r.get("dt")
                S["sec_per_step"].append(dt / 100 if dt else np.nan)
                hours += (dt or 0) / 3600
                S["hours"].append(hours)
            entry["step_note"] = "From energy.log; validation every 500 steps on 80 frames; wall-clock axis is cumulative logged block time."
        entry["series"] = {k: (v if k in ("step", "t") else rnd([np.nan if x is None else x for x in v])) for k, v in S.items()}
        entry["scores"] = read_scores(run, cutoff=float(getattr(args, "cutoff", 6.0) or 6.0))
        fin = lambda k: next((x for x in reversed(entry["series"][k]) if x is not None), None)
        ve, vf = np.array([np.nan if x is None else x for x in entry["series"]["valid_e"]]), np.array([np.nan if x is None else x for x in entry["series"]["valid_f"]])
        recent = [x for x in entry["series"]["sec_per_step"][-6:] if x is not None]
        entry["now"] = {"step": S["step"][-1], "valid_e": fin("valid_e"), "valid_f": fin("valid_f"), "train_e": fin("train_e"), "train_f": fin("train_f"), "lr": fin("lr"),
                        "best_valid_e": float(np.nanmin(ve)), "best_valid_f": float(np.nanmin(vf)), "sec_per_step_recent": float(np.mean(recent)) if recent else None,
                        "hours": S["hours"][-1]}
        if entry["now"]["sec_per_step_recent"] and not run.get("status"):  # only meaningful for a running job
            entry["now"]["eta_days_to_max_steps"] = (entry["config"]["max_steps"] - S["step"][-1]) * entry["now"]["sec_per_step_recent"] / 86400
        data["runs"].append(entry)
        n = entry["now"]
        print(f"{run['id']:14s} step {n['step']:6d}/{entry['config']['max_steps']}  valid E {n['valid_e']:.3f} (best {n['best_valid_e']:.3f}) kcal/mol  "
              f"valid F {n['valid_f']:.3f} (best {n['best_valid_f']:.3f}) kcal/mol/A  train E {n['train_e']:.3f} F {n['train_f']:.3f}  "
              f"lr {n['lr']}  {n['sec_per_step_recent'] or float('nan'):.1f} s/step  sigma {sigma:.1f}  scores {'yes' if entry['scores'] else 'no'}")
    # raw-vs-EMA weight diagnostic (results/revision/energy_training_dashboard/raw_vs_ema/README.md)
    rows = []
    for summary in sorted((OUT / "raw_vs_ema").glob("*/summary.json")):
        for split, m in json.loads(summary.read_text())["by_split"].items():
            rows.append({"weights": summary.parent.name, "split": split, "n": m["n"], "energy_mae": m["energy_mae_kcal_mol"],
                         "energy_bias": m.get("energy_bias_kcal_mol"), "energy_mae_bias_removed": m.get("energy_mae_bias_removed_kcal_mol"),
                         "force_mae": m["force_mae_kcal_mol_A"]})
    if rows:
        data["diagnostics"] = {"raw_vs_ema": rows, "run": "water_pbe_d4_avdz_energy_v3_slurm/2026-10-07_eVbz6kif", "step": 90000}
    (OUT / "dashboard_data.json").write_text(json.dumps(data, separators=(",", ":")))
    structures = {}  # molecule -> "split/index" -> {Z, R}; geometries are the same at both levels of theory
    for entry in data["runs"]:
        sc = entry.get("scores")
        if not sc:
            continue
        mol = structures.setdefault(entry["molecule"], {})
        for split, i in zip(sc["split"], sc["index"]):
            mol.setdefault(f"{split}/{i}", frame_structure(entry["molecule"], split, i, entry["theory"]))
    (OUT / "structures.json").write_text(json.dumps(structures, separators=(",", ":")))
    print("wrote", OUT / "structures.json", f"{(OUT / 'structures.json').stat().st_size / 1024:.0f} KB")
    print("wrote", OUT / "dashboard_data.json", f"{(OUT / 'dashboard_data.json').stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
