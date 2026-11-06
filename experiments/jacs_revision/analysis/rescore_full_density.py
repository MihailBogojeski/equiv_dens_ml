#!/usr/bin/env python3
"""Re-score DenSNet density errors against the full SCF density.

``csh_evaluate.py`` was written for the OMol25/CSH labels, which carry only
density-fitting (DF) coefficients, so it hardcodes ``projected_density=True``.
The water, malonaldehyde and published small-molecule models were trained with
``--projected_density=False``: their target is the full SCF density built from
the MO coefficients. Scoring them against the DF projection measures mostly the
DF incompleteness of the reference (~5e-3), not the model.

This evaluator puts three densities on the same quadrature points and reports,
per structure,

  afe        = int |rho_pred - rho_full| dV / int rho_full dV   (primary)
  afe_vs_df  = int |rho_pred - rho_DF|   dV / int rho_DF   dV   (old metric)
  df_floor   = int |rho_DF   - rho_full| dV / int rho_full dV   (DF baseline)

The points are the full Becke-partitioned level-2 atom-centred grid the
training code samples from (no random subsampling, no random rotation), so the
integrals are deterministic quadratures rather than 4000-point Monte Carlo
estimates. ``afe`` keeps the key name the downstream scripts read.

One label set is scored against any number of models in a single pass: the
grid, rho_full and rho_DF depend only on the labels, so they are computed once
per structure and shared, and the free-atom (SAD) density is computed once per
prior file. Models are evaluated in chunks of grid points; the density head is
pointwise given the predicted coefficients, so chunking changes only memory.

Single model:
  python experiments/jacs_revision/analysis/rescore_full_density.py \\
    --run-dir results/revision/water_pbe_d4_avdz/2026-08-26_DQX1HUW0 \\
    --np-dataset datasets/revision/water_labels/water_id_test_pbe_d4_avdz_base.npy \\
    --dens-dataset datasets/revision/water_labels/water_id_test_pbe_d4_avdz.npy \\
    --elements 1,8 --label water_id_test --with-df \\
    --out results/revision/rescore_fullref/eval/pbe_d4_avdz/water_id_test.json

Manifest (list of label sets, each with its models):
  python experiments/jacs_revision/analysis/rescore_full_density.py --manifest m.json [--only NAME]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _p in (str(_REPO), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from equiv_dens.data.density_dataset import AtomsDensityData  # noqa: E402
from equiv_dens.training.model_loader import load_model  # noqa: E402
from equiv_dens.utils import orbitals  # noqa: E402
from equiv_dens.utils.grids import spherical_grid  # noqa: E402
from experiments.jacs_revision.csh.csh_evaluate import (  # noqa: E402
    load_source_index,
    load_training_args,
)

BOHR = 0.52917721092
# Paths recorded by the 2024 training runs; the files now live in this repo.
LEGACY_ROOTS = ("/home/ml-dft/equiv_dens/",)
PATH_ARGS = (
    "orbitals_file",
    "radial_coeffs_file",
    "atom_dens_path",
    "pseudo_pot_path",
    "L0_coeffs_file",
    "calc_basis_path",
)
SAVE_EVERY = 25


def remap_path(value):
    """Map a legacy absolute path onto the repository when it is missing here."""
    if not isinstance(value, str) or os.path.exists(value):
        return value
    for root in LEGACY_ROOTS:
        if value.startswith(root):
            candidate = _REPO / value[len(root):]
            if candidate.exists():
                return str(candidate)
    return value


def resolve_checkpoint(run_dir: Path) -> Path:
    """Return the state-dict file ``load_model(train=False)`` will read."""
    latest = run_dir / "checkpoints" / "latest_checkpoint.pth"
    if latest.is_file():
        model_id = torch.load(latest, map_location="cpu", weights_only=False)["ID"]
        best = run_dir / f"best_{model_id}.pth"
        if best.is_file():
            return best
        for name in ("best_checkpoint_energy.pth", "best_checkpoint_density.pth", "latest_checkpoint.pth"):
            candidate = run_dir / "checkpoints" / name
            if candidate.is_file():
                return candidate
    bests = sorted(glob.glob(str(run_dir / "best_*.pth"))) or sorted(
        glob.glob(str(run_dir / "checkpoints" / "best_*.pth"))
    )
    if not bests:
        raise FileNotFoundError(f"no checkpoint under {run_dir}")
    return Path(bests[0])


def sha256(path: Path, limit: int = 1 << 20) -> str:
    """Hash of the first MiB; enough to tell checkpoints apart cheaply."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        h.update(handle.read(limit))
    return h.hexdigest()[:16]


def point_keys(props: dict, n_points: int) -> list[str]:
    """Keys whose second axis runs over grid points (batch size is one)."""
    keys = []
    for key, value in props.items():
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == 1 and value.shape[1] == n_points:
            keys.append(key)
    if "coords" not in keys or "coord_weights" not in keys:
        raise RuntimeError(f"grid keys not found among {sorted(props)}")
    return keys


def predict_density(model, props: dict, device: str, chunk: int) -> np.ndarray:
    """Model density on every grid point of a single-structure batch."""
    n_points = props["coords"].shape[1]
    keys = point_keys(props, n_points)
    out = np.empty(n_points, dtype=np.float64)
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        sub = {}
        for key, value in props.items():
            # DFTNetwork.forward copies every input key to its output, so a
            # reference density passed in would come back as the "prediction"
            # of a model without a density head. Never pass it.
            if key in ("density", "density_grad"):
                continue
            if key in keys:
                value = value[:, start:stop]
            sub[key] = value.to(device) if torch.is_tensor(value) else value
        # Not under no_grad: models with an energy head take forces by autograd
        # inside forward (see csh_evaluate.batch_absolute_fractional_error).
        result = model(sub)
        if "density" not in result:
            raise RuntimeError("model has no density head (density_weight == 0 in its args?)")
        out[start:stop] = result["density"][0].detach().double().cpu().numpy()
        del result, sub
    return out


def df_density(dataset: AtomsDensityData, idx: int, coords_bohr: np.ndarray, chunk: int) -> np.ndarray:
    """rho_DF at the grid points, on a private copy of the molecule.

    ``AtomsDensityData.sample_projected_density`` switches the cached molecule
    to the auxiliary basis in place, after which the full-density path would
    read MO coefficients against the wrong basis. A copy avoids that.
    """
    mol = dataset.mols[idx].copy()
    mol.basis = dataset.density_fitting["auxbasis"]
    mol.build(verbose=0)
    coeffs = np.concatenate([c[1] for c in dataset.density_fitting["df_coeffs"][idx]])
    rho = np.empty(len(coords_bohr), dtype=np.float64)
    for start in range(0, len(coords_bohr), chunk):
        stop = min(start + chunk, len(coords_bohr))
        rho[start:stop] = orbitals.sample_density_base(
            [mol], coords_bohr[None, start:stop], [coeffs], projected=True
        )[0].double().numpy()
    return rho


def merge_overrides(base_file: Path, override_file: Path) -> Path:
    """args.txt with single-valued ``--key=value`` lines replaced by the override file's.

    The published 2024 models are loaded for MD with a short override file
    (config/md/nn/<molecule>.txt: SAD prior type and path, basis files) on top
    of the architecture recorded in the run's own args.txt.
    """
    overrides = {}
    for line in override_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("--") and "=" in line:
            key = line.split("=", 1)[0]
            if key in ("--restart", "--load_from"):
                continue
            overrides[key] = line
    kept = [ln for ln in base_file.read_text().splitlines() if ln.strip().split("=", 1)[0] not in overrides]
    handle = tempfile.NamedTemporaryFile("w", suffix="_args.txt", delete=False)
    handle.write("\n".join(kept + list(overrides.values())) + "\n")
    handle.close()
    return Path(handle.name)


def _gpu_eval(mol, coords_bohr: np.ndarray, chunk: int, contract):
    """Evaluate AOs of ``mol`` on the GPU chunk by chunk and reduce each chunk with ``contract``.

    Torch cancels GPU jobs that idle for hours; the reference densities are the
    bulk of the arithmetic, so they run here rather than in PySCF on the CPU.
    """
    import cupy as cp
    from gpu4pyscf.dft import numint as gnum

    out = np.empty(len(coords_bohr), dtype=np.float64)
    for start in range(0, len(coords_bohr), chunk):
        stop = min(start + chunk, len(coords_bohr))
        ao = gnum.eval_ao(mol, cp.asarray(coords_bohr[start:stop]))  # (ngrid, nao), PySCF order
        out[start:stop] = cp.asnumpy(contract(ao))
        del ao
    return out


def full_density_gpu(dataset: AtomsDensityData, idx: int, coords_bohr: np.ndarray, chunk: int) -> np.ndarray:
    """rho_SCF from the stored MO coefficients and occupations."""
    import cupy as cp

    mol = dataset.mols[idx]
    if not mol._built:
        mol.build(verbose=0)
    mo = np.asarray(dataset.coeffs[idx]["mo_coeff"])
    occ = np.asarray(dataset.coeffs[idx]["mo_occ"])
    if occ.ndim == 1:
        mo, occ = mo[None], occ[None]
    blocks = [m[:, o > 0] * np.sqrt(o[o > 0]) for m, o in zip(mo, occ)]
    c_occ = cp.asarray(np.concatenate(blocks, axis=1))
    return _gpu_eval(mol, coords_bohr, chunk, lambda ao: ((ao @ c_occ) ** 2).sum(axis=1))


def df_density_gpu(dataset: AtomsDensityData, idx: int, coords_bohr: np.ndarray, chunk: int) -> np.ndarray:
    import cupy as cp

    mol = dataset.mols[idx].copy()
    mol.basis = dataset.density_fitting["auxbasis"]
    mol.build(verbose=0)
    coeffs = cp.asarray(np.concatenate([c[1] for c in dataset.density_fitting["df_coeffs"][idx]]))
    return _gpu_eval(mol, coords_bohr, chunk, lambda ao: ao @ coeffs)


def training_args(run_dir: Path, args_file: Path | None, override_file: Path | None = None):
    base = args_file or (run_dir / "args.txt")
    if not Path(base).is_file() and override_file is not None:
        # Some runs (e.g. ethanethiol 2025-06-08_cnRa8AqQ) keep only their
        # checkpoints; the override's single-valued lines then stand in for args.txt.
        empty = tempfile.NamedTemporaryFile("w", suffix="_args.txt", delete=False)
        empty.close()
        base = Path(empty.name)
    if override_file is not None:
        base = merge_overrides(base, override_file)
    args = load_training_args(str(base))
    for name in PATH_ARGS:
        if hasattr(args, name):
            setattr(args, name, remap_path(getattr(args, name)))
    return args


def checkpoint_args(run_dir: Path, override_file: Path | None):
    """Architecture from the Namespace pickled in latest_checkpoint.pth, as md.py loads it.

    The 2024 runs' args.txt files no longer describe their checkpoints (feature
    widths, SAD prior type); the pickled Namespace does. Options the current
    parser added since then take their defaults, and file paths come from the
    MD override file.
    """
    import copy

    saved = torch.load(run_dir / "checkpoints" / "latest_checkpoint.pth", map_location="cpu", weights_only=False)["args"]
    current = training_args(run_dir, None, override_file)
    args = copy.deepcopy(saved)
    for key, value in vars(current).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    # The override sets the SAD prior type for the 2024 runs. A run without an
    # args.txt borrowed another model's file (see training_args), so keep its own.
    own_args = (run_dir / "args.txt").is_file()
    for name in PATH_ARGS + (("atom_dens_type",) if own_args else ()):
        if hasattr(current, name):
            setattr(args, name, getattr(current, name))
    args.load_from = None
    # The pickled args are the energy stage's, where density_weight is 0 and
    # load_model would build no density head; the density weights are in the
    # state dict all the same.
    if getattr(args, "density_weight", 0) <= 0:
        args.density_weight = 1.0
    # L0_start (an L=0 residual block in the energy head) postdates the 2024
    # runs and defaults to True; building it would leave random, unloaded
    # energy-head weights. Take it from what the checkpoint actually holds.
    if "L0_start" not in vars(saved):
        state = torch.load(resolve_checkpoint(run_dir), map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        args.L0_start = any("L0_change" in key for key in state)
    return args


def make_dataset(args, np_path, dens_path, elements, with_density, grid_level, grid_only=False):
    """with_density: grid plus rho_SCF (CPU); grid_only: grid alone; neither: no grid at all."""
    if with_density:
        required = ["df_coeffs", "density"]
    elif grid_only:
        required = ["df_coeffs", "coords"]
    else:
        required = ["df_coeffs"]
    return AtomsDensityData(
        np_path=np_path,
        density_path=dens_path,
        orbitals_path=args.orbitals_file,
        radial_coeffs_file=args.radial_coeffs_file,
        atom_dens_path=args.atom_dens_path,
        atom_dens_type=args.atom_dens_type,
        required_properties=required,
        projected_density=False,
        density_n_samp=10**9,  # larger than any grid: keep every point
        grid_fn=partial(spherical_grid, level=grid_level),
        dtype=torch.float32,
        df_loss_weights=args.df_loss_weights,
        cutoff=args.cutoff,
        # "auto": the elements present in the labels, as md.py builds its dataset.
        all_atom_numbers=None if elements == "auto" else np.array([int(z) for z in elements.split(",")]),
    )


class ModelEntry:
    """One model scored on the current label set, with its own light dataset.

    The dataset carries the model's cutoff, SAD prior and basis files; it is
    built without the density property, so ``get_properties`` returns positions,
    neighbours and prior coefficients but no grid, which is injected instead.
    """

    def __init__(self, spec: dict, label_set: dict, device: str, grid_level: int, tmp_root: Path):
        self.spec = spec
        self.out = Path(spec["out"])
        run_dir = Path(spec["run_dir"]).resolve()
        self.checkpoint = Path(spec["checkpoint"]).resolve() if spec.get("checkpoint") else resolve_checkpoint(run_dir)
        override = Path(spec["args_override"]) if spec.get("args_override") else None
        if spec.get("args_source") == "checkpoint":
            args = checkpoint_args(run_dir, override)
        else:
            args = training_args(run_dir, Path(spec["args_file"]) if spec.get("args_file") else None, override)
        if spec.get("checkpoint"):
            from experiments.jacs_revision.analysis.score_learning_curve_ckpts import prepare_eval_run_dir

            run_dir = prepare_eval_run_dir(run_dir, self.checkpoint, tmp_root / self.out.stem)
        args.np_dataset = label_set["np"]
        args.dens_dataset = label_set["dens"]
        args.restart = str(run_dir)
        args.use_gpu = device.startswith("cuda")
        # density_fine_tuning trains with the charge constraint on (and, for a
        # direct model, positive coefficients), set only transiently and never
        # written to args.txt (training/utils.py). Rebuilt without them, a
        # fine-tuned checkpoint is evaluated under a different parameterisation:
        # direct models then integrate to ~8.5 N_e, and malonaldehyde PBE to
        # 0.989 N_e. Rebuild the model as it was fine-tuned.
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        self.fine_tuned = any("integral_constraint" in key for key in state)
        if self.fine_tuned:
            args.integral_constraint = "coeffs_in_coeffs_net"
            if not getattr(args, "remove_atom_density", False):
                args.positive_coeffs = True
        del state
        self.args = args
        self.dataset = make_dataset(args, label_set["np"], label_set["dens"], label_set["elements"], False, grid_level)
        if spec.get("args_source") == "checkpoint":
            # Old checkpoints lack scaling.std (the force-scaling constant),
            # which load_model refuses once a density head is built. It does
            # not touch the density; allow it and nothing else.
            args.ignore_missing_keywords = True
        self.model = load_model(args, self.dataset, train=False).to(device).eval()
        if spec.get("args_source") == "checkpoint":
            saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            saved = saved.get("model_state_dict", saved)
            missing = set(self.model.state_dict()) - set(saved)
            # Only the density network matters here; energy-head modules added
            # to the code after 2024 (e.g. L0_change) may be absent.
            fatal = sorted(k for k in missing if k.startswith(("density_repr_model.", "property_models.density")))
            if fatal:
                raise RuntimeError(f"{spec['name']}: checkpoint lacks density weights {fatal[:5]}")
            if missing:
                print(f"    {spec['name']}: {len(missing)} non-density keys absent from checkpoint (energy head / scaling), ignored")
        self.records = {}
        if self.out.is_file():
            try:
                previous = json.loads(self.out.read_text())
                prov = previous.get("provenance", {})
                if (prov.get("checkpoint_sha256_1MiB") == sha256(self.checkpoint)
                        and prov.get("fine_tuned_rebuilt_with_constraint", False) == self.fine_tuned):
                    self.records = {int(r["index"]): r for r in previous.get("records", [])}
            except (json.JSONDecodeError, KeyError):
                self.records = {}
        print(
            f"  model {spec['name']}: {self.checkpoint} ({sha256(self.checkpoint)}), cutoff {args.cutoff}, "
            f"SAD {Path(str(args.atom_dens_path)).name}, remove_atom_density={args.remove_atom_density}, "
            f"resume {len(self.records)} records",
            flush=True,
        )

    def write(self, label_set: dict, n_target: int, wall: float, complete: bool):
        records = [self.records[k] for k in sorted(self.records)]
        summary = summarise(records, self.spec.get("label", label_set["name"]))
        provenance = {
            "reference": "full SCF density from MO coefficients (projected_density=False)",
            "grid": "full Becke-partitioned level-2 atom-centred grid, no subsampling",
            "run_dir": self.spec["run_dir"],
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256_1MiB": sha256(self.checkpoint),
            "np_dataset": label_set["np"],
            "dens_dataset": label_set["dens"],
            "atom_dens_path": str(self.args.atom_dens_path),
            "remove_atom_density": getattr(self.args, "remove_atom_density", None),
            "trained_projected_density": getattr(self.args, "projected_density", None),
            "fine_tuned_rebuilt_with_constraint": self.fine_tuned,
            "positive_coeffs": getattr(self.args, "positive_coeffs", None),
            "cutoff": self.args.cutoff,
            "n_target": n_target,
            "complete": complete,
            "wall_s": wall,
        }
        self.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"summary": summary, "provenance": provenance, "records": records}, indent=1) + "\n")
        tmp.replace(self.out)


def summarise(records: list[dict], label: str) -> dict:
    def stats(key):
        values = np.array([r[key] for r in records if key in r and np.isfinite(r[key])])
        if not len(values):
            return {}
        return {
            f"{key}_mean": float(values.mean()),
            f"{key}_median": float(np.median(values)),
            f"{key}_p90": float(np.percentile(values, 90)),
            f"{key}_max": float(values.max()),
            f"{key}_n": int(len(values)),
        }

    summary = {"label": label, "n_structures": len(records), **stats("afe"), **stats("afe_vs_df"), **stats("df_floor")}
    if records:
        summary["int_full_minus_nelec_max"] = float(max(abs(r["int_full"] - r["n_elec"]) for r in records))
        natoms = np.array([r["natoms"] for r in records])
        summary["by_size"] = {}
        for lo, hi in [(0, 20), (20, 40), (40, 70), (70, 110), (110, 1000)]:
            mask = (natoms >= lo) & (natoms < hi)
            if mask.any():
                summary["by_size"][f"{lo}-{hi}"] = {
                    "n": int(mask.sum()),
                    "afe_mean": float(np.mean([r["afe"] for r, m in zip(records, mask) if m])),
                }
    return summary


def check_gpu_reference(reference, idx, coords, shift, chunk, with_df):
    """Once per label set: the GPU reference densities against the PySCF CPU path."""
    frame = coords - (shift.to(coords) if torch.is_tensor(shift) else shift)
    sub = frame[:, :20000]
    cpu_full = reference.sample_density([idx], sub)[0].double().numpy()
    gpu_full = full_density_gpu(reference, idx, sub[0].double().numpy() / BOHR, chunk)
    out = {"full_max_rel": float(np.abs(gpu_full - cpu_full).max() / np.abs(cpu_full).max())}
    if with_df:
        cpu_df = df_density(reference, idx, sub[0].double().numpy() / BOHR, chunk)
        gpu_df = df_density_gpu(reference, idx, sub[0].double().numpy() / BOHR, chunk)
        out["df_max_rel"] = float(np.abs(gpu_df - cpu_df).max() / np.abs(cpu_df).max())
    # The CPU path rounds grid coordinates to float32 (~1e-6 relative near nuclei);
    # a basis or ordering error would be O(1).
    if max(out.values()) > 1e-4:
        raise RuntimeError(f"GPU reference density disagrees with PySCF: {out}")
    return out


def score_label_set(label_set: dict, device: str, chunk: int, grid_level: int, limit: int, gpu_reference: bool = True,
                    ref_chunk: int = 50000):
    """Score every model listed under one label set."""
    t0 = time.time()
    gpu_ref = gpu_reference and device.startswith("cuda")
    print(f"== {label_set['name']}: {label_set['dens']}", flush=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="rescore_wrap_"))
    entries = [ModelEntry(spec, label_set, device, grid_level, tmp_root) for spec in label_set["models"]]
    with_df = bool(label_set.get("with_df", False))
    ref_args = entries[0].args
    reference = make_dataset(ref_args, label_set["np"], label_set["dens"], label_set["elements"],
                             not gpu_ref, grid_level, grid_only=gpu_ref)
    checked = None
    n_total = len(reference)
    indices = list(range(n_total if not limit else min(limit, n_total)))
    source_index = load_source_index(label_set["np"], n_total)

    for count, idx in enumerate(indices, start=1):
        pending = [e for e in entries if idx not in e.records]
        if not pending:
            continue
        ref = reference.get_properties([idx])
        coords = ref["coords"]
        weights = ref["coord_weights"][0].double().numpy()
        shift = ref.get("pos_shift", 0)
        frame = coords - (shift.to(coords) if torch.is_tensor(shift) else shift)
        coords_bohr = frame[0].double().numpy() / BOHR
        if gpu_ref:
            if checked is None:
                checked = check_gpu_reference(reference, idx, coords, shift, ref_chunk, with_df)
                print(f"  GPU reference check vs PySCF: {checked}", flush=True)
            rho_full = full_density_gpu(reference, idx, coords_bohr, ref_chunk)
        else:
            rho_full = ref["density"][0].double().numpy()
        n_full = float(weights @ rho_full)
        base = {
            "index": int(idx),
            "natoms": int((ref["atom_numbers"][0] > 0).sum()),
            "n_elec": int(ref["atom_numbers"][0].sum()),
            "n_grid": int(len(weights)),
            "int_full": n_full,
        }
        if source_index is not None and source_index[idx] >= 0:
            base["source_index"] = int(source_index[idx])
        rho_df = None
        if with_df:
            rho_df = (df_density_gpu if gpu_ref else df_density)(reference, idx, coords_bohr, ref_chunk if gpu_ref else chunk)
            n_df = float(weights @ rho_df)
            base["int_df"] = n_df
            base["df_floor"] = float(weights @ np.abs(rho_df - rho_full) / n_full)

        sad_cache: dict[str, torch.Tensor] = {}
        for entry in pending:
            props = entry.dataset.get_properties([idx])
            if not torch.allclose(props["batch_positions"], ref["batch_positions"], atol=1e-5):
                raise RuntimeError(f"{entry.spec['name']}: structure {idx} is centred differently from the reference")
            key = str(entry.args.atom_dens_path)
            if key not in sad_cache:
                sad_cache[key] = entry.dataset.sample_atom_density(
                    props["batch_positions"], props["batch_atom_numbers"], coords
                )
            props["coords"] = coords
            props["coord_weights"] = ref["coord_weights"]
            props["atom_density"] = sad_cache[key]
            rho_pred = predict_density(entry.model, props, device, chunk)
            record = dict(base)
            record["int_pred"] = float(weights @ rho_pred)
            record["afe"] = float(weights @ np.abs(rho_pred - rho_full) / n_full)
            if rho_df is not None:
                record["afe_vs_df"] = float(weights @ np.abs(rho_pred - rho_df) / base["int_df"])
            entry.records[idx] = record

        if gpu_ref:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()
        if count % SAVE_EVERY == 0 or count == len(indices):
            wall = time.time() - t0
            for entry in entries:
                entry.write(label_set, len(indices), wall, complete=len(entry.records) >= len(indices))
            done = [e.records[idx]["afe"] for e in entries if idx in e.records]
            print(
                f"  [{count}/{len(indices)}] {wall / count:.1f}s/structure, last afe "
                + " ".join(f"{a:.5f}" for a in done)
                + (f", df_floor {base['df_floor']:.5f}" if with_df else ""),
                flush=True,
            )
    wall = time.time() - t0
    for entry in entries:
        entry.write(label_set, len(indices), wall, complete=len(entry.records) >= len(indices))
        s = summarise([entry.records[k] for k in sorted(entry.records)], entry.spec["name"])
        print(
            f"  {entry.spec['name']}: afe {s.get('afe_mean', float('nan')):.5f}"
            + (f", afe_vs_df {s['afe_vs_df_mean']:.5f}, df_floor {s['df_floor_mean']:.5f}" if "afe_vs_df_mean" in s else "")
            + f" over {s['n_structures']} -> {entry.out}",
            flush=True,
        )
    del entries
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=None, help="JSON list of label sets with models")
    ap.add_argument("--only", action="append", default=None, help="label-set names to run (repeatable)")
    ap.add_argument("--run-dir", type=Path, default=None, help="single-model mode: training run directory")
    ap.add_argument("--args-file", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None, help="score this .pth instead of best_<ID>.pth")
    ap.add_argument("--np-dataset")
    ap.add_argument("--dens-dataset")
    ap.add_argument("--elements", default="1,6,7,8,16")
    ap.add_argument("--label", default="test")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--with-df", action="store_true", help="also report afe_vs_df and df_floor")
    ap.add_argument("--chunk", type=int, default=150000, help="grid points per model call")
    ap.add_argument("--grid-level", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cpu-reference", action="store_true", help="rho_SCF and rho_DF with PySCF on the CPU")
    ap.add_argument("--ref-chunk", type=int, default=50000, help="grid points per GPU reference-density block")
    cli = ap.parse_args()

    if cli.manifest:
        label_sets = json.loads(cli.manifest.read_text())
        if cli.only:
            label_sets = [ls for ls in label_sets if ls["name"] in set(cli.only)]
    else:
        if not (cli.run_dir and cli.np_dataset and cli.dens_dataset and cli.out):
            ap.error("single-model mode needs --run-dir, --np-dataset, --dens-dataset and --out")
        label_sets = [
            {
                "name": cli.label,
                "np": cli.np_dataset,
                "dens": cli.dens_dataset,
                "elements": cli.elements,
                "with_df": cli.with_df,
                "models": [
                    {
                        "name": cli.label,
                        "run_dir": str(cli.run_dir),
                        "args_file": str(cli.args_file) if cli.args_file else None,
                        "checkpoint": str(cli.checkpoint) if cli.checkpoint else None,
                        "out": str(cli.out),
                        "label": cli.label,
                    }
                ],
            }
        ]
    for label_set in label_sets:
        score_label_set(label_set, cli.device, cli.chunk, cli.grid_level, cli.limit, not cli.cpu_reference, cli.ref_chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
