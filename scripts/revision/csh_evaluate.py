#!/usr/bin/env python3
"""Evaluate a CSH-trained DenSNet model, reporting HELM-comparable metrics.

Primary metric is the paper's absolute fractional error: the integrated density
error normalised by the electron count, so structures of very different size are
weighted equally.

Results are broken down by molecule size because the CSH test splits are much
larger than the training structures, so the aggregate number would otherwise
hide whether the model is extrapolating or just interpolating.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from equiv_dens.data.density_dataset import AtomsDensityData  # noqa: E402
from equiv_dens.training.model_loader import load_model  # noqa: E402
from equiv_dens.training.parse_command_line_arguments import (  # noqa: E402
    parse_command_line_arguments,
)


def load_training_args(args_file: str):
    """Re-read a saved args.txt.

    The saved file records `=None` for unset options, which the typed parser
    rejects, and the parser also reads sys.argv, which would pick up this
    script's own flags. Both are worked around the same way the existing
    evaluator does: strip those lines into a temp file and parse with an empty
    argv.
    """
    import tempfile

    from collections import Counter, OrderedDict

    # Saved with underscores but declared with hyphens, so they never round-trip.
    # They only affect benchmarking, not evaluation.
    drop_prefixes = ("--n_warmup", "--n_runs")

    lines = []
    for line in Path(args_file).read_text().strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or "=None" in line:
            continue
        if line.startswith(drop_prefixes):
            continue
        # args.txt is written by splitting on whitespace, so "kcal/mol" comes
        # back truncated to "kcal" and no longer matches the allowed choices.
        for flag in ("--energy_unit_out", "--energy_unit_in"):
            if line == f"{flag}=kcal":
                line = f"{flag}=kcal/mol"
        lines.append(line)

    # Multi-value options are saved as repeated `--flag=value` lines, which
    # argparse overwrites instead of accumulating: `--order` would come back as
    # [5] rather than [1, 3, 5], silently rebuilding a different architecture
    # than the checkpoint. Regroup them into the multi-line form.
    counts = Counter(ln.split("=", 1)[0] for ln in lines if "=" in ln)
    repeated = {flag for flag, n in counts.items() if n > 1}
    grouped: OrderedDict = OrderedDict()
    rebuilt = []
    for line in lines:
        flag = line.split("=", 1)[0]
        if "=" in line and flag in repeated:
            grouped.setdefault(flag, []).append(line.split("=", 1)[1])
        else:
            rebuilt.append(line)
    for flag, values in grouped.items():
        rebuilt.append(flag)
        rebuilt.extend(values)
    lines = rebuilt

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(lines))
        tmp = handle.name

    saved_argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        parsed = parse_command_line_arguments(arg_file=tmp)
    finally:
        sys.argv = saved_argv
    return parsed[0] if isinstance(parsed, tuple) else parsed


def load_source_index(np_dataset: str, n_total: int) -> np.ndarray | None:
    """The `source_index` column of a base npy, or None if it has none.

    Returns None rather than raising for datasets assembled before this was
    recorded, so an older split still evaluates -- it just cannot be joined to
    geometries safely, and the caller says so.
    """
    try:
        raw = np.load(np_dataset, allow_pickle=True)
    except Exception:
        return None
    data = raw.item() if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == () else raw
    try:
        values = np.asarray(data["source_index"], dtype=int)
    except (KeyError, IndexError, TypeError):
        return None
    return values if values.shape[0] == n_total else None


def absolute_fractional_error(pred, ref, weights):
    """int |rho_pred - rho_ref| dV / int rho_ref dV, per structure."""
    num = torch.sum(torch.abs(pred - ref) * weights, dim=-1)
    den = torch.sum(ref * weights, dim=-1)
    return num / den


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-file", required=True, help="args.txt of the trained model")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="training run directory to restore from (the one holding checkpoints/)",
    )
    parser.add_argument("--np-dataset", required=True)
    parser.add_argument("--dens-dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--elements",
        default="1,6,7,8,16",
        help="element set the model was trained on; the per-element buffers are "
        "registered from this, so it must match training or the state dict will not load "
        "(e.g. --elements 1,6,7,8,9,16,17,35 for the 8-element model)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="test")
    cli = parser.parse_args()

    args = load_training_args(cli.args_file)
    args.np_dataset = cli.np_dataset
    args.dens_dataset = cli.dens_dataset
    args.density_subsamples = cli.n_samples
    if cli.run_dir:
        args.restart = cli.run_dir

    dataset = AtomsDensityData(
        np_path=cli.np_dataset,
        density_path=cli.dens_dataset,
        orbitals_path=args.orbitals_file,
        radial_coeffs_file=args.radial_coeffs_file,
        atom_dens_path=args.atom_dens_path,
        atom_dens_type=args.atom_dens_type,
        required_properties=["df_coeffs", "density"],
        projected_density=True,
        density_n_samp=cli.n_samples,
        dtype=torch.float32,
        # The per-element coefficient-weight buffers are registered from the
        # dataset, so this has to match training or the state dict will not load.
        df_loss_weights=args.df_loss_weights,
        cutoff=args.cutoff,
        all_atom_numbers=np.array([int(z) for z in cli.elements.split(",")]),
    )
    n_total = len(dataset)
    indices = list(range(n_total if not cli.limit else min(cli.limit, n_total)))
    print(f"evaluating {len(indices)} of {n_total} structures from {cli.np_dataset}")

    # Row number here is not the frame's number in the geometry file: assembly
    # drops frames ORCA failed on, so the two agree only for a split that came
    # back complete. Anything joining these errors to per-geometry quantities
    # needs the original, which the assembler stores alongside the coordinates.
    source_index = load_source_index(cli.np_dataset, n_total)
    if source_index is None:
        print(
            "  note: this np_dataset predates source_index, so results are keyed by row "
            "number; that is only safe to join against geometries if nothing was dropped"
        )

    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    model = load_model(args, dataset, train=False)
    model = model.to(device).eval()

    records = []
    for start in range(0, len(indices), cli.batch_size):
        batch = indices[start : start + cli.batch_size]
        props = dataset.get_properties(batch)
        props = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in props.items()
        }
        # Deliberately not `torch.no_grad()`. A model with an energy head gets its
        # forces by differentiating the energy with respect to the positions, so
        # under no_grad that head raises "element 0 of tensors does not require
        # grad" and no density is ever returned. The CSH models are density-only
        # and never hit it; every model in the water campaign has the head, so
        # this is the one path that has to work for the OOD figure.
        #
        # The positions are left exactly as the dataset produced them, requiring
        # no grad. DFTNetwork clones them and flips the flag itself, and that only
        # works on a leaf -- handing it a tensor that already requires grad makes
        # the clone a non-leaf and the model fails on the way back out.
        #
        # The graph is built and discarded per batch and only the density is read
        # from it, so the cost is one batch of activations.
        out = model(props)

        ref = props["density"]
        weights = props["coord_weights"]
        pred = out["density"]
        with torch.no_grad():
            afe = absolute_fractional_error(pred, ref, weights).cpu().numpy()

        natoms = (props["batch_atom_numbers"] > 0).sum(-1).cpu().numpy()
        n_elec = props["batch_atom_numbers"].sum(-1).cpu().numpy()
        for j, idx in enumerate(batch):
            record = {
                "index": int(idx),
                "natoms": int(natoms[j]),
                "n_elec": int(n_elec[j]),
                "afe": float(afe[j]),
            }
            if source_index is not None and source_index[idx] >= 0:
                record["source_index"] = int(source_index[idx])
            records.append(record)
        print(
            f"  [{start + len(batch)}/{len(indices)}] "
            f"AFE batch mean {float(np.mean(afe)):.5f}",
            flush=True,
        )

    afe_all = np.array([r["afe"] for r in records])
    natoms = np.array([r["natoms"] for r in records])
    summary = {
        "label": cli.label,
        "n_structures": len(records),
        "afe_mean": float(afe_all.mean()),
        "afe_median": float(np.median(afe_all)),
        "afe_p90": float(np.percentile(afe_all, 90)),
        "afe_max": float(afe_all.max()),
        "by_size": {},
    }
    edges = [(0, 20), (20, 40), (40, 70), (70, 110), (110, 1000)]
    for lo, hi in edges:
        mask = (natoms >= lo) & (natoms < hi)
        if mask.sum():
            summary["by_size"][f"{lo}-{hi}"] = {
                "n": int(mask.sum()),
                "afe_mean": float(afe_all[mask].mean()),
            }

    print(json.dumps(summary, indent=2))
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(f"wrote {cli.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
