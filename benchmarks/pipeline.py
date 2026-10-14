"""Full-model numerical snapshots and synchronized prepared-input timings.

Examples (always run reference and candidate in separate Python processes)::

  python -m benchmarks.pipeline --source-root REFERENCE generate --output fixture.pt
  python -m benchmarks.pipeline --source-root REFERENCE snapshot --fixture fixture.pt --output ref.pt
  python -m benchmarks.pipeline --source-root CANDIDATE compare --fixture fixture.pt --reference ref.pt --output comparison.json

Timing is a single arm/block. The coordinator owns the shared measurement lock,
five paired alternating arm rounds and performance decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

from benchmarks.fixtures import (CASES, MODES, build_model, clone_tree, load_fixture,
                                  make_fixture, select_source, tensor_hash)
from benchmarks.precision import precision_metadata, use_full_precision

KERNEL_BACKEND = "broadcast"
DENSITY_BACKEND = "legacy"
ENERGY_BACKEND = "legacy"


def provenance(source_root, fixture):
    imported = {}
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("equiv_dens") or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(source_root):
            raise AssertionError(f"Mixed source import: {name}: {path}")
        if path.is_file():
            imported[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    instrument_root = Path(__file__).resolve().parent
    benchmark_sources = {}
    for name, module in sorted(sys.modules.items()):
        if not (name.startswith("benchmarks") or name == "__main__"):
            continue
        if not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if name == "__main__" and not path.is_relative_to(instrument_root):
            continue
        if not path.is_relative_to(instrument_root):
            raise AssertionError(f"Mixed benchmark import: {name}: {path}")
        if path.is_file():
            benchmark_sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"source_root": str(source_root),
            "revision": subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
            "dirty_patch_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(source_root), "diff", "--", "src"])).hexdigest(),
            "imports": imported, "benchmark_sources": benchmark_sources,
            "fixture_hash": fixture["fixture_hash"], "state_hash": fixture["state_hash"],
            "torch": torch.__version__, "threads": torch.get_num_threads(), "host": os.uname().nodename,
            "kernel_backend": KERNEL_BACKEND,
            "density_backend": DENSITY_BACKEND, "energy_backend": ENERGY_BACKEND,
            "precision": precision_metadata(),
            "allocation": os.environ.get("SLURM_JOB_ID"), "runtime_image": os.environ.get("APPTAINER_CONTAINER")}


def initialized_model(fixture, mode, device="cpu", training=None):
    model = build_model(fixture["spec"], mode, kernel_backend=KERNEL_BACKEND,
                        density_backend=DENSITY_BACKEND, energy_backend=ENERGY_BACKEND).to(device)
    model.load_state_dict(fixture["state_dict"], strict=True)
    if tensor_hash(model.state_dict()) != fixture["state_hash"]:
        raise AssertionError("Loaded model tensors differ from saved state")
    if training is not None:
        model.train(training)
    return model


def nonzero(value, description):
    if not torch.isfinite(value).all() or value.abs().max().item() <= 1e-12:
        raise AssertionError(f"Nonfinite or degenerate {description}")


def parameter_gradients(model):
    return {name: None if p.grad is None else p.grad.detach().cpu().clone()
            for name, p in model.named_parameters()}


def loss_for(output, mode):
    if mode == "coefficients":
        parts = [output["df_coeffs"].square().mean()]
        for field in ("radial_width", "radial_scale"):
            parts.extend(value.square().mean() * 0.01 for atom in output[field] for value in atom.values())
        return sum(parts)
    if mode == "density":
        return (output["density"] - 0.23).square().mean() + (output["density_grad"] - 0.07).square().mean() * 0.1
    if mode == "ef":
        return output["energy"].square().mean()
    target = torch.linspace(-0.031, 0.043, output["forces"].numel(),
                            dtype=output["forces"].dtype, device=output["forces"].device).reshape_as(output["forces"])
    return (output["forces"] - target).square().mean()


def evaluate_contract(fixture, mode, training, device="cpu"):
    model = initialized_model(fixture, mode, device, training)
    inputs = clone_tree(fixture["atoms"], device)
    grid_calls = []
    hook = model.property_models["density"].register_forward_hook(lambda *_: grid_calls.append(1))
    output = model(inputs)
    hook.remove()
    if len(grid_calls) != int(mode == "density"):
        raise AssertionError("Unexpected grid evaluation in requested-output mode")
    fields = ["df_coeffs", "spherical_coeffs", "radial_width", "radial_scale"]
    if mode == "density":
        fields += ["density", "density_grad"]
        nonzero(output["density"], "density")
        nonzero(output["density_grad"], "spatial density gradient")
    elif mode in ("ef", "force_training"):
        fields += ["energy", "forces"]
        nonzero(output["energy"], "energy")
        nonzero(output["forces"], "forces")
        if output["forces"].requires_grad != training:
            raise AssertionError("Force graph policy mismatch")
    nonzero(output["df_coeffs"], "density coefficients")
    result = {"outputs": clone_tree({key: output[key] for key in fields}, "cpu"),
              "training": training, "mode": mode, "grid_calls": len(grid_calls),
              "forces_graph": bool(output.get("forces", torch.tensor(0)).requires_grad)}
    if mode == "density" and training:
        named_parameters = list(model.density_repr_model.named_parameters())
        for field in ("density", "density_grad"):
            gradients = torch.autograd.grad((output[field]-0.07).square().mean(),
                                            [p for _, p in named_parameters],
                                            retain_graph=True, allow_unused=True)
            result[f"{field}_loss_gradients"] = {
                name: None if gradient is None else gradient.detach().cpu()
                for (name, _), gradient in zip(named_parameters, gradients)}
            nonzero(torch.cat([gradient.flatten() for gradient in gradients if gradient is not None]),
                    f"{field}-only parameter loss gradients")
    if mode in ("ef", "force_training") and training:
        coefficient_nodes = [v for atom in output["spherical_coeffs"] for v in atom.values() if v.requires_grad]
        if coefficient_nodes:
            sensitivity = torch.autograd.grad(output["energy"].sum(), coefficient_nodes, retain_graph=True, allow_unused=True)
            result["density_energy_sensitivity"] = [None if value is None else value.detach().cpu() for value in sensitivity]
            nonzero(torch.cat([v.flatten() for v in sensitivity if v is not None]), "density-mediated energy sensitivity")
    if training or mode == "coefficients":
        loss = loss_for(output, mode)
        loss.backward()
        result["loss"] = loss.detach().cpu()
        result["gradients"] = parameter_gradients(model)
        gradients = [value.flatten() for value in result["gradients"].values() if value is not None]
        nonzero(torch.cat(gradients), f"{mode} parameter loss gradients")
    if training and mode in ("coefficients", "density", "force_training"):
        if mode == "force_training" and any(p.requires_grad or p.grad is not None for p in model.density_repr_model.parameters()):
            raise AssertionError("Frozen density-stage parameters received gradients")
        before = clone_tree(model.state_dict(), "cpu")
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        optimizer.step()
        after = clone_tree(model.state_dict(), "cpu")
        for key in before:
            if mode == "force_training" and key.startswith("density_repr_model.") and not torch.equal(before[key], after[key]):
                raise AssertionError("Optimizer changed frozen density state")
        changed = [key for key in before if not torch.equal(before[key], after[key])]
        expected_prefix = "property_models.energy." if mode == "force_training" else "density_repr_model."
        if not any(key.startswith(expected_prefix) for key in changed):
            raise AssertionError(f"{mode} loss did not update requested parameters")
        result["updated_state"] = after
        result["optimizer_state"] = clone_tree(optimizer.state_dict(), "cpu")
        result["changed_parameters"] = changed
    return result


def density_finite_difference_checks(fixture, device="cpu", steps=(1e-3, 3e-4, 1e-4, 3e-5)):
    from equiv_dens.utils.base import to_angstrom
    expansion_model = initialized_model(fixture, "density", device, False)
    density = expansion_model(clone_tree(fixture["atoms"], device))
    spatial = density["density_grad"][:, 0].detach()
    spatial_rows = []
    for step in steps:
        differences = torch.zeros_like(spatial)
        for axis in range(3):
            minus, plus = clone_tree(fixture["atoms"], device), clone_tree(fixture["atoms"], device)
            minus["coords"][:, 0, axis] -= step
            plus["coords"][:, 0, axis] += step
            rho_minus = expansion_model(minus)["density"][:, 0]
            rho_plus = expansion_model(plus)["density"][:, 0]
            differences[:, axis] = (rho_plus-rho_minus) / (2*step) * to_angstrom
        spatial_rows.append({"step_angstrom": step, "max_abs_error": (spatial-differences).abs().max().item(),
                             "finite_difference": differences.cpu()})
    gate = 1e-8 if fixture["spec"]["dtype"] == "float64" else 1e-4
    return {"spatial": spatial_rows,
            "spatial_fd_pass": min(row["max_abs_error"] for row in spatial_rows) <= gate,
            "gate_abs": gate, "spatial_bohr_per_angstrom": to_angstrom}


def finite_difference_checks(fixture, device="cpu", steps=(1e-3, 3e-4, 1e-4, 3e-5)):
    """Independent central differences of the COMPLETE energy and grid density.

    The explicit DensityExpansion gradient is per bohr; coordinates are angstrom.
    Test all force components and three nonzero grid-gradient components.
    """
    spatial_result = density_finite_difference_checks(fixture, device, steps)
    if not fixture["spec"]["learned_radial"]:
        return {**spatial_result, "force_fd_pass": None, "force_loss_mixed_fd_pass": None,
                "energy_exclusion": "Reference energy encoder indexes empty fixed-radial lists"}
    model = initialized_model(fixture, "ef", device, False)
    baseline = model(clone_tree(fixture["atoms"], device))
    forces = baseline["forces"].detach()
    rows = []
    for step in steps:
        differences = torch.zeros_like(forces)
        for index in range(forces.numel()):
            minus, plus = clone_tree(fixture["atoms"], device), clone_tree(fixture["atoms"], device)
            minus["positions"].view(-1)[index] -= step
            plus["positions"].view(-1)[index] += step
            # Batch centers remain consistent even when this energy encoder
            # presently only reads position-derived features and coefficients.
            flat_index = int(minus["batch_idx_pos"][index // 3]) * 3 + index % 3
            minus["batch_positions"].view(-1)[flat_index] -= step
            plus["batch_positions"].view(-1)[flat_index] += step
            en_minus = model(minus)["energy"].detach().sum()
            en_plus = model(plus)["energy"].detach().sum()
            differences.view(-1)[index] = -(en_plus - en_minus) / (2*step)
        rows.append({"step_angstrom": step, "max_abs_error": (forces-differences).abs().max().item(),
                     "finite_difference": differences.cpu()})
    gate = 1e-8 if fixture["spec"]["dtype"] == "float64" else 1e-4
    # Independent mixed-derivative probe: finite difference a force-only loss
    # along the largest nonzero trainable energy-parameter gradient component.
    force_model = initialized_model(fixture, "force_training", device, True)
    force_loss = loss_for(force_model(clone_tree(fixture["atoms"], device)), "force_training")
    force_loss.backward()
    named = [(name, parameter) for name, parameter in force_model.named_parameters()
             if name.startswith("property_models.energy.") and parameter.grad is not None]
    parameter_name, parameter = max(named, key=lambda pair: float(pair[1].grad.abs().max()))
    parameter_index = int(parameter.grad.abs().reshape(-1).argmax())
    derivative = parameter.grad.reshape(-1)[parameter_index].detach().item()
    original = parameter.detach().reshape(-1)[parameter_index].item()
    mixed_rows = []
    for step in steps:
        values = []
        for sign in (-1, 1):
            with torch.no_grad():
                parameter.reshape(-1)[parameter_index] = original + sign*step
            value = loss_for(force_model(clone_tree(fixture["atoms"], device)), "force_training")
            values.append(value.detach().item())
        difference = (values[1]-values[0])/(2*step)
        mixed_rows.append({"step_parameter": step, "finite_difference": difference,
                           "max_abs_error": abs(derivative-difference)})
    with torch.no_grad():
        parameter.reshape(-1)[parameter_index] = original
    return {**spatial_result, "forces": rows,
            "force_fd_pass": min(row["max_abs_error"] for row in rows) <= gate,
            "force_loss_parameter": parameter_name, "force_loss_parameter_index": parameter_index,
            "force_loss_parameter_gradient": derivative, "force_loss_mixed": mixed_rows,
            "force_loss_mixed_fd_pass": min(row["max_abs_error"] for row in mixed_rows) <= gate,
            "gate_abs": gate}


def snapshot(fixture, device="cpu", finite_differences=True):
    contracts, exclusions = {}, {}
    for mode in MODES:
        if not fixture["spec"]["learned_radial"] and mode in ("ef", "force_training"):
            # Preserve this actual source failure as a capability diagnostic.
            try:
                evaluate_contract(fixture, mode, mode == "force_training", device)
            except IndexError as error:
                exclusions[mode] = {"supported": False, "exception": type(error).__name__,
                                    "message": str(error), "reason": "energy encoder indexes empty fixed-radial lists"}
            else:
                raise AssertionError("Fixed-radial energy capability changed; review the reference exclusion")
            continue
        for training in ((True,) if mode == "force_training" else (False, True)):
            key = f"{mode}/{'train' if training else 'eval'}"
            contracts[key] = evaluate_contract(fixture, mode, training, device)
    result = {"contracts": contracts, "exclusions": exclusions,
              "fixture_hash": fixture["fixture_hash"], "state_hash": fixture["state_hash"]}
    if finite_differences:
        result["finite_differences"] = finite_difference_checks(fixture, device)
    # Scripted successive positions, with unchanged basis and saved model state.
    frames = []
    if fixture["spec"]["learned_radial"]:
        model = initialized_model(fixture, "ef", device, False)
        for displacement in (0.0, 0.013, -0.021):
            atoms = clone_tree(fixture["atoms"], device)
            atoms["positions"][0, 0, 0] += displacement
            atoms["batch_positions"][0, 0, 0] += displacement
            output = model(atoms)
            frames.append(clone_tree({key: output[key] for key in ("energy", "forces", "df_coeffs")}, "cpu"))
        nonzero(frames[1]["forces"]-frames[0]["forces"], "successive-frame force response")
    result["successive_frames"] = frames
    return result


def compare_snapshots(reference, candidate, dtype="float64"):
    failures, metrics = [], {}
    def visit(left, right, path=""):
        if isinstance(left, torch.Tensor):
            derivative = any(word in path for word in ("gradients", "sensitivity", "forces", "density_grad"))
            atol, rtol = ((1e-8, 1e-6) if derivative else (1e-9, 1e-7)) if dtype == "float64" else ((1e-4, 1e-3) if derivative else (1e-5, 1e-4))
            if not isinstance(right, torch.Tensor) or left.shape != right.shape:
                failures.append(path + ": type/shape mismatch")
            elif left.is_floating_point():
                metrics[path] = float((left-right).abs().max()) if left.numel() else 0.0
                if not torch.allclose(left, right, atol=atol, rtol=rtol):
                    failures.append(path + f": max_abs_error={metrics[path]} atol={atol} rtol={rtol}")
            elif not torch.equal(left, right):
                failures.append(path + ": integer tensors differ")
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                failures.append(path + ": dictionary keys differ")
            else:
                for key in left:
                    visit(left[key], right[key], f"{path}/{key}")
        elif isinstance(left, (list, tuple)):
            if not isinstance(right, (list, tuple)) or len(left) != len(right):
                failures.append(path + ": sequence differs")
            else:
                for index, (a, b) in enumerate(zip(left, right)):
                    visit(a, b, f"{path}/{index}")
        elif left != right:
            failures.append(path + f": metadata differs ({left!r}, {right!r})")
    for key in ("fixture_hash", "state_hash", "contracts", "exclusions", "successive_frames"):
        visit(reference[key], candidate[key], key)
    return {"protocol": "P1", "passed": not failures, "failures": failures, "max_abs_errors": metrics,
            "reference_finite_differences": _json_tree(reference.get("finite_differences")),
            "candidate_finite_differences": _json_tree(candidate.get("finite_differences"))}


def prepared_step(fixture, mode, device="cpu"):
    """Return a warm-timer callable and model; construction occurs outside timer.

    Force training replays Adam updates from the saved state. Arms must have the
    same warmup/iteration counts; each new block process restarts this sequence.
    Includes DFTNetwork input cloning, zeroing, force loss, backward and step.
    """
    training = mode in ("coefficients_training", "density_training", "force_training")
    requested_mode = mode.removesuffix("_training") if mode != "force_training" else mode
    if not fixture["spec"]["learned_radial"] and requested_mode in ("ef", "force_training"):
        raise ValueError("Fixed-radial E/F is unsupported by the reference energy encoder; see snapshot exclusions")
    model = initialized_model(fixture, requested_mode, device, training)
    atoms = clone_tree(fixture["atoms"], device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3) if training else None
    def step():
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        # Coefficient evaluation has no no-force property wrapper to disable AD.
        if requested_mode == "coefficients" and not training:
            with torch.no_grad():
                output = model(atoms)
        else:
            output = model(atoms)
        if optimizer is not None:
            loss_for(output, requested_mode).backward()
            optimizer.step()
        return output
    return step, model


def _json_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_tree(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_tree(child) for child in value]
    return value


def main():
    global KERNEL_BACKEND, DENSITY_BACKEND, ENERGY_BACKEND
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--kernel-backend", choices=("broadcast", "matmul"), default="broadcast")
    parser.add_argument("--density-backend", choices=("legacy", "contracted"), default="legacy")
    parser.add_argument("--energy-backend", choices=("legacy", "device"), default="legacy")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--case", choices=CASES, default="tiny")
    generate.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    generate.add_argument("--fixed-radial", action="store_true")
    generate.add_argument("--features", type=int)
    generate.add_argument("--order", type=int)
    generate.add_argument("--modules", type=int)
    generate.add_argument("--energy-normalize", type=int, choices=(0, 1, 2))
    generate.add_argument("--grid-points", type=int)
    generate.add_argument("--output", type=Path, required=True)
    for name in ("snapshot", "compare", "benchmark"):
        sub = commands.add_parser(name)
        sub.add_argument("--fixture", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "compare":
            sub.add_argument("--reference", type=Path, required=True)
        if name != "benchmark":
            sub.add_argument("--skip-finite-differences", action="store_true")
        else:
            sub.add_argument("--mode", choices=MODES, required=True)
            sub.add_argument("--warmup", type=int, default=10)
            sub.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    KERNEL_BACKEND = args.kernel_backend
    DENSITY_BACKEND, ENERGY_BACKEND = args.density_backend, args.energy_backend
    root = select_source(args.source_root)
    use_full_precision()
    torch.set_num_threads(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "generate":
        fixture = make_fixture(args.case, getattr(torch, args.dtype), not args.fixed_radial,
                               num_features=args.features, order=args.order, num_modules=args.modules,
                               energy_normalize=args.energy_normalize, grid_points=args.grid_points)
        fixture["provenance"] = provenance(root, fixture)
        torch.save(fixture, args.output)
        print(json.dumps({"output": str(args.output), "fixture_hash": fixture["fixture_hash"], "state_hash": fixture["state_hash"]}))
        return
    fixture = load_fixture(args.fixture)
    if args.command in ("snapshot", "compare"):
        result = snapshot(fixture, args.device, not args.skip_finite_differences)
        result["provenance"] = provenance(root, fixture)
        if args.command == "snapshot":
            torch.save(result, args.output)
            print(json.dumps({"output": str(args.output), "contracts": list(result["contracts"]),
                              "finite_differences": _json_tree(result.get("finite_differences"))}))
        else:
            reference = torch.load(args.reference, weights_only=False, map_location="cpu")
            compared = compare_snapshots(reference, result, fixture["spec"]["dtype"])
            compared["reference_provenance"] = reference["provenance"]
            compared["candidate_provenance"] = result["provenance"]
            args.output.write_text(json.dumps(compared, indent=2) + "\n")
            print(json.dumps({"output": str(args.output), "passed": compared["passed"], "failures": compared["failures"]}))
            if not compared["passed"]:
                raise SystemExit(1)
    else:
        from benchmarks.timing import synchronize, timed_block
        started = time.perf_counter()
        step, model = prepared_step(fixture, args.mode, args.device)
        synchronize(args.device)
        cold = time.perf_counter() - started
        for _ in range(args.warmup):
            step()
        synchronize(args.device)
        timing = timed_block(step, device=args.device, iterations=args.iterations)
        result = {"mode": args.mode, "case": fixture["spec"]["case"], "device": args.device,
                  "warmup_calls": args.warmup, "cold_model_seconds": cold, **timing,
                  "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "boundary": "prepared device inputs -> requested outputs; force-training includes zero_grad/loss/backward/Adam step",
                  "excludes": "fixture deserialization, host preparation, transfers, model construction, logging",
                  "provenance": provenance(root, fixture), "final_state_hash": tensor_hash(model.state_dict())}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "mode": args.mode, "seconds_per_call": result["seconds_per_call"]}))


if __name__ == "__main__":
    main()
