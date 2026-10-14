"""Three-backend, paper-system DenSnet numerical and latency instrument.

Each invocation runs one backend in a fresh qualified runtime. All outputs,
loss gradients and first SGD updates are archived for an independent gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

MODES = ("ef", "calculator", "density", "density_update", "energy_force_update")
BACKENDS = ("torch-eager", "torch-compile", "jax")
LR = 1e-5


def numpy_tree(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return np.asarray(value).copy()
    if isinstance(value, dict):
        return {k: numpy_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [numpy_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(numpy_tree(v) for v in value)
    return value


def generate(root, cases, grid_points=1000, batch_size=1):
    import torch
    from benchmarks.application_fixtures import application_spec, application_atoms, build_application_model
    from benchmarks.fixtures import tensor_hash
    from benchmarks.precision import use_full_precision
    from equiv_dens_jax.convert import export_module
    from equiv_dens_jax.serialization import save_tree
    use_full_precision()
    torch.set_num_threads(1)
    for case in cases:
        spec = application_spec(case)
        inputs, frames = application_atoms(grid_points, batch_size)
        atoms = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
        model = build_application_model(spec, "ef")
        model.no_force_props = ["density"]
        output = model(atoms)
        for field in ("density", "density_grad", "energy", "forces"):
            prediction = output[field].detach()
            offset = torch.linspace(.011, .029, prediction.numel(), dtype=torch.float64).reshape(prediction.shape)
            atoms[field] = prediction + offset
        node, state, skipped = export_module(model)
        payload = {"model": node, "state": state, "inputs": numpy_tree(atoms),
                   "torch_state": numpy_tree(model.state_dict()), "spec": spec, "frames": frames,
                   "metadata": {"case": case, "input_hash": tensor_hash(atoms),
                                "state_hash": tensor_hash(model.state_dict()), "skipped": skipped,
                                "atoms_per_molecule": 9, "batch_size": batch_size,
                                "grid_points": grid_points,
                                "parameter_count": sum(p.numel() for p in model.parameters()),
                                "weights": "identical initialized arrays; no accuracy claim",
                                "targets": "initial prediction plus deterministic nonzero offsets"}}
        save_tree(root / case, payload)
        print("GENERATED", json.dumps(payload["metadata"]), flush=True)


def frame_inputs(bundle, index):
    """Only coordinate values change; compressed topology and labels stay fixed."""
    result = dict(bundle["inputs"])
    positions = bundle["frames"]["positions"]
    batch_size = result["batch_atom_numbers"].shape[0]
    indices = (index + np.arange(batch_size)) % len(positions)
    result["batch_positions"] = positions[indices].copy()
    result["positions"] = result["batch_positions"].reshape(1, -1, 3)
    return result


def calculator_inputs(bundle, index):
    """Host geometry packing, including the complete directed molecular graph."""
    from equiv_dens_jax.data import prepare_atoms
    frame = frame_inputs(bundle, index)
    prepared = prepare_atoms({"atom_numbers": frame["batch_atom_numbers"],
                              "positions": frame["batch_positions"]})
    return dict(frame, **prepared)


def compare(paths, output):
    from equiv_dens_jax.serialization import load_tree
    left, right = (load_tree(path.with_suffix("")) for path in paths)
    rows = []
    def visit(a, b, path=""):
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                raise AssertionError("Key mismatch at " + path)
            for key in a:
                visit(a[key], b[key], path + "/" + str(key))
        else:
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape:
                raise AssertionError("Shape mismatch at " + path)
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            passed = finite and bool(np.allclose(a, b, rtol=1e-8, atol=1e-8))
            rows.append({"field": path, "pass": passed,
                         "max_abs": float(np.max(np.abs(a-b), initial=0)),
                         "max_scale": float(np.max(np.abs(a), initial=0))})
    visit(left, right)
    report = {"pass": bool(rows) and all(row["pass"] for row in rows),
              "atol": 1e-8, "rtol": 1e-8, "rows": rows}
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["pass"]:
        raise AssertionError("Backend numerical equivalence failed; see " + str(output))


def worker(args):
    from equiv_dens_jax.serialization import load_tree, save_tree
    bundle = load_tree(args.fixture)
    update = args.mode.endswith("_update")
    density = args.mode in ("density", "density_update")
    fields = ("density", "density_grad") if density else ("energy", "forces")
    weights = (1., .1) if density else (.2, .3)
    prefix = "density_repr_model." if density else "property_models.energy."
    started = time.perf_counter()
    diagnostics = {}
    if args.backend.startswith("torch"):
        import torch
        from benchmarks.application_fixtures import build_application_model
        from benchmarks.precision import use_full_precision
        use_full_precision()
        torch.set_num_threads(1)
        model = build_application_model(bundle["spec"], "ef" if args.mode == "calculator" else args.mode).to(args.device)
        model.load_state_dict({k: torch.as_tensor(v, device=args.device) for k, v in bundle["torch_state"].items()}, strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(update and name.startswith(prefix))
        def arrays(tree):
            if isinstance(tree, np.ndarray):
                return torch.as_tensor(tree, device=args.device)
            if isinstance(tree, dict):
                return {k: arrays(v) for k, v in tree.items()}
            return tree
        data = arrays(bundle["inputs"])
        frames = [arrays(frame_inputs(bundle, i)) for i in range(len(bundle["frames"]["positions"]))]
        active = {k: p for k, p in model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.SGD(active.values(), lr=LR) if update else None
        compiled = args.backend == "torch-compile"
        if compiled:
            torch._dynamo.config.suppress_errors = False
            compile_threads = int(os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "1"))
            torch._inductor.config.compile_threads = compile_threads
            torch._dynamo.utils.counters.clear()
            if model.calculate_forces:
                model.enable_compile()
            else:  # density: compile representation + coefficient head, the grid expansion stays eager
                from equiv_dens.nn.dft_network import DFTNetwork
                repr_model, dens_head = model.density_repr_model[0], model.density_repr_model[1]
                expansion = model.property_models["density"]
                zs = data["batch_atom_numbers"].max(0)[0].tolist()
                dens_head.warm_caches(zs, data["positions"].device, data["positions"].dtype)
                coeffs_fn = torch.compile(lambda a: dens_head(repr_model(a)), dynamic=True)
        def model_call(atoms):
            if compiled and not model.calculate_forces:
                inputs = {k: v for k, v in atoms.items()
                          if isinstance(v, torch.Tensor) and k not in DFTNetwork.label_keys}
                out = dict(atoms)
                with torch.set_grad_enabled(update):
                    out.update(coeffs_fn(dict(inputs)))
                    return expansion(out, density_grad=True)
            return model(dict(atoms))
        def execute(index=0):
            atoms = (arrays(calculator_inputs(bundle, index)) if args.mode == "calculator" else
                     frames[index % len(frames)] if args.mode == "ef" else data)
            if update:
                optimizer.zero_grad(set_to_none=True)
            out = model_call(atoms)
            result = {field: out[field] for field in fields}
            if update:
                loss = sum(weight * (out[field]-data[field]).square().mean()
                           for field, weight in zip(fields, weights))
                loss.backward()
                optimizer.step()
                result["loss"] = loss.detach()
            return numpy_tree(result) if args.mode == "calculator" else result
        def parameters():
            return dict(model.named_parameters())
        def gradients():
            return {k: p.grad if p.grad is not None else torch.zeros_like(p) for k, p in active.items()}
        def compiler_diagnostics():
            counters = {category: dict(values) for category, values in torch._dynamo.utils.counters.items()}
            return {"backend": "inductor", "fullgraph": False, "compile_strategy": args.compile_strategy,
                    "dynamic": True, "compile_threads": compile_threads,
                    "suppress_errors": torch._dynamo.config.suppress_errors, "counters": counters}
        def synchronize(value=None):
            if args.device == "cuda":
                torch.cuda.synchronize()
        def memory():
            if args.device == "cpu":
                return {"live_bytes": None, "peak_live_bytes": None, "reserved_bytes": None}
            return {"live_bytes": torch.cuda.memory_allocated(),
                    "peak_live_bytes": torch.cuda.max_memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved()}
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        backend_metadata = {"version": torch.__version__, "cuda": torch.version.cuda,
                            "matmul_precision": "highest; TF32 disabled",
                            "compile_threads": int(os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "1")) if args.backend == "torch-compile" else None,
                            "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu"}
    else:
        import jax
        import jax.numpy as jnp
        from equiv_dens_jax.pipeline import Pipeline
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        if jax.devices()[0].platform != ("gpu" if args.device == "cuda" else "cpu"):
            raise RuntimeError("JAX device does not match the requested benchmark device")
        pipeline = Pipeline(bundle["model"], bundle["state"])
        data = pipeline.prepare(bundle["inputs"])
        frames = [pipeline.prepare(frame_inputs(bundle, i)) for i in range(len(bundle["frames"]["positions"]))]
        parameters_state = pipeline.state["params"]
        buffers = pipeline.state["buffers"]
        active_names = tuple(k for k in parameters_state if k.startswith(prefix))
        def core(parameters, atoms):
            out = pipeline.forward({"params": parameters, "buffers": buffers}, atoms,
                                   training=update, properties=("density",) if density else ("energy",),
                                   forces=not density)
            result = {field: out[field] for field in fields}
            if update:
                result["loss"] = sum(weight * jnp.mean((out[field]-data[field])**2)
                                     for field, weight in zip(fields, weights))
            return result
        if update:
            @jax.jit
            def operation(parameters, atoms):
                active_parameters = {k: parameters[k] for k in active_names}
                def objective(value):
                    out = core(dict(parameters, **value), atoms)
                    return out["loss"], out
                (_, out), grad = jax.value_and_grad(objective, has_aux=True)(active_parameters)
                new = dict(parameters)
                new.update({k: parameters[k]-LR*value for k, value in grad.items()})
                return new, out, grad
        else:
            operation = jax.jit(core)
        last_gradient = {}
        def execute(index=0):
            nonlocal parameters_state, last_gradient
            atoms = (pipeline.prepare(calculator_inputs(bundle, index)) if args.mode == "calculator" else
                     frames[index % len(frames)] if args.mode == "ef" else data)
            if update:
                parameters_state, out, last_gradient = operation(parameters_state, atoms)
            else:
                out = operation(parameters_state, atoms)
            return numpy_tree(out) if args.mode == "calculator" else out
        def parameters():
            return parameters_state
        def gradients():
            return last_gradient
        def synchronize(value=None):
            jax.block_until_ready((value, parameters_state, last_gradient) if update else value)
        def memory():
            stats = jax.devices()[0].memory_stats() or {}
            return {"live_bytes": stats.get("bytes_in_use"), "peak_live_bytes": stats.get("peak_bytes_in_use"),
                    "native_stats": stats}
        backend_metadata = {"version": jax.__version__, "matmul_precision": "highest",
                            "device": str(jax.devices()[0])}
    setup_seconds = time.perf_counter()-started
    first_start = time.perf_counter()
    first = execute(0)
    synchronize(first)
    cold_seconds = time.perf_counter()-first_start
    regression = {"outputs": numpy_tree(first),
                  "one_step_params": numpy_tree(parameters()) if update else {},
                  "gradients": numpy_tree(gradients()) if update else {}}
    if args.mode in ("ef", "calculator"):
        regression["changed_coordinates"] = {}
        for index in (1, 9):
            result = execute(index)
            synchronize(result)
            regression["changed_coordinates"][str(index)] = numpy_tree(result)
    save_tree(args.output.with_suffix(""), regression)
    for index in range(args.warmup):
        synchronize(execute(index+1))
    if args.backend == "torch-compile":
        diagnostics["before_timing"] = compiler_diagnostics()
    before = memory()
    times = []
    for index in range(args.iterations):
        start = time.perf_counter()
        result = execute(index+args.warmup+1)
        synchronize(result)
        times.append(1000. * (time.perf_counter()-start))
    if args.backend == "torch-compile":
        diagnostics["after_timing"] = compiler_diagnostics()
        if diagnostics["before_timing"] != diagnostics["after_timing"]:
            raise AssertionError("Torch compiled execution recompiled during timed calls")
        breaks = diagnostics["after_timing"]["counters"].get("graph_break", {})
        if breaks:
            raise AssertionError("Unexpected graph breaks in strict fullgraph workload")
    final = numpy_tree(result)
    if not all(np.isfinite(value).all() for value in final.values()):
        raise AssertionError("Nonfinite outputs")
    if update:
        final_params = numpy_tree(parameters())
        for name, value in final_params.items():
            if not np.isfinite(value).all():
                raise AssertionError("Nonfinite updated parameter " + name)
            if not name.startswith(prefix) and not np.array_equal(value, bundle["state"]["params"][name]):
                raise AssertionError("Frozen parameter changed " + name)
        if not any(not np.array_equal(value, bundle["state"]["params"][name])
                   for name, value in final_params.items() if name.startswith(prefix)):
            raise AssertionError("Requested parameter stage did not update")
    record = {"backend": args.backend, "mode": args.mode, "case": bundle["metadata"]["case"],
              "compile_strategy": args.compile_strategy if args.backend == "torch-compile" else None,
              "times_ms": times, "setup_seconds": setup_seconds, "cold_first_call_seconds": cold_seconds,
              "memory_before_timing": before, "memory_after_timing": memory(),
              "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "metadata": bundle["metadata"], "backend_metadata": backend_metadata,
              "compiler_diagnostics": diagnostics,
              "provenance": {"host": os.uname().nodename, "allocation": os.getenv("SLURM_JOB_ID"),
                             "runtime": os.getenv("APPTAINER_CONTAINER"),
                             "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                             "precision": "float64", "optimizer": "SGD; no momentum/decay/EMA/Lookahead",
                             "learning_rate": LR,
                             "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"backend": args.backend, "mode": args.mode,
                      "median_ms": float(np.median(times)), "cold_seconds": cold_seconds}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", type=Path)
    parser.add_argument("--cases", nargs="+", default=["ethanol-paper"])
    parser.add_argument("--grid-points", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compile-strategy", choices=("fullgraph",), default="fullgraph")
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--comparison-output", type=Path)
    args = parser.parse_args()
    if args.generate:
        generate(args.generate, args.cases, args.grid_points, args.batch_size)
    elif args.compare:
        compare(args.compare, args.comparison_output)
    else:
        worker(args)


if __name__ == "__main__":
    main()
