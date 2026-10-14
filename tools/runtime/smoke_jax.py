#!/usr/bin/env python3
"""Small numerical/runtime smoke check for the DenSNet JAX environment."""

import argparse
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import sys
import traceback


VERSION_DISTRIBUTIONS = {
    "jax": "jax",
    "jaxlib": "jaxlib",
    "jax-cuda12-plugin": "jax-cuda12-plugin",
    "jax-cuda12-pjrt": "jax-cuda12-pjrt",
    "numpy": "numpy",
    "scipy": "scipy",
    "flax": "flax",
    "optax": "optax",
    "orbax-checkpoint": "orbax-checkpoint",
    "jax-md": "jax-md",
    "jraph": "jraph",
    "e3nn-jax": "e3nn-jax",
    "cuequivariance-jax": "cuequivariance-jax",
    "cuequivariance-ops-jax-cu12": "cuequivariance-ops-jax-cu12",
    "pyscf": "pyscf",
}


def _max_abs_error(left, right):
    import numpy as np

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return float("inf")
    if left_array.size == 0:
        return 0.0
    return float(np.max(np.abs(left_array - right_array)))


def _record_close(checks, name, left, right, tolerance, note):
    error = _max_abs_error(left, right)
    checks[name] = {
        "passed": math.isfinite(error) and error <= tolerance,
        "max_abs_error": error if math.isfinite(error) else None,
        "tolerance": tolerance,
        "scope": note,
    }


def _tree_is_finite(tree):
    import jax
    import numpy as np

    leaves = jax.tree_util.tree_leaves(tree)
    return bool(leaves) and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def _tree_max_abs_error(left, right):
    import jax

    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return float("inf")
    errors = [_max_abs_error(a, b) for a, b in zip(left_leaves, right_leaves)]
    return max(errors, default=0.0)


def _record_tree_close(checks, name, left, right, tolerance, note):
    error = _tree_max_abs_error(left, right)
    checks[name] = {
        "passed": math.isfinite(error) and error <= tolerance,
        "max_abs_error": error if math.isfinite(error) else None,
        "tolerance": tolerance,
        "scope": note,
    }


def _run_numerical_checks(jax, jnp, flax_nn, optax, device_name, report):
    import numpy as np

    checks = report["checks"]
    eager_jit_tolerance = 1e-8
    finite_difference_tolerance = 1e-7

    def energy(parameter, positions):
        radial = jnp.sum(positions**2, axis=-1)
        return parameter * jnp.sum(jnp.exp(-radial)) + 0.1 * jnp.sum(positions**4)

    parameter = jnp.asarray(1.3, dtype=jnp.float64)
    positions = jnp.asarray(
        [[0.7, -0.4, 0.9], [-0.6, 0.8, 0.3]], dtype=jnp.float64
    )
    target_forces = jnp.asarray(
        [[0.1, -0.2, 0.3], [-0.1, 0.15, -0.25]], dtype=jnp.float64
    )

    force = lambda p, r: -jax.grad(energy, argnums=1)(p, r)
    hessian = jax.hessian(energy, argnums=1)
    force_loss = lambda p: jnp.mean((force(p, positions) - target_forces) ** 2)

    eager_energy = energy(parameter, positions)
    jit_energy = jax.jit(energy)(parameter, positions)
    eager_force = force(parameter, positions)
    jit_force = jax.jit(force)(parameter, positions)
    eager_hessian = hessian(parameter, positions)
    jit_hessian = jax.jit(hessian)(parameter, positions)
    eager_mixed_gradient = jax.grad(force_loss)(parameter)
    jit_mixed_gradient = jax.jit(jax.grad(force_loss))(parameter)

    _record_close(
        checks, "energy_eager_vs_jit", eager_energy, jit_energy,
        eager_jit_tolerance, "Analytic scalar energy; eager/JIT parity.",
    )
    _record_close(
        checks, "forces_eager_vs_jit", eager_force, jit_force,
        eager_jit_tolerance, "Negative coordinate gradient; eager/JIT parity.",
    )
    _record_close(
        checks, "hessian_eager_vs_jit", eager_hessian, jit_hessian,
        eager_jit_tolerance, "Coordinate Hessian; eager/JIT parity.",
    )
    _record_close(
        checks, "force_loss_parameter_gradient_eager_vs_jit",
        eager_mixed_gradient, jit_mixed_gradient, eager_jit_tolerance,
        "Gradient of mean squared force error with respect to scalar energy parameter.",
    )
    checks["energy_force_hessian_finite"] = {
        "passed": all(
            np.all(np.isfinite(np.asarray(value)))
            for value in (eager_energy, eager_force, eager_hessian, eager_mixed_gradient)
        ),
        "scope": "Finite eager energy, force, Hessian, and force-loss parameter gradient.",
    }

    finite_difference_step = 1e-5
    finite_difference_force = np.empty((2, 3), dtype=np.float64)
    positions_host = np.asarray(positions)
    for atom in range(positions_host.shape[0]):
        for axis in range(positions_host.shape[1]):
            plus = positions_host.copy()
            minus = positions_host.copy()
            plus[atom, axis] += finite_difference_step
            minus[atom, axis] -= finite_difference_step
            finite_difference_force[atom, axis] = -(
                float(energy(parameter, jnp.asarray(plus, dtype=jnp.float64)))
                - float(energy(parameter, jnp.asarray(minus, dtype=jnp.float64)))
            ) / (2.0 * finite_difference_step)
    _record_close(
        checks, "forces_vs_central_finite_difference", eager_force,
        finite_difference_force, finite_difference_tolerance,
        "Central finite difference of energy, h=1e-5.",
    )

    class TinyNetwork(flax_nn.Module):
        @flax_nn.compact
        def __call__(self, inputs):
            hidden = flax_nn.Dense(4, dtype=jnp.float64, param_dtype=jnp.float64)(inputs)
            return flax_nn.Dense(1, dtype=jnp.float64, param_dtype=jnp.float64)(jnp.tanh(hidden))

    network = TinyNetwork()
    network_inputs = jnp.asarray(
        [[0.2, -0.1], [0.5, 0.7], [-0.3, 0.4]], dtype=jnp.float64
    )
    network_targets = jnp.asarray([[0.1], [-0.2], [0.3]], dtype=jnp.float64)
    params = network.init(jax.random.PRNGKey(0), network_inputs)["params"]
    network_apply = lambda p, x: network.apply({"params": p}, x)
    eager_output = network_apply(params, network_inputs)
    jit_output = jax.jit(network_apply)(params, network_inputs)
    _record_close(
        checks, "flax_forward_eager_vs_jit", eager_output, jit_output,
        eager_jit_tolerance, "Tiny Flax Dense/tanh/Dense forward pass.",
    )

    def network_loss(p):
        prediction = network_apply(p, network_inputs)
        return jnp.mean((prediction - network_targets) ** 2)

    optimizer = optax.adam(learning_rate=1e-2)
    optimizer_state = optimizer.init(params)

    def train_step(p, state):
        loss_value, gradients = jax.value_and_grad(network_loss)(p)
        updates, new_state = optimizer.update(gradients, state, p)
        new_params = optax.apply_updates(p, updates)
        return loss_value, new_params, new_state, gradients

    eager_loss, eager_updated_params, eager_optimizer_state, eager_network_grads = train_step(
        params, optimizer_state
    )
    jit_loss, jit_updated_params, jit_optimizer_state, jit_network_grads = jax.jit(train_step)(
        params, optimizer_state
    )
    _record_close(
        checks, "flax_optax_loss_eager_vs_jit", eager_loss, jit_loss,
        eager_jit_tolerance, "Scalar MSE loss for one Adam update.",
    )
    _record_tree_close(
        checks, "flax_optax_updated_params_eager_vs_jit", eager_updated_params,
        jit_updated_params, eager_jit_tolerance, "Flax parameters after one Optax Adam update.",
    )
    _record_tree_close(
        checks, "flax_optax_state_eager_vs_jit", eager_optimizer_state,
        jit_optimizer_state, eager_jit_tolerance, "Optax Adam state after one update.",
    )
    checks["flax_optax_output_trees_finite"] = {
        "passed": all(
            _tree_is_finite(tree)
            for tree in (
                eager_output,
                eager_network_grads,
                eager_updated_params,
                eager_optimizer_state,
                jit_output,
                jit_network_grads,
                jit_updated_params,
                jit_optimizer_state,
            )
        ) and bool(np.isfinite(np.asarray(eager_loss))),
        "scope": "Finite Flax outputs, gradients, updated parameters, and optimizer states.",
    }

    report["inputs"] = {
        "energy_parameter": 1.3,
        "positions": positions_host.tolist(),
        "dtype": "float64",
        "device": device_name,
        "finite_difference_step": finite_difference_step,
        "eager_jit_tolerance": eager_jit_tolerance,
        "finite_difference_tolerance": finite_difference_tolerance,
    }


def _run_cueq_cuda(jax, jnp, report):
    """Check one uniform CUDA polynomial against independent analytic results.

    This is an operator availability/derivative check, not an equivariant model
    qualification. API: NVIDIA's Segmented Polynomial in JAX tutorial.
    """
    import cuequivariance as cue
    import cuequivariance_jax as cuex

    descriptor = cue.SegmentedTensorProduct.from_subscripts('u,u,u')
    descriptor.add_path(None, None, None, c=1.0, dims={'u': 32})
    polynomial = cue.SegmentedPolynomial(
        inputs=list(descriptor.operands[:2]), outputs=[descriptor.operands[2]],
        operations=[(cue.Operation([0, 1, 2]), descriptor)],
    )
    shape = [jax.ShapeDtypeStruct((32,), jnp.float32)]

    def product(x, y):
        return cuex.segmented_polynomial(polynomial, [x, y], shape,
                                        method='uniform_1d')[0]

    x = jnp.linspace(0.1, 0.9, 32, dtype=jnp.float32)
    y = jnp.linspace(0.2, 1.2, 32, dtype=jnp.float32)
    compiled = jax.jit(product)
    lowered = str(compiled.lower(x, y).compiler_ir(dialect='stablehlo'))
    if 'custom_call' not in lowered:
        raise RuntimeError('CuEq uniform_1d did not lower to a CUDA custom call')
    checks = report['checks']
    _record_close(checks, 'cueq_cuda_product', compiled(x, y), x * y, 1e-6,
                  'Explicit uniform_1d custom call; elementwise polynomial.')
    loss = lambda a, b: jnp.sum(product(a, b)**2)
    gradient = jax.grad(loss, argnums=(0, 1))
    first = jax.jit(gradient)(x, y)
    _record_tree_close(checks, 'cueq_cuda_first_derivative', first,
                       (2*x*y**2, 2*y*x**2), 2e-6,
                       'CUDA polynomial gradients vs analytic expressions.')
    second = jax.jit(lambda a, b: jax.jvp(gradient, (a, b),
                          (jnp.ones_like(a), jnp.ones_like(b)))[1])(x, y)
    _record_tree_close(checks, 'cueq_cuda_second_derivative', second,
                       (2*y**2 + 4*x*y, 2*x**2 + 4*x*y), 1e-5,
                       'CUDA polynomial Hessian-vector product vs analytic expressions.')
    mixed = jax.jit(jax.grad(lambda b: jnp.mean(gradient(x, b)[0]**2)))(y)
    _record_close(checks, 'cueq_cuda_force_loss_mixed_gradient', mixed,
                  16*x**2*y**3/32, 2e-6,
                  'Mixed force-loss gradient through the CUDA polynomial.')
    report['scopes']['cuequivariance'] = 'Uniform_1d polynomial and its derivatives only; no complete backbone.'
    report['scopes']['cuda_kernels'] = 'Explicit CUDA custom call tested; full future operator coverage remains unqualified.'


def _import_runtime(report):
    import jax

    jax.config.update("jax_enable_x64", True)
    report["jax_enable_x64"] = bool(jax.config.read("jax_enable_x64"))
    report["jax_disable_jit"] = bool(jax.config.jax_disable_jit)
    if report["jax_disable_jit"]:
        raise RuntimeError("JAX JIT is disabled; compiler qualification would be invalid")
    import jax.numpy as jnp
    from flax import linen as flax_nn
    import optax
    import orbax.checkpoint
    import jax_md
    import jraph
    import e3nn_jax
    import cuequivariance
    from pyscf import gto as pyscf_gto

    imported = {
        "jax": jax,
        "jax.numpy": jnp,
        "flax.linen": flax_nn,
        "optax": optax,
        "orbax.checkpoint": orbax.checkpoint,
        "jax_md": jax_md,
        "jraph": jraph,
        "e3nn_jax": e3nn_jax,
        "cuequivariance": cuequivariance,
        "pyscf.gto": pyscf_gto,
    }
    # The CUDA-enabled CuEq frontend eagerly loads libcuda, even for CPU work.
    # Keep core JAX usable on CPU-only nodes; CUDA qualification must load and
    # execute that frontend and may not downgrade a library error to a skip.
    if report["device_requested"] == "cuda":
        import cuequivariance_jax
        imported["cuequivariance_jax"] = cuequivariance_jax
    else:
        report["optional_imports"] = {
            "cuequivariance_jax": {
                "status": "not_tested",
                "reason": "CUDA-enabled frontend requires NVIDIA driver libraries; qualify with --device cuda on a GPU node.",
            }
        }
        report["scopes"]["cuequivariance"] = "Descriptor package imported; CUDA-enabled JAX frontend requires GPU-node qualification."
    report["imports"] = {name: True for name in imported}
    report["versions"] = {
        name: metadata.version(distribution)
        for name, distribution in VERSION_DISTRIBUTIONS.items()
    }
    report["checks"]["jax_enable_x64"] = {
        "passed": report["jax_enable_x64"],
        "scope": "JAX float64 mode enabled before numerical checks.",
    }
    return jax, jnp, flax_nn, optax


def _select_device(jax, requested):
    devices = jax.devices()
    device_rows = [
        {
            "platform": str(device.platform),
            "kind": str(device.device_kind),
            "id": int(device.id),
            "description": str(device),
        }
        for device in devices
    ]
    if requested == "cuda":
        gpu_devices = [
            device for device in devices
            if str(device.platform).lower() in {"gpu", "cuda"}
        ]
        if not gpu_devices:
            raise RuntimeError(
                "CUDA was requested, but JAX did not expose an actual GPU device; "
                f"visible devices: {device_rows}"
            )
        selected = gpu_devices[0]
    else:
        cpu_devices = [device for device in devices if str(device.platform).lower() == "cpu"]
        if not cpu_devices:
            raise RuntimeError(f"CPU was requested, but JAX exposed no CPU device: {device_rows}")
        selected = cpu_devices[0]
    return selected, device_rows


def _write_report(report, output_path):
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if output_path is None or output_path == "-":
        print(rendered)
    else:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", help="Write JSON here; omit or use '-' for stdout.")
    args = parser.parse_args(argv)

    # This must be selected before importing JAX; no implicit CPU fallback for CUDA.
    os.environ["JAX_PLATFORMS"] = args.device
    report = {
        "status": "failed",
        "device_requested": args.device,
        "jax_platforms": os.environ["JAX_PLATFORMS"],
        "imports": {},
        "versions": {},
        "devices": [],
        "checks": {},
        "scopes": {
            "cuequivariance": "Import check only; no cuEquivariance operator was executed.",
            "cuda_kernels": "UNQUALIFIED; import success does not qualify CUDA kernels.",
            "pyscf": "Import check only; no DFT calculation was executed.",
            "jax_runtime": "Tiny analytic derivatives and MLP update only; no project workflow qualification.",
        },
    }

    try:
        jax, jnp, flax_nn, optax = _import_runtime(report)
        selected_device, devices = _select_device(jax, args.device)
        report["devices"] = devices
        report["device_selected"] = {
            "platform": str(selected_device.platform),
            "kind": str(selected_device.device_kind),
            "id": int(selected_device.id),
            "description": str(selected_device),
        }
        _run_numerical_checks(
            jax, jnp, flax_nn, optax, selected_device.platform, report
        )
        if args.device == 'cuda':
            _run_cueq_cuda(jax, jnp, report)
        report["status"] = (
            "passed"
            if all(check.get("passed", False) for check in report["checks"].values())
            else "failed"
        )
    except Exception as error:  # Preserve a machine-readable failure report.
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()

    try:
        _write_report(report, args.output)
    except Exception as error:
        print(f"failed to write smoke report: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
