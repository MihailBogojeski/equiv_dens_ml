#!/usr/bin/env python3
"""Qualify tiny DenSNet/Torch capabilities without checkpoints or datasets.

Run inside the selected runtime, from a checkout mounted into the container:
    python tools/runtime/smoke_torch.py --device cpu
    python tools/runtime/smoke_torch.py --device cuda

Only an analytic torch.func probe is compiled. This does not qualify compilation
of the DenSNet pipeline, real model accuracy, production MD, or performance.
CUDA tensor-product higher derivatives are a separate future-backbone gate:
an explicitly unsupported API is reported, never treated as a passing gate.
Stdout contains one JSON document; dependency diagnostics go to stderr.
"""

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback


class UnsupportedHigherDerivative(RuntimeError):
    """The selected CUDA primitive explicitly lacks a higher derivative."""


class Checks:
    def __init__(self):
        self.results = []

    def run(self, name, function, *, required=True):
        result = {"name": name, "required": required}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                details = function()
            result.update(status="passed", details=details or {})
        except UnsupportedHigherDerivative as error:
            result.update(status="unsupported", error=str(error))
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc())
        self.results.append(result)
        print(f"[{result['status']}] {name}", file=sys.stderr, flush=True)
        return result["status"] == "passed"

    def omit(self, name, reason, *, required=False):
        self.results.append({"name": name, "required": required,
                             "status": "not_tested", "reason": reason})

    def passed(self, name):
        return any(row["name"] == name and row["status"] == "passed"
                   for row in self.results)


def import_module(name):
    module = importlib.import_module(name)
    return {"module": name, "path": getattr(module, "__file__", None)}


def package_versions():
    versions = {}
    for name in ("torch", "torchvision", "torchaudio", "numpy", "scipy", "sympy",
                 "schnetpack", "pyscf", "gpu4pyscf-cuda12x", "cupy-cuda12x",
                 "cutensor-cu12", "mace-torch", "e3nn", "cuequivariance",
                 "cuequivariance-torch", "cuequivariance-ops-torch-cu12"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_compat():
    if any(name == "schnetpack" or name.startswith("schnetpack.")
           for name in sys.modules):
        raise AssertionError("SchNetPack was imported before equiv_dens.compat")
    import equiv_dens.compat  # noqa: F401
    import torch.utils.data.dataloader as dataloader

    if not hasattr(dataloader, "T_co"):
        raise AssertionError("Project compatibility patch did not expose T_co")
    return {"project_compat_before_schnetpack": True, "T_co_available": True}


def finite(tensor):
    import torch

    if not bool(torch.isfinite(tensor).all()):
        raise AssertionError("Nonfinite numerical result")


def compare(actual, reference, *, atol=1e-9, rtol=1e-7):
    import torch

    finite(actual)
    finite(reference)
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    return float((actual - reference).abs().max().detach().cpu())


def check_device(device):
    import torch

    torch.set_num_threads(1)
    details = {"torch": torch.__version__, "python": sys.version,
               "torch_cuda_build": torch.version.cuda, "requested_device": device}
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA qualification requires an actual accessible GPU")
        tensor = torch.ones(2, device="cuda") * 2
        compare(tensor.cpu(), torch.full((2,), 2.0))
        torch.cuda.synchronize()
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        details.update(device_index=torch.cuda.current_device(),
                       device_name=properties.name,
                       compute_capability=[properties.major, properties.minor])
    return details


def check_vision_audio(device):
    import torch
    import torchvision
    import torchaudio

    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.],
                          [2., 2., 3., 3.]], device=device)
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    if keep.cpu().tolist() != [0, 2]:
        raise AssertionError("torchvision native NMS returned unexpected indices")
    if not torchaudio._extension._IS_TORCHAUDIO_EXT_AVAILABLE:
        raise AssertionError("torchaudio native extension is absent")
    waveform = torch.linspace(-0.5, 0.5, 32, device=device).reshape(1, -1)
    coeffs = torch.tensor([1., 0.], device=device)
    filtered = torchaudio.functional.lfilter(waveform, coeffs, coeffs)
    error = compare(filtered, waveform, atol=1e-6, rtol=1e-6)
    return {"device": device, "torchvision": torchvision.__version__,
            "torchaudio": torchaudio.__version__, "native_nms": True,
            "torchaudio_native_extension": True, "lfilter_max_abs_error": error}


def check_cli(repo):
    process = subprocess.run([sys.executable, str(repo / "run.py"), "--help"],
                             cwd=repo, capture_output=True, text=True, timeout=60)
    if process.returncode != 0:
        raise RuntimeError(f"CLI --help failed ({process.returncode}): {process.stderr}")
    if "train" not in process.stdout or "gxtb-md" not in process.stdout:
        raise AssertionError("CLI help did not expose the expected project commands")
    return {"entry_point": str(repo / "run.py"), "help_exit_code": 0,
            "scope": "Help and dependency imports; no workflow dispatched"}


def check_representation(device):
    import torch
    from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
    from equiv_dens.nn.representation import EquivariantSphericalHarmonics

    torch.manual_seed(1729)
    # The real contract is {Z: [(Z, radial label, angular momentum), ...]}.
    # Supply a small CG matrix instead of the class's much larger L10 default.
    model = EquivariantSphericalHarmonics(
        orbital_basis={1: [(1, 0, 0)], 8: [(8, 0, 0), (8, 1, 1)]},
        order=1, mixing_order=1, num_features=4, num_basis_functions=4,
        num_radial_components=2, num_modules=1, cutoff=6.0,
        clebsch_gordan=ClebschGordanMatrix(order_max=1), normalize=0,
    ).to(device=device, dtype=torch.float64).eval()
    positions = torch.tensor([[[0.13, -0.20, 0.11], [1.53, 0.72, -0.27],
                               [-1.14, 0.85, 0.51]]], device=device,
                             dtype=torch.float64, requires_grad=True)
    atom_numbers = torch.tensor([[8, 1, 1]], device=device, dtype=torch.long)
    idx_i = torch.tensor([0, 0, 1, 1, 2, 2], device=device, dtype=torch.long)
    idx_j = torch.tensor([1, 2, 0, 2, 0, 1], device=device, dtype=torch.long)

    def evaluate(coords):
        # forward mutates its atom dictionary, so every call gets a fresh one.
        return model({"positions": coords, "atom_numbers": atom_numbers,
                      "idx_i": idx_i, "idx_j": idx_j,
                      "atom_mask": torch.ones_like(atom_numbers, dtype=coords.dtype)})

    def scalar(coords):
        # An invariant synthetic readout, not a physical energy/density head.
        return sum(feature.square().mean()
                   for feature in evaluate(coords)["sph_repr"])

    features = evaluate(positions)["sph_repr"]
    shapes = [list(feature.shape) for feature in features]
    if shapes != [[1, 3, 1, 4], [1, 3, 3, 4]]:
        raise AssertionError(f"Unexpected sph_repr shapes: {shapes}")
    for feature in features:
        finite(feature)
    value = scalar(positions)
    gradient, = torch.autograd.grad(value, positions, create_graph=True)
    direction = torch.tensor([[[0.2, -0.3, 0.4], [-0.7, 0.1, 0.2],
                               [0.5, 0.2, -0.6]]], device=device, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)
    hvp, = torch.autograd.grad((gradient * direction).sum(), positions,
                              create_graph=True, retain_graph=True)
    finite(gradient)
    finite(hvp)
    if (float(gradient.detach().abs().max()) < 1e-10
            or float(hvp.detach().abs().max()) < 1e-10):
        raise AssertionError("The representation derivative probe has no useful signal")
    step = 1e-4
    detached = positions.detach()
    fd_first = (scalar(detached + step * direction)
                - scalar(detached - step * direction)) / (2 * step)
    first_error = compare((gradient * direction).sum(), fd_first,
                          atol=2e-6, rtol=3e-4)

    def position_gradient(coords):
        coords = coords.detach().requires_grad_(True)
        return torch.autograd.grad(scalar(coords), coords)[0].detach()

    fd_hvp = (position_gradient(detached + step * direction)
              - position_gradient(detached - step * direction)) / (2 * step)
    hvp_error = compare(hvp, fd_hvp, atol=3e-6, rtol=3e-4)
    force_loss = gradient.square().mean()
    parameter_gradients = torch.autograd.grad(
        force_loss, tuple(model.parameters()), allow_unused=True)
    used = [grad for grad in parameter_gradients if grad is not None]
    if not used or max(float(grad.abs().max()) for grad in used) < 1e-10:
        raise AssertionError("Force-loss parameter differentiation has no useful signal")
    for grad in used:
        finite(grad)
    translated = scalar(detached + detached.new_tensor([0.3, -0.1, 0.2]))
    translation_error = compare(translated, value.detach(), atol=1e-9, rtol=1e-7)
    return {"device": device, "dtype": "float64", "sph_repr_shapes": shapes,
            "finite_difference_step": step, "directional_first_max_abs_error": first_error,
            "hessian_vector_max_abs_error": hvp_error,
            "force_loss_parameter_gradients": len(used),
            "translation_invariant_readout_max_abs_error": translation_error,
            "scope": "Actual tiny representation and synthetic readout; no production head"}


def analytic_probe(weights, positions, target):
    import torch

    def energy(parameters, coords):
        displacement = coords[1] - coords[0]
        return (0.5 * parameters[0] * displacement.square().sum()
                + 0.25 * parameters[1] * displacement.pow(4).sum())

    def loss(parameters, coords, expected_forces):
        forces = -torch.func.grad(energy, argnums=1)(parameters, coords)
        return (forces - expected_forces).square().mean()

    return (energy(weights, positions),
            -torch.func.grad(energy, argnums=1)(weights, positions),
            loss(weights, positions, target),
            torch.func.grad(loss, argnums=0)(weights, positions, target),
            torch.compiler.is_compiling())


def analytic_reference(weights, positions, target):
    import torch

    displacement = positions[1] - positions[0]
    derivative = weights[0] * displacement + weights[1] * displacement.pow(3)
    forces = torch.stack([derivative, -derivative])
    residual = forces - target
    loss_gradient = torch.stack([
        (2 * residual * torch.stack([displacement, -displacement])).mean(),
        (2 * residual * torch.stack([displacement.pow(3), -displacement.pow(3)])).mean(),
    ])
    energy = (0.5 * weights[0] * displacement.square().sum()
              + 0.25 * weights[1] * displacement.pow(4).sum())
    return energy, forces, residual.square().mean(), loss_gradient


def check_analytic(device, *, compile_probe=False):
    import torch

    weights = torch.tensor([0.7, 0.2], device=device, dtype=torch.float64)
    positions = torch.tensor([[0.1, -0.2, 0.3], [0.8, 0.4, -0.1]],
                             device=device, dtype=torch.float64)
    target = torch.full_like(positions, 0.05)
    probe = analytic_probe
    # model_loader sets suppress_errors=True on import. Explicitly override it
    # so qualification cannot silently degrade to an eager implementation.
    with torch._dynamo.config.patch(suppress_errors=False):
        if compile_probe:
            probe = torch.compile(analytic_probe, backend="inductor", fullgraph=True,
                                  dynamic=False)
        errors = []
        for scale in (1.0, 1.1):
            inputs = weights * scale, positions * scale, target
            output = probe(*inputs)
            if bool(output[-1]) != compile_probe:
                raise AssertionError("Compiler execution marker disagrees with requested mode")
            reference = analytic_reference(*inputs)
            eager = analytic_probe(*inputs)
            cpu = analytic_probe(*(item.cpu() for item in inputs))
            for index in range(4):
                errors.append(compare(output[index], reference[index], atol=1e-10, rtol=1e-9))
                compare(output[index], eager[index], atol=1e-10, rtol=1e-9)
                compare(output[index].cpu(), cpu[index], atol=1e-10, rtol=1e-9)
    return {"device": device, "dtype": "float64", "independent_analytic_parity": True,
            "eager_cpu_reference_parity": True, "input_cases": 2,
            "max_abs_error": max(errors), "compiled": compile_probe,
            "backend": "inductor" if compile_probe else None,
            "fullgraph": compile_probe, "compiler_execution_marker": compile_probe,
            "scope": "Analytic torch.func energy, force, force-loss and parameter gradient only"}


def check_gpu_dipole():
    import cupy as cp
    import numpy as np
    from pyscf import gto, lib
    from equiv_dens.integral.dipole_gpu import int1e_r_gpu

    lib.num_threads(1)
    molecule = gto.M(atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
                     basis="sto3g", verbose=0)
    reference = molecule.intor("int1e_r", comp=3)
    actual = int1e_r_gpu(molecule, molecule, xp=cp, r_origin=(0., 0., 0.))
    if not isinstance(actual, cp.ndarray):
        raise AssertionError("GPU dipole result was not a CuPy device array")
    cp.cuda.get_current_stream().synchronize()
    error = float(np.max(np.abs(cp.asnumpy(actual) - reference)))
    if not np.isfinite(error) or error > 1e-10:
        raise AssertionError(f"CuPy/PySCF dipole error {error} exceeds 1e-10 Bohr")
    return {"molecule": "water", "basis": "sto3g", "origin_bohr": [0., 0., 0.],
            "result_shape": list(actual.shape), "cupy_device": actual.device.id,
            "max_abs_error_bohr": error, "tolerance_bohr": 1e-10,
            "cupy": cp.__version__, "cuda_runtime": cp.cuda.runtime.runtimeGetVersion()}


def check_gpu_rhf():
    import cupy as cp
    import numpy as np
    from pyscf import gto, lib, scf
    from gpu4pyscf.scf.hf import RHF as GPURHF

    lib.num_threads(1)
    molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", unit="Angstrom",
                     basis="sto3g", verbose=0)
    cpu = scf.RHF(molecule).density_fit(auxbasis="weigend")
    gpu = GPURHF(molecule).density_fit(auxbasis="weigend")
    for mean_field in (cpu, gpu):
        mean_field.chkfile = None
        mean_field.conv_tol = 1e-11
        mean_field.max_cycle = 30
        mean_field.init_guess = "1e"
        mean_field.verbose = 0
    cpu_energy = float(cpu.kernel())
    gpu_energy = float(gpu.kernel())
    if not cpu.converged or not gpu.converged:
        raise AssertionError("Tiny matched CPU/GPU DF RHF did not converge")
    if not isinstance(gpu.mo_coeff, cp.ndarray):
        raise AssertionError("gpu4pyscf coefficients are not CuPy device arrays")
    cp.cuda.get_current_stream().synchronize()
    error = abs(cpu_energy - gpu_energy)
    if not np.isfinite(error) or error > 2e-8:
        raise AssertionError(f"DF RHF CPU/GPU energy error {error} exceeds 2e-8 Hartree")
    return {"molecule": "H2", "bond_angstrom": 0.74, "basis": "sto3g",
            "auxbasis": "weigend", "cpu_energy_hartree": cpu_energy,
            "gpu_energy_hartree": gpu_energy, "max_abs_error_hartree": error,
            "tolerance_hartree": 2e-8, "cpu_converged": True, "gpu_converged": True,
            "scope": "Tiny density-fitted RHF; no DFT/AIMD capability claim"}


def pbe_mean_field(*, gpu=False):
    from pyscf import dft, gto, lib

    lib.num_threads(1)
    molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", unit="Angstrom",
                     basis="sto3g", verbose=0)
    if gpu:
        from gpu4pyscf.dft.rks import RKS
    else:
        RKS = dft.RKS
    mean_field = RKS(molecule).density_fit(auxbasis="weigend")
    mean_field.xc = "PBE"
    mean_field.grids.level = 0
    mean_field.chkfile = None
    mean_field.conv_tol = 1e-10
    mean_field.max_cycle = 30
    mean_field.init_guess = "1e"
    mean_field.verbose = 0
    return mean_field


def check_cpu_pbe():
    import numpy as np

    mean_field = pbe_mean_field()
    energy = float(mean_field.kernel())
    if not mean_field.converged or not np.isfinite(energy) or not -2.0 < energy < -0.1:
        raise AssertionError(f"Tiny CPU DF PBE did not converge to a sensible energy: {energy}")
    return {"molecule": "H2", "bond_angstrom": 0.74, "basis": "sto3g",
            "auxbasis": "weigend", "xc": "PBE", "grids_level": 0,
            "conv_tol": 1e-10, "energy_hartree": energy, "converged": True,
            "scope": "Tiny DF PBE calculation; no production-grid accuracy claim"}


def check_gpu_pbe(cpu_reference):
    import cupy as cp
    import numpy as np

    if not cpu_reference:
        raise RuntimeError("Matched CPU DF PBE prerequisite failed")
    mean_field = pbe_mean_field(gpu=True)
    energy = float(mean_field.kernel())
    if not mean_field.converged or not isinstance(mean_field.mo_coeff, cp.ndarray):
        raise AssertionError("GPU DF PBE did not converge with CuPy orbital coefficients")
    cp.cuda.get_current_stream().synchronize()
    error = abs(energy - cpu_reference["energy_hartree"])
    if not np.isfinite(error) or error > 1e-7:
        raise AssertionError(f"Matched CPU/GPU DF PBE differs by {error} Hartree (>1e-7)")
    return {**cpu_reference, "gpu_energy_hartree": energy,
            "cpu_energy_hartree": cpu_reference["energy_hartree"],
            "max_abs_error_hartree": error, "tolerance_hartree": 1e-7,
            "gpu_converged": True, "gpu_mo_coeff_is_cupy": True,
            "scope": "Tiny GPU libxc/DF PBE calculation; no AIMD claim"}


def check_d4():
    import numpy as np
    from dftd4.interface import DampingParam, DispersionModel
    from pyscf import gto

    molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", unit="Angstrom",
                     basis="sto3g", verbose=0)
    numbers = np.asarray(molecule.atom_charges(), dtype=np.int32)
    positions = molecule.atom_coords(unit="Bohr")
    damping = DampingParam(method="pbe")

    def evaluate(coords, *, gradient):
        return DispersionModel(numbers, coords, charge=0.0).get_dispersion(
            damping, grad=gradient)

    result = evaluate(positions, gradient=True)
    energy = float(result["energy"])
    gradient = np.asarray(result["gradient"])
    if (gradient.shape != (2, 3) or not np.isfinite(energy)
            or not np.isfinite(gradient).all() or energy >= 0
            or np.max(np.abs(gradient)) < 1e-12):
        raise AssertionError("D4 energy/gradient is invalid or the probe has no derivative signal")
    step = 1e-4
    delta = np.zeros_like(positions)
    delta[1, 2] = step
    derivative = (float(evaluate(positions + delta, gradient=False)["energy"])
                  - float(evaluate(positions - delta, gradient=False)["energy"])) / (2 * step)
    np.testing.assert_allclose(gradient[1, 2], derivative, atol=1e-9, rtol=1e-4)
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=1e-10, rtol=0)
    return {"molecule": "H2", "bond_angstrom": 0.74, "damping_method": "PBE",
            "model": "D4", "energy_hartree": energy,
            "gradient_hartree_per_bohr": gradient.tolist(),
            "finite_difference_step_bohr": step,
            "derivative_max_abs_error_hartree_per_bohr": abs(gradient[1, 2] - derivative),
            "scope": "Actual D4 library energy/gradient and finite-difference parity"}


def tensor_product_probe(*, higher=False):
    import torch
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    import cuequivariance_ops_torch  # noqa: F401 - require compiled package

    torch.manual_seed(2718)
    irreps = cue.Irreps(cue.O3, "2x0e + 1x1o")
    batch_size = 2
    # Explicit methods and effective-backend checks follow the pinned v0.12.0
    # API: https://github.com/NVIDIA/cuEquivariance/tree/v0.12.0/cuequivariance_torch
    # The documented unshared external-weight shape is (batch, weight_numel),
    # also used by FullyConnectedTensorProductConv for message-passing edges.
    modules = [cuet.FullyConnectedTensorProduct(
        irreps, irreps, irreps, layout=cue.mul_ir, shared_weights=False,
        internal_weights=False, device="cuda", dtype=torch.float64,
        math_dtype=torch.float64, method=method,
    ) for method in ("fused_tp", "naive")]
    fused, naive = modules
    if fused.f.method != "fused_tp" or "FusedTP" not in type(fused.f.m).__name__:
        raise AssertionError("Requested CuEq fused CUDA primitive silently fell back")
    if (naive.f.method != "naive"
            or type(naive.f.m).__name__ != "SegmentedPolynomialNaive"):
        raise AssertionError("Requested CuEq naive reference backend was not selected")
    if any(module.shared_weights or module.internal_weights for module in modules):
        raise AssertionError("CuEq primitive did not select unshared external weights")
    if fused.weight_numel != naive.weight_numel:
        raise AssertionError("CuEq fused/reference weight dimensions differ")
    inputs = (torch.randn(batch_size, irreps.dim, device="cuda", dtype=torch.float64,
                          requires_grad=True),
              torch.randn(batch_size, irreps.dim, device="cuda", dtype=torch.float64,
                          requires_grad=True),
              torch.randn(batch_size, fused.weight_numel, device="cuda",
                          dtype=torch.float64, requires_grad=True))
    directions = tuple(torch.randn_like(item) for item in inputs)
    outputs, derivatives, second, force_gradients = [], [], [], []
    for module in modules:
        output = module(*inputs)
        if output.shape != (batch_size, irreps.dim):
            raise AssertionError(f"Unexpected CuEq tensor-product shape: {output.shape}")
        scalar = output.square().sum() + 0.13 * output.cos().sum()
        gradient = torch.autograd.grad(scalar, inputs, create_graph=higher,
                                       retain_graph=higher)
        outputs.append(output)
        derivatives.append(gradient)
        if higher:
            hessian_vector = torch.autograd.grad(
                sum((grad * direction).sum()
                    for grad, direction in zip(gradient, directions)),
                inputs, retain_graph=True)
            force_gradient, = torch.autograd.grad(gradient[0].square().mean(), inputs[2])
            second.append(hessian_vector)
            force_gradients.append(force_gradient)
    error = compare(outputs[0], outputs[1], atol=2e-9, rtol=2e-8)
    first_errors = [compare(left, right, atol=2e-8, rtol=2e-7)
                    for left, right in zip(*derivatives)]
    result = {"device": "cuda", "dtype": "float64", "irreps": str(irreps),
              "layout": "mul_ir", "method": fused.f.method,
              "effective_backend": type(fused.f.m).__name__,
              "reference_method": naive.f.method,
              "reference_effective_backend": type(naive.f.m).__name__,
              "reference_backend": "CuEq naive PyTorch with identical inputs/weights",
              "batch_size": batch_size,
              "input_shapes": [list(item.shape) for item in inputs[:2]],
              "output_shape": list(outputs[0].shape),
              "weight_numel": fused.weight_numel,
              "weight_shape": list(inputs[2].shape),
              "shared_weights": fused.shared_weights,
              "internal_weights": fused.internal_weights,
              "shared_weight_path_qualified": False,
              "forward_max_abs_error": error,
              "first_derivative_max_abs_error": max(first_errors),
              "scope": "CUDA tensor-product primitive with unshared per-sample "
                       "external weights used in message passing; no shared-weight "
                       "path or future backbone/model is qualified"}
    if higher:
        second_errors = [compare(left, right, atol=2e-7, rtol=2e-6)
                         for left, right in zip(*second)]
        mixed_error = compare(*force_gradients, atol=2e-7, rtol=2e-6)
        result.update(second_derivative_max_abs_error=max(second_errors),
                      force_loss_weight_gradient_max_abs_error=mixed_error,
                      higher_derivative_gate=True)
    torch.cuda.synchronize()
    return result


def check_tensor_product_higher():
    try:
        return tensor_product_probe(higher=True)
    except (NotImplementedError, RuntimeError) as error:
        # Only explicit missing derivative support is an optional limitation.
        # OOM, missing symbols, numerical errors and arbitrary API failures fail.
        if re.search(r"derivative for .+ is not implemented|double backward.*not "
                     r"supported|does not support double backward|once_differentiable|"
                     r"higher[- ]order.*not (implemented|supported)", str(error), re.I):
            raise UnsupportedHigherDerivative(
                "CUDA CuEq higher derivatives are unavailable; future force-training "
                f"backbone gate remains closed: {type(error).__name__}: {error}") from error
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--skip-compile", action="store_true",
                        help="Leave compiler capability explicitly unqualified")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "src"))
    checks = Checks()
    # Keep tiny probes within small allocations and caches out of project data.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["MAX_JOBS"] = "1"
    with tempfile.TemporaryDirectory(prefix="densnet-runtime-smoke-") as cache:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
        os.environ["TORCH_EXTENSIONS_DIR"] = cache + "/extensions"
        ready = checks.run("requested_device", lambda: check_device(args.device))
        checks.run("project_compat_before_schnetpack", check_compat)
        for name in (
            "schnetpack", "schnetpack.md", "schnetpack.md.calculators",
            "equiv_dens.cli", "equiv_dens.training.parse_command_line_arguments",
            "equiv_dens.training.trainer", "equiv_dens.training.model_loader",
            "equiv_dens.data.density_dataset", "equiv_dens.md.dft_network_calculator",
            "equiv_dens.md.gxtb_calculator", "equiv_dens.md.md_console_logger",
            "mace.modules", "e3nn.o3", "dftd4.pyscf", "dftpy",
            "rdkit.Chem", "apsw", "h5py", "sklearn", "pandas", "matplotlib",
            "tensorboardX", "wandb", "memspectrum", "cuequivariance_torch",
        ):
            checks.run("import:" + name, lambda name=name: import_module(name))
        checks.run("project_cli_help", lambda: check_cli(repo))
        checks.run("cpu_pyscf_df_pbe", check_cpu_pbe)
        cpu_pbe_reference = checks.results[-1].get("details")
        checks.run("d4_energy_gradient", check_d4)
        if ready:
            checks.run("torchvision_torchaudio_native_abi",
                       lambda: check_vision_audio(args.device))
            checks.run("densnet_representation_eager_derivatives",
                       lambda: check_representation(args.device))
            checks.run("analytic_torch_func_derivatives",
                       lambda: check_analytic(args.device))
            if args.skip_compile:
                checks.omit("analytic_inductor_fullgraph", "Explicit --skip-compile")
            else:
                checks.run("analytic_inductor_fullgraph",
                           lambda: check_analytic(args.device, compile_probe=True))
        else:
            for name in ("torchvision_torchaudio_native_abi",
                         "densnet_representation_eager_derivatives",
                         "analytic_torch_func_derivatives", "analytic_inductor_fullgraph"):
                checks.omit(name, "Requested device unavailable", required=True)
        cuda_checks = ("cupy_dipole_vs_pyscf", "gpu4pyscf_df_rhf_vs_cpu",
                       "gpu4pyscf_df_pbe_vs_cpu",
                       "cuequivariance_fused_forward_first_derivatives")
        if args.device == "cuda" and ready:
            checks.run(cuda_checks[0], check_gpu_dipole)
            checks.run(cuda_checks[1], check_gpu_rhf)
            checks.run(cuda_checks[2], lambda: check_gpu_pbe(cpu_pbe_reference))
            first_passed = checks.run(cuda_checks[3], tensor_product_probe)
            if first_passed:
                checks.run("cuequivariance_higher_derivative_backbone_gate",
                           check_tensor_product_higher, required=False)
            else:
                checks.omit("cuequivariance_higher_derivative_backbone_gate",
                            "Forward/first derivative prerequisite failed")
        else:
            for name in cuda_checks:
                checks.omit(name, "CUDA scope not requested" if args.device == "cpu"
                            else "Requested CUDA device unavailable",
                            required=args.device == "cuda")
            checks.omit("cuequivariance_higher_derivative_backbone_gate",
                        "Requires successful CUDA primitive check")
        # Optional unsupported capabilities can leave a future gate closed;
        # actual numerical/API failures, even in those probes, fail this run.
        failed = any(row["status"] == "failed" or
                     (row["required"] and row["status"] != "passed")
                     for row in checks.results)
        result = {"schema_version": 1, "status": "failed" if failed else "passed",
                  "requested_device": args.device, "skip_compile": args.skip_compile,
                  "compiler_qualified": checks.passed("analytic_inductor_fullgraph"),
                  "cuda_checks_qualified": all(checks.passed(name) for name in cuda_checks),
                  "cuequivariance_higher_derivative_backbone_gate": checks.passed(
                      "cuequivariance_higher_derivative_backbone_gate"),
                  "scope": "Synthetic runtime capabilities; no performance/model/MD claim",
                  "packages": package_versions(),
                  "checks": checks.results}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
