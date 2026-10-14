"""Optional current properties through the actual native Pipeline boundary.

Primitive Torch references are required and loaded as portable arrays. Models
compose their current descriptors; no Torch runtime or evaluator is imported.
The DFTpy interface is exercised with a prescribed potential, not external
pseudopotential assets. Analytic DF data use the qualified s/p PySCF basis
because the source's squeezed radial helper rejects the learned ragged fixture.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from equiv_dens_jax import functionals
from equiv_dens_jax.losses import ErrorDict
from equiv_dens_jax.pipeline import Pipeline
from equiv_dens_jax.serialization import load_tree
from equiv_dens_jax.training import Optimizer, initialize, make_step

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
ROOT = Path(__file__).resolve().parents[2]


def required_bundle(environment, relative):
    path = Path(os.environ.get(environment, str(ROOT / relative)))
    if not (path / "manifest.json").is_file():
        pytest.fail(f"Required independent Torch reference missing: {path}; generate it in the separate Torch runtime")
    return load_tree(path)


@pytest.fixture(scope="module")
def references():
    return {
        "pipeline": required_bundle("DENSNET_JAX_PIPELINE_REFERENCE", "scratch/jax_pipeline/pipeline-tiny"),
        "head": required_bundle("DENSNET_JAX_OPTIONAL_HEAD_REFERENCE", "scratch/jax_pipeline/density-head-positive"),
        "df": required_bundle("DENSNET_JAX_OPTIONAL_DF_REFERENCE", "scratch/jax_pipeline/density-df"),
        "moments": required_bundle("DENSNET_JAX_ENERGY_REFERENCE", "scratch/jax-pipeline/energy-reference")["moments"],
        "functional": required_bundle("DENSNET_JAX_FUNCTIONAL_REFERENCE", "scratch/jax-pipeline/functional-reference")["cases"]["minimum_only"],
    }


def assert_close(actual, expected, *, atol=1e-9, rtol=1e-8):
    assert np.shape(actual) == np.shape(expected)
    assert np.isfinite(np.asarray(actual)).all()
    assert np.isfinite(np.asarray(expected)).all()
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def descriptor(kind, path, *, attrs=None, children=None):
    return {"type": kind, "path": path, "attrs": attrs or {}, "children": children or {}, "params": {}, "buffers": {}}


def relocate(node, state, prefix):
    """Keep exported canonical identities while placing a descriptor in a model."""
    node = deepcopy(node)
    def visit(current):
        current["path"] = prefix + ("." + current["path"] if current["path"] else "")
        for kind in ("params", "buffers"):
            current[kind] = {name: prefix + "." + key for name, key in current[kind].items()}
        for sub in current["children"].values():
            visit(sub)
    visit(node)
    return node, {kind: {prefix + "." + name: value for name, value in state[kind].items()}
                  for kind in ("params", "buffers")}


def pipeline_shell(references, density_node, properties, state):
    node = deepcopy(references["pipeline"]["model"])
    node["params"], node["buffers"] = {}, {}
    node["attrs"].update(calculate_forces=False, force_props=[], no_force_props=list(properties), remove_atom_density=False)
    node["children"]["density_repr_model"] = density_node
    node["children"]["property_models"] = descriptor("ModuleDict", "property_models", children=properties)
    for name in ("conversions_in", "conversions_out"):
        node["children"][name] = descriptor("UnitConversion", name)
    node["children"]["scaling"] = descriptor("VarianceScaling", "scaling", attrs={"std": 1.0})
    return Pipeline(node, state)


class HostGrid:
    def __init__(self, metadata):
        self.shape = tuple(metadata["shape"])
        self.lattice, self.volume, self.point_volume = (metadata[key] for key in ("lattice", "volume", "point_volume"))
        self.reciprocal = SimpleNamespace(gg=metadata["gg"], coords=metadata["reciprocal_coords"], mask=metadata["reciprocal_mask"])
    def get_reciprocal_grid(self):
        return self.reciprocal


class PrescribedPotential:
    def __init__(self, fields):
        self.fields = np.asarray(fields)
        self.calls = []
    def restart(self, *, grid, ions):
        self.current = ions.index
        self.calls.append(("restart", self.current))
    def __call__(self, density):
        self.calls.append(("density", self.current, np.asarray(density).copy()))
        self.vreal = self.fields[self.current].copy()


def functional_pipeline(references):
    head = references["head"]
    head_node, state = relocate(head["node"], head["state"], "density_repr_model")
    case = references["functional"]
    lda, lda_state = relocate(case["node"], case["state"], "property_models.energy_min.1")
    for kind in state:
        state[kind].update(lda_state[kind])
    expansion = descriptor("DensityExpansion", "property_models.energy_min.0", attrs={
        "orbital_basis": head_node["attrs"]["orbital_basis"], "expansion_constraint": "sq", "density_grad": False})
    sequence = descriptor("Sequential", "property_models.energy_min", children={"0": expansion, "1": lda})
    pipeline = pipeline_shell(references, head_node, {"energy_min": sequence}, state)
    atoms = deepcopy(head["atoms"])
    atoms["positions"] = atoms["batch_positions"].copy()
    atoms["shifted_positions"] = atoms["batch_positions"].copy()
    shape = tuple(case["grid_metadata"]["shape"])
    grid_points = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1).reshape(-1, 3)
    # Synthetic positive-density qualification, away from atom/grid coincidence.
    coords = 0.11 + grid_points * 0.13
    atoms["coords"] = np.broadcast_to(coords, (2, *coords.shape)).copy()
    atoms["coord_weights"] = np.full((2, len(coords)), case["grid_metadata"]["point_volume"])
    atoms["density"] = np.full((2, len(coords)), -99.0)  # A stale placeholder must never reach the host callback.
    atoms["idx"] = np.arange(2)
    potential = PrescribedPotential(case["atoms"]["v_real"])
    atoms.update(grid=HostGrid(case["grid_metadata"]), pseudo_pot=potential, dftpy_grid=object(),
                 ions=[SimpleNamespace(index=0), SimpleNamespace(index=1)])
    return pipeline, atoms, potential, shape


def test_functional_host_preparation_predict_and_refresh_current_prefix(references):
    pipeline, raw, potential, shape = functional_pipeline(references)
    prepared = pipeline.prepare_functionals(raw, properties=("energy_min",))
    assert len(potential.calls) == 4
    assert all(name not in prepared for name in ("grid", "pseudo_pot", "dftpy_grid", "ions"))
    prefix = pipeline._apply(pipeline.node["children"]["density_repr_model"], pipeline.state, prepared,
                             layouts=pipeline.layouts, training=True)
    sequence = pipeline.node["children"]["property_models"]["children"]["energy_min"]
    expanded = pipeline._apply(sequence["children"]["0"], pipeline.state, prefix,
                               layouts=pipeline.layouts, training=True)
    rho = np.asarray(expanded["density"]).reshape((2, *shape))
    assert np.isfinite(rho).all() and rho.min() > 0
    for index in range(2):
        np.testing.assert_array_equal(potential.calls[2 * index + 1][2], rho[index])
    expected = functionals.apply_functional(sequence["children"]["1"], pipeline.state, expanded)["energy_min"]
    # Prepared numeric data must retain the explicit field without host refresh.
    assert_close(pipeline.predict(prepared, properties=("energy_min",), jit=True)["energy_min"], expected)
    assert len(potential.calls) == 4
    actual = pipeline.predict(raw, properties=("energy_min",), jit=True)
    assert_close(actual["energy_min"], expected)
    assert len(potential.calls) == 8
    # Changing weights must affect the density observed by the next host call.
    key = next(name for name in pipeline.state["params"] if name.endswith("weight"))
    pipeline.state = {**pipeline.state, "params": {**pipeline.state["params"], key: pipeline.state["params"][key] + 0.002}}
    updated = pipeline.predict(raw, properties=("energy_min",), jit=True)
    assert len(potential.calls) == 12
    assert not np.array_equal(potential.calls[9][2], potential.calls[5][2])
    assert not np.array_equal(updated["energy_min"], actual["energy_min"])


def test_functional_detached_field_parameter_gradient_and_density_training_step(references):
    pipeline, raw, potential, _ = functional_pipeline(references)
    prepared = pipeline.prepare_functionals(raw, properties=("energy_min",))
    def loss(params, field):
        state = {"params": params, "buffers": pipeline.state["buffers"]}
        result = pipeline.forward(state, dict(prepared, v_real=field), training=True, properties=("energy_min",), forces=False)
        return jnp.mean(result["energy_min"])
    value, (gradient, field_gradient) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(pipeline.state["params"], prepared["v_real"])
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(gradient))
    np.testing.assert_array_equal(field_gradient, np.zeros_like(field_gradient))
    norm = jnp.sqrt(sum(jnp.sum(value**2) for value in gradient.values()))
    assert float(norm) > 1e-7
    direction = jax.tree.map(lambda value: value / norm, gradient)
    step = 1e-5
    shifted = lambda amount: jax.tree.map(lambda value, tangent: value + amount * tangent, pipeline.state["params"], direction)
    numerical = (loss(shifted(step), prepared["v_real"]) - loss(shifted(-step), prepared["v_real"])) / (2 * step)
    assert_close(numerical, norm, atol=1e-6, rtol=1e-5)
    optimizer = Optimizer(name="sgd", learning_rate=1e-7, ema_decay=None, lookahead_k=0)
    initial = initialize(pipeline, stage="density", optimizer=optimizer)
    update = make_step(pipeline, ErrorDict({"energy_min": 1.0}), stage="density", optimizer=optimizer)
    trained, errors = update(initial, prepared)
    assert np.isfinite(float(errors["loss"]))
    assert int(trained["step"]) == 1
    assert any(not np.array_equal(initial["params"][key], trained["params"][key]) for key in gradient)
    assert len(potential.calls) == 4, "Compiled differentiation/training must not call a host pseudopotential"
    assert "torch" not in sys.modules


def analytic_pipeline(references, name):
    case = references["moments"][name]
    observable, observable_state = relocate(case["node"], case["state"], "property_models.dipole_moment")
    # A portable current fixed-DF head, configured with the independent analytic
    # fixture's supported shell metadata. The model uses no learned radial path.
    head = deepcopy(references["df"]["node"])
    basis = observable["attrs"]["orbital_basis"]
    radial, vector = {}, []
    for slot, z in enumerate(np.max(case["atoms"]["batch_atom_numbers"], axis=0)):
        counts = {}
        radial[int(z)] = []
        for _, _, L in basis[int(z)]:
            key = (int(z), L)
            index = counts.get(L, 0)
            counts[L] = index + 1
            radial[int(z)].append(([float(case["atoms"]["radial_width"][slot][key][0, 0, 0, index])],
                                  [float(case["atoms"]["radial_scale"][slot][key][0, 0, 0, index])]))
            coefficient = case["atoms"]["spherical_coeffs"][slot][key][:, 0, :, index]
            if L == 1:
                coefficient = coefficient[:, [2, 0, 1]]
            if L % 2:
                coefficient = -coefficient
            vector.append(coefficient)
    head["attrs"].update(orbital_basis=basis, radial_coeffs=radial)
    head, state = relocate(head, references["df"]["state"], "density_repr_model")
    for kind in state:
        state[kind].update(observable_state[kind])
    pipeline = pipeline_shell(references, head, {"dipole_moment": observable}, state)
    raw = deepcopy(case["atoms"])
    for key in ("spherical_coeffs", "radial_width", "radial_scale"):
        raw.pop(key)
    raw["positions"] = raw["batch_positions"].copy()
    raw["df_coeffs"] = np.concatenate(vector, axis=1)
    return pipeline, raw, case


@pytest.mark.parametrize("name", ("analytic_sp", "analytic_free_df", "analytic_free_mo"))
def test_analytic_prepare_forward_predict_actual_pyscf_and_free_atom_reference(references, name):
    pipeline, raw, case = analytic_pipeline(references, name)
    prepared = pipeline.prepare_observables(raw, properties=("dipole_moment",))
    for basis_name in ("atom_df_coeffs_basis", "atom_mo_coeffs_basis"):
        if basis_name in raw:
            assert basis_name not in prepared, "QC basis descriptors must retain Python scalar types outside JIT"
            assert basis_name in pipeline.host_metadata
    assert_close(prepared["dipole_integrals"], case["integrals"], atol=1e-10, rtol=1e-9)
    if "atom_dipole_moment" in case:
        assert_close(prepared["atom_dipole_moment"], case["atom_dipole_moment"], atol=1e-10, rtol=1e-9)
    def evaluate(inputs):
        return pipeline.forward(pipeline.state, inputs, training=True, properties=("dipole_moment",), forces=False)["dipole_moment"]
    assert_close(evaluate(prepared), case["output"])
    assert_close(jax.jit(evaluate)(prepared), case["output"])
    assert_close(pipeline.predict(raw, properties=("dipole_moment",), jit=True)["dipole_moment"], case["output"])
    def loss(vector):
        output = evaluate(dict(prepared, df_coeffs=vector))
        return jnp.sum(output**2) + 0.13 * jnp.sum(jnp.cos(output))
    coefficients = prepared["df_coeffs"]
    gradient = jax.grad(loss)(coefficients)
    assert np.isfinite(np.asarray(gradient)).all() and float(jnp.linalg.norm(gradient)) > 1e-8
    direction = jnp.linspace(0.17, 0.43, coefficients.size).reshape(coefficients.shape)
    step = 1e-5
    numerical = (loss(coefficients + step * direction) - loss(coefficients - step * direction)) / (2 * step)
    assert_close(numerical, jnp.sum(gradient * direction), atol=1e-8, rtol=1e-6)
    assert "torch" not in sys.modules


def test_prepared_analytic_prediction_uses_explicit_free_atom_terms(references):
    pipeline, raw, case = analytic_pipeline(references, "analytic_free_mo")
    prepared = pipeline.prepare_observables(raw, properties=("dipole_moment",))
    assert "atom_mo_coeffs_basis" not in prepared
    assert pipeline.host_metadata["atom_mo_coeffs_basis"] == "sto-3g"
    result = pipeline.predict(prepared, properties=("dipole_moment",), jit=True)
    assert_close(result["dipole_moment"], case["output"])


def test_grid_dipole_predict_resolves_and_evaluates_density_prefix(references):
    pipeline, raw, _, _ = functional_pipeline(references)
    sequence = pipeline.node["children"]["property_models"]["children"]["energy_min"]
    expansion = deepcopy(sequence["children"]["0"])
    expansion["path"] = "property_models.density"
    observable = descriptor("DipoleMomentCalc", "property_models.dipole_moment")
    pipeline = pipeline_shell(references, pipeline.node["children"]["density_repr_model"],
                              {"density": expansion, "dipole_moment": observable}, pipeline.state)
    result = pipeline.predict(raw, properties=("dipole_moment",), jit=True)
    density = np.maximum(np.asarray(result["density"]), 0)
    numbers = raw["batch_atom_numbers"]
    center = np.sum(numbers[..., None] * raw["batch_positions"], axis=1, keepdims=True) / np.sum(numbers, axis=1)[:, None, None]
    density *= numbers.sum(axis=1, keepdims=True) / np.maximum(np.sum(density * raw["coord_weights"], axis=1, keepdims=True), 1e-30)
    expected = np.sum(numbers[..., None] * (raw["batch_positions"] - center), axis=1)
    expected -= np.sum((density * raw["coord_weights"])[..., None] * (raw["coords"] - center), axis=1)
    assert_close(result["dipole_moment"], expected)
    assert pipeline.resolve_properties(("dipole_moment",)) == ("density", "dipole_moment")
