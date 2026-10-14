"""Torch-free tests for static density layouts, coefficient packing and GTO evaluation."""
import jax
import jax.numpy as jnp
import numpy as np
import os
from pathlib import Path

jax.config.update("jax_enable_x64", True)

from equiv_dens_jax.density import (
    apply_density_head,
    apply_density_expansion,
    coefficients_dict_to_vector,
    coefficients_to_dict,
    prepare_layout,
)
from equiv_dens_jax.serialization import load_tree
from equiv_dens_jax.gaussians import BOHR_ANGSTROM, PYSCF_GTO_FACTOR


def _required_reference_paths(env_name, default_names):
    import pytest
    raw = os.environ.get(env_name)
    paths = ([Path(value) for value in raw.split(os.pathsep) if value] if raw else
             [Path("scratch/jax_pipeline") / name for name in default_names])
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        pytest.fail(
            f"required Torch-generated reference bundle(s) missing: {missing}; "
            f"set {env_name} to path-separated bundles or generate the default fixtures "
            "with tests/jax_pipeline/generate_density_reference.py"
        )
    return paths


def _layout_node(basis, **attrs):
    return {"type": "DensityExpansion", "path": "", "attrs": {"orbital_basis": basis, **attrs},
            "children": {}, "params": {}, "buffers": {}}


def test_layout_and_vector_roundtrip_preserve_pyscf_p_order():
    basis = {1: [(1, 1, 0), (1, 1, 1)]}
    nums = jnp.array([[1, 1], [1, 1]])
    atoms = {"batch_atom_numbers": nums}
    layout = prepare_layout(_layout_node(basis), atoms)
    coeffs = {"spherical_coeffs": [
        {(1, 0): jnp.array([[[[2.0]]], [[[3.0]]]]),
         (1, 1): jnp.array([[[[4.0], [5.0], [6.0]]], [[[7.0], [8.0], [9.0]]]])},
        {(1, 0): jnp.array([[[[10.0]]], [[[11.0]]]]),
         (1, 1): jnp.array([[[[12.0], [13.0], [14.0]]], [[[15.0], [16.0], [17.0]]]])},
    ]}
    atoms.update(coeffs)
    vec = coefficients_dict_to_vector(atoms, layout)
    np.testing.assert_allclose(vec[0], [2, -6, -4, -5, 10, -14, -12, -13])
    radial = {1: [[[1.2], [2.0]], [[1.3], [2.1]]]} 
    unpacked = coefficients_to_dict(vec, layout, radial)
    repacked = coefficients_dict_to_vector(unpacked, layout)
    np.testing.assert_allclose(repacked, vec)


def test_expansion_eager_jit_and_explicit_gradient_match_s_orbital():
    basis = {1: [(1, 1, 0)]}
    node = _layout_node(basis, density_grad=True)
    nums = jnp.ones((1, 1), dtype=jnp.int32)
    coords = jnp.array([[[0.2, -0.1, 0.3], [0.8, 0.4, -0.2]]], dtype=jnp.float64)
    positions = jnp.array([[[0.0, 0.0, 0.0]]], dtype=jnp.float64)
    atoms = {"batch_atom_numbers": nums, "batch_positions": positions, "coords": coords,
             "spherical_coeffs": [{(1, 0): jnp.array([[[[1.7]]]], dtype=jnp.float64)}],
             "radial_width": [{(1, 0): jnp.array([[[[0.8]]]], dtype=jnp.float64)}],
             "radial_scale": [{(1, 0): jnp.array([[[[1.2]]]], dtype=jnp.float64)}]}
    layout = prepare_layout(node, atoms)
    run = lambda a: apply_density_expansion(node, {}, a, layout)
    eager = run(atoms)
    compiled = jax.jit(run)(atoms)
    np.testing.assert_allclose(eager["density"], compiled["density"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(eager["density_grad"], compiled["density_grad"], atol=1e-12, rtol=1e-12)
    d = np.linalg.norm(np.asarray(coords[0])-np.asarray(positions[0, 0]), axis=-1)
    r_bohr = d / BOHR_ANGSTROM
    expected = 1.2 / PYSCF_GTO_FACTOR * 1.7 * np.exp(-0.8 * r_bohr**2)
    np.testing.assert_allclose(eager["density"][0], expected, atol=1e-11, rtol=1e-11)
    # The explicit derivative is with respect to grid coordinates, in Angstrom units.
    expected_grad = (expected[:, None] * (-2 * 0.8 * r_bohr[:, None])
                     * np.asarray(coords[0]) / d[:, None])
    np.testing.assert_allclose(eager["density_grad"][0], expected_grad, atol=1e-10, rtol=1e-10)


def test_expansion_constraints_keep_historical_gradient_untransformed():
    basis = {1: [(1, 1, 0)]}
    atoms = {"batch_atom_numbers": jnp.ones((1, 1), dtype=jnp.int32),
             "batch_positions": jnp.zeros((1, 1, 3)), "coords": jnp.array([[[0.5, 0., 0.]]]),
             "coord_weights": jnp.ones((1, 1)),
             "spherical_coeffs": [{(1, 0): jnp.ones((1,1,1,1))}],
             "radial_width": [{(1, 0): jnp.ones((1,1,1,1))}],
             "radial_scale": [{(1, 0): jnp.ones((1,1,1,1))}]}
    base = apply_density_expansion(_layout_node(basis, density_grad=True), {}, atoms)
    square = apply_density_expansion(_layout_node(basis, density_grad=True, expansion_constraint="sq"), {}, atoms)
    np.testing.assert_allclose(square["density"], base["density"]**2)
    np.testing.assert_allclose(square["density_grad"], base["density_grad"])


def test_grid_scaling_factor_override_is_dynamic_and_keeps_legacy_gradient():
    basis = {1: [(1, 1, 0)]}
    node = _layout_node(
        basis, density_grad=True, integral_constraint="grid", grid_scaling_factor=0.8)
    node["buffers"] = {"grid_scaling_factor": "grid_factor"}
    state = {"buffers": {"grid_factor": jnp.asarray([0.1])}}
    atoms = {
        "batch_atom_numbers": jnp.ones((1, 1), dtype=jnp.int32),
        "batch_positions": jnp.zeros((1, 1, 3), dtype=jnp.float64),
        "coords": jnp.array([[[0.4, 0.1, 0.0], [0.8, -0.2, 0.3]]], dtype=jnp.float64),
        "coord_weights": jnp.array([[0.7, 1.3]], dtype=jnp.float64),
        "spherical_coeffs": [{(1, 0): jnp.array([[[[1.2]]]], dtype=jnp.float64)}],
        "radial_width": [{(1, 0): jnp.array([[[[0.9]]]], dtype=jnp.float64)}],
        "radial_scale": [{(1, 0): jnp.array([[[[1.1]]]], dtype=jnp.float64)}],
    }
    layout = prepare_layout(node, atoms)
    run = jax.jit(lambda a: apply_density_expansion(node, state, a, layout))
    raw_node = _layout_node(basis, density_grad=True)
    raw = apply_density_expansion(raw_node, {}, atoms, layout)
    raw_integral = jnp.sum(raw["density"] * atoms["coord_weights"], axis=1, keepdims=True)
    scaling = jnp.sum(atoms["batch_atom_numbers"], axis=-1, keepdims=True) / raw_integral

    def density_at(factor):
        return run({**atoms, "density_grid_scaling_factor": factor})["density"]

    for factor in (0.25, 0.7):
        expected = raw["density"] * (scaling * (1.0 - factor) + factor)
        np.testing.assert_allclose(density_at(jnp.asarray(factor)), expected, rtol=1e-12, atol=1e-12)
    # The explicit gradient is computed before and remains unscaled by the
    # historical grid integral constraint, even as the scalar changes.
    result = run({**atoms, "density_grid_scaling_factor": jnp.asarray(0.25)})
    np.testing.assert_allclose(result["density_grad"], raw["density_grad"], rtol=1e-12, atol=1e-12)
    grad_factor = jax.grad(lambda factor: jnp.sum(density_at(factor)))(jnp.asarray(0.25))
    expected_grad = jnp.sum(raw["density"] * (1.0 - scaling))
    np.testing.assert_allclose(grad_factor, expected_grad, rtol=1e-12, atol=1e-12)


def _assert_tree_close(actual, expected, atol=1e-9, rtol=1e-8):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_close(actual[key], expected[key], atol, rtol)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected):
            _assert_tree_close(x, y, atol, rtol)
    else:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol)


def test_torch_density_head_reference_bundles():
    """Required cross-process references generated by the Torch-only helper."""
    paths = _required_reference_paths("DENSITY_REFERENCE_BUNDLES", (
        "density-head-learned", "density-head-fixed", "density-head-positive",
        "density-head-constraint", "density-head-weighted", "density-head-core",
        "density-head-delta"))
    for path in paths:
        ref = load_tree(path)
        assert ref["metadata"].get("generator_sha256"), f"missing generator hash in {path}"
        expected_loss_definition = (
            "sum_squares(radial_width,radial_scale,spherical_coeffs)"
            if ref["metadata"].get("variant") == "delta" else
            "sum_squares(df_coeffs,radial_width,radial_scale,spherical_coeffs)"
        )
        assert ref["metadata"].get("loss_definition") == expected_loss_definition, (
            f"unexpected head reference loss definition in {path}")
        node, state, atoms = ref["node"], ref["state"], ref["atoms"]
        layout = prepare_layout(node, atoms)
        result = apply_density_head(node, state, atoms, layout)
        for key, expected in ref["expected"].items():
            _assert_tree_close(result[key], expected)
        ref_output_loss = 0.0
        if "df_coeffs" in ref["expected"]:
            ref_output_loss += sum(float(np.square(np.asarray(x)).sum())
                                   for x in [ref["expected"]["df_coeffs"]])
        for name in ("radial_width", "radial_scale", "spherical_coeffs"):
            ref_output_loss += sum(float(np.square(np.asarray(v)).sum())
                                   for atom in ref["expected"][name] for v in atom.values())

        def objective(params, fs):
            run_atoms = dict(atoms, sph_repr=fs)
            output = apply_density_head(node, {"params": params, "buffers": state["buffers"]}, run_atoms, layout)
            terms = [jnp.sum(output["df_coeffs"]**2)] if "df_coeffs" in output else []
            terms += [jnp.sum(v**2) for name in ("radial_width", "radial_scale")
                      for atom in output[name] for v in atom.values()]
            terms += [jnp.sum(v**2) for atom in output["spherical_coeffs"] for v in atom.values()]
            return sum(terms)
        fs = atoms["sph_repr"]
        grad_params, grad_features = jax.grad(objective, argnums=(0,1))(state["params"], fs)
        try:
            np.testing.assert_allclose(objective(state["params"],fs), ref["loss"], atol=1e-8, rtol=1e-7)
        except AssertionError as exc:
            raise AssertionError(
                f"density head loss mismatch in {path}: JAX={float(objective(state['params'],fs))}, "
                f"Torch-loss={float(np.asarray(ref['loss']))}, output-derived={ref_output_loss}"
            ) from exc
        missing_grads = set(ref["parameter_grads"]) - set(grad_params)
        assert not missing_grads, f"JAX density parameter gradients missing {sorted(missing_grads)} in {path}"
        for name, expected in ref["parameter_grads"].items():
            np.testing.assert_allclose(grad_params[name], expected, atol=1e-8, rtol=1e-7)
        _assert_tree_close(grad_features, ref["feature_grads"], atol=1e-8, rtol=1e-7)


def test_torch_df_and_freeatom_reference_bundles():
    paths = _required_reference_paths("DENSITY_DF_REFERENCE_BUNDLES",
                                      ("density-df", "density-freeatom"))
    for path in paths:
        ref = load_tree(path)
        node,state,atoms=ref["node"],ref["state"],ref["atoms"]
        layout=prepare_layout(node,atoms)
        result=apply_density_head(node,state,atoms,layout)
        for key,expected in ref["expected"].items():
            _assert_tree_close(result[key],expected)
        def objective(vector):
            run_atoms=dict(atoms,df_coeffs=vector)
            output=apply_density_head(node,state,run_atoms,layout)
            return sum(jnp.sum(v**2) for name in ("spherical_coeffs","radial_width","radial_scale")
                       for atom in output[name] for v in atom.values())
        grad=jax.grad(objective)(atoms["df_coeffs"])
        np.testing.assert_allclose(objective(atoms["df_coeffs"]),ref["loss"],atol=1e-8,rtol=1e-7)
        np.testing.assert_allclose(grad,ref["input_grad"],atol=1e-8,rtol=1e-7)


def test_torch_density_expansion_reference_bundle():
    path = os.environ.get("DENSITY_EXPANSION_REFERENCE_BUNDLE")
    if not path:
        default = Path("scratch/jax_pipeline/density-expansion-sp")
        import pytest
        if not default.is_dir():
            pytest.fail(
                "required Torch-generated density expansion reference is missing; "
                "set DENSITY_EXPANSION_REFERENCE_BUNDLE or generate "
                "scratch/jax_pipeline/density-expansion-sp with "
                "tests/jax_pipeline/generate_density_reference.py"
            )
        path = str(default)
    ref = load_tree(Path(path))
    node, atoms = ref["node"], ref["atoms"]
    layout = prepare_layout(node, atoms)
    result = apply_density_expansion(node, {}, atoms, layout)
    for key, expected in ref["expected"].items():
        try:
            _assert_tree_close(result[key], expected, atol=1e-8, rtol=1e-8)
        except AssertionError as exc:
            raise AssertionError(f"density expansion mismatch in {key}") from exc
    for L, expected in ref.get("components", {}).items():
        component = apply_density_expansion(node, {}, ref["atoms"], layout, eval_L=[L])
        try:
            _assert_tree_close({k: component[k] for k in expected}, expected, atol=1e-8, rtol=1e-8)
        except AssertionError as exc:
            raise AssertionError(f"density expansion component L={L} mismatch") from exc
