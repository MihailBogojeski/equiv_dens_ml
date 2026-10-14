"""Current-geometry host preparation for the native JAX MD adapter."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from equiv_dens_jax.data import prepare_atoms
from equiv_dens_jax.md import Calculator
from equiv_dens_jax.md_preparation import make_md_preparation


def test_portable_atomic_assets_match_legacy_sad_preparation(tmp_path):
    from equiv_dens_jax.serialization import save_tree
    basis_path,atom_path=_source_files(tmp_path)
    reference=make_md_preparation(_config(basis_path,atom_path))
    basis_bundle,atom_bundle=tmp_path/"basis-bundle",tmp_path/"atom-bundle"
    # This is trusted fixture conversion; the new callback reads numeric
    # manifests and arrays without enabling object deserialization.
    save_tree(basis_bundle,np.load(basis_path,allow_pickle=True).item())
    save_tree(atom_bundle,np.load(atom_path,allow_pickle=True).item())
    config=_config(basis_bundle,atom_bundle)
    config["trusted_legacy"]=False
    portable=make_md_preparation(config)
    data=_prepared([[-.37,.12,0.],[.39,-.1,.03]])
    old,new=reference(data),portable(data)
    for field in ("coords","coord_weights","atom_density","atom_density_grad",
                  "atom_mo_coeffs","atom_mo_coeffs_occ","atom_dipole_moment"):
        np.testing.assert_array_equal(new[field],old[field])


def _source_files(tmp_path):
    basis_path = tmp_path / "orbital_basis.npy"
    atom_path = tmp_path / "free_atoms.npy"
    np.save(basis_path, {"H": [(1, 1, 0)]}, allow_pickle=True)
    np.save(atom_path, {
        1: {"mo_basis": "sto-3g", "mo_coeff": np.array([[1.0]]),
            "mo_occ": np.array([1.0])},
    }, allow_pickle=True)
    return basis_path, atom_path


def _config(basis_path, atom_path):
    return {
        "basis_path": str(basis_path),
        "atom_density_path": str(atom_path),
        "trusted_legacy": True,
        "atom_density_type": "mo_coeffs",
        "remove_atom_density": True,
        "density_grad": True,
        "include_atom_dipole": True,
        "grid_level": 0,
        "grid_count": 16,
        "seed": 17,
    }


def _prepared(positions):
    return prepare_atoms({
        "atom_numbers": np.array([[1, 1]], dtype=np.int64),
        "positions": np.asarray(positions, dtype=np.float64)[None],
    })


def test_callback_refreshes_centered_grid_sad_and_mo_dipole_inputs(tmp_path):
    basis_path, atom_path = _source_files(tmp_path)
    prepare = make_md_preparation(_config(basis_path, atom_path))
    positions = np.array([[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]])
    first = prepare(_prepared(positions))
    again = prepare(_prepared(positions))

    assert first["coords"].shape == (1, 16, 3)
    assert first["coord_weights"].shape == (1, 16)
    assert first["atom_density"].shape == (1, 16)
    assert first["atom_density_grad"].shape == (1, 16, 3)
    assert first["atom_mo_coeffs"].shape == (1, 2, 2)
    assert first["atom_mo_coeffs_occ"].shape == (1, 2)
    assert first["atom_mo_coeffs_basis"] == {1: "sto-3g"}
    assert first["atom_dipole_moment"].shape == (1, 3)
    for field in ("coords", "coord_weights", "atom_density", "atom_density_grad"):
        np.testing.assert_array_equal(first[field], again[field])
    assert np.isfinite(first["atom_density"]).all()
    assert np.isfinite(first["atom_density_grad"]).all()

    # Translation changes the grid coordinates with the live geometry, while
    # radial SAD values and quadrature weights translate covariantly.
    shift = np.array([3.0, -1.5, 0.25])
    translated = prepare(_prepared(positions + shift))
    np.testing.assert_allclose(translated["coords"], first["coords"] + shift, atol=1e-12)
    np.testing.assert_allclose(translated["coord_weights"], first["coord_weights"],
                               rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(translated["atom_density"], first["atom_density"], atol=1e-12)
    np.testing.assert_allclose(translated["atom_density_grad"], first["atom_density_grad"], atol=1e-12)


def test_callback_runs_through_native_calculator_after_geometry_centering(tmp_path):
    basis_path, atom_path = _source_files(tmp_path)
    prepare = make_md_preparation(_config(basis_path, atom_path))

    class PipelineStub:
        def predict(self, atoms, **kwargs):
            self.atoms = atoms
            assert "coords" in atoms and "atom_density" in atoms
            return {"energy": np.array([0.0]), "forces": np.zeros((1, 2, 3))}

    pipeline = PipelineStub()
    calculator = Calculator(pipeline, center=True, prepare_properties=prepare, jit=False)
    positions = np.array([[10.0, -2.0, 0.0], [10.74, -2.0, 0.0]])
    result = calculator.calculate(np.array([1, 1]), positions)
    np.testing.assert_allclose(pipeline.atoms["batch_positions"],
                               [[[-0.37, 0, 0], [0.37, 0, 0]]], atol=1e-12)
    assert result["forces"].shape == (1, 2, 3)


def test_callback_adds_df_density_and_analytic_dipole_inputs(tmp_path):
    basis_path = tmp_path / "orbital_basis.npy"
    atom_path = tmp_path / "free_atom_df.npy"
    np.save(basis_path, {"H": [(1, 1, 0)]}, allow_pickle=True)
    np.save(atom_path, {
        1: {"df_basis": {1: "sto-3g"}, "df_coeffs": np.array([[0.4]])},
    }, allow_pickle=True)
    prepare = make_md_preparation({
        "basis_path": str(basis_path), "atom_density_path": str(atom_path),
        "trusted_legacy": True, "atom_density_type": "df_coeffs",
        "remove_atom_density": True, "include_atom_dipole": True,
        "grid_level": 0, "grid_count": 8, "seed": 3,
    })
    positions = np.array([[[-0.37, 0, 0], [0.37, 0, 0]],
                          [[-0.4, 0.1, 0], [0.4, -0.1, 0]]])
    batch = prepare_atoms({"atom_numbers": np.array([[1, 1], [1, 1]]),
                           "positions": positions})
    result = prepare(batch)
    assert result["atom_density"].shape == (2, 8)
    assert result["atom_df_coeffs"].shape == (2, 2)
    assert result["atom_df_coeffs_basis"] == {1: "sto-3g"}
    assert result["atom_dipole_moment"].shape == (2, 3)
    assert np.isfinite(result["atom_density"]).all()
    assert np.isfinite(result["atom_dipole_moment"]).all()


def test_angstrom_and_bohr_input_geometry_produce_identical_internal_fields(tmp_path):
    basis_path, atom_path = _source_files(tmp_path)
    angstrom_callback = make_md_preparation(_config(basis_path, atom_path))
    bohr_config = _config(basis_path, atom_path)
    bohr_config["position_unit"] = "bohr"
    bohr_callback = make_md_preparation(bohr_config)
    positions_angstrom = np.array([[-0.37, 0, 0], [0.37, 0, 0]], dtype=np.float64)
    atoms_angstrom = _prepared(positions_angstrom)
    atoms_bohr = _prepared(positions_angstrom / 0.529177210903)
    angstrom = angstrom_callback(atoms_angstrom)
    bohr = bohr_callback(atoms_bohr)

    # Geometry passed onward is preserved in each caller's units, while all
    # PySCF grid and SAD work uses the internal Angstrom coordinate contract.
    np.testing.assert_allclose(angstrom["batch_positions"], atoms_angstrom["batch_positions"])
    np.testing.assert_allclose(bohr["batch_positions"], atoms_bohr["batch_positions"])
    for field in ("coords", "coord_weights", "atom_density", "atom_density_grad",
                  "atom_dipole_moment"):
        np.testing.assert_allclose(angstrom[field], bohr[field], rtol=1e-12, atol=1e-12)


def test_spline_mode_exports_logr_splines_without_density_subtraction(tmp_path):
    basis_path = tmp_path / "orbital_basis.npy"
    atom_path = tmp_path / "free_atom_splines.npy"
    np.save(basis_path, {"H": [(1, 1, 0)]}, allow_pickle=True)
    radii = np.geomspace(0.05, 20.0, 9)
    values = np.exp(-radii)
    np.save(atom_path, {
        1: {"x_samples": radii, "y_samples": values, "spline_order": 3},
    }, allow_pickle=True)
    prepare = make_md_preparation({
        "basis_path": str(basis_path), "atom_density_path": str(atom_path),
        "trusted_legacy": True, "atom_density_type": "spline",
        "remove_atom_density": False, "density_expansion": False,
    })
    result = prepare(_prepared([[-0.37, 0, 0], [0.37, 0, 0]]))
    assert "atom_density" not in result
    assert set(result["logr_splines"]) == {1}
    assert np.isfinite(result["logr_splines"][1](np.log([0.2, 1.0]))).all()


def test_sad_enrichment_reuses_existing_qc_grid_without_regenerating(tmp_path):
    basis_path, atom_path = _source_files(tmp_path)
    config = _config(basis_path, atom_path)
    positions = np.array([[-0.37, 0, 0], [0.37, 0, 0]])
    base = make_md_preparation(config)(_prepared(positions))
    qc_atoms = _prepared(positions)
    qc_atoms["coords"] = base["coords"].copy()
    qc_atoms["coord_weights"] = base["coord_weights"].copy()

    reuse_config = dict(config, density_expansion=False)
    enriched = make_md_preparation(reuse_config)(qc_atoms)
    np.testing.assert_array_equal(enriched["coords"], qc_atoms["coords"])
    np.testing.assert_array_equal(enriched["coord_weights"], qc_atoms["coord_weights"])
    np.testing.assert_allclose(enriched["atom_density"], base["atom_density"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(enriched["atom_density_grad"], base["atom_density_grad"],
                               atol=1e-12, rtol=1e-12)

    with pytest.raises(ValueError, match="existing coords and coord_weights"):
        make_md_preparation(reuse_config)(_prepared(positions))


def test_callback_rejects_missing_or_incompatible_free_atom_sources(tmp_path):
    basis_path, atom_path = _source_files(tmp_path)
    with pytest.raises(ValueError, match="atom_density_path"):
        make_md_preparation({"basis_path": str(basis_path), "trusted_legacy": True,
                             "atom_density_type": "df_coeffs"})
    with pytest.raises(ValueError, match="free-atom dipoles require"):
        make_md_preparation({"basis_path": str(basis_path), "trusted_legacy": True,
                             "include_atom_dipole": True})
    config = _config(basis_path, atom_path)
    config["trusted_legacy"] = False
    with pytest.raises(ValueError, match="trusted_legacy"):
        make_md_preparation(config)
