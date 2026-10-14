"""Host-only PySCF data preparation tests; no Torch reference imports."""
import numpy as np
import pytest

from equiv_dens_jax.preparation import (
    BOHR_ANGSTROM,
    batch_pyscf_grids,
    build_pyscf_molecules,
    load_array_mapping,
    load_legacy_qc_results,
    make_pyscf_grid,
    prepare_dataset_batches,
    sample_molecular_density,
    sample_sad_coeff_density,
    sample_sad_spline,
)


def test_legacy_object_loading_requires_explicit_trust_and_npz_stays_safe(tmp_path):
    object_path = tmp_path / "atoms.npy"
    np.save(object_path, {"positions": np.zeros((1, 1, 3)), "atom_numbers": np.ones((1, 1))})
    with pytest.raises(ValueError, match="trusted_legacy"):
        load_array_mapping(object_path)
    loaded = load_array_mapping(object_path, trusted_legacy=True)
    np.testing.assert_array_equal(loaded["atom_numbers"], [[1]])

    qc_path = tmp_path / "QC-results.npy"
    np.save(qc_path, np.asarray([({"atom": "H 0 0 0"}, {"mo_coeff": np.ones((1, 1))})], dtype=object))
    with pytest.raises(ValueError, match="trusted_legacy"):
        load_legacy_qc_results(qc_path)
    records = load_legacy_qc_results(qc_path, trusted_legacy=True)
    assert records[0][0]["atom"] == "H 0 0 0"

    numeric_path = tmp_path / "atoms.npz"
    np.savez(numeric_path, atom_numbers=np.ones((1, 1), dtype=np.int64), positions=np.zeros((1, 1, 3)))
    np.testing.assert_array_equal(load_array_mapping(numeric_path)["atom_numbers"], [[1]])


def _hydrogen_molecule(basis="sto-3g"):
    from pyscf import gto
    mol = gto.M(atom="H 0 0 0", basis=basis, unit="Angstrom", spin=1)
    mol.build()
    return mol


def test_atomic_core_mo_sad_preserves_rotations_and_translations():
    # Equal p occupations give a spherical oxygen density without any SCF.
    entries={8:{"mo_basis":"sto-3g","mo_coeff":np.eye(5),
                "mo_occ":np.array([2.,2.,4/3,4/3,4/3])}}
    rng=np.random.default_rng(1923)
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    direction=rng.normal(size=(200,3));direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    grid=(direction*np.geomspace(.02,3.,200)[:,None])[None]
    positions=np.zeros((1,1,3));numbers=np.array([[8]])
    reference,_=sample_sad_coeff_density(positions,numbers,grid,entries,
                                        coefficient_type="mo_coeffs",density_grad=True)
    shift=np.array([.7,-.4,.2])
    actual,_=sample_sad_coeff_density(positions@q.T+shift,numbers,grid@q.T+shift,entries,
                                     coefficient_type="mo_coeffs",density_grad=True)
    np.testing.assert_allclose(actual[...,0],reference[...,0],atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(actual[...,1:],reference[...,1:]@q.T,atol=1e-10,rtol=1e-10)


def test_molecular_mo_and_projected_density_return_values_and_spatial_gradient():
    mol = _hydrogen_molecule()
    coords_ang = np.array([[[0.31, -0.17, 0.42], [0.53, 0.29, -0.21]]])
    coeffs = [{"mo_coeff": np.array([[1.0]]), "mo_occ": np.array([1.0])}]
    sampled = sample_molecular_density([mol], coords_ang, coeffs, density_grad=True)
    assert sampled.shape == (1, 2, 4)
    assert np.isfinite(sampled).all()

    projected = sample_molecular_density([mol], coords_ang, [np.array([0.75])],
                                         projected=True, density_grad=True)
    assert projected.shape == (1, 2, 4)
    # The explicit PySCF derivative is with respect to Bohr coordinates.
    h_ang = 2e-5
    for axis in range(3):
        plus, minus = coords_ang.copy(), coords_ang.copy()
        plus[0, 0, axis] += h_ang
        minus[0, 0, axis] -= h_ang
        rho_plus = sample_molecular_density([mol], plus, coeffs)[0, 0]
        rho_minus = sample_molecular_density([mol], minus, coeffs)[0, 0]
        finite_difference_per_bohr = (rho_plus - rho_minus) / (2 * h_ang / BOHR_ANGSTROM)
        np.testing.assert_allclose(sampled[0, 0, axis + 1], finite_difference_per_bohr,
                                   atol=2e-6, rtol=2e-5)


def test_saved_molecule_records_and_deterministic_pyscf_grid():
    records = [({"atom": "H 0 0 0", "basis": "sto-3g", "spin": 1}, {})]
    mols = build_pyscf_molecules(records)
    coords, weights = make_pyscf_grid(mols[0], level=0, count=12, seed=4)
    coords_again, weights_again = make_pyscf_grid(mols[0], level=0, count=12, seed=4)
    assert coords.shape == (12, 3)
    assert weights.shape == (12,)
    np.testing.assert_array_equal(coords, coords_again)
    np.testing.assert_array_equal(weights, weights_again)
    batch_coords, batch_weights = batch_pyscf_grids(mols, level=0, count=12, seed=4)
    np.testing.assert_array_equal(batch_coords[0], coords)
    np.testing.assert_array_equal(batch_weights[0], weights)


def test_prepare_dataset_batches_keeps_centered_grid_and_targets_aligned(tmp_path):
    atom_path = tmp_path / "atoms.npy"
    np.save(atom_path, {
        "positions": np.array([[[0.2, -0.1, 0.3]]]),
        "atom_numbers": np.array([[1]]),
        "energy": np.array([[0.25]]),
        "forces": np.array([[[0.1, 0.2, 0.3]]]),
    })
    basis_path = tmp_path / "orbitals.npy"
    np.save(basis_path, {"H": [(1, 0, 0)]})
    qc_path = tmp_path / "QC-results.npy"
    record = ({"atom": "H 0.2 -0.1 0.3", "basis": "sto-3g", "spin": 1},
              {"mo_coeff": np.array([[1.0]]), "mo_occ": np.array([1.0])})
    np.save(qc_path, np.asarray([record], dtype=object))
    batch, = prepare_dataset_batches(atom_path, basis_path, qc_path, count=6,
                                    seed=8, trusted_legacy=True, grid_level=0,
                                    density_grad=True)
    assert batch["batch_positions"].shape == (1, 1, 3)
    np.testing.assert_allclose(batch["batch_positions"], 0.0, atol=1e-14)
    assert batch["coords"].shape == (1, 6, 3)
    assert batch["density"].shape == (1, 6)
    assert batch["density_grad"].shape == (1, 6, 3)
    np.testing.assert_allclose(batch["energy"], [[0.25]])
    np.testing.assert_allclose(batch["forces"], [[[0.1, 0.2, 0.3]]])


def test_sad_spline_values_and_explicit_gradient_match_log_radius_formula():
    log_r = np.linspace(-2.0, 2.0, 20)
    radii = np.exp(log_r)
    values = 2.0 + 3.0 * log_r + 0.5 * log_r**2
    entries = {1: {"x_samples": radii, "y_samples": values, "spline_order": 3}}
    positions = np.zeros((1, 1, 3))
    numbers = np.ones((1, 1), dtype=np.int64)
    coords = np.array([[[0.4, 0.2, -0.1], [-0.3, 0.5, 0.2]]])
    result = sample_sad_spline(positions, numbers, coords, entries, density_grad=True)
    r_bohr = np.linalg.norm(coords, axis=-1) / BOHR_ANGSTROM
    log_eval = np.log(r_bohr)
    expected = 2.0 + 3.0 * log_eval + 0.5 * log_eval**2
    expected_grad = ((3.0 + log_eval)[..., None] * coords
                     / (r_bohr[..., None]**2 * BOHR_ANGSTROM))
    np.testing.assert_allclose(result[..., 0], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(result[..., 1:], expected_grad, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("coefficient_type", ["mo_coeffs", "df_coeffs"])
def test_sad_coefficient_densities_include_atomwise_values_and_gradients(coefficient_type):
    positions = np.array([[[0.13, -0.21, 0.08]]])
    numbers = np.ones((1, 1), dtype=np.int64)
    coords = positions + np.array([[[0.2, 0.1, -0.3], [0.5, -0.2, 0.1]]])
    if coefficient_type == "mo_coeffs":
        entries = {1: {"mo_basis": "sto-3g", "mo_coeff": np.array([[1.0]]),
                       "mo_occ": np.array([1.0])}}
    else:
        entries = {1: {"df_basis": "sto-3g", "df_coeffs": np.array([0.8])}}
    total, atomwise = sample_sad_coeff_density(positions, numbers, coords, entries,
                                               coefficient_type=coefficient_type,
                                               density_grad=True)
    assert total.shape == (1, 2, 4)
    assert atomwise.shape == (1, 1, 2, 4)
    np.testing.assert_allclose(total, atomwise[:, 0], atol=1e-12, rtol=1e-12)
    shift = np.array([3.0, -1.2, 0.4])
    translated, _ = sample_sad_coeff_density(
        positions + shift, numbers, coords + shift, entries,
        coefficient_type=coefficient_type, density_grad=True)
    np.testing.assert_allclose(translated, total, atol=1e-9, rtol=1e-9)

