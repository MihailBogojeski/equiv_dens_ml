"""Exact adapter/layout contracts plus independent open-boundary neighbors."""
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.fixtures import REFERENCE_ROOT
from benchmarks.geometry import (fixture, load_reference, make_calculator,
                                 make_system)
from equiv_dens.md import dft_network_calculator as md
from equiv_dens.utils import base
from schnetpack import properties


REFERENCE = REFERENCE_ROOT


@pytest.fixture(scope='module')
def reference():
    if not (REFERENCE / 'src/equiv_dens/utils/base.py').is_file():
        pytest.skip('Immutable paper reference checkout is unavailable')
    return load_reference(REFERENCE)


def assert_inputs(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if torch.is_tensor(left[key]):
            torch.testing.assert_close(left[key], right[key], atol=1e-9, rtol=1e-7)
        else:
            assert left[key] == right[key]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape', [(1, 8), (4, 8), (1, 32)])
def test_same_prepared_inputs_reference(reference, dtype, shape):
    _, ref_md = reference
    before = make_calculator(ref_md, 'reference')
    after = make_calculator(md, 'native')
    for system in fixture(shape[1], shape[0], dtype=dtype):
        assert_inputs(before._generate_input(system), after._generate_input(system))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_layout_padding_order_and_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware required')
    numbers = torch.tensor([[8, 1, 1, 0], [6, 8, 1, 0]], device=device)
    positions = torch.arange(24, dtype=torch.float64, device=device).view(2, 4, 3)
    positions.requires_grad_()
    layout = base.TorchAtomBatchLayout(numbers, positions.dtype)
    expected_nums, expected_props = base.compress_batch_atoms(
        numbers.cpu().numpy(), {'positions': positions.detach().cpu().numpy()})
    np.testing.assert_array_equal(layout.numpy_numbers, expected_nums)
    packed = layout.pack_positions(positions)
    torch.testing.assert_close(packed.cpu(), torch.from_numpy(expected_props['positions']))
    assert layout.first_positions == {1: 0, 8: 2, 6: 3}
    assert layout.counts == [3, 3]
    packed.sum().backward()
    torch.testing.assert_close(positions.grad, (numbers > 0).unsqueeze(-1).expand_as(positions).double())
    assert layout.matches(numbers.clone(), positions.dtype)
    assert not layout.matches(numbers.flip(0), positions.dtype)
    assert not layout.matches(numbers[:, :3], positions.dtype)
    assert not layout.matches(numbers.to(torch.int32), positions.dtype)
    assert not layout.matches(numbers, torch.float32)
    assert not layout.matches(numbers, positions.dtype, 'cuda' if device == 'cpu' else 'cpu')


def brute_force(positions, cutoff):
    # Independent scalar enumeration; ordering matches upper pairs then reverse.
    pairs = [(i, j) for i in range(len(positions)) for j in range(i + 1, len(positions))
             if float(np.linalg.norm(positions[i] - positions[j])) < cutoff]
    return pairs + [(j, i) for i, j in pairs]


@pytest.mark.parametrize('backend', ['reference', 'native'])
@pytest.mark.parametrize('natoms', [1, 2, 5])
def test_neighbors_brute_force_strict_cutoff(backend, natoms):
    positions = torch.tensor([[0., 0, 0], [1., 0, 0], [2.0, 0, 0],
                              [2.001, 0, 0], [8., 0, 0]], dtype=torch.float64)[:natoms]
    atoms = {'positions': positions[None], 'atom_numbers': torch.ones(1, natoms, dtype=torch.long)}
    ii, jj, shifts = base.TorchNeighborList(1.0, backend=backend).get_neighbors(atoms)
    assert list(zip(ii[0].tolist(), jj[0].tolist())) == brute_force(positions.numpy(), 1.0)
    assert shifts[0].shape == (len(ii[0]), 3)
    assert not shifts[0].any()


def test_reference_single_pair_shape_is_normalized(reference):
    ref_base, _ = reference
    args = (torch.zeros(2, dtype=torch.bool), torch.tensor([[0., 0, 0], [0.5, 0, 0]]),
            torch.eye(3), torch.empty(0, 3, dtype=torch.long), 1.)
    ii, jj, _ = ref_base.neighbor_pairs(*args)
    # Qualified Torch accepts the legacy scalar index, but the helper shape is
    # inconsistent. The correction is a vector contract, not an observed crash.
    assert ii.ndim == jj.ndim == 0
    ii, jj, shifts = base.neighbor_pairs(*args)
    assert ii.shape == jj.shape == (1,)
    assert shifts.shape == (1, 3)


def test_disconnected_molecule_offsets_fail_before(reference):
    _, ref_md = reference
    numbers = torch.ones((2, 3), dtype=torch.long)
    positions = torch.tensor([[[0., 0, 0], [.5, 0, 0], [9., 0, 0]],
                              [[0., 0, 0], [.5, 0, 0], [9., 0, 0]]])
    system = make_system(positions, numbers)
    expected_i, expected_j = torch.tensor([0, 1, 3, 4]), torch.tensor([1, 0, 4, 3])
    original = make_calculator(ref_md, 'reference', cutoff=1)._generate_input(system)
    assert not torch.equal(original['idx_i'], expected_i)
    for backend in ('reference', 'native'):
        corrected = make_calculator(md, backend, cutoff=1)._generate_input(system)
        torch.testing.assert_close(corrected['idx_i'], expected_i)
        torch.testing.assert_close(corrected['idx_j'], expected_j)


def test_empty_first_molecule_offsets():
    numbers = torch.ones((2, 3), dtype=torch.long)
    positions = torch.tensor([[[0., 0, 0], [4., 0, 0], [9., 0, 0]],
                              [[0., 0, 0], [.5, 0, 0], [9., 0, 0]]])
    for backend in ('reference', 'native'):
        prepared = make_calculator(md, backend, cutoff=1)._generate_input(make_system(positions, numbers))
        torch.testing.assert_close(prepared['idx_i'], torch.tensor([3, 4]))


def test_batch_individual_centering_unpack():
    positions = torch.tensor([[[1., 0, 0], [0, 1, 0], [2, 0, 1], [7, 4, 2]],
                              [[4., 2, 1], [1, 2, 3], [0, 0, 0], [4, 4, 3]]], dtype=torch.float64)
    numbers = torch.tensor([[8, 1, 1, 0], [6, 8, 1, 0]])
    prepared = make_calculator()._generate_input(make_system(positions, numbers))
    centered = positions - base.center_of_mass(positions, numbers)
    expected_nums, expected = base.compress_batch_atoms(numbers.numpy(), {'positions': centered.numpy()})
    torch.testing.assert_close(prepared['batch_positions'], torch.from_numpy(expected['positions']))
    torch.testing.assert_close(prepared['batch_atom_numbers'], torch.from_numpy(expected_nums).long())
    assert not prepared['positions'].requires_grad
    weighted_sum = (prepared['batch_positions'] * prepared['batch_atom_numbers'].unsqueeze(-1)).sum(1)
    torch.testing.assert_close(weighted_sum, torch.zeros_like(weighted_sum), atol=1e-14, rtol=0)
    for i in range(2):
        individual = make_calculator()._generate_input(make_system(positions[i:i+1], numbers[i:i+1]))
        mask = prepared['batch_atom_mask'][i]
        # Species insertion order depends on earlier batch members; compare the
        # same atoms after stable species ordering, preserving ties.
        batch_order = prepared['batch_atom_numbers'][i, mask].argsort(stable=True)
        individual_order = individual['atom_numbers'][0].argsort(stable=True)
        torch.testing.assert_close(prepared['batch_positions'][i, mask][batch_order],
                                   individual['positions'][0, individual_order])
    prepared['forces'] = torch.arange(prepared['positions'].numel(), dtype=torch.float64).view_as(prepared['positions'])
    original_forces = prepared['forces'].clone()
    unpacked = base.batch_compressed_atoms(prepared, ['forces'])['forces']
    torch.testing.assert_close(unpacked.flatten(0, 1)[prepared['batch_idx_pos']], original_forces[0])
    assert not unpacked[~prepared['batch_atom_mask']].any()


def test_moved_frames_cache_refresh_and_distance_gradients():
    calc = make_calculator(cutoff=1.)
    numbers = torch.tensor([[8, 1, 1]])
    positions = torch.tensor([[[0., 0, 0], [.5, 0, 0], [2., 0, 0]]], dtype=torch.float64)
    prepared = calc._generate_input(make_system(positions, numbers))
    layout = calc._geometry_layout
    assert len(prepared['idx_i']) == 2
    positions[:, -1, 0] = .75
    prepared = calc._generate_input(make_system(positions, numbers))
    assert calc._geometry_layout is layout
    assert len(prepared['idx_i']) == 6
    coords = prepared['positions'].requires_grad_()
    distances, _ = base.calculate_distances_and_directions(coords, prepared['idx_i'], prepared['idx_j'])
    grad = torch.autograd.grad(distances.square().sum(), coords)[0]
    independent = torch.zeros_like(coords)
    for i, j in zip(prepared['idx_i'].tolist(), prepared['idx_j'].tolist()):
        independent[:, i] += 2 * (coords[:, i].detach() - coords[:, j].detach())
        independent[:, j] += 2 * (coords[:, j].detach() - coords[:, i].detach())
    torch.testing.assert_close(grad, independent)
    assert grad.abs().max() > 0
    changed = numbers.clone()
    changed[0, 0] = 6
    calc._generate_input(make_system(positions, changed))
    assert calc._geometry_layout is not layout
    layout = calc._geometry_layout
    calc._generate_input(make_system(positions.float(), changed))
    assert calc._geometry_layout is not layout


def test_grid_policy_unchanged():
    calls = []
    calc = make_calculator()
    calc.density_expansion = True
    calc.pyscf_grid = False
    calc.cache_grid = True
    calc.grid_cache_threshold = .1
    calc._cached_positions = None
    calc._cached_grid_coords = None
    calc._cached_grid_weights = None
    calc._cached_atom_density = None
    calc.grid_spec = {}
    def sample(spec, count, numbers, positions):
        calls.append(positions.clone())
        return positions[:, :1].clone(), torch.ones(positions.shape[0], 1).to(positions)
    calc.grid_sampling_fn = sample
    frame = fixture(8)[0]
    before = calc._generate_input(frame)
    after = calc._generate_input(frame)
    assert len(calls) == 1
    torch.testing.assert_close(before['coords'], after['coords'])
    frame[properties.R][:, -1, 0] += 1.
    calc._generate_input(frame)
    assert len(calls) == 2


def test_layout_refresh_on_batch_order_and_atom_count():
    calc = make_calculator()
    numbers = torch.tensor([[8, 1, 1], [6, 1, 1]])
    positions = torch.tensor([[[0., 0, 0], [.5, 0, 0], [0, .5, 0]],
                              [[0., 0, 0], [.7, 0, 0], [0, .7, 0]]])
    calc._generate_input(make_system(positions, numbers))
    layout = calc._geometry_layout
    system = make_system(positions.flip(0), numbers.flip(0))
    actual = calc._generate_input(system)
    assert calc._geometry_layout is not layout
    assert_inputs(actual, make_calculator(md, 'reference')._generate_input(system))
    layout = calc._geometry_layout
    larger_positions = torch.cat((positions, torch.zeros(2, 1, 3)), 1)
    larger_numbers = torch.cat((numbers, torch.ones(2, 1, dtype=torch.long)), 1)
    system = make_system(larger_positions, larger_numbers)
    actual = calc._generate_input(system)
    assert calc._geometry_layout is not layout
    assert_inputs(actual, make_calculator(md, 'reference')._generate_input(system))


def test_native_periodic_falls_back_to_existing_semantics():
    atoms = {'positions': torch.tensor([[[0., 0, 0], [.4, .2, 0]]]),
             'atom_numbers': torch.tensor([[1, 8]]), 'cell': torch.eye(3) * 3}
    before = base.TorchNeighborList(1., 'reference').get_neighbors(atoms, pbc=True)
    after = base.TorchNeighborList(1., 'native').get_neighbors(atoms, pbc=True)
    for ref_lists, actual_lists in zip(before, after):
        for ref, actual in zip(ref_lists, actual_lists):
            torch.testing.assert_close(ref, actual)
