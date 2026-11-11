import equiv_dens.compat  # noqa: F401 - apply T_co patch before schnetpack import

import os
import time
import warnings
import torch
import torch.nn as nn
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from schnetpack.md.calculators import MDCalculator, MDCalculatorError
from schnetpack import properties
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
import ase.units
from pyscf import gto, dft

try:
    from equiv_dens.inference import FastInferenceWrapper, InferenceSettings
    HAS_FAST_INFERENCE = True
except ImportError:
    HAS_FAST_INFERENCE = False


# MD calculator class
class DFTNetworkCalculator(MDCalculator):
    def __init__(self,
                 model,
                 required_properties,
                 force_key,
                 energy_key='energy',
                 verbose=0,
                 n_jobs=10,
                 density_expansion=False,
                 position_unit="Angstrom",
                 energy_unit="kcal/mol",
                 grid_spec=None,
                 grid_sampling_fn=None,
                 use_gpu=False,
                 detach=True,
        cutoff=7.937658158457616,
        pyscf_grid=False,
        atom_dens=None,
        atom_dens_type=None,
        remove_atom_density=False,
        dpm_intor=False,
        enable_tf32=False,
        enable_inference_mode=True,
        cache_grid=True,
        grid_cache_threshold=0.1,
        use_fast_inference=False,
        compile_model=False,
        geometry_backend="reference",
        static_compile=False,
        ):
        # energy prediction model
        super().__init__(
            required_properties,
            force_key,
            energy_unit,
            position_unit,
            energy_key=energy_key,
        )
        self.model = model
        # density prediction model
        self.grid_sampling_fn = grid_sampling_fn
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.cutoff = cutoff
        self.use_gpu = use_gpu
        self.pyscf_grid = pyscf_grid
        # print('calculator grid spec', grid_spec)
        self.grid_spec = {}
        if self.use_gpu and not self.pyscf_grid:
            for key in grid_spec.keys():
                self.grid_spec[key] = (grid_spec[key][0].cuda(),
                                       grid_spec[key][1].cuda())
        else:
            self.grid_spec = grid_spec
        self.density_expansion = density_expansion
        self.detach = detach
        self.atom_dens = atom_dens
        self.atom_dens_type = atom_dens_type or 'spline'
        self.remove_atom_density = remove_atom_density
        self.dpm_intor = dpm_intor
        if geometry_backend not in ("reference", "native"):
            raise ValueError("Geometry backend must be reference or native")
        self.geometry_backend = geometry_backend
        self._geometry_layout = None
        self._geometry_neighbors = None
        # static compile: fixed shapes and CUDA graphs, the neighbor pairs are padded to a grow-only capacity
        self.static_compile = static_compile
        self._pair_capacity = 0
        if static_compile:
            self.model.enable_compile(static=True)
        
        # Performance optimizations
        self.enable_tf32 = enable_tf32
        self.enable_inference_mode = enable_inference_mode
        self.cache_grid = cache_grid
        self.grid_cache_threshold = grid_cache_threshold
        
        # Grid caching state
        self._cached_grid_coords = None
        self._cached_grid_weights = None
        self._cached_positions = None
        self._cached_atom_density = None
        
        # Set TF32 explicitly (model_loader enables it globally at import)
        torch.backends.cuda.matmul.allow_tf32 = self.enable_tf32
        torch.backends.cudnn.allow_tf32 = self.enable_tf32
        torch.set_float32_matmul_precision('high' if self.enable_tf32 else 'highest')
        if self.verbose > 0:
            print(f"TF32 {'enabled' if self.enable_tf32 else 'disabled'}")

        # Optional FastInferenceWrapper integration
        self.use_fast_inference = use_fast_inference
        self.compile_model = compile_model
        self._inference_wrapper = None
        
        if self.use_fast_inference and HAS_FAST_INFERENCE:
            settings = InferenceSettings(
                tf32=self.enable_tf32,
                compile=self.compile_model,
                cuda_graph=False,  # CUDA graphs don't work well with MD (dynamic shapes)
            )
            self._inference_wrapper = FastInferenceWrapper(self.model, settings)
            if self.verbose > 0:
                print(f"FastInferenceWrapper enabled (tf32={self.enable_tf32}, compile={self.compile_model})")

    def calculate(self, system):
        """
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation and uses the results to update the system state.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        """
        # set model to evaluation mode to disable graph creation
        start = time.time()
        self.model.eval()

        inputs = self._generate_input(system)

        # Run model forward pass with optional inference mode for speed
        start_model = time.time()
        compute_forces = 'forces' in self.required_properties
        
        if self._inference_wrapper is not None:
            # Use FastInferenceWrapper if available
            results = self._inference_wrapper(inputs, compute_forces=compute_forces)
        elif self.enable_inference_mode and not compute_forces:
            # Use inference mode when we don't need gradients for forces
            with torch.inference_mode():
                results = self.model(inputs)
        else:
            # Need gradients for force computation
            results = self.model(inputs)
        
        start_coeffs = time.time()
        vector_coeffs = orbitals.coeffs_dict_to_vector(results, self.model.density_repr_model[0].orbital_basis,
                                                       results['batch_atom_numbers'])
        start_other = time.time()
        results['spherical_coeffs'] = vector_coeffs['spherical_coeffs']
        results['radial_width'] = vector_coeffs['radial_width']
        results['radial_scale'] = vector_coeffs['radial_scale']
        results = utils.batch_compressed_atoms(results, ['positions', 'forces'])
        if 'forces' in results:
            # packed (element-grouped) order -> system atom order
            forces = results['forces'].masked_fill(~self._unpack_mask.unsqueeze(-1), 0)
            idx = self._unpack_idx.unsqueeze(-1).expand_as(forces)
            results['forces'] = forces.new_zeros(forces.shape[0], self._natoms, 3).scatter_add_(1, idx, forces)
        self.results = {}
        for p in self.required_properties:
            if p not in results:
                raise MDCalculatorError(
                    "Requested property {:s} not in " "results".format(p)
                )
            else:
                # Detach properties if requested
                self.results[p] = results[p].detach() if self.detach else results[p]
        self._update_system(system)

    def _can_use_cached_grid(self, positions):
        """Check if cached grid can be reused based on position displacement."""
        if not self.cache_grid or self._cached_positions is None:
            return False
        
        # Compute max displacement from cached positions
        displacement = torch.max(torch.abs(positions - self._cached_positions))
        return displacement.item() < self.grid_cache_threshold
    
    def _generate_input(self, system):
        """
        Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
        formatted input for the schnetpack model.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            dict(torch.Tensor): Schnetpack inputs in dictionary format.
        """
        sys_mols = self._get_system_molecules(
            system
        )
        positions = sys_mols[properties.R]
        atom_types = sys_mols[properties.Z]
        if self.use_gpu:
            positions = positions.cuda()
            atom_types = atom_types.cuda()
        natoms = int(sys_mols[properties.n_atoms][0])
        self._natoms = natoms
        positions = positions.view(-1, natoms, 3)
        atom_types = atom_types.view(-1, natoms)
        center = torch.sum(positions * atom_types.unsqueeze(-1), 1) / torch.sum(atom_types, 1).unsqueeze(-1)
        # inputs = {'positions': positions + 10,
        native_geometry = self.geometry_backend == "native"
        if native_geometry:
            layout = self._geometry_layout
            if layout is None or not layout.matches(atom_types, positions.dtype, positions.device):
                layout = utils.TorchAtomBatchLayout(atom_types, positions.dtype, positions.device)
                self._geometry_layout = layout
            atom_numbers = layout.numpy_numbers
            self._unpack_idx, self._unpack_mask = layout.gather_indices, layout.mask
            # Preserve the original MD adapter's detached coordinate boundary.
            positions = layout.pack_positions(positions - center.unsqueeze(1)).detach()
        else:
            props = {'positions': (positions - center.unsqueeze(1)).cpu().numpy(),
                     'indices': np.broadcast_to(np.arange(natoms), atom_types.shape)}
            atom_numbers, props = utils.compress_batch_atoms(atom_types.cpu().numpy(), props)
            self._unpack_idx = torch.from_numpy(props['indices']).long().to(positions.device)
            self._unpack_mask = torch.from_numpy(atom_numbers > 0).to(positions.device)
            positions = torch.from_numpy(props['positions']).to(positions)
        inputs = {}
        if self.density_expansion:
            # Check if we can use cached grid coordinates
            use_cached = self._can_use_cached_grid(positions)
            
            if use_cached and self._cached_grid_coords is not None:
                # Reuse cached grid
                sample_coords = self._cached_grid_coords
                coord_weights = self._cached_grid_weights
            else:
                # Generate new grid
                if self.pyscf_grid:
                    sample_coords, coord_weights = get_pyscf_coords(self.grid_spec, 10000000000,
                                                                    atom_numbers,
                                                                    positions)
                else:
                    sample_coords, coord_weights = self.grid_sampling_fn(self.grid_spec, 10000000000,
                                                                         atom_numbers,
                                                                         positions)
                # Cache for next step
                if self.cache_grid:
                    self._cached_grid_coords = sample_coords
                    self._cached_grid_weights = coord_weights
                    self._cached_positions = positions.clone()
            
            inputs['coords'] = sample_coords
            inputs['coord_weights'] = coord_weights
            if self.remove_atom_density and self.atom_dens is not None:
                # Check if we can use cached atom density
                if use_cached and self._cached_atom_density is not None:
                    inputs['atom_density'] = self._cached_atom_density
                else:
                    positions_for_dens = positions.view(-1, atom_numbers.shape[1], 3).cpu()
                    atom_numbers_t = torch.tensor(atom_numbers).type(torch.long)
                    sample_coords_cpu = sample_coords.cpu() if sample_coords.is_cuda else sample_coords
                    atom_dens, _ = orbitals.sample_atom_density(
                        positions=positions_for_dens,
                        atom_numbers=atom_numbers_t,
                        coords=sample_coords_cpu,
                        atom_dens_type=self.atom_dens_type,
                        atom_dens_dict=self.atom_dens,
                    )
                    inputs['atom_density'] = atom_dens.to(positions)
                    if self.cache_grid:
                        self._cached_atom_density = inputs['atom_density']

        # dpm_intor + remove_atom_density needs atom_mo_coeffs/atom_df_coeffs for intor_dipole_moment_free_atom
        if (not self.density_expansion and self.dpm_intor and self.remove_atom_density
                and self.atom_dens is not None and 'coeffs' in self.atom_dens_type):
            atom_dens_t = self.atom_dens_type
            (inputs['atom_' + atom_dens_t],
             inputs['atom_' + atom_dens_t + '_occ'],
             inputs['atom_' + atom_dens_t + '_basis']) = orbitals.join_atom_coeffs(
                torch.LongTensor(atom_numbers), self.atom_dens, atom_dens_t)

        inputs['positions'] = positions
        inputs['atom_numbers_first_positions'] = (dict(layout.first_positions) if native_geometry
                                                  else utils.get_atom_num_first_positions(atom_numbers))
        inputs['atom_numbers'] = (layout.numbers.clone() if native_geometry
                                 else torch.tensor(atom_numbers).to(positions).type(torch.long))
        inputs['atom_mask'] = inputs['atom_numbers'] > 0

        if native_geometry:
            if self._geometry_neighbors is None:
                self._geometry_neighbors = utils.TorchNeighborList(self.cutoff, backend="native")
            nl = self._geometry_neighbors
            nl.cutoff = self.cutoff
        else:
            nl = utils.TorchNeighborList(self.cutoff)
        idx_is, idx_js, _ = nl.get_neighbors(inputs)
        prev_max = 0
        for i in range(len(idx_is)):
            idx_is[i] += prev_max
            idx_js[i] += prev_max
            # Offset by all real atoms, including isolated atoms absent in edges.
            prev_max += (layout.counts[i] if native_geometry
                         else int(np.count_nonzero(atom_numbers[i])))

        if not native_geometry:
            atom_batch_idx = np.zeros_like(atom_numbers)
            for i in range(len(atom_numbers)):
                atom_batch_idx[i, :] = i
            atom_batch_idx = torch.tensor(atom_batch_idx).to(positions).type(torch.long)

        idx_is = torch.cat(idx_is, dim=0)
        idx_js = torch.cat(idx_js, dim=0)
        inputs['idx_i'] = idx_is
        inputs['idx_j'] = idx_js
        inputs['batch_atom_numbers'] = inputs['atom_numbers'] * 1
        inputs['batch_atom_mask'] = (inputs['atom_mask'] * 1).type(torch.bool)
        inputs['batch_positions'] = inputs['positions'] * 1
        inputs['positions'] = positions.view(1, -1, *inputs['positions'].shape[2:])
        inputs['atom_numbers'] = inputs['batch_atom_numbers'].flatten()
        inputs['atom_mask'] = inputs['batch_atom_mask'].flatten()
        if native_geometry:
            inputs['batch_idx_pos'] = layout.batch_idx_pos.clone()
        else:
            batch_nz = inputs['atom_mask'].to(inputs['positions'])
            batch_idx_pos = batch_nz * torch.arange(len(batch_nz)).to(batch_nz)
            inputs['batch_idx_pos'] = batch_idx_pos[inputs['atom_mask']].to(torch.long)
        inputs['atom_numbers'] = inputs['atom_numbers'][inputs['atom_mask']].view(1, -1)
        if native_geometry:
            inputs['atom_batch_idx'] = layout.atom_batch_idx.clone()
        else:
            inputs['atom_batch_idx'] = atom_batch_idx.flatten()
            inputs['atom_batch_idx'] = inputs['atom_batch_idx'][inputs['atom_mask']].view(1, -1)
        inputs['positions'] = inputs['positions'][:, inputs['atom_mask']]
        if self.static_compile:
            self._pad_pairs(inputs)

        return inputs

    def _pad_pairs(self, inputs):
        """Pad the neighbor pairs to a grow-only capacity (10% headroom, multiples of 128, at most all pairs) so
        that the pair count of the static compile only changes when the capacity grows. The padding pairs are pairs
        of the same molecule outside of the cutoff: their radial basis (cutoff function) and with it their messages
        and gradients are exactly zero."""
        idx_i, idx_j = inputs['idx_i'], inputs['idx_j']
        b = inputs['atom_batch_idx'][0]
        N = b.shape[0]
        n_all = int(np.sum(np.bincount(b.cpu().numpy()) ** 2)) - N  # ordered pairs within the molecules
        self._pair_capacity = min(max(self._pair_capacity, -(-int(1.1 * idx_i.numel()) // 128) * 128), n_all)
        pad = self._pair_capacity - idx_i.numel()
        if pad > 0:
            outside = b[:, None] == b[None, :]
            outside.fill_diagonal_(False)
            outside[idx_i, idx_j] = False
            far_i, far_j = outside.nonzero().T
            inputs['idx_i'] = torch.cat([idx_i, far_i[:pad]])
            inputs['idx_j'] = torch.cat([idx_j, far_j[:pad]])


def get_pyscf_coords(grid_spec, density_n_samp, atom_numbers, positions):
    """
    Get density grid coordinates using PySCF gen_grid.

    Args:
    idx (list of int): index of molecule(s) to get coordinates for
    Returns:
    coords (torch.Tensor): coordinates of grid points
    weights (torch.Tensor): integration weights of grid points
    """
    # mol = utils.npy_to_ase(dataset.atoms['positions'][0:1], dataset.atoms['atom_numbers'][0:1])[0]
    # utils.npy_to_ase(dataset.atoms['positions'][0:1], dataset.atoms['atom_numbers'][0:1])[0]
    start = time.time()
    max_len = 0
    all_coords = []
    all_weights = []
    atom_numbers = atom_numbers.astype(int)
    pos = positions.detach().cpu().numpy()
    if atom_numbers.ndim < 2:
        atom_numbers = atom_numbers.unsqueeze(0)
        pos = pos.unsqueeze(0)
    for i in range(atom_numbers.shape[0]):
        loop_start = time.time()
        atom = [(atom_numbers[i, j], pos[i, j]) for j in range(atom_numbers.shape[1])]
        mol = gto.Mole(atom=atom)
        if not mol._built:
            build_start = time.time()
            mol.build()
        rot_spec = grid_spec
        coords, weights = dft.gen_grid.get_partition(mol, rot_spec)
        # print('coords shape', coords.shape)
        # print('weights shape', weights)
        if density_n_samp > coords.shape[0]:
            coords = torch.tensor(coords).to(positions)
            weights = torch.tensor(weights).to(positions)
        else:
            rand_idx = np.random.choice(np.arange(coords.shape[0]),
                                        size=density_n_samp, replace=False)
            coords = torch.tensor(coords[:, rand_idx]).to(positions)
            weights = torch.tensor(weights[:, rand_idx]).to(positions)
        all_coords.append(coords)
        all_weights.append(weights)
    pad_coords = nn.utils.rnn.pad_sequence(all_coords, batch_first=True, padding_value=0) * utils.to_angstrom
    pad_weights = nn.utils.rnn.pad_sequence(all_weights, batch_first=True, padding_value=0)
    return pad_coords, pad_weights


def geometry_to_model_inputs(atom_numbers, positions, cutoff, use_gpu=False):
    """Build DenSNet inputs from batched atom numbers and positions tensors."""
    if not isinstance(atom_numbers, np.ndarray):
        atom_numbers = np.asarray(atom_numbers, dtype=np.int64)
    if atom_numbers.ndim == 1:
        atom_numbers = atom_numbers[None, :]
    if positions.ndim == 2:
        positions = positions.unsqueeze(0)
    if use_gpu and not positions.is_cuda:
        positions = positions.cuda()
    inputs = {
        "positions": positions,
        "atom_numbers_first_positions": utils.get_atom_num_first_positions(atom_numbers),
        "atom_numbers": torch.tensor(atom_numbers, dtype=torch.long, device=positions.device),
    }
    inputs["atom_mask"] = inputs["atom_numbers"] > 0
    nl = utils.TorchNeighborList(cutoff)
    idx_is, idx_js, _ = nl.get_neighbors(inputs)
    prev_max = 0
    for i in range(len(idx_is)):
        idx_is[i] += prev_max
        idx_js[i] += prev_max
        max_i = torch.max(idx_is[i]).item() if idx_is[i].numel() > 0 else prev_max
        max_j = torch.max(idx_js[i]).item() if idx_js[i].numel() > 0 else prev_max
        prev_max = max(max_i, max_j) + 1
    atom_batch_idx = np.zeros_like(atom_numbers)
    for i in range(len(atom_numbers)):
        atom_batch_idx[i, :] = i
    atom_batch_idx = torch.tensor(atom_batch_idx, dtype=torch.long, device=positions.device)
    empty = torch.tensor([], dtype=torch.long, device=positions.device)
    idx_is = torch.cat(idx_is, dim=0) if idx_is else empty
    idx_js = torch.cat(idx_js, dim=0) if idx_js else empty
    inputs["idx_i"] = idx_is
    inputs["idx_j"] = idx_js
    inputs["batch_atom_numbers"] = inputs["atom_numbers"] * 1
    inputs["batch_atom_mask"] = inputs["atom_mask"].to(torch.bool)
    inputs["batch_positions"] = inputs["positions"] * 1
    inputs["positions"] = positions.view(1, -1, *inputs["positions"].shape[2:])
    inputs["atom_numbers"] = inputs["batch_atom_numbers"].flatten()
    inputs["atom_mask"] = inputs["batch_atom_mask"].flatten()
    batch_nz = inputs["atom_mask"].to(inputs["positions"])
    batch_idx_pos = batch_nz * torch.arange(len(batch_nz), device=batch_nz.device)
    inputs["batch_idx_pos"] = batch_idx_pos[inputs["atom_mask"]].to(torch.long)
    inputs["atom_numbers"] = inputs["atom_numbers"][inputs["atom_mask"]].view(1, -1)
    inputs["atom_batch_idx"] = atom_batch_idx.flatten()
    inputs["atom_batch_idx"] = inputs["atom_batch_idx"][inputs["atom_mask"]].view(1, -1)
    inputs["positions"] = inputs["positions"][:, inputs["atom_mask"]]
    return inputs


def ase_atoms_to_model_inputs(atoms, cutoff, use_gpu=False):
    """Build DenSNet energy/force inputs from an ASE Atoms object."""
    atom_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)[None, :]
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32).unsqueeze(0)
    return geometry_to_model_inputs(atom_numbers, positions, cutoff, use_gpu)


def model_needs_density_grid(model):
    """True if ``model`` evaluates a density head that reads grid coordinates.

    ``DFTNetwork.forward`` unconditionally runs every property model the
    checkpoint was built with (see ``model_loader.build_model``), regardless
    of what a caller actually wants back. A checkpoint trained with
    ``density_weight > 0`` (or a core-density basis) carries a ``'density'``
    or ``'core_density'`` entry in ``model.property_models``, and that head's
    forward pass indexes ``inputs['coords']`` (see
    ``equiv_dens.nn.property_output.density.DensityOutput.forward``). If the
    calculator does not attach an integration grid in that case, inference
    crashes with ``KeyError: 'coords'`` -- the failure this function exists
    to avoid (see ``relax_water_clusters.py`` / ``submit_hbond_relax.sbatch``).
    """
    property_models = getattr(model, "property_models", None)
    if not property_models:
        return False
    try:
        keys = set(property_models.keys())
    except AttributeError:
        return False
    return bool(keys & {"density", "core_density"})


class DenSNetCalculator(Calculator):
    """ASE energy+force calculator wrapping a loaded DenSNet model.

    Model energies are interpreted as ``energy_unit`` (paper default kcal/mol)
    and converted to eV / eV/Å for ASE.
    """

    implemented_properties = ["energy", "forces", "dipole"]

    def __init__(
        self,
        model,
        cutoff=7.937658158457616,
        use_gpu=False,
        energy_unit="kcal/mol",
        grid_spec=None,
        grid_sampling_fn=None,
        pyscf_grid=False,
        need_density_grid=False,
        atom_dens=None,
        atom_dens_type=None,
        remove_atom_density=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_gpu = bool(use_gpu and torch.cuda.is_available())
        self.model = model.cuda() if self.use_gpu else model.cpu()
        self.model.eval()
        self.cutoff = float(cutoff)
        self.energy_unit = energy_unit
        self.grid_spec = grid_spec or {}
        self.grid_sampling_fn = grid_sampling_fn
        self.pyscf_grid = bool(pyscf_grid)
        self.need_density_grid = bool(need_density_grid)
        self.atom_dens = atom_dens
        self.atom_dens_type = atom_dens_type or "spline"
        self.remove_atom_density = bool(remove_atom_density)

    def _grid_spec_from_atoms(self, atom_numbers, positions):
        """Build a PySCF atomic-grid spec from the current geometry."""
        from equiv_dens.utils.grids import spherical_grid

        pos = positions.detach().cpu().numpy() if torch.is_tensor(positions) else np.asarray(positions)
        return spherical_grid(
            {"atom_numbers": np.asarray(atom_numbers), "positions": pos}
        )

    def _add_density_grid(self, inputs, atom_numbers, positions):
        """Attach integration grid coordinates expected by the density head."""
        if self.grid_sampling_fn is not None:
            spec = self.grid_spec
            if torch.is_tensor(positions):
                # The grid template comes from the dataset on the CPU; the sampler
                # multiplies it by a rotation built on the positions' device.
                dev = positions.device
                spec = {
                    key: (tuple(v.to(dev) if torch.is_tensor(v) else v for v in val)
                          if isinstance(val, tuple)
                          else (val.to(dev) if torch.is_tensor(val) else val))
                    for key, val in spec.items()
                }
            sample_coords, coord_weights = self.grid_sampling_fn(
                spec, 10_000_000_000, atom_numbers, positions
            )
        else:
            spec = self.grid_spec or self._grid_spec_from_atoms(atom_numbers, positions)
            cpu_spec = {}
            for key, val in spec.items():
                if isinstance(val, tuple):
                    cpu_spec[key] = tuple(
                        v.detach().cpu().numpy() if torch.is_tensor(v) else v
                        for v in val
                    )
                elif torch.is_tensor(val):
                    cpu_spec[key] = val.detach().cpu().numpy()
                else:
                    cpu_spec[key] = val
            sample_coords, coord_weights = get_pyscf_coords(
                cpu_spec, 10_000_000_000, atom_numbers, positions.detach().cpu()
            )
            if self.use_gpu:
                sample_coords = sample_coords.cuda()
                coord_weights = coord_weights.cuda()
        inputs["coords"] = sample_coords
        inputs["coord_weights"] = coord_weights
        if self.remove_atom_density and self.atom_dens is not None:
            positions_for_dens = positions.view(-1, atom_numbers.shape[1], 3).cpu()
            atom_numbers_t = torch.tensor(atom_numbers).type(torch.long)
            sample_coords_cpu = (
                sample_coords.cpu() if sample_coords.is_cuda else sample_coords
            )
            atom_dens, _ = orbitals.sample_atom_density(
                positions=positions_for_dens,
                atom_numbers=atom_numbers_t,
                coords=sample_coords_cpu,
                atom_dens_type=self.atom_dens_type,
                atom_dens_dict=self.atom_dens,
            )
            inputs["atom_density"] = atom_dens.to(positions)
        return inputs

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atom_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)[None, :]
        positions = torch.tensor(
            atoms.get_positions(), dtype=torch.float32, device="cuda" if self.use_gpu else "cpu"
        ).unsqueeze(0)
        inputs = geometry_to_model_inputs(atom_numbers, positions, self.cutoff, self.use_gpu)
        # Analytic dipoles restore the subtracted free-atom density from its
        # coefficients, just as AtomsDensityData and DFTNetworkCalculator do.
        # This head also runs during energy/force calls, without a density grid.
        if self.remove_atom_density and self.atom_dens is not None and self.atom_dens_type in {"mo_coeffs", "df_coeffs"}:
            key = "atom_" + self.atom_dens_type
            inputs[key], inputs[key + "_occ"], inputs[key + "_basis"] = orbitals.join_atom_coeffs(
                torch.as_tensor(atom_numbers, dtype=torch.long), self.atom_dens, self.atom_dens_type
            )
        # Energy/forces do not need a real-space grid: DFTNetwork.forward skips
        # density/core_density when ``coords`` is absent. An intor dipole
        # lives on spherical_coeffs and also does not need a grid. Attach
        # one only when a density expansion is required.
        want_dipole = "dipole" in properties
        if self.need_density_grid:
            inputs = self._add_density_grid(inputs, atom_numbers, positions)
        results = self.model(inputs)
        energy = results["energy"].detach().cpu().numpy().reshape(-1)[0]
        forces = np.asarray(results["forces"].detach().cpu().numpy()).reshape(len(atoms), 3)
        if str(self.energy_unit).startswith("kcal"):
            factor = float(ase.units.kcal / ase.units.mol)
            energy = float(energy) * factor
            forces = forces * factor
        self.results = {"energy": float(energy), "forces": forces}
        if want_dipole:
            dipole = None
            if "dipole_moment" in results:
                dipole = np.asarray(
                    results["dipole_moment"].detach().cpu().numpy(), dtype=float
                ).reshape(-1)[:3]
            else:
                if "coords" not in inputs:
                    inputs = self._add_density_grid(inputs, atom_numbers, positions)
                    results = self.model(inputs)
                if "density" in results and "coords" in inputs:
                    # Quadrature in float64: the electronic moment sums rho*w*(r-c)
                    # over ~1e6 points whose absolute magnitude is thousands of e A
                    # for an 80-atom chain, and float32 GPU reductions are
                    # order-nondeterministic at ~1e-6 relative, i.e. ~1e-3 e A
                    # of call-to-call noise in the dipole (observed 8.6e-4 e A on
                    # the polythiophene 8-mer; 1e-5 on ethanol-sized molecules).
                    packed = {
                        "density": results["density"].to(torch.float64),
                        "coord_weights": inputs["coord_weights"].to(torch.float64),
                        "coords": inputs["coords"].to(torch.float64),
                        "batch_positions": inputs["batch_positions"].to(torch.float64),
                        "batch_atom_numbers": inputs["batch_atom_numbers"],
                    }
                    packed = orbitals.calc_dipole_moment(packed)
                    dipole = packed["dipole_moment"].detach().cpu().numpy().reshape(-1)[:3]
            if dipole is None:
                raise RuntimeError(
                    "DenSNetCalculator cannot form a dipole: "
                    f"dipole_moment={'dipole_moment' in results}, "
                    f"density={'density' in results}, "
                    f"coords={'coords' in inputs}"
                )
            self.results["dipole"] = np.asarray(dipole, dtype=float)


_STALE_DATA_PREFIXES = ("/home/ml-dft/equiv_dens/",)
_PAPER_SAD_PRIOR = "datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy"


def remap_stale_dataset_path(current, repo):
    """Map leftover cluster paths onto this checkout when the file is missing."""
    if current is None:
        return None
    from pathlib import Path

    repo = Path(repo)
    path = Path(current)
    if path.exists():
        return str(path)
    text = str(current)
    for prefix in _STALE_DATA_PREFIXES:
        if text.startswith(prefix):
            local = repo / text[len(prefix):]
            if local.exists():
                return str(local)
    local = repo / "datasets" / path.name
    if local.exists():
        return str(local)
    return None


def peek_energy_state_dict(restart):
    """Read energy-head weights from a paper run dir (best_*.pth or latest)."""
    from pathlib import Path

    restart = Path(restart)
    candidates = [restart / "checkpoints" / "latest_checkpoint.pth"]
    root_bests = sorted(
        restart.glob("best_*.pth"),
        key=lambda path: (path.name != "best_ext.pth", path.name),
    )
    candidates.extend(root_bests)
    ckpt_dir = restart / "checkpoints"
    if ckpt_dir.is_dir():
        candidates.extend(sorted(ckpt_dir.glob("best_*.pth")))
    for path in candidates:
        if not path.is_file():
            continue
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(blob, dict):
            continue
        for key in ("model", "state_dict", "model_state_dict"):
            inner = blob.get(key)
            if isinstance(inner, dict):
                blob = inner
                break
        if any(str(key).startswith("property_models.energy") for key in blob):
            return blob
    return None


def infer_append_atom_density(state):
    """Return True when the energy head was trained with concatenated SAD coeffs."""
    if not state:
        return None
    weight = state.get("property_models.energy.input_layer.0.weight")
    if weight is None:
        return None
    width = int(weight.shape[1])
    if width == 60:
        return True
    if width == 30:
        return False
    return None


def resolve_inference_np_dataset(explicit, args_np_dataset, repo):
    """Choose the npy that defines which elements the orbital basis includes.

    ``load_densnet_calculator`` used to replace a missing caller path with
    ``ethanol_train_10.npy`` even when the args file already named the
    training set. Ethanol is C/H/O; a thiophene checkpoint is C/H/S, so the
    L=0 energy width became 27 instead of 30.
    """
    from pathlib import Path

    repo = Path(repo)
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    if args_np_dataset:
        candidates.append(args_np_dataset)
    candidates.append(repo / "datasets" / "ethanol_train_10.npy")
    for current in candidates:
        remapped = remap_stale_dataset_path(current, repo)
        if remapped is not None:
            return remapped
        path = Path(current)
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "No inference np_dataset found; pass np_dataset= or set --np_dataset "
        "in the DenSNet args file"
    )


def should_enable_inference_density_head(ckpt_state, args):
    """True when an energy-only ext checkpoint still carries dens_model weights.

    Phase-2 / ext runs set ``density_weight=0``, so ``load_model`` used to
    omit ``property_models.density``. The spherical-coefficient network stays
    in ``density_repr_model.1``; rebuilding the (parameter-free) expansion
    head is what lets ``get_dipole_moment`` evaluate ρ on a grid.
    """
    return _checkpoint_has_dens_model(ckpt_state)


def should_enable_inference_intor_head(ckpt_state, args):
    """True when spherical coefficients can form an analytic (intor) dipole.

    Energy-only ext checkpoints zero ``dipole_moment_weight`` and
    ``dpm_intor``. The dens_model that produces those coefficients is still
    in ``density_repr_model.1``. Building ``DipoleMomentIntorCalc`` avoids
    the CPU Becke grid that idled the L40S on Torch gl* nodes.
    """
    return _checkpoint_has_dens_model(ckpt_state)


def _checkpoint_has_dens_model(ckpt_state):
    if not ckpt_state:
        return False
    for key in ckpt_state:
        name = str(key)
        if name.startswith("property_models.density") or name.startswith(
            "density_repr_model.1"
        ):
            return True
    return False


def load_densnet_calculator(
    restart,
    args_file=None,
    np_dataset=None,
    use_gpu=False,
    repo=None,
    pyscf_grid=None,
):
    """Load a paper DenSNet run directory and return an ASE calculator.

    ``use_gpu`` defaults to False so revision CPU jobs do not touch occupied
    node GPUs. ``pyscf_grid`` overrides the args file: energy/force MD should
    pass False so each step does not rebuild a DFT integration grid.
    """
    import tempfile
    from pathlib import Path

    from equiv_dens.data.density_dataset import AtomsDensityData
    from equiv_dens.training.model_loader import load_model
    from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments

    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
    restart = Path(restart)

    # An explicitly supplied args_file wins. It used to be third in this list,
    # behind the run directory's own args.txt, so passing --args-file had no
    # effect whenever the checkpoint shipped one -- which the published models
    # all do. That silently reinstated the 2024 argument set, and the flags that
    # did not exist in 2024 then took their modern defaults.
    candidates = []
    if args_file is not None:
        candidates.append(Path(args_file))
    candidates += [
        restart / "args.txt",
        repo / "config" / "training" / "ethanol_all_001.txt",
        repo / "config" / "md" / "nn" / "ethanol_500ps.txt",
    ]

    def _is_arch_file(path):
        if not path.is_file():
            return False
        text = path.read_text()
        return "--order" in text or "--num_features" in text

    args_path = next((p for p in candidates if _is_arch_file(p)), None)
    if args_path is None:
        args_path = next((p for p in candidates if p.is_file()), None)
    if args_path is None:
        raise FileNotFoundError(f"No DenSNet args file found next to {restart}")

    raw = args_path.read_text()
    drop_prefixes = ("--n_warmup", "--n_runs")
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=None" in stripped:
            continue
        if stripped.startswith(drop_prefixes):
            continue
        # args.txt is written by splitting on whitespace, so "kcal/mol" comes
        # back truncated to "kcal" and no longer matches the allowed choices.
        for flag in ("--energy_unit_out", "--energy_unit_in"):
            if stripped == f"{flag}=kcal":
                stripped = f"{flag}=kcal/mol"
        lines.append(stripped)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        args, _ = parse_command_line_arguments(arg_file=tmp_path)
    finally:
        os.unlink(tmp_path)

    latest = restart / "checkpoints" / "latest_checkpoint.pth"
    ckpt_state = peek_energy_state_dict(restart)
    if latest.is_file():
        blob = torch.load(str(latest), map_location="cpu", weights_only=False)
        ckpt_args = blob.get("args") if isinstance(blob, dict) else None
        if ckpt_args is not None:
            skip = {
                "restart",
                "use_gpu",
                "np_dataset",
                "dens_dataset",
                "np_dataset_valid",
                "dens_dataset_valid",
                "np_dataset_test",
                "dens_dataset_test",
                "atom_dens_path",
                "atom_dens_type",
                "orbitals_file",
                "radial_coeffs_file",
                "pseudo_pot_path",
                # Architecture switches added after these checkpoints were
                # written are absent from their stored args, so copying the
                # stored args wholesale leaves them at today's defaults. They
                # are inferred from the weights below instead.
                "L0_start",
                "append_atom_density",
                "num_en_basis_functions",
            }
            for key, value in vars(ckpt_args).items():
                if key not in skip:
                    setattr(args, key, value)
    args.ignore_unexpected_keywords = True

    # Read the architecture off the weights rather than trusting flags. A
    # checkpoint predating a flag cannot record it, and the default silently
    # builds layers it never trained: L0_start=True adds radial_L0_map, which
    # then stays randomly initialised through a non-strict load. That produced
    # energies that changed from one load of the same checkpoint to the next,
    # and forces whose error exactly equalled the reference force magnitude --
    # the signature of a model predicting nothing at all.
    if ckpt_state is not None:
        energy_keys = [k for k in ckpt_state if k.startswith("property_models.energy")]
        if energy_keys:
            args.L0_start = any("radial_L0_map" in k for k in energy_keys)
            rbf = ckpt_state.get("property_models.energy.radial_basis_functions.logc")
            if rbf is not None:
                args.num_en_basis_functions = int(rbf.shape[0])
        append = infer_append_atom_density(ckpt_state)
        if append is not None:
            args.append_atom_density = append
            if append and getattr(args, "atom_dens_type", "spline") == "spline":
                args.atom_dens_type = "mo_coeffs"
        if should_enable_inference_density_head(ckpt_state, args):
            if float(getattr(args, "density_weight", 0.0) or 0.0) <= 0.0:
                args.density_weight = 1.0
        if should_enable_inference_intor_head(ckpt_state, args):
            args.dpm_intor = True
            if float(getattr(args, "dipole_moment_weight", 0.0) or 0.0) <= 0.0:
                args.dipole_moment_weight = 1.0

    args.restart = str(restart)
    # A finished energy checkpoint already includes the frozen density model.
    # Do not overlay it with a possibly newer external training source.
    args.load_from = None
    args.use_gpu = bool(use_gpu)
    if pyscf_grid is not None:
        args.pyscf_grid = bool(pyscf_grid)
    args.np_dataset = resolve_inference_np_dataset(
        np_dataset, getattr(args, "np_dataset", None), repo
    )
    args.dens_dataset = None
    args.np_dataset_test = args.np_dataset
    args.dens_dataset_test = None
    # Only fill these in when the args do not already name a file that exists.
    # atom_dens_path in particular was overwritten unconditionally with the
    # revision prior, so a published model was evaluated against a different
    # free-atom reference than the one it learned to correct.
    for attr in (
        "orbitals_file",
        "radial_coeffs_file",
        "atom_dens_path",
        "pseudo_pot_path",
        "atomic_energies",
    ):
        remapped = remap_stale_dataset_path(getattr(args, attr, None), repo)
        if remapped is not None:
            setattr(args, attr, remapped)
    atom_dens_fallback = (
        _PAPER_SAD_PRIOR
        if getattr(args, "append_atom_density", False)
        else "datasets/revision/sad_pbe_augccpvdz.npy"
    )
    for attr, rel in (
        ("orbitals_file", "datasets/augccpvqzjkfit_orbital_basis_df.npy"),
        ("radial_coeffs_file", "datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy"),
        ("atom_dens_path", atom_dens_fallback),
        ("pseudo_pot_path", "pseudo_potentials"),
    ):
        current = getattr(args, attr, None)
        if current and Path(current).exists():
            continue
        path = repo / rel
        if path.exists():
            setattr(args, attr, str(path))

    # Ask for forces when the dataset carries them. load_model only rebuilds the
    # VarianceScaling factor -- the training-set force standard deviation the
    # model's outputs are expressed in -- when 'forces' is a required property.
    # Without it the factor defaults to 1 and every energy and force comes back
    # in normalised units, which looks like a model that has stopped responding
    # to geometry rather than like a missing constant.
    required_properties = []
    try:
        probe = np.load(args.np_dataset, allow_pickle=True).item()
        if isinstance(probe, dict) and "forces" in probe:
            required_properties = ["energy", "forces"]
    except Exception:
        pass
    if not required_properties and getattr(args, "output_scaling", False):
        warnings.warn(
            f"{args.np_dataset} carries no forces, so the output scaling cannot be "
            "reconstructed and predictions stay in normalised units. Pass "
            "np_dataset=<the training set this checkpoint was fitted on>.",
            RuntimeWarning,
        )

    dataset = AtomsDensityData(
        np_path=args.np_dataset,
        density_path=args.dens_dataset,
        required_properties=required_properties,
        orbitals_path=args.orbitals_file,
        density_n_samp=getattr(args, "density_subsamples", 10000),
        radial_coeffs_file=args.radial_coeffs_file,
        L0_coeffs_file=getattr(args, "L0_coeffs_file", None),
        pyscf_grid=getattr(args, "pyscf_grid", False),
        atom_dens_path=args.atom_dens_path,
        atom_dens_type=getattr(args, "atom_dens_type", "spline"),
        cutoff=args.cutoff,
        verbose=getattr(args, "verbose", 0),
        timing=False,
        use_gpu=args.use_gpu,
        projected_density=getattr(args, "projected_density", False),
        df_loss_weights=getattr(args, "df_loss_weights", False),
        density_grad=getattr(args, "density_grad", False),
        calc_basis_path=getattr(args, "calc_basis_file", None),
        dtype=args.dtype,
    )
    def _build():
        return load_model(args, dataset, train=False)

    ckpt_width = None
    if ckpt_state is not None:
        w = ckpt_state.get("property_models.energy.input_layer.0.weight")
        if w is not None:
            ckpt_width = int(w.shape[1])

    # append_atom_density doubles the energy head's L=0 input width by
    # concatenating the prior's own coefficients. Which setting a checkpoint
    # used cannot be read off a single tensor, so try loading once and flip
    # append_atom_density on a size mismatch before giving up. For thiophene
    # and ethanethiol the widths are 30 without and 60 with.
    model = None
    for attempt in range(2):
        try:
            model = _build()
            break
        except RuntimeError as exc:
            if attempt == 0 and "size mismatch" in str(exc):
                args.append_atom_density = not getattr(args, "append_atom_density", False)
                continue
            raise

    if ckpt_width is not None and "energy" in getattr(model, "property_models", {}):
        built = int(model.property_models["energy"].input_layer[0].weight.shape[1])
        if built != ckpt_width:
            raise RuntimeError(
                f"energy head expects an L=0 width of {ckpt_width} but the model "
                f"builds {built} with append_atom_density both ways; the prior at "
                f"{args.atom_dens_path} is probably not the one it was trained with"
            )

    # load_model restores the weights itself, but non-strict, so a shape or name
    # disagreement leaves randomly initialised tensors behind and returns
    # quietly. Re-checking here turns that into an error at load time rather
    # than a plausible-looking number in a results file.
    if ckpt_state is not None:
        live = model.state_dict()
        stale = [
            k for k, v in ckpt_state.items()
            if k in live and tuple(live[k].shape) != tuple(v.shape)
        ]
        # The packed representation builds its Clebsch-Gordan tables on the fly (cg_product.py); checkpoints
        # written by the earlier code carry them as buffers (*.clebsch_gordan.cg_matrix / cg_masked), which are
        # not weights and have nothing to match.
        absent = [k for k in ckpt_state if k not in live and ".clebsch_gordan.cg_" not in k]
        if stale or absent:
            raise RuntimeError(
                f"checkpoint does not fit the model built from {args_path}: "
                f"{len(stale)} shape mismatches, {len(absent)} unmatched tensors "
                f"(e.g. {(stale + absent)[:3]})"
            )

    model.eval()
    grid_spec = dataset.grid_spec
    if use_gpu and not getattr(args, "pyscf_grid", False):
        grid_spec = {
            key: (grid_spec[key][0].cuda(), grid_spec[key][1].cuda())
            for key in grid_spec.keys()
        }
    return DenSNetCalculator(
        model,
        cutoff=args.cutoff,
        use_gpu=args.use_gpu,
        energy_unit=getattr(args, "energy_unit_out", "kcal/mol"),
        grid_spec=grid_spec,
        grid_sampling_fn=dataset.sampling_fn,
        pyscf_grid=getattr(args, "pyscf_grid", False),
        need_density_grid=False,
        atom_dens=dataset.atom_dens,
        atom_dens_type=getattr(args, "atom_dens_type", "spline"),
        remove_atom_density=getattr(args, "remove_atom_density", False),
    )
