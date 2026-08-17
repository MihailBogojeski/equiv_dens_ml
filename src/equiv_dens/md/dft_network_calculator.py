import equiv_dens.compat  # noqa: F401 - apply T_co patch before schnetpack import

import time
import torch
import torch.nn as nn
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from schnetpack.md.calculators import MDCalculator, MDCalculatorError
from schnetpack import properties
import numpy as np
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
        enable_tf32=True,
        enable_inference_mode=True,
        cache_grid=True,
        grid_cache_threshold=0.1,
        use_fast_inference=False,
        compile_model=False,
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
        
        # Apply TF32 optimization if enabled
        if self.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if self.verbose > 0:
                print("TF32 enabled for faster matrix operations")
        
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
        positions = positions.view(-1, natoms, 3)
        atom_types = atom_types.view(-1, natoms)
        center = torch.sum(positions * atom_types.unsqueeze(-1), 1) / torch.sum(atom_types, 1).unsqueeze(-1)
        # inputs = {'positions': positions + 10,
        props = {'positions': (positions - center.unsqueeze(1)).cpu().numpy()}
        atom_numbers, props = utils.compress_batch_atoms(atom_types.cpu().numpy(), props)
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
        inputs['atom_numbers_first_positions'] = utils.get_atom_num_first_positions(atom_numbers)
        inputs['atom_numbers'] = torch.tensor(atom_numbers).to(positions).type(torch.long)
        inputs['atom_mask'] = inputs['atom_numbers'] > 0

        nl = utils.TorchNeighborList(self.cutoff)
        idx_is, idx_js, _ = nl.get_neighbors(inputs)
        prev_max = 0
        for i in range(len(idx_is)):
            idx_is[i] += prev_max
            idx_js[i] += prev_max
            # Handle empty neighbor lists (e.g. single-atom or edge-case configs)
            max_i = torch.max(idx_is[i]).item() if idx_is[i].numel() > 0 else prev_max
            max_j = torch.max(idx_js[i]).item() if idx_js[i].numel() > 0 else prev_max
            prev_max = max(max_i, max_j) + 1

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
        batch_nz = inputs['atom_mask'].to(inputs['positions'])
        batch_idx_pos = batch_nz * torch.arange(len(batch_nz)).to(batch_nz)
        inputs['batch_idx_pos'] = batch_idx_pos[inputs['atom_mask']].to(torch.long)
        inputs['atom_numbers'] = inputs['atom_numbers'][inputs['atom_mask']].view(1, -1)
        inputs['atom_batch_idx'] = atom_batch_idx.flatten()
        inputs['atom_batch_idx'] = inputs['atom_batch_idx'][inputs['atom_mask']].view(1, -1)
        inputs['positions'] = inputs['positions'][:, inputs['atom_mask']]

        return inputs


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
