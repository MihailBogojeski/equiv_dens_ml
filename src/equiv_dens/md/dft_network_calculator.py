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


def ase_atoms_to_model_inputs(atoms, cutoff, use_gpu=False):
    """Build DenSNet energy/force inputs from an ASE Atoms object."""
    atom_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)[None, :]
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32).unsqueeze(0)
    if use_gpu:
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


class DenSNetCalculator(Calculator):
    """ASE energy+force calculator wrapping a loaded DenSNet model.

    Model energies are interpreted as ``energy_unit`` (paper default kcal/mol)
    and converted to eV / eV/Å for ASE.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model,
        cutoff=7.937658158457616,
        use_gpu=False,
        energy_unit="kcal/mol",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_gpu = bool(use_gpu and torch.cuda.is_available())
        self.model = model.cuda() if self.use_gpu else model.cpu()
        self.model.eval()
        self.cutoff = float(cutoff)
        self.energy_unit = energy_unit

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        inputs = ase_atoms_to_model_inputs(atoms, self.cutoff, self.use_gpu)
        results = self.model(inputs)
        energy = results["energy"].detach().cpu().numpy().reshape(-1)[0]
        forces = np.asarray(results["forces"].detach().cpu().numpy()).reshape(len(atoms), 3)
        if str(self.energy_unit).startswith("kcal"):
            factor = float(ase.units.kcal / ase.units.mol)
            energy = float(energy) * factor
            forces = forces * factor
        self.results = {"energy": float(energy), "forces": forces}


def load_densnet_calculator(
    restart,
    args_file=None,
    np_dataset=None,
    use_gpu=False,
    repo=None,
):
    """Load a paper DenSNet run directory and return an ASE calculator.

    ``use_gpu`` defaults to False so revision CPU jobs do not touch occupied
    node GPUs.
    """
    import tempfile
    from pathlib import Path

    from equiv_dens.data.density_dataset import AtomsDensityData
    from equiv_dens.training.model_loader import load_model
    from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments

    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
    restart = Path(restart)
    if np_dataset is None:
        np_dataset = repo / "datasets" / "ethanol_train_10.npy"

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
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=None" in stripped:
            continue
        lines.append(stripped)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        args, _ = parse_command_line_arguments(arg_file=tmp_path)
    finally:
        os.unlink(tmp_path)

    latest = restart / "checkpoints" / "latest_checkpoint.pth"
    ckpt_state = None
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
        if isinstance(blob, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if isinstance(blob.get(key), dict):
                    ckpt_state = blob[key]
                    break
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

    args.restart = str(restart)
    args.use_gpu = bool(use_gpu)
    args.np_dataset = str(np_dataset)
    args.dens_dataset = None
    args.np_dataset_test = str(np_dataset)
    args.dens_dataset_test = None
    # Only fill these in when the args do not already name a file that exists.
    # atom_dens_path in particular was overwritten unconditionally with the
    # revision prior, so a published model was evaluated against a different
    # free-atom reference than the one it learned to correct.
    for attr, rel in (
        ("orbitals_file", "datasets/augccpvqzjkfit_orbital_basis_df.npy"),
        ("radial_coeffs_file", "datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy"),
        ("atom_dens_path", "datasets/revision/sad_pbe_augccpvdz.npy"),
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

    model = _build()

    # append_atom_density doubles the energy head's L=0 input width by
    # concatenating the prior's own coefficients. Which setting a checkpoint
    # used cannot be read off a single tensor, so build, compare against the
    # checkpoint, and flip once if it disagrees. For the thiophene model the
    # widths are 30 without and 60 with.
    ckpt_width = None
    if ckpt_state is not None:
        w = ckpt_state.get("property_models.energy.input_layer.0.weight")
        if w is not None:
            ckpt_width = int(w.shape[1])
    if ckpt_width is not None and "energy" in getattr(model, "property_models", {}):
        built = int(model.property_models["energy"].input_layer[0].weight.shape[1])
        if built != ckpt_width:
            args.append_atom_density = not getattr(args, "append_atom_density", False)
            model = _build()
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
        absent = [k for k in ckpt_state if k not in live]
        if stale or absent:
            raise RuntimeError(
                f"checkpoint does not fit the model built from {args_path}: "
                f"{len(stale)} shape mismatches, {len(absent)} unmatched tensors "
                f"(e.g. {(stale + absent)[:3]})"
            )

    model.eval()
    return DenSNetCalculator(
        model,
        cutoff=args.cutoff,
        use_gpu=args.use_gpu,
        energy_unit=getattr(args, "energy_unit_out", "kcal/mol"),
    )
