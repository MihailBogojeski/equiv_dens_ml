import time
import torch
import torch.nn as nn
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from schnetpack.md.calculators import MDCalculator, MDCalculatorError
from schnetpack import properties
import numpy as np
from pyscf import gto, dft


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

        # for key in inputs.keys():
        #     if isinstance(inputs[key], torch.Tensor) or isinstance(inputs[key], np.ndarray):
        #         print(key, 'type', inputs[key].type())
        #     else:
        #         print(key, 'type', type(inputs[key]))

        start_model = time.time()
        results = self.model(inputs)
        # print('avg_distance', torch.mean(results['distances']))
        # print('avg_force', torch.mean(torch.norm(results['forces'], dim=-1)))
        # print('avg_energy', torch.mean(results['energy']))
        # print('avg_distance', torch.mean(results['distances']))
        # print('avg_force', torch.mean(torch.norm(results['forces'], dim=-1)))
        # print('avg_energy', torch.mean(results['energy']))
        # print('Model time:', time.time() - start_model)
        print('density integral', torch.sum(results['density'] * results['coord_weights'], -1))
        start_coeffs = time.time()
        vector_coeffs = orbitals.coeffs_dict_to_vector(results, self.model.density_repr_model[0].orbital_basis,
                                                       results['batch_atom_numbers'])
        # print('Coeffs time:', time.time() - start_coeffs)
        start_other = time.time()
        results['spherical_coeffs'] = vector_coeffs['spherical_coeffs']
        results['radial_width'] = vector_coeffs['radial_width']
        results['radial_scale'] = vector_coeffs['radial_scale']
        results = utils.batch_compressed_atoms(results, ['positions', 'forces'])
        print('forces', results['forces'])
        self.results = {}
        for p in self.required_properties:
            # if p in ['spherical_coeffs', 'radial_width', 'radial_scale']:
            #     self.results[p] = []
            #     for L in range(len(coeffs[p])):
            #         self.results[p].append(coeffs[p][L].detach())
            # elif p not in results:
            if p not in results:
                raise MDCalculatorError(
                    "Requested property {:s} not in " "results".format(p)
                )
            else:
                # Detach properties if requested
                self.results[p] = results[p].detach() if self.detach else results[p]
        # print('system before', system.properties)
        self._update_system(system)
        # print('system after', system.properties)
        # print('Other time:', time.time() - start_other)

        # print('Step time:', time.time() - start)

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
            # print('grid spec', self.grid_spec)
            if self.pyscf_grid:
                sample_coords, coord_weights = get_pyscf_coords(self.grid_spec, 10000000000,
                                                                atom_numbers,
                                                                positions)
            else:
                sample_coords, coord_weights = self.grid_sampling_fn(self.grid_spec, 10000000000,
                                                                     atom_numbers,
                                                                     positions)
            inputs['coords'] = sample_coords
            inputs['coord_weights'] = coord_weights

        inputs['positions'] = positions
        inputs['atom_numbers_first_positions'] = utils.get_atom_num_first_positions(atom_numbers)
        inputs['atom_numbers'] = torch.tensor(atom_numbers).to(positions).type(torch.long)
        inputs['atom_mask'] = inputs['atom_numbers'] > 0

        nl = utils.TorchNeighborList(self.cutoff)
        print(inputs['positions'])
        idx_is, idx_js, _ = nl.get_neighbors(inputs)
        # print('inputs positions shape', inputs['positions'].shape)
        # print('idx_is', idx_is)
        prev_max = 0
        for i in range(len(idx_is)):
            idx_is[i] += prev_max
            idx_js[i] += prev_max
            print('idx_is shape', idx_is[i].shape)
            max_i = torch.max(idx_is[i])
            max_j = torch.max(idx_is[i])
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
