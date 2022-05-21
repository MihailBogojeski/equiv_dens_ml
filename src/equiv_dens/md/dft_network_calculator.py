import time
import torch
import equiv_dens.utils.base as utils
import equiv_dens.utils.orbitals as orbitals
from schnetpack.md.calculators import MDCalculator, MDCalculatorError


# MD calculator class
class DFTNetworkCalculator(MDCalculator):
    def __init__(self,
                 model,
                 required_properties,
                 force_handle,
                 atoms_data=None,
                 verbose=0,
                 n_jobs=10,
                 density_expansion=False,
                 position_conversion="A",
                 force_conversion="kcal/mol/A",
                 grid_spec=None,
                 grid_sampling_fn=None,
                 property_conversion={},
                 use_gpu=False,
                 detach=True):
        # energy prediction model
        super().__init__(
            required_properties,
            force_handle,
            detach=detach,
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            property_conversion=property_conversion,
        )
        self.model = model
        # density prediction model
        self.grid_sampling_fn = grid_sampling_fn
        self.verbose = verbose
        self.n_jobs = n_jobs
        print('calculator grid spec', grid_spec)
        self.grid_spec = {}
        if use_gpu:
            for key in grid_spec.keys():
                self.grid_spec[key] = (grid_spec[key][0].cuda(),
                                       grid_spec[key][1].cuda())
        else:
            self.grid_spec = grid_spec
        self.density_expansion = density_expansion

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
        results = self.model(inputs)
        # print('density integral', torch.sum(results['density'] * results['coord_weights'], -1))
        vector_coeffs = orbitals.coeffs_dict_to_vector(results, self.model.density_repr_model[0].orbital_basis,
                                                                          results['atom_numbers'])
        results['spherical_coeffs'] = vector_coeffs['spherical_coeffs']
        results['radial_width'] = vector_coeffs['radial_width']
        results['radial_scale'] = vector_coeffs['radial_scale']
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
                self.results[p] = results[p].detach()
        # print('system before', system.properties)
        self._update_system(system)
        # print('system after', system.properties)

        print('Step time:', time.time() - start)

    def _generate_input(self, system):
        """
        Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
        formatted input for the schnetpack model.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            dict(torch.Tensor): Schnetpack inputs in dictionary format.
        """
        positions, atom_types, atom_masks, cells, pbc = self._get_system_molecules(
            system
        )
        center = torch.sum(positions * system.masses, 2) / torch.sum(system.masses, 2)
        # inputs = {'positions': positions + 10,
        inputs = {'positions': positions - center.permute(1, 0, 2),
                  'atom_numbers': atom_types,
                  'atom_mask': atom_types != 0,
                  }
        if self.density_expansion:
            # print('grid spec', self.grid_spec)
            sample_coords, coord_weights = self.grid_sampling_fn(self.grid_spec, 10000000000,
                                                                 utils.numbers_to_symbols(atom_types[0].squeeze().detach().cpu().numpy()),
                                                                 inputs['positions'])
            inputs['coords'] = sample_coords
            inputs['coord_weights'] = coord_weights

        return inputs
