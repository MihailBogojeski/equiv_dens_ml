import torch.nn as nn
import numpy as np
import torch
from torch.utils._pytree import tree_leaves
from equiv_dens.utils.orbitals import CoeffsDict
from equiv_dens.utils.scaling import UnitConversion, VarianceScaling


class DFTNetwork(nn.Module):
    """
    Neural network for computing molecular electronic densities and density-derived properties
    """
    # batch entries that are only targets (not passed into the compiled region)
    label_keys = ('coords', 'coord_weights', 'density', 'density_grad', 'atom_density', 'atom_density_grad', 'energy',
                  'forces', 'batch_forces', 'dipole_moment')

    def __init__(self, density_repr_model,
                 property_model_dict,
                 calculate_forces_dict=None,
                 verbose=0,
                 memory=False,
                 conversions_in=UnitConversion(),
                 conversions_out=UnitConversion(),
                 scaling=VarianceScaling(),
                 remove_atom_density=False,):
        super().__init__()
        self.density_repr_model = density_repr_model
        self.property_models = nn.ModuleDict(property_model_dict)
        self.verbose = verbose
        self.conversions_in = conversions_in
        self.conversions_out = conversions_out
        self.memory = memory
        self.scaling = scaling
        if calculate_forces_dict is None:
            calculate_forces_dict = {key: False for key in property_model_dict.keys()}
            self.calculate_forces = False
        else:
            self.calculate_forces = np.any(list(calculate_forces_dict.values()))

        # separate properties that require forces to calculate them first
        self.force_props = [key for key in calculate_forces_dict if calculate_forces_dict[key]]
        self.no_force_props = [key for key in calculate_forces_dict if not calculate_forces_dict[key]]
        print('force props', self.force_props)
        print('no force props', self.no_force_props)
        self.remove_atom_density = remove_atom_density
        self.compiled = False
        # Clebsch-Gordan buffers are not stored in checkpoints anymore, ignore them in old ones
        self._register_load_state_dict_pre_hook(self._drop_cg_keys)
        if self.verbose > 0:
            print('force props', self.force_props)
            print('no force props', self.no_force_props)

    @staticmethod
    def _drop_cg_keys(state_dict, prefix, *args):
        for key in [k for k in state_dict if k.startswith(prefix) and 'clebsch_gordan.' in k]:
            del state_dict[key]

    def transform_input(self, data):
        """
        Convert and scale the input data to improve training.

        Args:
            data: Dictionary containing a collection of atoms and their various properties
        Returns:
            data: Updated dictionary containing the requested adjusted properties
        """
        data = self.conversions_in(data)
        data = self.scaling(data)
        return data

    def transform_density(self, data):
        """
        Transform density by removing the free atom densities if required.

        Args:
            data: Dictionary containing a collection of atoms and their various properties
        Returns:
            data: Updated dictionary containing the original properties
        """
        if 'density' in data.keys() and self.remove_atom_density:
            data['density'] -= data['atom_density']
        return data

    def transform_back_input(self, data):
        """
        Convert and scale the transformed input data back to its original form.

        Args:
            data: Dictionary containing a collection of atoms and their various properties
        Returns:
            data: Updated dictionary containing the original properties
        """
        data = self.scaling.transform_back(data)
        data = self.conversions_out(data)
        return data

    def transform_back_density(self, data):
        """
        Transform the density back to it's original form with the atom densities added back.

        Args:
            data: Dictionary containing a collection of atoms and their various properties
        Returns:
            data: Updated dictionary containing the original properties
        """
        if 'density' in data.keys() and self.remove_atom_density:
            data['density'] += data['atom_density']
        return data


    def _core(self, positions, atoms):
        """Representation and the property models that need forces; returns (energy sum, (atoms, energy sum))."""
        atoms = dict(atoms)
        atoms['positions'] = positions
        atoms = self.density_repr_model(atoms)
        # run the models that require forces first, then turn off gradient for the positions
        for key in self.force_props:
            atoms = self.property_models[key](atoms)
        energy = atoms['energy'].sum() if 'energy' in atoms else positions.sum() * 0
        # func.grad only returns tensors as aux outputs
        aux = {k: v for k, v in atoms.items() if all(isinstance(t, torch.Tensor) for t in tree_leaves(v))}
        return energy, (aux, energy)

    def _forces(self, positions, atoms):
        return torch.func.grad_and_value(self._core, has_aux=True)(positions, atoms)

    def enable_compile(self):
        """Compile the representation and force heads, the forces are then computed with torch.func.grad."""
        self.compiled = self.calculate_forces
        # the density coefficient head inside the compiled region gets the packed layout of each batch as data
        self.coeffs_head = ([m for m in self.density_repr_model.modules() if hasattr(m, 'batch_layout')] + [None])[0]
        for key in self.force_props:
            self.property_models[key].calculate_forces = False
        # sizes that happen to be equal in one batch (atoms, slots, pairs, ...) must not share a symbol
        torch.fx.experimental._config.use_duck_shape = False
        self._forces = torch.compile(self._forces, dynamic=True)

    def forward(self, data):
        """
        Compute the electron density and needed density-derived properties.

        inputs:
            atoms: Dictionary containing a collection of atoms and their various properties
        outputs:
            atoms: Updated dictionary containing the requested predicted properties
        """
        if self.memory:
            print('dft network forward start:')
            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
            print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        atoms = {}
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                atoms[key] = data[key].clone()
            else:
                atoms[key] = data[key]

        if self.compiled:
            # forces from torch.func.grad inside the compiled region, since compiled autograd.grad has no double backward.
            # Only the model inputs go in: labels and python values would add guards and recompiles.
            positions = atoms['positions']
            inputs = {k: v for k, v in atoms.items()
                      if isinstance(v, torch.Tensor) and k not in self.label_keys and k != 'positions'}
            if self.coeffs_head is not None:
                zs, inputs['coeff_layout'] = self.coeffs_head.batch_layout(atoms)
            grad, (energy, (aux, _)) = self._forces(positions, inputs)
            atoms.update(aux)
            atoms['positions'] = positions
            if self.coeffs_head is not None:
                atoms['slot_z'] = zs
                atoms = CoeffsDict(atoms, self.coeffs_head.layout)
            atoms['forces'] = -grad
        else:
            if self.calculate_forces:
                atoms['positions'].requires_grad_(True)
            zs = atoms['batch_atom_numbers'].max(0)[0].tolist()
            atoms = self._core(atoms['positions'], atoms)[1][0]
            # the tensor-only aux dict lost the lazy conversion of the packed coefficients, restore it
            atoms['slot_z'] = zs
            atoms = CoeffsDict(atoms, [m.layout for m in self.modules() if hasattr(m, 'warm_caches')][0])

        atoms['positions'].requires_grad_(False)
        if self.training:
            for key in self.no_force_props:
                atoms = self.property_models[key](atoms)
                if key == 'density' and self.remove_atom_density:
                    atoms['density'] += atoms['atom_density']
                    if self.property_models['density'].density_grad:
                        atoms['density_grad'] += atoms['atom_density_grad']
                if self.memory:
                    print('dft network forward', key, ':')
                    print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
                    print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        else:
            with torch.no_grad():
                for key in self.no_force_props:
                    atoms = self.property_models[key](atoms)
                    if key == 'density' and self.remove_atom_density:
                        atoms['density'] += atoms['atom_density']
                        if self.property_models['density'].density_grad:
                            atoms['density_grad'] += atoms['atom_density_grad']
                    if self.memory:
                        print('dft network forward', key, ':')
                        print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
                        print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        if not self.training:
            atoms = self.scaling.transform_back(atoms)
            atoms = self.conversions_out(atoms)

        return atoms
