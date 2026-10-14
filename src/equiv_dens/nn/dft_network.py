import torch.nn as nn
import numpy as np
import torch
from torch.utils._pytree import tree_leaves
from equiv_dens.utils.orbitals import CoeffsDict
from equiv_dens.utils.scaling import UnitConversion, VarianceScaling


def exact_unbacked_output_strides():
    """Make inductor return graph outputs whose strides depend on data-dependent (unbacked) sizes with exactly the strides
    of the traced graph. Inductor cannot report such strides to the backward compile, which then assumes the traced
    ones; a different layout chosen by inductor (copy only then) would break the saved tensors of the backward."""
    from torch._inductor import ir
    from torch._inductor.graph import GraphLowering
    from torch._inductor.custom_graph_pass import CustomGraphPass
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
    if getattr(GraphLowering.output, 'exact_unbacked_strides', False):
        return
    output = GraphLowering.output

    def exact_output(self, target, args, kwargs):
        output(self, target, args, kwargs)
        fx_args = self.current_node.args[0]
        fx_args = fx_args if isinstance(fx_args, (tuple, list)) else (fx_args,)
        outs = []
        for r, node in zip(self.graph_outputs, fx_args):
            val = node.meta.get('val') if isinstance(node, torch.fx.Node) else None
            if isinstance(val, torch.Tensor) and isinstance(r, ir.IRNode) and free_unbacked_symbols(val.stride()):
                # also fixes a still flexible layout (no copy then)
                r = ir.ExternKernel.require_exact_strides(r, [t.node.expr if isinstance(t, torch.SymInt) else t
                                                              for t in val.stride()])
            outs.append(r)
        self.graph_outputs = outs

    exact_output.exact_unbacked_strides = True
    GraphLowering.output = exact_output

    class CacheKey(CustomGraphPass):
        """No-op pass whose uuid puts the patched lowering into the inductor cache key."""
        def __call__(self, graph):
            pass

        def uuid(self):
            return 'equiv_dens exact unbacked output strides 1'

    torch._inductor.config.post_grad_custom_post_pass = CacheKey()


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
        # the density coefficient head of the representation model (its packed layout is passed in as data)
        self.coeffs_head = ([m for m in self.density_repr_model.modules() if hasattr(m, 'batch_layout')] + [None])[0]
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

    # data-dependent dims of the compiled-region inputs and their sizes: molecules B, atom columns A, atoms N,
    # neighbor pairs P, packed shells S<L> of every L and coefficients C
    batch_dims = {'positions': {1: 'N'}, 'atom_numbers': {1: 'N'}, 'atom_batch_idx': {1: 'N'}, 'batch_idx_pos': {0: 'N'},
                  'atom_mask': {0: 'BA'}, 'idx_i': {0: 'P'}, 'idx_j': {0: 'P'}, 'neighbor_batch_idx': {0: 'P'},
                  'batch_atom_numbers': {0: 'B', 1: 'A'}, 'batch_atom_mask': {0: 'B', 1: 'A'},
                  'batch_positions': {0: 'B', 1: 'A'}, 'n_electrons': {0: 'B'}, 'idx': {0: 'B'}, '_idx': {0: 'B'}}
    layout_dims = {'atom': {0: 'S'}, 'e': {0: 'S'}, 'width0': {0: 'S'}, 'scale0': {0: 'S'}, 'start': {1: 'A'},
                   'perm': {0: 'C'}, 'inv': {0: 'C'}, 'sign': {0: 'C'}}

    @staticmethod
    def mark(t, dims, suffix=''):
        for d, size in dims.items():
            # the size of this batch is the hint for the size heuristics of inductor (not a guard)
            torch._dynamo.decorators.mark_unbacked(t, d, shape_id=size + suffix, hint_override=max(t.shape[d], 2))

    def mark_sizes(self, inputs):
        """Mark the batch-dependent sizes unbacked: no guards on their values and no 0/1 specialization (batch size 1,
        a single atom, no neighbor pairs, an L without shells), one graph serves every batch. Dims of the same size
        share one symbol."""
        for k, v in inputs.items():
            if k in self.batch_dims:
                self.mark(v, self.batch_dims[k])
        for k, v in inputs.get('coeff_layout', {}).items():
            for L, t in enumerate(v if isinstance(v, list) else [v]):
                self.mark(t, self.layout_dims[k], str(L) if isinstance(v, list) else '')

    def compiled_inputs(self, atoms):
        """Inputs of the compiled region: the model input tensors (no labels, no python values) and the packed layout
        of the batch. Returns (zs, inputs)."""
        inputs = {k: v for k, v in atoms.items()
                  if isinstance(v, torch.Tensor) and k not in self.label_keys and k != 'positions'}
        zs = None
        if self.coeffs_head is not None:
            zs, inputs['coeff_layout'] = self.coeffs_head.batch_layout(atoms)
        self.mark_sizes({'positions': atoms['positions'], **inputs})
        return zs, inputs

    def enable_compile(self):
        """Compile the representation and force heads, the forces are then computed with torch.func.grad."""
        self.compiled = self.calculate_forces
        for key in self.force_props:
            self.property_models[key].calculate_forces = False
        # inductor with data-dependent sizes: the uniform-value constant folding checks contiguity it cannot decide, and
        # cat inputs realized into the cat output are saved for the backward with wrong strides
        torch._inductor.config.joint_graph_constant_folding = False
        torch._inductor.config.force_pointwise_cat = True
        exact_unbacked_output_strides()
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
            zs, inputs = self.compiled_inputs(atoms)
            grad, (energy, (aux, _)) = self._forces(positions, inputs)
            atoms.update(aux)
            atoms['positions'] = positions
            atoms['forces'] = -grad
        else:
            if self.calculate_forces:
                atoms['positions'].requires_grad_(True)
            zs = atoms['batch_atom_numbers'].max(0)[0].tolist()
            atoms = self._core(atoms['positions'], atoms)[1][0]
        if self.coeffs_head is not None:
            # the tensor-only aux dict lost the lazy conversion of the packed coefficients, restore it
            atoms['slot_z'] = zs
            atoms = CoeffsDict(atoms, self.coeffs_head.layout)

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
