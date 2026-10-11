"""Complete native JAX composition of the current DenSnet modules."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .common import attr, child
from .serialization import load_tree


_ENERGY = {"SphericalHarmonicsEnergyNetwork", "SphericalHarmonicsEmbeddingEnergyNetwork",
           "SphericalLinearEnergyNetwork", "RepresentationEnergyNetwork"}
_HEADS = {"DensityCoeffsNetwork", "DFDensityCoeffs", "FreeAtomDensityCoeffs"}
_DISTANCE = {"angstrom_to_angstrom": 1.0, "angstrom_to_bohr": 1 / 0.52917721092,
             "bohr_to_angstrom": 0.52917721092}
_ENERGY_UNITS = {"hartree_to_kcal": 627.5096080305927,
                 "millihartree_to_kcal": 0.6275096080305927,
                 "eV_to_kcal": 23.060541945329334,
                 "kelvin_to_kcal": 0.001987191686485529,
                 "kcal_to_kcal": 1.0, "kcal_to_hartree": 1 / 627.5096080305927,
                 "kcal_to_millihartree": 1 / 0.6275096080305927,
                 "kcal_to_eV": 1 / 23.060541945329334,
                 "kcal_to_kelvin": 1 / 0.001987191686485529}


def _factor(node, name, factors):
    if name in node["children"] and child(node, name)["type"] == "Identity":
        return 1.0
    descriptor = attr(node, name)
    if isinstance(descriptor, dict) and descriptor.get("function") in factors:
        return factors[descriptor["function"]]
    if descriptor is None:
        return 1.0
    raise NotImplementedError(f"Unsupported conversion: {node['path']}.{name}: {descriptor}")


def unit_conversion(node, atoms):
    out = dict(atoms)
    distance = _factor(node, "dist_conversion_func", _DISTANCE)
    energy = _factor(node, "en_conversion_func", _ENERGY_UNITS)
    out["positions"] = out["positions"] * distance
    if "energy" in out:
        out["energy"] = out["energy"] * energy
    if "forces" in out:
        out["forces"] = out["forces"] * energy / distance
    return out


def _arrays(tree):
    return jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree)


def numerical_inputs(atoms):
    """Separate host QC/basis objects from array arguments to compiled code."""
    numerical,host={},{}
    for name,value in atoms.items():
        # PySCF basis tuples use Python integer angular labels. Passing their
        # otherwise numeric leaves through JIT turns those labels into arrays.
        if name=="basis" or name.endswith("_basis"):
            host[name]=value
            continue
        leaves=jax.tree.leaves(value)
        valid=all(isinstance(v,(int,float,bool,np.number)) or
                  (hasattr(v,"dtype") and hasattr(v,"shape") and
                   np.dtype(v.dtype).kind in "biufc") for v in leaves)
        (numerical if valid else host)[name]=value
    return numerical,host


class Pipeline:
    """A static model tree plus array state; layouts are prepared on the host.

    ``forward`` matches DFTNetwork.forward's already-transformed input contract.
    ``predict(convert_inputs=True)`` additionally applies its transform_input.
    Torch is needed only for the separate one-time conversion operation.
    """

    def __init__(self, node, state, *, metadata=None):
        if node["type"] != "DFTNetwork":
            raise ValueError("Pipeline requires a converted DFTNetwork")
        self.node = node
        self.state = _arrays(state)
        self.metadata = metadata or {}
        self.layouts = {}
        self.layout_signature = None
        self._compiled = {}
        self._observable_caches = {}
        self.host_metadata = {}

    @classmethod
    def load(cls, path):
        bundle = load_tree(path)
        return cls(bundle["model"], bundle["state"], metadata=bundle.get("metadata"))

    def prepare(self, atoms):
        """Prepare static species/index metadata outside all differentiated code."""
        from . import density, energy, observables
        host = jax.tree.map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, atoms)
        if "batch_atom_numbers" not in host:
            raise ValueError("Use data.prepare_atoms or supply current compressed-batch metadata")
        numbers = np.asarray(host["batch_atom_numbers"])
        if numbers.ndim!=2 or numbers.dtype.kind not in "iu" or np.any(numbers<0):
            raise ValueError("batch_atom_numbers must be a nonnegative integer [B,N] array")
        def validate_species(node):
            if node["type"]=="Embedding" and np.any(numbers>=attr(node,"Zmax")):
                raise ValueError(f"Atomic number exceeds embedding Zmax={attr(node,'Zmax')}")
            for sub in node["children"].values():
                validate_species(sub)
        validate_species(self.node)
        signature = (numbers.shape, numbers.dtype.str, numbers.tobytes())
        if signature != self.layout_signature:
            layouts = {}
            energy_nodes = []
            def visit(node):
                if node["type"] in _HEADS or node["type"] == "DensityExpansion":
                    layouts[node["path"]] = density.prepare_layout(node, host)
                elif node["type"] in _ENERGY:
                    energy_nodes.append(node)
                elif node["type"] in {"DipoleMomentCalc","DipoleMomentIntorCalc"}:
                    layouts[node["path"]] = observables.prepare_layout(node,host)
                for sub in node["children"].values():
                    visit(sub)
            visit(self.node)
            if energy_nodes:
                post_density = self._apply(child(self.node, "density_repr_model"), self.state,
                                           _arrays(atoms), layouts=layouts, training=False)
                host_post = jax.tree.map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, post_density)
                for node in energy_nodes:
                    layouts[node["path"]] = energy.prepare_layout(node, host_post)
            self.layouts = layouts
            self.layout_signature = signature
            self._compiled.clear()
        numerical,self.host_metadata=numerical_inputs(atoms)
        return _arrays(numerical)

    def _apply(self, node, state, atoms, *, layouts, training):
        kind = node["type"]
        if kind == "Identity":
            return dict(atoms)
        if kind == "Sequential":
            for sub in node["children"].values():
                atoms = self._apply(sub, state, atoms, layouts=layouts, training=training)
            return atoms
        if kind == "EquivariantSphericalHarmonics":
            from .representation import apply_representation
            return apply_representation(node, state, atoms)
        if kind in _HEADS:
            from .density import apply_density_head
            return apply_density_head(node, state, atoms, layout=layouts[node["path"]])
        if kind == "DensityExpansion":
            from .density import apply_density_expansion
            return apply_density_expansion(node, state, atoms, layout=layouts[node["path"]])
        if kind in _ENERGY:
            from .energy import apply_energy
            return apply_energy(node, state, atoms, layout=layouts[node["path"]])
        if kind in {"DipoleMomentCalc", "DipoleMomentIntorCalc"}:
            from .observables import apply_observable
            return apply_observable(node, state, atoms,layout=layouts.get(node["path"]))
        if kind == "LDAFunctional":
            from .functionals import apply_functional
            return apply_functional(node, state, atoms)
        raise NotImplementedError(f"JAX pipeline does not implement {kind} at {node['path']}")

    def resolve_properties(self,properties):
        nodes=child(self.node,"property_models")["children"]
        configured=tuple(attr(self.node,"force_props",()))+tuple(attr(self.node,"no_force_props",()))
        requested=tuple(configured if properties is None else properties)
        if len(set(requested))!=len(requested):
            raise ValueError("Requested properties must be unique")
        fields=set(requested)
        for name in requested:
            if name not in nodes:
                raise KeyError(f"Model has no {name} property")
            if nodes[name]["type"]=="DipoleMomentCalc":
                if "density" not in nodes:
                    raise ValueError("Grid dipoles require the density property")
                fields.add("density")
        if "density" in fields and "core_density" in nodes:
            fields.add("core_density")
        order=tuple(dict.fromkeys((*configured,*nodes)))
        return tuple(name for name in order if name in fields)

    def forward(self, state, atoms, *, training=False, properties=None, forces=None, layouts=None):
        layouts = self.layouts if layouts is None else layouts
        if self.layout_signature is None and layouts is self.layouts:
            raise RuntimeError("Call prepare(atoms) outside JIT before forward")
        density_node = child(self.node, "density_repr_model")
        property_nodes = child(self.node, "property_models")["children"]
        requested = self.resolve_properties(properties)
        calculate_forces = bool(attr(self.node, "calculate_forces", False)) if forces is None else forces
        if calculate_forces and "energy" not in requested:
            raise ValueError("Forces require the energy property")

        def density_forward(inputs):
            return self._apply(density_node, state, dict(inputs), layouts=layouts, training=training)

        out = density_forward(atoms)
        if calculate_forces:
            def total_energy(positions):
                inputs = dict(atoms, positions=positions)
                prediction = density_forward(inputs)
                prediction = self._apply(property_nodes["energy"], state, prediction,
                                         layouts=layouts, training=training)
                return jnp.sum(prediction["energy"])
            # Do not detach density parameters or positions when density weights
            # are frozen. Parameter exclusion belongs to optimizer updates.
            out["forces"] = -jax.grad(total_energy)(atoms["positions"])

        for key in requested:
            out = self._apply(property_nodes[key], state, out, layouts=layouts, training=training)
            if key == "density" and attr(self.node, "remove_atom_density", False):
                out["density"] = out["density"] + out["atom_density"]
                if "density_grad" in out and attr(property_nodes[key], "density_grad", False):
                    out["density_grad"] = out["density_grad"] + out["atom_density_grad"]
        if not training:
            out = self.transform_back_input(out)
        return out

    def transform_input(self, atoms):
        out = unit_conversion(child(self.node, "conversions_in"), atoms)
        scale = attr(child(self.node, "scaling"), "std", 1)
        for key in ("energy", "forces"):
            if key in out:
                out[key] = out[key] / scale
        return out

    def transform_back_input(self, atoms):
        """Apply the same output-unit conversion to predictions and labels."""
        out=dict(atoms)
        scale=attr(child(self.node,"scaling"),"std",1)
        for key in ("energy","forces"):
            if key in out:
                out[key]=out[key]*scale
        return unit_conversion(child(self.node,"conversions_out"),out)

    def prepare_observables(self, atoms, *, properties=None):
        """Prepare current analytic integrals on the host for coefficient-only AD.

        Call again after geometry or parameters change. This does not freeze
        the density network or its position dependence in the learned energy.
        """
        from .observables import prepare_integrals,prepare_free_atom_dipole
        previous_metadata=self.host_metadata
        atoms=self.prepare(atoms)
        # A previous prepare may already have removed these host-only fields.
        self.host_metadata={**previous_metadata,**self.host_metadata}
        nodes=child(self.node,"property_models")["children"]
        requested=tuple(nodes) if properties is None else tuple(properties)
        analytic=[node for name,node in nodes.items() if name in requested and node["type"]=="DipoleMomentIntorCalc"]
        if len(analytic)>1:
            raise ValueError("Multiple analytic dipoles require distinct integral fields")
        if analytic:
            density=self._apply(child(self.node,"density_repr_model"),self.state,atoms,
                                layouts=self.layouts,training=False)
            node=analytic[0]
            atoms=dict(atoms,dipole_integrals=jnp.asarray(prepare_integrals(node,density,layout=self.layouts[node["path"]])))
            if attr(node,"remove_atom_density",False):
                host_atoms={**self.host_metadata,**density}
                atoms["atom_dipole_moment"]=prepare_free_atom_dipole(host_atoms)
        return atoms

    def prepare_functionals(self,atoms,*,properties=None):
        """Refresh detached grid potentials from host objects at current weights.

        Portable precomputed numeric fields are explicit inputs and are left
        alone. This runs outside differentiation/JIT, as does source DFTpy.
        """
        from .functionals import prepare_functional_inputs
        metadata=self.host_metadata
        atoms=self.prepare(atoms)
        self.host_metadata={**metadata,**self.host_metadata}
        nodes=child(self.node,"property_models")["children"]
        requested=tuple(nodes) if properties is None else tuple(properties)
        def contains_functional(node):
            return node["type"]=="LDAFunctional" or any(contains_functional(sub) for sub in node["children"].values())
        targets=[nodes[key] for key in requested if contains_functional(nodes[key])]
        if not targets or "functional_grid" in atoms and "v_real" in atoms:
            return atoms
        if len(targets)>1:
            raise ValueError("Multiple grid functionals require distinct prepared input fields")
        current=self._apply(child(self.node,"density_repr_model"),self.state,atoms,
                            layouts=self.layouts,training=False)
        current={**self.host_metadata,**current}
        def prefix(node,data):
            if node["type"]=="LDAFunctional":
                return prepare_functional_inputs(node,data),True
            if node["type"]=="Sequential":
                for sub in node["children"].values():
                    data,done=prefix(sub,data)
                    if done:
                        return data,True
                return data,False
            return self._apply(node,self.state,data,layouts=self.layouts,training=False),False
        prepared,found=prefix(targets[0],current)
        if not found:
            raise ValueError("Functional property must expose a Sequential prefix")
        return dict(atoms,functional_grid=_arrays(prepared["functional_grid"]),
                    v_real=jnp.asarray(prepared["v_real"]))

    def predict(self, atoms, *, properties=None, forces=None, jit=True, convert_inputs=False):
        atoms = self.prepare(atoms)
        if convert_inputs:
            atoms = self.transform_input(atoms)
        layouts = self.layouts
        nodes=child(self.node,"property_models")["children"]
        requested=self.resolve_properties(properties)
        atoms=self.prepare_functionals(atoms,properties=requested)
        # Explicitly prepared integrals/free-atom terms need no host recompute.
        # Raw inputs use the source-compatible host cache below.
        analytic=[name for name in requested if nodes[name]["type"]=="DipoleMomentIntorCalc"
                  and "dipole_integrals" not in atoms]
        core_properties=tuple(name for name in requested if name not in analytic)
        if forces is None:
            forces=bool(attr(self.node,"calculate_forces",False)) and "energy" in core_properties
        if jit:
            key = (core_properties, bool(forces))
            if key not in self._compiled:
                self._compiled[key] = jax.jit(
                    lambda state, inputs: self.forward(state, inputs, properties=core_properties,
                                                       forces=forces, layouts=layouts))
            function = self._compiled[key]
        else:
            function = lambda state, inputs: self.forward(state, inputs, properties=core_properties,
                                                           forces=forces, layouts=layouts)
        result=function(self.state, atoms)
        if analytic:
            from .observables import apply_host_observable,HostDipoleCache
            result={**self.host_metadata,**result}
            for name in analytic:
                node=nodes[name]
                cache=self._observable_caches.setdefault(node["path"],HostDipoleCache())
                result=apply_host_observable(node,self.state,result,layout=layouts[node["path"]],cache=cache)
        return result
