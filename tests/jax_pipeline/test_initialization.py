"""Native fresh initialization rules, independent of Torch RNG equivalence."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import jax
from equiv_dens_jax.initialization import (
    initialize_parameters, initialization_rules, InitializationRuleConflict,
)
from equiv_dens_jax.serialization import load_tree

jax.config.update("jax_enable_x64",True)
LAYERS = Path(os.environ.get("DENSNET_JAX_LAYER_FIXTURES","scratch/jax_pipeline/layers-production"))
PIPELINE = Path(os.environ.get("DENSNET_JAX_PIPELINE_REFERENCE","scratch/jax_pipeline/pipeline-tiny"))
ENERGY = Path(os.environ.get("DENSNET_JAX_ENERGY_REFERENCE","scratch/jax-pipeline/energy-reference"))


def layer_fixture(name):
    path = LAYERS/f"{name}.json"
    if not path.is_file():
        pytest.fail(f"Generate production layer fixtures before initialization qualification: {path}")
    spec = json.loads(path.read_text())
    assert spec["schema"] == 1
    with np.load(path.with_suffix(".npz")) as arrays:
        state = {group:{key:arrays[f"state/{group}/{key}"].copy() for key in keys}
                 for group,keys in spec["state_keys"].items()}
    return spec["node"],state


def assert_rules(model,template,result):
    rules = initialization_rules(model,template)
    assert set(result["params"]) == set(template["params"]) == set(rules)
    for key,rule in rules.items():
        value = np.asarray(result["params"][key])
        assert value.shape == template["params"][key].shape
        assert value.dtype == template["params"][key].dtype
        assert np.isfinite(value).all()
        if rule[0] == "constant":
            np.testing.assert_array_equal(value,np.full(value.shape,rule[1],dtype=value.dtype))
        elif rule[0] == "uniform":
            assert np.all(value >= rule[1]) and np.all(value <= rule[2])
        else:
            flat = value.reshape(value.shape[0],-1)
            gram = flat @ flat.T if flat.shape[0] <= flat.shape[1] else flat.T @ flat
            np.testing.assert_allclose(gram,np.eye(gram.shape[0]),atol=1e-12,rtol=1e-12)
    for key,value in result["buffers"].items():
        np.testing.assert_array_equal(value,template["buffers"][key])


@pytest.mark.parametrize("name",[
    "Swish","ShiftedSoftplus","ExponentialGaussianRadialBasisFunctions",
    "self_parityTrue","pair_parityFalse","interaction_norm1","residual_norm2",
    "representation_shared_params",
])
def test_source_initialization_rules_and_determinism(name):
    model,template = layer_fixture(name)
    result = initialize_parameters(model,template,seed=184)
    assert_rules(model,template,result)
    # Independent source invariants, rather than trusting the rule inspector.
    if name in ("Swish","ShiftedSoftplus"):
        np.testing.assert_array_equal(result["params"][model["params"]["alpha"]],np.ones(3))
        np.testing.assert_array_equal(result["params"][model["params"]["beta"]],np.full(3,1.702 if name == "Swish" else 1.))
    elif name == "ExponentialGaussianRadialBasisFunctions":
        value = result["params"][model["params"]["_alpha"]]
        np.testing.assert_allclose(jax.nn.softplus(value),0.5,atol=1e-15)
    elif name == "self_parityTrue":
        bounds = {"keepcoeff_0":np.sqrt(3),"keepcoeff_1":np.sqrt(2),"keepcoeff_2":np.sqrt(4.5),
                  "mixcoeff_0_1_1":np.sqrt(2),"mixcoeff_0_2_2":np.sqrt(4.5),
                  "mixcoeff_1_2_1":np.sqrt(2),"mixcoeff_1_2_3":np.sqrt(6)}
        assert set(model["params"]) == set(bounds)
        for local,bound in bounds.items():
            assert np.all(np.abs(result["params"][model["params"][local]]) <= bound)
    elif name == "interaction_norm1":
        # The fourth radial map keeps Linear's original initialization.
        radial = model["children"]["radial_fn"]["children"]["3"]
        value = np.asarray(result["params"][radial["params"]["weight"]])
        assert np.all(np.abs(value) <= 0.5)
        assert not np.allclose(value@value.T,np.eye(3))
    elif name == "residual_norm2":
        second = model["children"]["linear2"]["children"]["linear"]
        for linear in second["children"].values():
            np.testing.assert_array_equal(result["params"][linear["params"]["weight"]],0)
    repeated = initialize_parameters(model,template,seed=184)
    for key,value in result["params"].items():
        np.testing.assert_array_equal(value,repeated["params"][key])
    # This must ignore every original trained/random parameter value.
    corrupted = {"params":{key:np.full(value.shape,np.nan,dtype=value.dtype) for key,value in template["params"].items()},"buffers":template["buffers"]}
    clean = initialize_parameters(model,corrupted,seed=184)
    for key,value in result["params"].items():
        np.testing.assert_array_equal(value,clean["params"][key])
    rules = initialization_rules(model,template)
    if any(rule[0] != "constant" for rule in rules.values()):
        changed = initialize_parameters(model,template,seed=185)
        assert any(not np.array_equal(value,changed["params"][key]) for key,value in result["params"].items())
    if name == "representation_shared_params":
        def shared(index):
            return model["children"]["module"]["children"][str(index)]["children"]["residual_post_x"]["children"]["stack"]["children"]["0"]["children"]["activation_pre"]["params"]["alpha"]
        assert shared(0) == shared(1)
        np.testing.assert_array_equal(result["params"][shared(0)],np.ones(3))
    assert "torch" not in sys.modules
    jax.clear_caches()


def test_complete_pipeline_source_head_overrides():
    if not (PIPELINE/"manifest.json").is_file():
        pytest.fail("Generate complete Torch pipeline architecture template first")
    reference = load_tree(PIPELINE)
    model,template = reference["model"],reference["state"]
    result = initialize_parameters(model,template,seed=50)
    assert_rules(model,template,result)
    def walk(node):
        if node["type"] == "DensityCoeffsNetwork" and node["attrs"].get("pred_radial_coeffs"):
            for group in ("radial_width","radial_scale"):
                for linear in node["children"][group]["children"].values():
                    np.testing.assert_array_equal(result["params"][linear["params"]["weight"]],0)
                    bias = result["params"][linear["params"]["bias"]]
                    if bias.size:
                        assert np.any(np.asarray(bias) != 0), "Source density radial biases remain random"
        for sub in node["children"].values():
            walk(sub)
    walk(model)
    assert "torch" not in sys.modules
    jax.clear_caches()


def test_unsupported_parameter_contract_and_ambiguous_alias_reject():
    node = {"type":"Unknown","path":"unknown","attrs":{},"params":{"weight":"w"},"buffers":{},"children":{}}
    state = {"params":{"w":np.ones((2,2))},"buffers":{}}
    with pytest.raises(ValueError,match="No source initialization contract"):
        initialize_parameters(node,state)
    children = {name:{"type":"Swish","path":name,"attrs":{"initial_alpha":initial},"params":{"alpha":"shared","beta":name},"buffers":{},"children":{}}
                for name,initial in (("a",1.),("b",2.))}
    parent = {"type":"ModuleList","path":"","attrs":{},"params":{},"buffers":{},"children":children}
    state = {"params":{key:np.ones(2) for key in ("shared","a","b")},"buffers":{}}
    with pytest.raises(InitializationRuleConflict,match="Shared parameter"):
        initialize_parameters(parent,state)
    tied = {"type":"Swish","path":"tied","attrs":{},"params":{"alpha":"shared","beta":"shared"},"buffers":{},"children":{}}
    with pytest.raises(InitializationRuleConflict,match="Shared parameter"):
        initialize_parameters(tied,{"params":{"shared":np.ones(2)},"buffers":{}})


def test_lda_optional_energy_offset_initialization():
    node = {"type":"LDAFunctional","path":"functional","attrs":{},
            "params":{"en_offset":"functional.en_offset"},"buffers":{},"children":{}}
    state = {"params":{"functional.en_offset":np.full((1,),np.nan)},"buffers":{}}
    result = initialize_parameters(node,state,seed=50)
    np.testing.assert_array_equal(result["params"]["functional.en_offset"],np.zeros(1))
    assert "torch" not in sys.modules
    jax.clear_caches()


@pytest.mark.parametrize("name",["spherical","embedding","linear","representation"])
def test_other_energy_head_source_initialization(name):
    if not (ENERGY/"manifest.json").is_file():
        pytest.fail("Generate independent Torch energy architecture templates first")
    case = load_tree(ENERGY)["energy"][name]
    result = initialize_parameters(case["node"],case["state"],seed=184)
    assert_rules(case["node"],case["state"],result)
    for local,key in case["node"]["params"].items():
        if local.startswith(("radial_scale_filters_","radial_width_filters_")):
            np.testing.assert_array_equal(result["params"][key],np.ones_like(case["state"]["params"][key]))
    assert "torch" not in sys.modules
    jax.clear_caches()
