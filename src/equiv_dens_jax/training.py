"""Functional staged/joint training and portable optimizer checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp

from .serialization import save_tree, load_tree


@dataclass(frozen=True)
class Optimizer:
    name: str = "sgd"
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    energy_weight_decay: float | None = None
    selective_weight_decay: bool = True
    momentum: float = 0.0
    clip_norm: float = 0.0
    ema_decay: float | None = 0.999
    ema_start_epoch: int = 0
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    offset_learning_rate_factor: float = 100.0

    def __post_init__(self):
        if self.name not in {"adam", "adamw", "amsgrad", "sgd"}:
            raise ValueError(f"Unsupported optimizer: {self.name}")
        if self.learning_rate <= 0 or self.epsilon <= 0:
            raise ValueError("learning_rate and epsilon must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must lie in [0,1)")
        if self.ema_decay is not None and not 0 <= self.ema_decay < 1:
            raise ValueError("EMA decay must lie in [0,1)")
        if self.lookahead_k < 0 or not 0 <= self.lookahead_alpha <= 1:
            raise ValueError("Invalid Lookahead configuration")


def active_parameters(params, stage="joint",model=None):
    if stage not in {"density", "energy", "joint"}:
        raise ValueError("stage must be density, energy or joint")
    frozen=set()
    if model is not None:
        def visit(node):
            for local,trainable in node.get("trainable",{}).items():
                if not trainable:
                    frozen.add(node["params"][local])
            for sub in node["children"].values():
                visit(sub)
        visit(model)
        core=model["children"]["property_models"]["children"].get("core_density")
        if core and core.get("attrs",{}).get("core_basis_ratio",0)>0:
            frozen.update(name for name in params if name.startswith("density_repr_model."))
    prefixes = ("density_repr_model.", "property_models.core_density.") if stage == "density" else ("property_models.energy.",)
    selected = tuple(name for name in params if name not in frozen and (stage=="joint" or name.startswith(prefixes)))
    if not selected:
        raise ValueError(f"No parameters selected for {stage} stage")
    return selected


def initialize(pipeline, *, stage="joint", optimizer=Optimizer(), seed=0):
    names = active_parameters(pipeline.state["params"],stage,pipeline.node)
    zeros = {name:jnp.zeros_like(pipeline.state["params"][name]) for name in names}
    params = dict(pipeline.state["params"])
    return {"step": jnp.asarray(0,dtype=jnp.int32), "params":params,
            "optimizer":{"count":jnp.asarray(0,dtype=jnp.int32),"m":zeros,"v":zeros,
                         "vmax":zeros,"slow":{name:params[name] for name in names},
                         "param_count":{name:jnp.asarray(0,dtype=jnp.int32) for name in names}},
            "rng":jax.random.PRNGKey(seed), "ema":dict(params)}


def _update(parameters, gradients, state, config, learning_rate=None):
    if config.clip_norm > 0:
        norm = jnp.sqrt(sum(jnp.sum(g*g) for g in gradients.values()))
        factor = jnp.minimum(1., config.clip_norm/(norm+1e-6))
        gradients = {name:g*factor for name,g in gradients.items()}
    def decay(name):
        if config.selective_weight_decay and not (
            "weight" in name and "radial_fn" not in name and "embedding" not in name):
            return 0.
        if "energy" in name and config.energy_weight_decay not in (None,0):
            return config.energy_weight_decay
        return config.weight_decay
    if config.name in {"adam", "amsgrad", "sgd"}:
        gradients = {name:g+decay(name)*parameters[name] for name,g in gradients.items()}
    count = state["count"]+1
    # Torch leaves all state of structurally unused parameters untouched.
    first,second,maximum_second,slow=(dict(state[key]) for key in ("m","v","vmax","slow"))
    parameter_counts=dict(state.get("param_count",{name:state["count"] for name in state["m"]}))
    updated={}
    for name,gradient in gradients.items():
        parameter_count=parameter_counts[name]+1
        parameter_counts[name]=parameter_count
        if config.name == "sgd":
            # Source SGD initializes its momentum buffer with the first gradient.
            first[name] = jnp.where(parameter_count==1, gradient,
                                   config.momentum*state["m"][name]+gradient)
            second[name] = state["v"][name]
            maximum_second[name] = state["vmax"][name]
            direction = first[name] if config.momentum else gradient
        else:
            first[name] = config.beta1*state["m"][name]+(1-config.beta1)*gradient
            second[name] = config.beta2*state["v"][name]+(1-config.beta2)*gradient**2
            maximum_second[name] = jnp.maximum(state["vmax"][name],second[name])
            mean = first[name]/(1-config.beta1**parameter_count)
            variance = (maximum_second[name] if config.name=="amsgrad" else second[name])/(1-config.beta2**parameter_count)
            direction = mean/(jnp.sqrt(variance)+config.epsilon)
        base = parameters[name]
        rate = (config.learning_rate if learning_rate is None else learning_rate)*(config.offset_learning_rate_factor if name=="en_offset" else 1.)
        if config.name == "adamw":
            base = base*(1-rate*decay(name))
        updated[name] = base-rate*direction
    if config.lookahead_k:
        # Source Lookahead synchronizes the whole optimizer group, including
        # a previously trained parameter absent from this batch's gradient.
        for name,parameter in parameters.items():
            fast=updated.get(name,parameter)
            # Legacy wrapper initializes slow weights *after* the first fast
            # update, and updates at counters 0,k,2k rather than k-1,2k-1.
            previous = jnp.where(state["count"]==0,fast,state["slow"][name])
            do_slow = state["count"] % config.lookahead_k == 0
            slow[name] = jnp.where(do_slow,previous+config.lookahead_alpha*(fast-previous),previous)
            fast = jnp.where(do_slow,slow[name],fast)
            updated[name] = fast
    return updated,{"count":count,"m":first,"v":second,"vmax":maximum_second,"slow":slow,
                    "param_count":parameter_counts}


def required_properties(loss, pipeline):
    fields = {field for field,weight in loss.loss_weights.items() if weight>0}
    if "forces" in fields:
        fields.add("energy")
    if "density_grad" in fields:
        fields.add("density")
    if "df_coeffs" in fields:
        fields.remove("df_coeffs")
    nodes = pipeline.node["children"]["property_models"]["children"]
    if "core_density" in nodes and "density" in fields:
        fields.add("core_density")
    return pipeline.resolve_properties(tuple(name for name in nodes if name in fields)), "forces" in fields


def make_step(pipeline, loss, *, stage="joint", optimizer=Optimizer(), jit=True):
    """Build one complete update; prepare the pipeline's layout before calling."""
    if pipeline.layout_signature is None:
        raise RuntimeError("Prepare a batch before constructing a training step")
    names = active_parameters(pipeline.state["params"],stage,pipeline.node)
    properties, forces = required_properties(loss,pipeline)
    layouts = pipeline.layouts
    used_names=None

    def step(train_state, data):
        nonlocal used_names
        rng,next_key = jax.random.split(train_state["rng"])
        active = {name:train_state["params"][name] for name in names}
        def objective(parameters,inputs):
            params = dict(train_state["params"])
            params.update(parameters)
            state = {"params":params,"buffers":pipeline.state["buffers"]}
            prediction = pipeline.forward(state,inputs,training=True,properties=properties,
                                          forces=forces,layouts=layouts)
            errors = loss.compute(prediction,inputs,key=next_key,
                                  loss_weights_override=inputs.get("training_loss_weights"))
            return errors["loss"],errors
        if used_names is None:
            from .parameter_usage import used_parameters
            used_names=used_parameters(lambda params,inputs:objective(params,inputs)[0],active,data)
        active={name:active[name] for name in used_names}
        (_,errors),gradients = jax.value_and_grad(lambda params:objective(params,data),has_aux=True)(active)
        group_parameters={name:train_state["params"][name] for name in names}
        updated,opt_state = _update(group_parameters,gradients,train_state["optimizer"],optimizer,
                                   data.get("training_learning_rate"))
        parameters = dict(train_state["params"])
        parameters.update(updated)
        ema = dict(train_state["ema"])
        for name in names:
            decay = 0. if optimizer.ema_decay is None else optimizer.ema_decay
            epoch = data.get("training_epoch",jnp.asarray(0))
            alpha = jnp.where(epoch>optimizer.ema_start_epoch,1-decay,1.)
            ema[name] = ema[name]-alpha*(ema[name]-parameters[name])
        return {"step":train_state["step"]+1,"params":parameters,"optimizer":opt_state,
                "rng":rng,"ema":ema},errors
    return jax.jit(step) if jit else step


def save_checkpoint(path, pipeline, train_state, *, stage="joint", optimizer=Optimizer(), loss=None, metadata=None):
    save_tree(path,{"model":pipeline.node,"state":{"params":train_state["params"],"buffers":pipeline.state["buffers"]},
                    "training":train_state,"stage":stage,"optimizer_config":vars(optimizer),
                    "loss_config":None if loss is None else loss.configuration(),
                    "metadata":{**pipeline.metadata,**(metadata or {})}})


def load_checkpoint(path):
    from .pipeline import Pipeline
    from .losses import ErrorDict
    tree = load_tree(path)
    pipeline = Pipeline(tree["model"],tree["state"],metadata=tree.get("metadata"))
    training = jax.tree.map(lambda x:jnp.asarray(x) if hasattr(x,"dtype") else x,tree["training"])
    optimizer = Optimizer(**tree["optimizer_config"])
    loss = ErrorDict(**tree["loss_config"]) if tree.get("loss_config") is not None else None
    return pipeline,training,tree["stage"],optimizer,loss
