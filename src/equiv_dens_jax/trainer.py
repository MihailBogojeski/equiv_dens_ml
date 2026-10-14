"""Native host training loop: shuffling, validation, EMA and plateau scheduling.

The loop and NumPy RNG state live in the same portable checkpoint as optimizer
state. JAX compilation is cached by static batch layout. Stage changes reset
optimizer moments explicitly, retaining the learned model weights.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from .training import initialize,make_step,required_properties,save_checkpoint
from .common import attr,child


class Trainer:
    def __init__(self,pipeline,loss,optimizer,*,stage="joint",seed=0,jit=True,
                 state=None,convert_inputs=False,scheduler=None,
                 grid_scaling_start=10000,grid_scaling_annealing=1.0,jit_validation=False):
        self.pipeline=pipeline
        self.loss=loss
        self.optimizer=optimizer
        self.stage=stage
        self.seed=seed
        self.jit=jit
        self.jit_validation=jit_validation
        self.state=dict(state) if state is not None else initialize(pipeline,stage=stage,optimizer=optimizer,seed=seed)
        self.convert_inputs=convert_inputs
        self.scheduler={"factor":.5,"patience":10,"threshold":1e-4,"min_lr":0.,"eps":1e-8,**(scheduler or {})}
        self.scheduler_enabled=scheduler is not None
        if not 0<self.scheduler["factor"]<1 or self.scheduler["patience"]<0:
            raise ValueError("Invalid plateau scheduler configuration")
        self.loop=self.state.pop("loop",{"epoch":0,"cursor":0,"order":[],"batch_count":0,
             "best":math.inf,"bad_epochs":0,"learning_rate":optimizer.learning_rate,
             "rng":np.random.default_rng(seed).bit_generator.state})
        self.loop=jax.tree.map(lambda x:np.asarray(x) if hasattr(x,"dtype") else x,self.loop)
        self.loop.setdefault("loss_weights",dict(loss.loss_weights))
        self.loop.setdefault("validation_loss_weights",dict(self.loop["loss_weights"]))
        self.grid_scaling_start=grid_scaling_start
        self.grid_scaling_annealing=grid_scaling_annealing
        if not np.isfinite(grid_scaling_annealing) or grid_scaling_annealing<0:
            raise ValueError("grid_scaling_annealing must be finite and nonnegative")
        density=child(pipeline.node,"property_models")["children"].get("density")
        self.grid_node=density if density and density["type"]=="DensityExpansion" else None
        if self.grid_node is not None:
            name=self.grid_node["buffers"].get("grid_scaling_factor")
            initial=pipeline.state["buffers"][name] if name else attr(self.grid_node,"grid_scaling_factor",0)
            self.loop.setdefault("grid_scaling_factor",float(np.asarray(initial).reshape(-1)[0]))
        self.rng=np.random.default_rng()
        self.rng.bit_generator.state=self.loop["rng"]
        self.updates={}
        self.validations={}

    def _grid_inputs(self,data):
        if self.grid_node is None:
            return data
        return dict(data,density_grid_scaling_factor=jnp.asarray(self.loop["grid_scaling_factor"]))

    def _prepare(self,batch):
        self.pipeline.state={"params":self.state["params"],"buffers":self.pipeline.state["buffers"]}
        data=self.pipeline.prepare(batch)
        if self.convert_inputs:
            data=self.pipeline.transform_input(data)
        data=self._grid_inputs(data)
        properties,_=required_properties(self.loss,self.pipeline)
        data=self.pipeline.prepare_functionals(data,properties=properties)
        data=self.pipeline.prepare_observables(data,properties=properties)
        return dict(data,training_epoch=jnp.asarray(self.loop["epoch"]),
                    training_learning_rate=jnp.asarray(self.loop["learning_rate"]),
                    training_loss_weights={key:jnp.asarray(value) for key,value in self.loop["loss_weights"].items()})

    def step(self,batches,*,shuffle=True):
        count=len(batches)
        if not count:
            raise ValueError("No training batches")
        if self.loop["batch_count"] not in (0,count):
            raise ValueError("Changing batch count on resume would change the saved data order")
        if not len(self.loop["order"]):
            self.loop["order"]=self.rng.permutation(count) if shuffle else np.arange(count)
            self.loop["batch_count"]=count
        index=int(self.loop["order"][int(self.loop["cursor"])])
        data=self._prepare(batches[index])
        signature=self.pipeline.layout_signature
        if signature not in self.updates:
            self.updates[signature]=make_step(self.pipeline,self.loss,stage=self.stage,optimizer=self.optimizer,jit=self.jit)
        self.state,metrics=self.updates[signature](self.state,data)
        # Loss values can be ready before independent optimizer outputs.
        # Completed-step timing and checkpoint boundaries include both.
        jax.block_until_ready((self.state,metrics))
        host={key:float(value) for key,value in metrics.items()}
        if not all(np.isfinite(value) for value in host.values()):
            raise FloatingPointError("Nonfinite training loss or metrics")
        if self.grid_node is not None and self.loop["epoch"]>self.grid_scaling_start:
            self.loop["grid_scaling_factor"]*=self.grid_scaling_annealing
            name=self.grid_node["buffers"].get("grid_scaling_factor")
            if name:
                buffers=dict(self.pipeline.state["buffers"])
                buffers[name]=jnp.full_like(buffers[name],self.loop["grid_scaling_factor"])
                self.pipeline.state={**self.pipeline.state,"buffers":buffers}
        self.loop["validation_loss_weights"]=dict(self.loop["loss_weights"])
        # Preserve per-update decay, floor crossing, and validation-before-decay.
        for field,weight in self.loop["loss_weights"].items():
            if weight>self.loss.weights_min[field]:
                self.loop["loss_weights"][field]=weight*self.loss.weights_decay[field]
        self.loop["cursor"]=int(self.loop["cursor"])+1
        completed_epoch=self.loop["cursor"]==count
        if completed_epoch:
            self.loop["cursor"]=0
            self.loop["epoch"]=int(self.loop["epoch"])+1
            self.loop["order"]=[]
        self.loop["rng"]=self.rng.bit_generator.state
        return host,completed_epoch

    def validate(self,batches,*,ema=True,exclude_energy_min=True):
        properties,forces=required_properties(self.loss,self.pipeline)
        original=self.pipeline.state
        values=[]
        try:
            parameters=self.state["ema"] if ema and self.optimizer.ema_decay is not None else self.state["params"]
            self.pipeline.state={"params":parameters,"buffers":original["buffers"]}
            for batch in batches:
                data=self.pipeline.prepare(batch)
                if self.convert_inputs:
                    data=self.pipeline.transform_input(data)
                data=self._grid_inputs(data)
                data=self.pipeline.prepare_functionals(data,properties=properties)
                data=self.pipeline.prepare_observables(data,properties=properties)
                weights=self.loop["validation_loss_weights"]
                exclude=exclude_energy_min and weights.get("energy_min",0)!=sum(weights.values())
                # Deterministic held-out pair sampling; training uses saved RNG.
                if self.jit_validation:
                    signature=(self.pipeline.layout_signature,exclude)
                    if signature not in self.validations:
                        layouts=self.pipeline.layouts
                        def evaluate(state,inputs,effective_weights,*,_layouts=layouts,_exclude=exclude):
                            prediction=self.pipeline.forward(state,inputs,training=False,
                                properties=properties,forces=forces,layouts=_layouts)
                            labels=self.pipeline.transform_back_input(inputs)
                            return self.loss.compute(prediction,labels,key=jax.random.PRNGKey(self.seed),
                                exclude_energy_min=_exclude,loss_weights_override=effective_weights)
                        self.validations[signature]=jax.jit(evaluate)
                    errors=self.validations[signature](self.pipeline.state,data,
                        {name:jnp.asarray(value) for name,value in weights.items()})
                else:
                    prediction=self.pipeline.forward(self.pipeline.state,data,training=False,properties=properties,forces=forces)
                    labels=self.pipeline.transform_back_input(data)
                    errors=self.loss.compute(prediction,labels,key=jax.random.PRNGKey(self.seed),
                                             exclude_energy_min=exclude,loss_weights_override=weights)
                jax.block_until_ready(errors)
                values.append({name:float(value) for name,value in errors.items()})
        finally:
            self.pipeline.state=original
        if not values:
            raise ValueError("No validation batches")
        return {name:float(np.mean([row[name] for row in values])) for name in values[0]}

    def schedule(self,metric):
        if not np.isfinite(metric):
            raise FloatingPointError("Nonfinite validation metric")
        threshold=self.scheduler["threshold"]
        if metric<self.loop["best"]*(1-threshold):
            self.loop["best"]=metric
            self.loop["bad_epochs"]=0
        else:
            self.loop["bad_epochs"]=int(self.loop["bad_epochs"])+1
        if self.scheduler_enabled and self.loop["bad_epochs"]>self.scheduler["patience"]:
            rate=max(self.scheduler["min_lr"],self.loop["learning_rate"]*self.scheduler["factor"])
            if self.loop["learning_rate"]-rate>self.scheduler["eps"]:
                self.loop["learning_rate"]=rate
            self.loop["bad_epochs"]=0

    def change_stage(self,stage,*,optimizer=None,loss=None):
        self.pipeline.state={"params":self.state["params"],"buffers":self.pipeline.state["buffers"]}
        self.stage=stage
        self.optimizer=optimizer or self.optimizer
        self.loss=loss or self.loss
        fresh=initialize(self.pipeline,stage=stage,optimizer=self.optimizer,seed=self.seed)
        fresh.update(step=self.state["step"],rng=self.state["rng"],ema=self.state["ema"])
        self.state=fresh
        self.loop.update(best=math.inf,bad_epochs=0,learning_rate=self.optimizer.learning_rate,
                         loss_weights=dict(self.loss.loss_weights),validation_loss_weights=dict(self.loss.loss_weights))
        self.updates.clear()
        self.validations.clear()

    def save(self,path):
        state=dict(self.state,loop=self.loop)
        save_checkpoint(path,self.pipeline,state,stage=self.stage,optimizer=self.optimizer,loss=self.loss,
                        metadata={"native_loop":{"scheduler":self.scheduler if self.scheduler_enabled else None,
                                                 "seed":self.seed,"convert_inputs":self.convert_inputs,
                                                 "grid_scaling_start":self.grid_scaling_start,
                                                 "grid_scaling_annealing":self.grid_scaling_annealing,
                                                 "jit_validation":self.jit_validation}})
