"""Native training, inference and MD command line, independent of Torch imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description="Current DenSnet pipeline in native JAX")
    parser.add_argument("--precision",choices=("float64","float32"),default="float64")
    parser.add_argument("--platform",choices=("cpu","cuda"))
    commands=parser.add_subparsers(dest="command",required=True)
    initialize_cmd=commands.add_parser("init",help="Initialize fresh JAX weights from a portable architecture template")
    initialize_cmd.add_argument("--model","--template",dest="model",type=Path,required=True)
    initialize_cmd.add_argument("--seed",type=int,default=0)
    initialize_cmd.add_argument("--output",type=Path,required=True)
    prepare=commands.add_parser("prepare",help="Prepare native density targets from existing QC results")
    prepare.add_argument("--atoms",type=Path,required=True)
    prepare.add_argument("--basis",type=Path,required=True)
    prepare.add_argument("--qc",type=Path,required=True)
    prepare.add_argument("--count",type=int)
    prepare.add_argument("--seed",type=int,default=0)
    prepare.add_argument("--grid-level",type=int,default=2)
    prepare.add_argument("--projected",action="store_true")
    prepare.add_argument("--density-grad",action="store_true")
    prepare.add_argument("--trusted-legacy",action="store_true")
    prepare.add_argument("--atomic-config",type=Path,help="JSON for frozen-atom SAD targets on the prepared QC grid")
    prepare.add_argument("--output",type=Path,required=True)
    infer=commands.add_parser("infer")
    infer.add_argument("--model",type=Path,required=True)
    infer.add_argument("--data",type=Path,required=True)
    infer.add_argument("--output",type=Path,required=True)
    infer.add_argument("--properties",nargs="+",default=["energy"])
    infer.add_argument("--forces",action=argparse.BooleanOptionalAction,default=True)
    infer.add_argument("--jit",action=argparse.BooleanOptionalAction,default=True)
    infer.add_argument("--convert-inputs",action="store_true")
    train=commands.add_parser("train")
    train.add_argument("--model",type=Path,required=True)
    train.add_argument("--data",type=Path,required=True)
    train.add_argument("--config",type=Path,required=True)
    train.add_argument("--output",type=Path,required=True)
    length=train.add_mutually_exclusive_group()
    length.add_argument("--steps",type=int)
    length.add_argument("--epochs",type=int)
    train.add_argument("--validation",type=Path)
    train.add_argument("--resume",action="store_true")
    train.add_argument("--jit",action=argparse.BooleanOptionalAction,default=True)
    train.add_argument("--convert-inputs",action="store_true")
    md=commands.add_parser("md")
    md.add_argument("--model",type=Path,required=True)
    md.add_argument("--geometry",type=Path,required=True)
    md.add_argument("--output",type=Path,required=True)
    md.add_argument("--steps",type=int,required=True)
    md.add_argument("--timestep-fs",type=float,default=.5)
    md.add_argument("--temperature-k",type=float,default=300.)
    md.add_argument("--seed",type=int,default=0)
    md.add_argument("--energy-to-ev",type=float,required=True)
    md.add_argument("--properties",nargs="+",default=["energy"])
    md.add_argument("--cutoff",type=float)
    md.add_argument("--preparation-config",type=Path,help="JSON for current-geometry grid/free-atom host preparation")
    md.add_argument("--jit",action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args(argv)
    if args.platform:
        import os
        os.environ["JAX_PLATFORMS"]=args.platform
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update("jax_enable_x64",args.precision=="float64")
    jax.config.update("jax_default_matmul_precision","highest")
    from .pipeline import Pipeline
    from .data import load_batches
    from .serialization import save_tree,load_tree
    if args.command=="prepare":
        from .preparation import prepare_dataset_batches
        batches=prepare_dataset_batches(args.atoms,args.basis,args.qc,count=args.count,seed=args.seed,
                  trusted_legacy=args.trusted_legacy,projected=args.projected,density_grad=args.density_grad,
                  grid_level=args.grid_level)
        if args.atomic_config:
            from .md_preparation import make_md_preparation
            config=json.loads(args.atomic_config.read_text())
            config={**config,"density_expansion":False,"position_unit":"angstrom"}
            enrich=make_md_preparation(config)
            batches=[enrich(batch) for batch in batches]
        save_tree(args.output,{"batches":batches,"metadata":{"seed":args.seed,"grid_level":args.grid_level,
            "grid_count":args.count,"quadrature":"uniform subsample with original weights; no rescaling"}})
        print(f"Prepared {len(batches)} native batches at {args.output}")
        return
    if args.command=="train" and args.resume:
        # Inspect host arrays before any JAX dtype conversion.
        saved=load_tree(args.model)
        saved_dtypes={str(value.dtype) for value in saved["training"]["params"].values()
                      if np.issubdtype(value.dtype,np.floating)}
        if saved_dtypes!={args.precision}:
            parser.error(f"Exact resume requires checkpoint precision {sorted(saved_dtypes)}; use --precision accordingly")
    pipeline=Pipeline.load(args.model)
    dtype=jnp.float64 if args.precision=="float64" else jnp.float32
    pipeline.state=jax.tree.map(lambda x:x.astype(dtype) if jnp.issubdtype(x.dtype,jnp.floating) else x,pipeline.state)
    if args.command=="init":
        from .initialization import initialize_parameters
        pipeline.state=initialize_parameters(pipeline.node,pipeline.state,seed=args.seed)
        save_tree(args.output,{"model":pipeline.node,"state":pipeline.state,
            "metadata":{**pipeline.metadata,"native_initialization_seed":args.seed}})
        print(f"Initialized native JAX weights at {args.output}")
        return
    if args.command=="md":
        from .md import run_md
        prepare_properties=None
        if args.preparation_config:
            from .md_preparation import make_md_preparation
            from .pipeline import _factor,_DISTANCE
            from .common import child
            config=json.loads(args.preparation_config.read_text())
            incoming=_factor(child(pipeline.node,"conversions_in"),"dist_conversion_func",_DISTANCE)
            expected_unit="angstrom" if incoming==1. else "bohr" if incoming==.52917721092 else None
            if expected_unit is None:
                parser.error("Host MD preparation requires internal Angstrom geometry and Angstrom or Bohr input")
            if config.get("position_unit",expected_unit)!=expected_unit:
                parser.error(f"MD preparation position_unit must match the model input: {expected_unit}")
            prepare_properties=make_md_preparation({**config,"position_unit":expected_unit})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        run_md(pipeline,args.geometry,args.output,steps=args.steps,timestep_fs=args.timestep_fs,
               temperature_k=args.temperature_k,seed=args.seed,energy_to_ev=args.energy_to_ev,
               properties=args.properties,cutoff=args.cutoff,jit=args.jit,prepare_properties=prepare_properties)
        return
    batches=load_batches(args.data)
    batches=jax.tree.map(lambda x:np.asarray(x,dtype=np.dtype(args.precision))
                         if isinstance(x,np.ndarray) and np.issubdtype(x.dtype,np.floating) else x,batches)
    if args.command=="infer":
        predictions=[]
        for batch in batches:
            values=pipeline.predict(batch,properties=args.properties,forces=args.forces,jit=args.jit,convert_inputs=args.convert_inputs)
            predictions.append({name:values[name] for name in ("energy","energy_min","forces","df_coeffs","density","density_grad","dipole_moment") if name in values})
        save_tree(args.output,{"predictions":predictions,"metadata":pipeline.metadata})
        print(f"Wrote {len(predictions)} predictions to {args.output}")
        return
    from .training import Optimizer,load_checkpoint
    from .trainer import Trainer
    from .losses import ErrorDict
    config=json.loads(args.config.read_text())
    if args.steps is not None and args.steps<0 or args.epochs is not None and args.epochs<0:
        parser.error("steps must be nonnegative")
    if args.resume:
        pipeline,train_state,stage,optimizer,loss=load_checkpoint(args.model)
    else:
        stage=config.get("stage","joint")
        optimizer=Optimizer(**config.get("optimizer",{}))
        loss=ErrorDict(**config["loss"])
        train_state=None
    if loss is None:
        parser.error("Checkpoint has no loss configuration")
    saved_loop=pipeline.metadata.get("native_loop",{}) if args.resume else {}
    trainer=Trainer(pipeline,loss,optimizer,stage=stage,state=train_state,jit=args.jit,
                    seed=saved_loop.get("seed",config.get("seed",0)),convert_inputs=saved_loop.get("convert_inputs",args.convert_inputs),
                    scheduler=saved_loop.get("scheduler",config.get("scheduler")),
                    grid_scaling_start=saved_loop.get("grid_scaling_start",config.get("grid_scaling_start",10000)),
                    grid_scaling_annealing=saved_loop.get("grid_scaling_annealing",config.get("grid_scaling_annealing",1.0)),
                    jit_validation=saved_loop.get("jit_validation",config.get("jit_validation",False)))
    validation=load_batches(args.validation) if args.validation else None
    if trainer.scheduler_enabled and validation is None:
        parser.error("A plateau scheduler requires --validation")
    steps=args.steps if args.steps is not None else (args.epochs*len(batches) if args.epochs is not None else 1)
    for iteration in range(steps):
        metrics,completed=trainer.step(batches,shuffle=config.get("shuffle",True))
        record={"step":int(trainer.state["step"]),"epoch":int(trainer.loop["epoch"]),"loss":metrics["loss"]}
        if completed and validation is not None:
            held_out=trainer.validate(validation)
            record["validation_loss"]=held_out["loss"]
            trainer.schedule(held_out["loss"])
        print(json.dumps(record))
    trainer.save(args.output)
    print(f"Saved native checkpoint to {args.output}")


if __name__=="__main__":
    main()
