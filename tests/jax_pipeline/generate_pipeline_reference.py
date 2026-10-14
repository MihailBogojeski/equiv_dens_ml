"""Collect immutable shared-parameter whole-pipeline references in Torch only."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import hashlib

import torch

from benchmarks.fixtures import fixture_spec, build_model, make_atoms, clone_tree, select_source
from benchmarks.precision import use_full_precision
from equiv_dens_jax.convert import export_module, to_numpy_tree
from equiv_dens_jax.serialization import save_tree


def generate(case, output, source=None):
    if source is not None:
        select_source(source)
    use_full_precision()
    from equiv_dens.training.errors import ErrorDict
    spec=fixture_spec(case,torch.float64)
    model=build_model(spec,"ef").train()
    model.no_force_props=["density"]
    atoms=make_atoms(case,torch.float64,grid_points=5)
    prediction=model(clone_tree(atoms))
    fields=("energy","forces","density","density_grad","df_coeffs")
    for field in fields:
        shape=prediction[field].shape
        offset=torch.linspace(.011,.029,prediction[field].numel(),dtype=torch.float64).reshape(shape)
        atoms[field]=prediction[field].detach()+offset
    config={"loss_weights":{"energy":.2,"forces":.3,"density":1.,"density_grad":.1},
            "loss_comp":{key:["mse"] for key in ("energy","forces","density","density_grad")}}
    output_train=model(clone_tree(atoms))
    error=ErrorDict(**config).compute(output_train,atoms)
    named=list(model.named_parameters())
    gradients=torch.autograd.grad(error["loss"],[p for _,p in named],allow_unused=True)
    node,state,skipped=export_module(model)
    expected={field:output_train[field] for field in fields}
    expected.update({key:output_train[key] for key in ("spherical_coeffs","radial_width","radial_scale")})
    payload={"model":node,"state":state,"inputs":to_numpy_tree(atoms),"loss_config":config,
             "expected":to_numpy_tree(expected),"errors":to_numpy_tree(error),
             "gradients":{name:None if g is None else to_numpy_tree(g) for (name,_),g in zip(named,gradients)},
             "metadata":{"case":case,"spec":spec,"torch":torch.__version__,"skipped_attributes":skipped,
                          "source":str(source or Path(__file__).resolve().parents[2]),
                          "allocation":__import__("os").environ.get("SLURM_JOB_ID")}}
    source_path=Path(source or Path(__file__).resolve().parents[2]).resolve()
    payload["metadata"]["source_revision"]=subprocess.check_output(
        ["git","-C",str(source_path),"rev-parse","HEAD"],text=True).strip()
    payload["metadata"]["source_files"]={str(path.relative_to(source_path)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((source_path/"src/equiv_dens").rglob("*.py"))}
    save_tree(output,payload)
    print(f"Saved {case}: {len(named)} parameter-gradient references to {output}")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=("tiny","batch","wide"),default="tiny")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-root",type=Path)
    args=parser.parse_args()
    generate(args.case,args.output,args.source_root)
