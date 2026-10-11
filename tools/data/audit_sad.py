#!/usr/bin/env python3
"""Check actual fixed-atom MO density O(3) covariance on the pilot's assets."""
import argparse
import json
from pathlib import Path
import numpy as np
from equiv_dens_jax.serialization import load_tree
from equiv_dens_jax.data import load_batches
from equiv_dens_jax.preparation import sample_sad_coeff_density


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    atomic=load_tree(args.data/"atomic_mo")
    rng=np.random.default_rng(9734)
    rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(rotation)<0:rotation[:,0]*=-1
    direction=rng.normal(size=(200,3));direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    coords=direction*np.geomspace(.02,5,200)[:,None]
    rows=[]
    cases=[(f"atomic-{z}",np.zeros((1,1,3)),np.array([[z]]),coords[None]) for z in atomic]
    batch=load_batches(args.data/"train")[0]
    cases.append(("prepared-water-batch",batch["batch_positions"],batch["batch_atom_numbers"],batch["coords"]))
    def evaluate(p,z,g):
        return sample_sad_coeff_density(p,z,g,atomic,coefficient_type="mo_coeffs",density_grad=True)[0]
    passed=True
    for name,positions,numbers,grid in cases:
        reference=evaluate(positions,numbers,grid)
        if name=="prepared-water-batch":
            np.testing.assert_allclose(reference[...,0],batch["atom_density"],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(reference[...,1:],batch["atom_density_grad"],atol=1e-10,rtol=1e-10)
        for sign in (1,-1):
            transform=sign*rotation;shift=np.array([.7,-.4,.2])
            actual=evaluate(positions@transform.T+shift,numbers,grid@transform.T+shift)
            expected_grad=reference[...,1:]@transform.T
            density_ok=np.allclose(actual[...,0],reference[...,0],atol=1e-8,rtol=1e-8)
            gradient_ok=np.allclose(actual[...,1:],expected_grad,atol=1e-8,rtol=1e-8)
            passed=passed and density_ok and gradient_ok
            rows.append({"case":name,"determinant":float(np.linalg.det(transform)),
                "density_pass":bool(density_ok),"gradient_pass":bool(gradient_ok),
                "density_max_abs_error":float(np.max(np.abs(actual[...,0]-reference[...,0]))),
                "gradient_max_abs_error":float(np.max(np.abs(actual[...,1:]-expected_grad)))})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={"status":"passed" if passed else "failed","seed":9734,"absolute_relative_gate":1e-8,
            "data":str(args.data.resolve()),"cases":rows}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    if not passed:raise SystemExit(1)


if __name__=="__main__":main()
