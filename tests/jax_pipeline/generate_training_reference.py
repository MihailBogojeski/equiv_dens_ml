"""Independent Torch loss/optimizer references, run outside the JAX runtime."""
from pathlib import Path
import argparse
import numpy as np
import torch

from equiv_dens_jax.convert import to_numpy_tree
from equiv_dens_jax.serialization import save_tree


def generate(output):
    from equiv_dens.training.errors import ErrorDict
    from equiv_dens.training.lookahead import Lookahead
    torch.set_default_dtype(torch.float64)
    rng=np.random.default_rng(21)
    data={"positions":torch.tensor(rng.normal(size=(2,3,3))),
          "batch_positions":torch.tensor(rng.normal(size=(2,3,3))),
          "batch_atom_numbers":torch.tensor([[8,1,1],[8,1,0]]),
          "coords":torch.tensor(rng.normal(size=(2,5,3))),
          "coord_weights":torch.tensor(rng.uniform(.1,.3,size=(2,5)))}
    shapes={"density":(2,5),"density_grad":(2,5,3),"energy":(2,1),
            "forces":(1,5,3),"df_coeffs":(2,8),"dipole_moment":(2,3)}
    predictions={}
    for name,shape in shapes.items():
        data[name]=torch.tensor(rng.uniform(.3,1.,size=shape))
        predictions[name]=(data[name]+torch.tensor(rng.uniform(.03,.09,size=shape))).requires_grad_()
    predictions["df_weights"]=torch.tensor(rng.uniform(.1,1.,size=(2,8)))
    components={"density":["mae","rmse","mse","mixed_dist_err","perc_mixed_dist_err",
        "perc_mae","perc_rmse","lda_mae","lda_rmse","hartree_mae","hartree_rmse",
        "kl_loss","dpm_loss","dpm_abs_loss","coulomb"],
        "density_grad":["mae","rmse","mse","norm_int","perc_norm_int","kinetic_vw"]}
    cases=[]
    # Freeze identical sampled pairs rather than equating unrelated RNG seeds.
    original=torch.randperm
    order=torch.tensor([3,0,4,1,2])
    torch.randperm=lambda *args,**kwargs:order.clone()
    try:
        for field in shapes:
            for component in components.get(field,["mae","rmse","mse"]):
                config={"loss_weights":{field:.7},"loss_comp":{field:[component]},
                        "weights_balance":1.7,"relative_en":True,"df_loss_weights":True}
                errors=ErrorDict(**config).compute(predictions,data)
                grads=torch.autograd.grad(errors["loss"],list(predictions.values())[:-1],
                                          allow_unused=True,retain_graph=True)
                cases.append({"config":config,"errors":to_numpy_tree(errors),
                              "gradients":{name:to_numpy_tree(g) if g is not None else None
                                           for name,g in zip(shapes,grads)}})
    finally:
        torch.randperm=original
    initial={"density_repr_model.linear.weight":np.array([[.1,-.4],[.6,.9]]),
             "density_repr_model.embedding.weight":np.array([.2,-.5]),
             "property_models.energy.linear.weight":np.array([-.7,.3]),
             "en_offset":np.array([.6])}
    optimizer_cases=[]
    for name in ("adam","amsgrad","adamw","sgd"):
        for lookahead_k in (0,3):
            params={key:torch.nn.Parameter(torch.tensor(value)) for key,value in initial.items()}
            groups=[]
            for key,p in params.items():
                decay=.03 if "weight" in key and "embedding" not in key else 0.
                if "energy" in key:
                    decay=.07
                groups.append({"params":[p],"weight_decay":decay,
                               "lr":.002*(100 if key=="en_offset" else 1)})
            if name=="sgd":
                opt=torch.optim.SGD(groups,lr=.002,momentum=.4)
            elif name=="adamw":
                opt=torch.optim.AdamW(groups,lr=.002)
            else:
                opt=torch.optim.Adam(groups,lr=.002,amsgrad=name=="amsgrad")
            if lookahead_k:
                opt=Lookahead(opt,k=lookahead_k,alpha=.5)
            updates=[]
            for step in range(7):
                gradients={key:torch.full_like(p,.08+.03*step)+p.detach()*.1 for key,p in params.items()}
                for key,p in params.items():
                    p.grad=gradients[key].clone()
                torch.nn.utils.clip_grad_norm_(list(params.values()),.3)
                opt.step()
                updates.append({"gradients":to_numpy_tree(gradients),"params":to_numpy_tree(params)})
            optimizer_cases.append({"config":{"name":name,"learning_rate":.002,"momentum":.4,
                "weight_decay":.03,"energy_weight_decay":.07,"lookahead_k":lookahead_k,"clip_norm":.3},
                "updates":updates})
    save_tree(output,{"data":to_numpy_tree(data),"predictions":to_numpy_tree(predictions),
                      "pairs":(order.numpy(),torch.roll(order,1).numpy()),"loss_cases":cases,
                      "initial":initial,"optimizer_cases":optimizer_cases,
                      "metadata":{"torch":torch.__version__,"seed":21}})
    print(f"Saved {len(cases)} loss and {len(optimizer_cases)} optimizer trajectories")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("scratch/jax_pipeline/training"))
    generate(parser.parse_args().output)
