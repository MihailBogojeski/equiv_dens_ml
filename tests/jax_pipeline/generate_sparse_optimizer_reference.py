"""Independent Torch oracle for intermittent None gradients and Lookahead."""
import argparse
from pathlib import Path

import numpy as np
import torch
from equiv_dens.training.lookahead import Lookahead
from equiv_dens_jax.serialization import save_tree


NAMES = ("density_repr_model.used.weight","property_models.energy.later.weight",
         "property_models.energy.zero.weight")
SCHEDULE = ((.2,None,0.),(-.1,None,None),(None,.3,0.),
            (None,None,None),(None,None,None),(.05,-.04,0.),(None,None,None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("scratch/jax_pipeline/sparse-optimizers"))
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    initial = {name:np.array([[-.1,.2],[.3,-.4]])+index*.4 for index,name in enumerate(NAMES)}
    cases = []
    for kind in ("sgd","adam","amsgrad","adamw"):
        for lookahead in (0,2):
            config = {"name":kind,"learning_rate":.03,"weight_decay":.07,
                      "momentum":.8,"beta1":.75,"beta2":.9,"epsilon":1e-7,
                      "lookahead_k":lookahead,"lookahead_alpha":.5,"ema_decay":None}
            params = {name:torch.nn.Parameter(torch.from_numpy(value.copy())) for name,value in initial.items()}
            if kind == "sgd":
                base = torch.optim.SGD(params.values(),lr=.03,weight_decay=.07,momentum=.8)
            else:
                constructor = torch.optim.AdamW if kind == "adamw" else torch.optim.Adam
                kwargs = {"amsgrad":True} if kind == "amsgrad" else {}
                base = constructor(params.values(),lr=.03,weight_decay=.07,betas=(.75,.9),eps=1e-7,**kwargs)
            optimizer = Lookahead(base,k=lookahead,alpha=.5) if lookahead else base
            updates = []
            for schedule in SCHEDULE:
                gradients = {}
                for name,value in zip(NAMES,schedule):
                    params[name].grad = None if value is None else torch.full_like(params[name],value)
                    if value is not None:
                        gradients[name] = np.full(initial[name].shape,value)
                optimizer.step()
                fast_state = {}
                for name,param in params.items():
                    fast_state[name] = {key:value.detach().numpy().copy() if isinstance(value,torch.Tensor) else value
                                        for key,value in base.state.get(param,{}).items()}
                slow = {name:optimizer.state[param]["slow_param"].detach().numpy().copy()
                        for name,param in params.items()} if lookahead else {}
                updates.append({"gradients":gradients,"params":{name:param.detach().numpy().copy() for name,param in params.items()},
                                "fast_state":fast_state,"slow":slow})
            cases.append({"config":config,"updates":updates})
    save_tree(args.output,{"initial":initial,"cases":cases,"torch":torch.__version__,"dtype":"float64"})
    print(f"Saved {len(cases)} sparse optimizer trajectories of seven updates to {args.output}")


if __name__ == "__main__":
    main()
