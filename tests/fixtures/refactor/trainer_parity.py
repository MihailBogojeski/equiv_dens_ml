"""Small CPU fixture for Trainer, optimizer, EMA, validation and cross-tree resume.

This tests the shared training engine using a synthetic density/energy model;
it does not stand in for validation against a completed JACS model checkpoint.
The caller supplies an isolated output directory and controls the package source.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import torch
from equiv_dens.training.trainer import Trainer
from equiv_dens.training.exponential_moving_average import ExponentialMovingAverage


class FixtureModel(torch.nn.Module):
    calculate_forces = False
    property_models = {}

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.3, -0.2], dtype=torch.float64))

    def transform_input(self, data):
        return data

    def transform_back_input(self, data):
        return data

    def forward(self, data):
        return {'density': data['x'] @ self.weight, 'energy': (data['x'] @ self.weight).sum().reshape(1)}


class FixtureErrors:
    relative_en = False

    def __init__(self, mode):
        self.loss_weights = {'density': float(mode != 'energy'), 'energy': float(mode != 'density')}

    def empty(self, fill_value=0):
        return {'loss': fill_value}

    def compute(self, predictions, data, **kwargs):
        loss = sum(self.loss_weights[k] * ((predictions[k]-data[k])**2).mean() for k in self.loss_weights)
        return {'loss': loss}


def plain(value):
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['density','joint','multiphase','energy'],required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    batch={'x':torch.tensor([[1.,2.],[-2.,1.]],dtype=torch.float64),
           'density':torch.tensor([0.8,-0.4],dtype=torch.float64),
           'energy':torch.tensor([0.4],dtype=torch.float64)}
    model=FixtureModel()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=1)
    trainer=Trainer(str(args.output/'run_fixture'),model,FixtureErrors(args.mode),
                    [optimizer],[scheduler],[batch],[[batch]],
                    args=SimpleNamespace(fix_hyperparams=False,fix_arguments=False),
                    hyperparam_args=[],restore=args.resume,
                    ema_params={'decay':0.9,'start_epoch':-1},
                    data_split_indices={'train':[0,1],'valid':[2]},
                    training_phases=['density','energy'] if args.mode=='multiphase' else None)
    if trainer.exponential_moving_average is None:
        trainer.exponential_moving_average=ExponentialMovingAverage(model,decay=0.9,start_epoch=-1)
    initial=plain(model.state_dict())
    prediction=plain(model(batch))
    trainer.train_iterator=iter([batch]);trainer.train_batch_num=-1
    trainer._train_step(use_gpu=False)
    trainer.step+=1
    valid,_=trainer._validate([batch],use_gpu=False,check_best=True)
    trainer.store_checkpoint()
    payload={'mode':args.mode,'initial':initial,'prediction':prediction,
             'trained':plain(model.state_dict()),'train_loss':plain(trainer.train_errors),
             'validation':plain(valid),'optimizer':plain(optimizer.state_dict()),
             'scheduler':plain(scheduler.state_dict()),
             'ema':plain(trainer.exponential_moving_average.ema),
             'splits':trainer.data_split_indices,'step':trainer.step}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/('resumed.json' if args.resume else 'step.json')).write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':main()
