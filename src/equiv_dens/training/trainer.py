import os
import torch
from tensorboardX import SummaryWriter
import math
import time
from equiv_dens.training.exponential_moving_average import ExponentialMovingAverage


class Trainer:
    """Class to train a model.
    This contains an internal training loop which takes care of validation and can be
    extended with custom functionality using hooks.
    Args:
       model_path (str): path to the model directory.
       model (torch.Module): model to be trained.
       loss_fn (callable): training loss function.
       optimizer (torch.optim.optimizer.Optimizer): training optimizer.
       train_loader (torch.utils.data.DataLoader): data loader for training set.
       validation_loader (torch.utils.data.DataLoader): data loader for validation set.
       keep_n_checkpoints (int, optional): number of saved checkpoints.
       checkpoint_interval (int, optional): intervals after which checkpoints is saved.
       hooks (list, optional): hooks to customize training process.
       loss_is_normalized (bool, optional): if True, the loss per data point will be
           reported. Otherwise, the accumulated loss is reported.
    """

    def __init__(
        self,
        model_path,
        model,
        error_dict,
        optimizers,
        schedulers,
        train_loader,
        validation_loaders,
        keep_n_checkpoints=3,
        checkpoint_interval=10,
        validation_interval=10,
        summary_interval=10,
        hooks=[],
        ema_params=None,
        args=None,
        restore=False,
        max_steps=100000,
        clip_norm=0,
        stop_at_learning_rate=1e-5,
        valid_check_best=None,
    ):
        self.model_path = model_path
        self.model_code = model_path.split('_')[-1]
        self.checkpoint_path = os.path.join(self.model_path, "checkpoints")
        self.best_model = os.path.join(self.model_path, "best_" + self.model_code + '.pth')
        self.train_loader = train_loader
        self.validation_loaders = validation_loaders
        self.validation_interval = validation_interval
        self.summary_interval = summary_interval
        self.keep_n_checkpoints = keep_n_checkpoints
        self.hooks = hooks
        self.ema_params = ema_params
        self.exponential_moving_average = None
        self.args = args
        self.max_steps = max_steps
        self.clip_norm = clip_norm
        self.stop_at_learning_rate = stop_at_learning_rate
        if valid_check_best is None:
            self.valid_check_best = [False] * len(validation_loaders)
            self.valid_check_best[0] = True
        else:
            self.valid_check_best = valid_check_best

        self._model = model
        self._module = model
        self._stop = False
        self.checkpoint_interval = checkpoint_interval

        self.error_dict = error_dict
        self.optimizers = optimizers
        self.schedulers = schedulers

        if os.path.exists(self.checkpoint_path) and restore:
            self.restore_checkpoint()
        else:
            os.makedirs(self.checkpoint_path)
            self.epoch = 0
            self.step = 0
            self.best_errors = self.error_dict.empty(fill_value=math.inf)
            self.valid_errors = [self.error_dict.empty(fill_value=math.inf) for i in range(len(self.validation_loaders))]

        self.train_errors = self.error_dict.empty()  # reset train error metrics
        self.summary = SummaryWriter(logdir=os.path.join(self.model_path, 'logs'), purge_step=self.step)

    def store_checkpoint(self):
        # move latest checkpoint (so it is not overwritten)
        if os.path.isfile(os.path.join(self.checkpoint_path, 'latest_checkpoint.pth')):
            os.rename(os.path.join(self.checkpoint_path, 'latest_checkpoint.pth'), os.path.join(
                self.checkpoint_path, 'checkpoint_' + str(self.step).zfill(10) + '.pth'))

        # overwrite latest checkpoint
        torch.save({
            'ID': self.model_code,
            'args': self.args,
            'step': self.step,
            'epoch': self.epoch,
            'best_errors': self.best_errors,
            'valid_errors': self.valid_errors,
            'model_state_dict': self._module.state_dict(),
            'optimizers_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
            'schedulers_state_dict': [scheduler.state_dict() for scheduler in self.schedulers],
            'exponential_moving_average': (self.exponential_moving_average.ema
                                           if self.exponential_moving_average is not None else None),
            'error_dict': self.error_dict,
        }, os.path.join(self.checkpoint_path, 'latest_checkpoint.pth'))
        self.summary.add_text('checkpoints', 'saved checkpoint', self.step)

        # remove oldest checkpoints
        if self.keep_n_checkpoints >= 0:  # for negative arguments, all checkpoints are kept
            for file in os.listdir(self.checkpoint_path):
                if file.startswith("checkpoint") and file.endswith('.pth'):
                    checkpoint_step = int(file.split('.pth')[0].split('_')[-1])
                    if checkpoint_step < self.step - self.checkpoint_interval * self.keep_n_checkpoints:
                        filename = os.path.join(self.checkpoint_path, file)
                        if os.path.isfile(filename):
                            os.remove(filename)

    def _aux_to(self, device, dtype):
        """
        Move the optimizers and schedulers to device before training.
        """
        for opt in self.optimizers:
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device).type(dtype)

        for sched in self.schedulers:
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device).type(dtype)

    def restore_checkpoint(self):
        checkpoint = torch.load(os.path.join(
            self.checkpoint_path, 'latest_checkpoint.pth'), map_location='cpu')
        self.args = checkpoint['args']  # overwrite args
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_errors = checkpoint['best_errors']
        self.valid_errors = checkpoint['valid_errors']
        self._module.load_state_dict(checkpoint['model_state_dict'])
        self.error_dict = checkpoint['error_dict']
        for i in range(len(self.optimizers)):
            self.optimizers[i].load_state_dict[checkpoint['optimizers_state_dict'][i]]
        for i in range(len(self.schedulers)):
            self.schedulers[i].load_state_dict[checkpoint['schedulers_state_dict'][i]]
        if self.ema_params is not None:
            checkpoint_ema = checkpoint['exponential_moving_average']
            self.exponential_moving_average = ExponentialMovingAverage(self._module, decay=self.ema_params['decay'],
                                                                       start_epoch=self.ema_params['start_epoch'])
            for key in self.exponential_moving_average.ema.keys():
                with torch.no_grad():
                    self.exponential_moving_average.ema[key].data.copy_(
                        checkpoint_ema[key].data)

    def run(self, n_steps, device='cpu', dtype=torch.float64):

        self._model.to(device)
        self._model.to(dtype)
        self._aux_to(device, dtype)

        if 'cuda' in device and torch.cuda.device_count() > 1:
            self._model = torch.nn.DataParallel(self._model)
            self.module = self._model.module
        else:
            self._module = self._model

        if device != 'cpu':
            print("Training on " + str(torch.cuda.device_count()) + " GPUs:")
        else:
            print("Training on the CPU:")

        if self.ema_params is not None and self.exponential_moving_average is not None:
            self.exponential_moving_average = ExponentialMovingAverage(self._module, decay=self.ema_params['decay'],
                                                                       start_epoch=self.ema_params['start_epoch'])

        if self.clip_norm > 0:
            self.gradient_norm = 0
        self.train_batch_num = -1
        # initialize state
        self._module.train()
        self.train_iterator = iter(self.train_loader)
        new_valid = False
        new_best = False
        start_time = time.time()

        while self.step < n_steps + 1:
            # get the next batch

            self._train_step(device)
            # run validation each validation_interval
            if self.step % self.validation_interval == 0:
                print('validation')
                new_valid = True
                self._module.eval()
                for i, valid_data_loader in enumerate(self.validation_loaders):
                    self.valid_errors[i], is_best = self._validate(valid_data_loader, device, check_best=self.valid_check_best[i])
                    if self.valid_check_best[i]:
                        new_best = is_best
                self._module.train()

            # write summary to console
            if self.step % self.summary_interval == 0:
                # write error summaries
                self.write_summary(new_valid, new_best)

                end_time = time.time()
                print("time elapsed:", end_time - start_time)
                start_time = end_time

                # reset train metrics
                if self.clip_norm > 0:
                    self.gradient_norm = 0
                self.train_errors = self.error_dict.empty()  # reset train error metrics
                self.train_batch_num = -1

            # increment step counter
            self.step += 1
            for key in self.error_dict.loss_weights.keys():
                if self.error_dict.loss_weights[key] > self.error_dict.weights_min['key']:
                    self.error_dict.loss_weights[key] *= self.error_dict.weights_decay[key]

            # save checkpoint (always the last step)
            if self.step % self.checkpoint_interval == 0:
                self.store_checkpoint()
                self.summary.add_text('checkpoints', 'saved checkpoint', self.step)

            # decide whether to stop the run based on learning rate
            stop_training = True
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    stop_training = stop_training and (
                        param_group['lr'] < self.stop_at_learning_rate)
            if self.step > self.max_steps:
                stop_training = True
            if stop_training:
                print("Learning rate is smaller than " +
                      str(self.stop_at_learning_rate) + "! Training stopped.")
                break

        # close summary writer
        self.summary.close()

    def _train_step(self, device):
        try:
            data = next(self.train_iterator)
        except StopIteration:
            self.epoch += 1
            self.train_iterator = iter(self.train_loader)
            return
        self.train_batch_num += 1
        # print('train loading time', time.time() - start_load)

        # send data to GPU
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

        # zero the parameter gradients
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # with torch.autograd.set_detect_anomaly(True):  # TODO!!! TURN THIS OFF AGAIN

        # forward step
        predictions = self._model(data)
        if 'density' in predictions.keys():
            print('train density intergal', torch.sum(predictions['density'] * predictions['coord_weights'], dim=1))
        if 'energy' in predictions.keys():
            print('pred energy', predictions['energy'].view((-1, )))
            print('true energy', data['energy'].view((-1, )))
        errors = self.error_dict.compute(predictions, data)

        # backward step
        errors['loss'].backward()

        # apply gradient clipping
        if self.clip_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(
                self._module.parameters(), self.clip_norm)
            self.gradient_norm += (norm - self.gradient_norm) / (self.train_batch_num + 1)

        # optimization step
        for optimizer in self.optimizers:
            optimizer.step()

        # update parameter averages
        if self.exponential_moving_average is not None:
            self.exponential_moving_average(self.epoch)

        # update train_errors (running average)
        for key in errors.keys():
            self.train_errors[key] += (errors[key].item() -
                                       self.train_errors[key]) / (self.train_batch_num + 1)

    def _validate(self, valid_data_loader, device, check_best=False):
        is_best = False
        # swap to exponentially averaged parameters for validation
        if self.exponential_moving_average is not None:
            self.exponential_moving_average.swap()

        # run once over the validation set
        valid_errors = self.error_dict.empty()
        for valid_batch_num, data in enumerate(valid_data_loader):
            # send data to GPU
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)

            # forward step
            predictions = self._model(data)
            # print('energy pred', predictions['energy'])
            if 'density' in predictions.keys():
                print('valid density intergal', torch.sum(predictions['density'] * predictions['coord_weights'], dim=1))
            if 'energy' in predictions.keys():
                print('pred energy', predictions['energy'].view((-1, )))
                print('true energy', data['energy'].view((-1, )))

            # print('spherical density integral', torch.sum(predictions['density'] * data['coord_weights'], dim=-1))
            # compute error metrics
            exclude_energy_min = check_best
            if 'energy_min' in self.error_dict.loss_weights.keys():
                if self.error_dict.loss_weights['energy_min'] == sum(self.error_dict.loss_weights.values()):
                    exclude_energy_min = False
            errors = self.error_dict.compute(predictions, data, exclude_energy_min=exclude_energy_min)

            # update valid_errors (running average)
            for key in errors.keys():
                valid_errors[key] += (errors[key].item() -
                                      valid_errors[key]) / (valid_batch_num + 1)

        # pass validation loss to learning rate scheduler
        if check_best:
            for scheduler in self.schedulers:
                scheduler.step(metrics=valid_errors['loss'])

            # save if it outperforms previous best
            if valid_errors['loss'] < self.best_errors['loss']:
                is_best = True
                self.best_errors = valid_errors
                torch.save(self._module.state_dict(), os.path.join(self.model_path, 'best_' + str(self.model_code) + '.pth'))
                # construct message for logging
                message = ''
                for key in self.best_errors.keys():
                    message += key + ': %.6f' % self.best_errors[key] + '\n'
                self.summary.add_text('best models', message, self.step)

        # swap back to original parameters for training
        if self.exponential_moving_average:
            self.exponential_moving_average.swap()

            # set model back to training mode
        return valid_errors, is_best

    def write_summary(self, new_valid, new_best):
        for key in self.train_errors.keys():
            self.summary.add_scalar(key + '/train', self.train_errors[key], self.step)

        if new_valid:
            for valid_err in self.valid_errors:
                for key in valid_err.keys():
                    self.summary.add_scalar(key + '/valid', valid_err[key], self.step)
                new_valid = False

        if new_best:
            for key in self.best_errors.keys():
                self.summary.add_scalar(key + '/best', self.best_errors[key], self.step)
            new_best = False

        if self.clip_norm > 0:
            self.summary.add_scalar('gradient/norm', self.gradient_norm, self.step)

        # write optional summaries for model parameters
        if self.args.write_parameter_summaries:
            for name, param in self._module.named_parameters():
                splitted_name = name.split('.', 1)
                if len(splitted_name) > 1:
                    first, last = splitted_name
                else:
                    first = 'nn'
                    last = splitted_name[0]
                if param.numel() > 1 and param.requires_grad:  # only tensors get written as histogram
                    self.summary.add_histogram(
                        first + '/' + last, param.clone().cpu().data.numpy(), self.step)

        # print progress to consoles
        progress_string = str(self.step).zfill(
            len(str(self.max_steps))) + "/" + str(self.max_steps)
        progress_string += " epoch: %6d" % self.epoch
        for key in self.error_dict.loss_weights.keys():
            if self.error_dict.loss_weights[key] > 0:
                progress_string += "\n  " + key + ":\n"
                progress_string += "    train mae: %10.6f" % self.train_errors[key + '_mae']
                progress_string += "    train loss: %10.6f" % self.train_errors['loss']
                for i in range(len(self.valid_errors)):
                    progress_string += "    valid " + str(i) + " mae: %10.6f" % self.valid_errors[i][key + '_mae']
                    if self.valid_check_best[i]:
                        progress_string += "    valid " + str(i) + " loss: %10.6f" % self.valid_errors[i][key + '_loss']
                progress_string += "     best mae: %10.6f" % self.best_errors[key + '_mae']
                progress_string += "    best loss: %10.6f" % self.best_errors['loss']
        print(progress_string)
