import os
import torch
from tensorboardX import SummaryWriter
import math
import time
from equiv_dens.training.exponential_moving_average import ExponentialMovingAverage
import sys


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
        hyperparam_args=None,
        restore=False,
        max_steps=100000,
        clip_norm=0,
        stop_at_learning_rate=1e-5,
        stop_at_learning_rate_patience=0,
        valid_check_best=None,
        verbose=0,
        timing=False,
        data_split_indices=None,
        grid_scaling_annealing=1.0,
        grid_scaling_start=10000,
        training_phases=None,
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
        self.hyperparam_args = hyperparam_args
        self.max_steps = max_steps
        self.clip_norm = clip_norm
        self.stop_at_learning_rate = stop_at_learning_rate
        self.stop_at_learning_rate_patience = stop_at_learning_rate_patience
        self.verbose = verbose
        self.timing = timing
        self.data_split_indices = data_split_indices
        self.grid_scaling_annealing = grid_scaling_annealing
        self.grid_scaling_start = grid_scaling_start
        self.training_phases = training_phases
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
        
        print('trainer restore', restore)
        if os.path.exists(self.checkpoint_path) and restore:
            self.restore_checkpoint()
        else:
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            self.epoch = 0
            self.step = 0
            self.best_errors = self.error_dict.empty(fill_value=math.inf)
            self.valid_errors = [self.error_dict.empty(fill_value=math.inf) for i in range(len(self.validation_loaders))]

        self.train_errors = self.error_dict.empty()  # reset train error metrics
        self.summary = SummaryWriter(logdir=os.path.join(self.model_path, 'logs'), purge_step=self.step)

    def store_checkpoint(self, best=False):
        if self.training_phases is None:
            phase = ''
        else:
            phase = '_' + self.training_phases[0]
        # move latest checkpoint (so it is not overwritten)
        if not best:
            if os.path.isfile(os.path.join(self.checkpoint_path, 'latest_checkpoint.pth')):
                os.rename(os.path.join(self.checkpoint_path, 'latest_checkpoint.pth'), os.path.join(
                    self.checkpoint_path, 'checkpoint_' + str(self.step).zfill(10) + phase + '.pth'))
            chk_name = 'latest_checkpoint.pth'
        else:
            chk_name = 'best_checkpoint.pth'

        # overwrite latest checkpoint
        checkpoint = {
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
            'data_split_indices': self.data_split_indices,
            'training_phases': self.training_phases
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_path, chk_name))
        self.summary.add_text('checkpoints', 'saved checkpoint', self.step)

        # remove oldest checkpoints
        if self.keep_n_checkpoints >= 0 and not best:  # for negative arguments, all checkpoints are kept
            for file in os.listdir(self.checkpoint_path):
                if file.startswith("checkpoint") and file.endswith('.pth'):
                    checkpoint_step = int(file.split('.pth')[0].split('_')[1])
                    checkpoint_phase = file.split('.pth')[0].split('_')[2]
                    if checkpoint_step < self.step - self.checkpoint_interval * self.keep_n_checkpoints \
                       and phase == checkpoint_phase:
                        filename = os.path.join(self.checkpoint_path, file)
                        if os.path.isfile(filename):
                            os.remove(filename)

    def _aux_to(self, use_gpu, dtype):
        """
        Move the optimizers and schedulers to device before training.
        """
        for opt in self.optimizers:
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        if use_gpu:
                            state[k] = v.type(dtype).cuda()
                        else:
                            state[k] = v.type(dtype)

        for sched in self.schedulers:
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        if use_gpu:
                            state[k] = v.type(dtype).cuda()
                        else:
                            state[k] = v.type(dtype)
        if self.exponential_moving_average is not None:
            for key in self.exponential_moving_average.ema.keys():
                if use_gpu:
                    self.exponential_moving_average.ema[key] = self.exponential_moving_average.ema[key].type(dtype).cuda()
                else:
                    self.exponential_moving_average.ema[key] = self.exponential_moving_average.ema[key].type(dtype)

    def restore_checkpoint(self):
        print('RESTORING CHECKPOINT')
        checkpoint = torch.load(os.path.join(
            self.checkpoint_path, 'latest_checkpoint.pth'), map_location='cpu')
        # self.args = checkpoint['args']  # overwrite args
        for arg in vars(checkpoint['args']):
            if self.args.fix_arguments:
                if arg in self.hyperparam_args:
                    print('loading hyperparam arg', arg)
                    setattr(self.args, arg, getattr(checkpoint['args'], arg))
            else:
                print('loading all arg', arg)
                setattr(self.args, arg, getattr(checkpoint['args'], arg))

        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_errors = checkpoint['best_errors']
        self.valid_errors = checkpoint['valid_errors']
        self._module.load_state_dict(checkpoint['model_state_dict'])
        self.error_dict = checkpoint['error_dict']
        self.training_phases = checkpoint['training_phases']
        if not hasattr(self.error_dict, 'relative_en'):
            self.error_dict.relative_en = False
        self.data_split_indices = checkpoint['data_split_indices']
        for i in range(len(self.optimizers)):
            self.optimizers[i].load_state_dict(checkpoint['optimizers_state_dict'][i])
        for i in range(len(self.schedulers)):
            self.schedulers[i].load_state_dict(checkpoint['schedulers_state_dict'][i])
        if self.ema_params is not None:
            checkpoint_ema = checkpoint['exponential_moving_average']
            if checkpoint_ema is not None:
                self.exponential_moving_average = ExponentialMovingAverage(self._module, decay=self.ema_params['decay'],
                                                                           start_epoch=self.ema_params['start_epoch'])
                for key in self.exponential_moving_average.ema.keys():
                    with torch.no_grad():
                        self.exponential_moving_average.ema[key].data.copy_(
                            checkpoint_ema[key].data)
            else:
                self.exponential_moving_average = None
                self.ema_params = None

    def run(self, n_steps, use_gpu=False, dtype=torch.float64):
        self._model.to(dtype)
        if use_gpu:
            self._model.cuda()

        self._aux_to(use_gpu, dtype)

        if use_gpu and torch.cuda.device_count() > 1:
            self._model = torch.nn.DataParallel(self._model)
            self._module = self._model.module
        else:
            self._module = self._model

        if use_gpu:
            print("Training on " + str(torch.cuda.device_count()) + " GPUs:")
        else:
            print("Training on the CPU:")

        if self.ema_params is not None and self.exponential_moving_average is None:
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
        stop_patience_count = 0
        start_time = time.time()
        while self.step < n_steps + 1:
            # get the next batch
            self._train_step(use_gpu)
            # run validation each validation_interval
            if self.step % self.validation_interval == 0:
                if self.verbose > 0:
                    print('validation')
                new_valid = True
                self._module.eval()
                for i, valid_data_loader in enumerate(self.validation_loaders):
                    if self.verbose > 0:
                        print('validation for', valid_data_loader)
                    if self._module.calculate_forces:
                        self.valid_errors[i], is_best = self._validate(valid_data_loader, use_gpu, check_best=self.valid_check_best[i])
                    else:
                        with torch.no_grad():
                            self.valid_errors[i], is_best = self._validate(valid_data_loader, use_gpu, check_best=self.valid_check_best[i])
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
                if self.error_dict.loss_weights[key] > self.error_dict.weights_min[key]:
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
            if stop_training:
                stop_training = stop_training and stop_patience_count > self.stop_at_learning_rate_patience
                stop_patience_count += 1
            if self.step > self.max_steps:
                print('Reached maximum number of steps! Training stopped.')
                break

            if stop_training:
                print("Learning rate is smaller than " +
                      str(self.stop_at_learning_rate) + "! Training stopped.")
                break
        # close summary writer
        self.summary.close()

    def _train_step(self, use_gpu):
        start = time.time()
        try:
            data = next(self.train_iterator)
        except StopIteration:
            self.epoch += 1
            self.train_iterator = iter(self.train_loader)
            return
        self.train_batch_num += 1
        # print('train loading time', time.time() - start_load)

        # send data to GPU
        if use_gpu:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda()
        if self.timing:
            print('train load time', time.time() - start)
        # zero the parameter gradients
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # for name, param in self._model.named_parameters():
        #     print('param grad', name, param)

        data = self._module.conversions_in(data)
        data = self._module.scaling(data)
        # print('model embedding layer before', self._model.density_repr_model[0].embedding.embedding.element_embedding)
        predictions = self._model(data)
        # print('model embedding layer after pred', self._model.density_repr_model[0].embedding.embedding.element_embedding)

        if self.verbose > 0:
            if 'density' in predictions.keys():
                print('train density intergal', torch.sum(predictions['density'] * predictions['coord_weights'], dim=1))
                print('true density intergal', torch.sum(data['density'] * data['coord_weights'], dim=1))
        if self.verbose > 0:
            if 'energy' in predictions.keys():
                print('pred energy', predictions['energy'].view((-1, )))
                print('true energy', data['energy'].view((-1, )))
            if 'forces' in predictions.keys():
                print('pred forces', predictions['forces'].sum((-1, -2)).view((-1, )))
                print('true forces', data['forces'].sum((-1, -2)).view((-1, )))

        if 'density' in data.keys() and torch.any(torch.isnan(data['density'])):
            print('Nans found in label density, skipping batch')
            return
        elif 'density' in predictions.keys() and torch.any(torch.isnan(predictions['density'])):
            print('num nans', torch.sum(torch.isnan(predictions['density'])))
            raise Exception('Nans found in predicted density')
            sys.exit()
        elif 'energy' in data.keys() and torch.any(torch.isnan(data['energy'])):
            print('Nans found in label energy, skipping batch')
            return
        elif 'energy' in predictions.keys() and torch.any(torch.isnan(predictions['energy'])):
            raise Exception('Nans found in predicted energy')
            sys.exit()
        elif 'forces' in data.keys() and torch.any(torch.isnan(data['forces'])):
            print('Nans found in label forces, skipping batch')
            return
        elif 'forces' in predictions.keys() and torch.any(torch.isnan(predictions['forces'])):
            raise Exception('Nans found in predicted forces')
            sys.exit()
        errors = self.error_dict.compute(predictions, data)
        # check for nans
        found_nans = False
        for key in errors.keys():
            if torch.any(torch.isnan(errors[key])):
                print('Nans found in', key, 'error')
                found_nans = True
                # raise Exception('Nans found in', key, 'error')
                # sys.exit()
        # backward step
        if found_nans:
            for k in errors.keys():
                print('key', k)
                print('nans', torch.any(torch.isnan(errors[k])))
            torch.save(predictions, self.model_code + '_pred_crash_dump_train_' + str(self.step) + '.pth')
            torch.save(data, self.model_code + '_true_crash_dump_train_' + str(self.step) + '.pth')
            torch.save(self._module.state_dict(), self.model_code + '_model_crash_dump_train_' + str(self.step) + '.pth')
            return

        # if self.verbose > 2:
        #     print('train step before backward:', torch.cuda.memory_summary())
        start_bw = time.time()
        # for key in ['df_coeffs', 'density', 'energy', 'forces']:
        #     if key in predictions.keys():
        #         print('prediction type', key, predictions[key].type())
        #         print('prediction grad fn', key, predictions[key].grad_fn)
        # print(errors['loss'].type())
        errors['loss'].backward()
        if self.timing:
            print('backward time', time.time() - start_bw)
        # if self.verbose > 2:
        #     print('train step after backward:', torch.cuda.memory_summary())

        # apply gradient clipping
        if self.clip_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(
                self._module.parameters(), self.clip_norm)
            self.gradient_norm += (norm - self.gradient_norm) / (self.train_batch_num + 1)

        # optimization step
        start_step = time.time()
        for optimizer in self.optimizers:
            optimizer.step()
        if self.timing:
            print('step time', time.time() - start_step)

        # print('model embedding layer after backward', self._model.density_repr_model[0].embedding.embedding.element_embedding)
        # update parameter averages
        if self.exponential_moving_average is not None:
            self.exponential_moving_average(self.epoch)

        if self.epoch > self.grid_scaling_start and 'density' in self._module.property_models.keys():
            density_expansion_model = self._module.property_models['density']
            density_expansion_model.grid_scaling_factor *= self.grid_scaling_annealing

        # update train_errors (running average)
        for key in errors.keys():
            self.train_errors[key] += (errors[key].item() -
                                       self.train_errors[key]) / (self.train_batch_num + 1)
        if self.timing:
            print('train step time', time.time() - start)

    def _validate(self, valid_data_loader, use_gpu, check_best=False):
        is_best = False
        # swap to exponentially averaged parameters for validation
        if self.exponential_moving_average is not None:
            self.exponential_moving_average.swap()

        # run once over the validation set
        valid_errors = self.error_dict.empty()
        for valid_batch_num, data in enumerate(valid_data_loader):
            start = time.time()
            # send data to GPU
            if use_gpu:
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cuda()
            if self.timing:
                print('valid load time', time.time() - start)

            # forward step
            # if self.verbose > 2:
            #     print('validate before prediction:', torch.cuda.memory_summary())
            # print('pre-conversion forces:', data['forces'])
            data = self._module.conversions_in(data)
            data = self._module.scaling(data)
            # print('post-conversion forces:', data['forces'])
            predictions = self._model(data)
            data = self._module.scaling.transform_back(data)
            data = self._module.conversions_out(data)
            # print('post-post-conversion forces:', data['forces'])
            # if self.verbose > 2:
            #     print('validate after prediction:', torch.cuda.memory_summary())
            # print('energy pred', predictions['energy'])
            if self.verbose > 0:
                if 'density' in predictions.keys():
                    print('valid density intergal', torch.sum(predictions['density'] * predictions['coord_weights'], dim=1))
                    print('true density intergal', torch.sum(data['density'] * data['coord_weights'], dim=1))
                if 'energy' in predictions.keys():
                    print('pred energy', predictions['energy'].view((-1, )))
                    print('true energy', data['energy'].view((-1, )))
                if 'forces' in predictions.keys():
                    print('pred forces', predictions['forces'].sum((-1, -2)).view((-1, )))
                    print('true forces', data['forces'].sum((-1, -2)).view((-1, )))
            if 'density' in data.keys() and torch.any(torch.isnan(data['density'])):
                print('Nans found in label density, skipping batch')
                continue
            elif 'density' in predictions.keys() and torch.any(torch.isnan(predictions['density'])):
                print('num nans', torch.sum(torch.isnan(predictions['density'])))
                raise Exception('Nans found in predicted density')
                sys.exit()
            if 'energy' in data.keys() and torch.any(torch.isnan(data['energy'])):
                print('Nans found in label energy, skipping batch')
                continue
            elif 'energy' in predictions.keys() and torch.any(torch.isnan(predictions['energy'])):
                raise Exception('Nans found in predicted energy')
                sys.exit()
            elif 'forces' in data.keys() and torch.any(torch.isnan(data['forces'])):
                print('Nans found in label forces, skipping batch')
                continue
            elif 'forces' in predictions.keys() and torch.any(torch.isnan(predictions['forces'])):
                raise Exception('Nans found in predicted forces')
                sys.exit()

            # print('spherical density integral', torch.sum(predictions['density'] * data['coord_weights'], dim=-1))
            # compute error metrics
            exclude_energy_min = check_best
            if 'energy_min' in self.error_dict.loss_weights.keys():
                if self.error_dict.loss_weights['energy_min'] == sum(self.error_dict.loss_weights.values()):
                    exclude_energy_min = False
            errors = self.error_dict.compute(predictions, data, exclude_energy_min=exclude_energy_min)
            # update valid_errors (running average)
            found_nans = False
            for key in errors.keys():
                if torch.any(torch.isnan(errors[key])):
                    print('Nans found in', key, 'error')
                    found_nans = True
                    # raise Exception('Nans found in', key, 'error')
                    # sys.exit()
            # backward step
            if found_nans:
                for k in errors.keys():
                    print('key', k)
                    print('nans', torch.any(torch.isnan(errors[k])))
                torch.save(predictions, self.model_code + '_pred_crash_dump_valid_' + str(self.step) + '.pth')
                torch.save(data, self.model_code + '_true_crash_dump_valid_' + str(self.step) + '.pth')
                torch.save(self._module.state_dict(), self.model_code + '_model_crash_dump_valid_' + str(self.step) + '.pth')
                continue
            for key in errors.keys():
                valid_errors[key] += (errors[key].item() -
                                      valid_errors[key]) / (valid_batch_num + 1)
            if self.timing:
                print('valid step time:', time.time() - start)
            predictions = None
            data = None
            errors = None

        # pass validation loss to learning rate scheduler
        if check_best:
            for scheduler in self.schedulers:
                scheduler.step(metrics=valid_errors['loss'])

            # save if it outperforms previous best
            if valid_errors['loss'] < self.best_errors['loss']:
                is_best = True
                self.best_errors = valid_errors
                torch.save(self._module.state_dict(), os.path.join(self.model_path, 'best_' + str(self.model_code) + '.pth'))
                self.store_checkpoint(best=True)
                # construct message for logging
                message = ''
                for key in self.best_errors.keys():
                    message += key + ': %.6f' % self.best_errors[key] + '\n'
                self.summary.add_text('best models', message, self.step)

        # swap back to original parameters for training
        if self.exponential_moving_average:
            self.exponential_moving_average.swap()

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
                progress_string += "    train rmse: %10.6f" % self.train_errors[key + '_rmse']
                progress_string += "    train loss: %10.6f" % self.train_errors['loss']
                for i in range(len(self.valid_errors)):
                    progress_string += "    valid " + str(i) + " mae: %10.6f" % self.valid_errors[i][key + '_mae']
                    if self.valid_check_best[i]:
                        progress_string += "    valid " + str(i) + " loss: %10.6f" % self.valid_errors[i][key + '_loss']
                progress_string += "     best mae: %10.6f" % self.best_errors[key + '_mae']
                progress_string += "     best rmse: %10.6f" % self.best_errors[key + '_rmse']
                progress_string += "    best loss: %10.6f" % self.best_errors['loss']
          
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                progress_string += "    lr: %10.6f" % param_group['lr']
        print(progress_string)
