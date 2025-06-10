from torch.optim.lr_scheduler import _LRScheduler


class InterpolatedDecayLR(_LRScheduler):
    """
    Learning rate scheduler that implements an interpolated decay schedule.
    alpha is the blending factor between linear and exponential decay.
    The learning rate is linearly increased during the warmup phase and then
    decays according to the interpolated decay schedule.
    The learning rate is calculated as:
        lr = (1 - alpha) * lr_linear + alpha * lr_exp
    where:
        lr_linear = base_lr + (lr_end - base_lr) * ratio
        lr_exp = base_lr * (lr_end / base_lr) ** ratio
    where ratio is the fraction of the total steps completed after warmup.
    """

    def __init__(self, optimizer, total_steps, lr_end, warmup_steps=1000, alpha=0.5, last_epoch=-1):
        self.total_steps = total_steps
        self.lr_end = lr_end
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = min(self.last_epoch, self.total_steps)
        
        # Handle warmup phase
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        
        # After warmup, calculate interpolated decay
        ratio = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        
        lrs = []
        for base_lr in self.base_lrs:
            lr_linear = base_lr + (self.lr_end - base_lr) * ratio
            lr_exp = base_lr * (self.lr_end / base_lr) ** ratio
            lr_blend = lr_linear ** (1 - self.alpha) * lr_exp ** self.alpha
            lrs.append(lr_blend)
        return lrs