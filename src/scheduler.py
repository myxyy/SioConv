import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class ExponentialDecayWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_scale: float,
        annealing_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_scale = warmup_scale
        self.annealing_steps = annealing_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        scale = max(1, min(
            np.exp(-(self.last_epoch - self.warmup_epochs)/self.annealing_steps * np.log(self.warmup_scale)),
            self.last_epoch / self.warmup_epochs
        ) * self.warmup_scale)
        return [base_lr * scale for base_lr in self.base_lrs]

class InvPropDecay(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        decay_time: float,
        last_epoch: int = -1,
    ):
        self.decay_time = decay_time
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        scale = 1 / (self.last_epoch / self.decay_time + 1)
        return [base_lr * scale for base_lr in self.base_lrs]