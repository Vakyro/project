"""
Learning rate schedulers for CLIPZyme.

Provides schedulers including warmup and cosine annealing.
"""

import math
from typing import Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine annealing.

    This is the scheduler used in CLIPZyme paper.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize warmup cosine scheduler.

        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup scheduler.

    Linearly increases learning rate from 0 to base_lr over warmup_steps.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        """
        Initialize linear warmup scheduler.

        Args:
            optimizer: Optimizer instance
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch number
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs


class ConstantLRScheduler(_LRScheduler):
    """
    Constant learning rate scheduler.

    Maintains constant learning rate throughout training.
    """

    def get_lr(self):
        """Return base learning rates."""
        return self.base_lrs


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'warmup_cosine',
    warmup_steps: int = 100,
    total_steps: Optional[int] = None,
    min_lr: float = 0.0,
    **kwargs
) -> _LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('warmup_cosine', 'warmup', 'constant', 'cosine')
        warmup_steps: Number of warmup steps
        total_steps: Total training steps (required for cosine)
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler arguments

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'warmup_cosine':
        if total_steps is None:
            raise ValueError("total_steps required for warmup_cosine scheduler")

        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr
        )

    elif scheduler_type == 'warmup':
        return LinearWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps
        )

    elif scheduler_type == 'constant':
        return ConstantLRScheduler(optimizer)

    elif scheduler_type == 'cosine':
        if total_steps is None:
            raise ValueError("total_steps required for cosine scheduler")

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=min_lr,
            **kwargs
        )

    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            **kwargs
        )

    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Available: 'warmup_cosine', 'warmup', 'constant', 'cosine', 'step', 'exponential'"
        )


# Export
__all__ = [
    'WarmupCosineScheduler',
    'LinearWarmupScheduler',
    'ConstantLRScheduler',
    'get_scheduler',
]
