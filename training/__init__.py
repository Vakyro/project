"""
Training infrastructure for CLIPZyme.

Provides callbacks, logging, schedulers, and training utilities.
"""

from .callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    MetricsLogger,
    ProgressBar,
    CallbackList,
)

from .logger import (
    TrainingLogger,
    WandbLogger,
    TensorBoardLogger,
    ConsoleLogger,
)

from .trainer import (
    CLIPZymeTrainer,
    TrainerConfig,
)

from .lr_scheduler import (
    WarmupCosineScheduler,
    get_scheduler,
)


__all__ = [
    # Callbacks
    'Callback',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateMonitor',
    'MetricsLogger',
    'ProgressBar',
    'CallbackList',
    # Loggers
    'TrainingLogger',
    'WandbLogger',
    'TensorBoardLogger',
    'ConsoleLogger',
    # Trainer
    'CLIPZymeTrainer',
    'TrainerConfig',
    # Scheduler
    'WarmupCosineScheduler',
    'get_scheduler',
]
