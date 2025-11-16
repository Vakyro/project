"""
Training callbacks for CLIPZyme.

Provides callbacks for monitoring, checkpointing, and early stopping.
"""

from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class Callback:
    """
    Base class for training callbacks.

    Callbacks allow you to customize behavior at different points
    in the training loop.
    """

    def on_train_begin(self, trainer, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, trainer, batch_idx: int, **kwargs):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        """Called at the end of each batch."""
        pass

    def on_validation_begin(self, trainer, **kwargs):
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self, trainer, metrics: Dict[str, float], **kwargs):
        """Called at the end of validation."""
        pass


class CallbackList(Callback):
    """
    Container for multiple callbacks.

    Manages a list of callbacks and calls them in order.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize callback list.

        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks or []

    def add(self, callback: Callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_begin(self, trainer, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(trainer, **kwargs)

    def on_train_end(self, trainer, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(trainer, **kwargs)

    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch, **kwargs)

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics, **kwargs)

    def on_batch_begin(self, trainer, batch_idx: int, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx, **kwargs)

    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, loss, **kwargs)

    def on_validation_begin(self, trainer, **kwargs):
        for callback in self.callbacks:
            callback.on_validation_begin(trainer, **kwargs)

    def on_validation_end(self, trainer, metrics: Dict[str, float], **kwargs):
        for callback in self.callbacks:
            callback.on_validation_end(trainer, metrics, **kwargs)


class EarlyStopping(Callback):
    """
    Early stopping callback.

    Stops training when a monitored metric stops improving.
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
            verbose: Print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.should_stop = False

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Check if training should stop."""
        current = metrics.get(self.monitor)

        if current is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return

        if self.best_score is None:
            self.best_score = current
            return

        if self.monitor_op(current - self.min_delta, self.best_score):
            # Improvement
            self.best_score = current
            self.counter = 0
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} improved to {current:.6f}"
                )
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} did not improve "
                    f"({self.counter}/{self.patience})"
                )

            if self.counter >= self.patience:
                self.should_stop = True
                trainer.should_stop = True
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: Stopping training after {epoch + 1} epochs"
                    )


class ModelCheckpoint(Callback):
    """
    Model checkpoint callback.

    Saves model checkpoints during training.
    """

    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_last: bool = True,
        filename_format: str = 'epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pt',
        verbose: bool = True
    ):
        """
        Initialize model checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
            save_last: Always save last checkpoint
            filename_format: Format string for checkpoint filename
            verbose: Print messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.filename_format = filename_format
        self.verbose = verbose

        self.best_score = None
        self.best_checkpoint_path = None

        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Save checkpoint if conditions are met."""
        current = metrics.get(self.monitor)

        should_save = False

        if not self.save_best_only:
            should_save = True
        elif current is not None:
            if self.best_score is None or self.monitor_op(current, self.best_score):
                self.best_score = current
                should_save = True

        if should_save:
            # Format filename
            filename = self.filename_format.format(
                epoch=epoch,
                **{k: v for k, v in metrics.items()}
            )
            checkpoint_path = self.checkpoint_dir / filename

            # Save checkpoint
            self._save_checkpoint(trainer, checkpoint_path, epoch, metrics)

            if current is not None and (self.best_score is None or current == self.best_score):
                self.best_checkpoint_path = checkpoint_path

                # Also save as best.pt
                best_path = self.checkpoint_dir / 'best.pt'
                self._save_checkpoint(trainer, best_path, epoch, metrics)

                if self.verbose:
                    logger.info(f"Saved best checkpoint: {best_path}")

        # Always save last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / 'last.pt'
            self._save_checkpoint(trainer, last_path, epoch, metrics)

    def _save_checkpoint(
        self,
        trainer,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Save checkpoint to disk."""
        checkpoint = {
            'epoch': epoch,
            'protein_encoder_state_dict': trainer.protein_encoder.state_dict(),
            'reaction_encoder_state_dict': trainer.reaction_encoder.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': metrics,
            'config': trainer.config.__dict__ if hasattr(trainer.config, '__dict__') else {},
        }

        if trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()

        if trainer.scaler is not None:
            checkpoint['scaler_state_dict'] = trainer.scaler.state_dict()

        torch.save(checkpoint, path)

        if self.verbose:
            logger.info(f"Saved checkpoint: {path}")


class LearningRateMonitor(Callback):
    """
    Learning rate monitor callback.

    Logs learning rate at each step.
    """

    def __init__(self, logging_interval: str = 'epoch'):
        """
        Initialize LR monitor.

        Args:
            logging_interval: 'epoch' or 'step'
        """
        self.logging_interval = logging_interval

    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        """Log learning rate after each batch."""
        if self.logging_interval == 'step':
            lr = trainer.optimizer.param_groups[0]['lr']
            if hasattr(trainer, 'logger') and trainer.logger is not None:
                trainer.logger.log({'learning_rate': lr}, step=trainer.global_step)

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Log learning rate after each epoch."""
        if self.logging_interval == 'epoch':
            lr = trainer.optimizer.param_groups[0]['lr']
            metrics['learning_rate'] = lr


class MetricsLogger(Callback):
    """
    Metrics logging callback.

    Logs metrics to console and/or file.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        console: bool = True
    ):
        """
        Initialize metrics logger.

        Args:
            log_file: Path to log file
            console: Log to console
        """
        self.log_file = Path(log_file) if log_file else None
        self.console = console

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Log metrics."""
        message = f"Epoch {epoch + 1}: " + ", ".join(
            f"{k}={v:.6f}" for k, v in metrics.items()
        )

        if self.console:
            logger.info(message)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {message}\n")


class ProgressBar(Callback):
    """
    Progress bar callback using tqdm.
    """

    def __init__(self):
        """Initialize progress bar."""
        self.epoch_pbar = None
        self.batch_pbar = None

    def on_train_begin(self, trainer, **kwargs):
        """Initialize epoch progress bar."""
        from tqdm import tqdm
        self.epoch_pbar = tqdm(
            total=trainer.config.max_epochs,
            desc="Training",
            position=0
        )

    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Initialize batch progress bar."""
        from tqdm import tqdm
        if hasattr(trainer, 'train_dataloader'):
            total_batches = len(trainer.train_dataloader)
            self.batch_pbar = tqdm(
                total=total_batches,
                desc=f"Epoch {epoch + 1}",
                position=1,
                leave=False
            )

    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        """Update batch progress bar."""
        if self.batch_pbar:
            self.batch_pbar.update(1)
            self.batch_pbar.set_postfix({'loss': f'{loss:.4f}'})

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Update epoch progress bar."""
        if self.batch_pbar:
            self.batch_pbar.close()
            self.batch_pbar = None

        if self.epoch_pbar:
            self.epoch_pbar.update(1)
            self.epoch_pbar.set_postfix(metrics)

    def on_train_end(self, trainer, **kwargs):
        """Close progress bars."""
        if self.epoch_pbar:
            self.epoch_pbar.close()
        if self.batch_pbar:
            self.batch_pbar.close()


class GradientClipping(Callback):
    """
    Gradient clipping callback.
    """

    def __init__(self, max_norm: float = 1.0):
        """
        Initialize gradient clipping.

        Args:
            max_norm: Maximum gradient norm
        """
        self.max_norm = max_norm

    def on_batch_end(self, trainer, batch_idx: int, loss: float, **kwargs):
        """Clip gradients."""
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(trainer.protein_encoder.parameters()) +
                list(trainer.reaction_encoder.parameters()),
                self.max_norm
            )


# Export
__all__ = [
    'Callback',
    'CallbackList',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateMonitor',
    'MetricsLogger',
    'ProgressBar',
    'GradientClipping',
]
