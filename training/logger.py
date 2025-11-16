"""
Training loggers for CLIPZyme.

Provides integration with WandB, TensorBoard, and console logging.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Base class for training loggers.
    """

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        raise NotImplementedError

    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            params: Dictionary of hyperparameters
        """
        pass

    def finish(self):
        """Finish logging session."""
        pass


class ConsoleLogger(TrainingLogger):
    """
    Simple console logger.
    """

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to console."""
        step_str = f"Step {step}: " if step is not None else ""
        metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        logger.info(f"{step_str}{metrics_str}")


class WandbLogger(TrainingLogger):
    """
    Weights & Biases logger.

    Requires: pip install wandb
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        resume: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            name: Run name
            config: Configuration dictionary
            tags: List of tags
            notes: Notes for this run
            resume: Resume mode ('allow', 'must', 'never')
            **kwargs: Additional arguments for wandb.init()
        """
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "WandB not installed. Install with: pip install wandb"
            )

        self.run = self.wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            resume=resume,
            **kwargs
        )

        logger.info(f"WandB initialized. Run: {self.run.name}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to WandB."""
        self.wandb.log(metrics, step=step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to WandB."""
        self.wandb.config.update(params)

    def finish(self):
        """Finish WandB run."""
        self.wandb.finish()
        logger.info("WandB run finished")


class TensorBoardLogger(TrainingLogger):
    """
    TensorBoard logger.

    Requires: pip install tensorboard
    """

    def __init__(
        self,
        log_dir: str = 'runs',
        comment: str = '',
        purge_step: Optional[int] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = ''
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment to append to log directory name
            purge_step: Step at which to purge events
            max_queue: Max events to queue before flushing
            flush_secs: Seconds between flushes
            filename_suffix: Suffix for log filename
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.SummaryWriter = SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard not installed. Install with: pip install tensorboard"
            )

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = self.SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix
        )

        logger.info(f"TensorBoard initialized. Log dir: {self.log_dir}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to TensorBoard."""
        if step is None:
            step = 0

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            elif isinstance(value, str):
                self.writer.add_text(key, value, step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        # Convert params to text format
        text = "\n".join(f"{k}: {v}" for k, v in params.items())
        self.writer.add_text('hyperparameters', text)

    def finish(self):
        """Close TensorBoard writer."""
        self.writer.close()
        logger.info("TensorBoard writer closed")


class MultiLogger(TrainingLogger):
    """
    Combines multiple loggers.
    """

    def __init__(self, loggers: list):
        """
        Initialize multi-logger.

        Args:
            loggers: List of TrainingLogger instances
        """
        self.loggers = loggers

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log to all loggers."""
        for logger_instance in self.loggers:
            logger_instance.log(metrics, step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to all loggers."""
        for logger_instance in self.loggers:
            logger_instance.log_hyperparameters(params)

    def finish(self):
        """Finish all loggers."""
        for logger_instance in self.loggers:
            logger_instance.finish()


def create_logger(
    logger_type: str = 'console',
    **kwargs
) -> TrainingLogger:
    """
    Create a logger instance.

    Args:
        logger_type: Type of logger ('console', 'wandb', 'tensorboard')
        **kwargs: Logger-specific arguments

    Returns:
        TrainingLogger instance
    """
    if logger_type == 'console':
        return ConsoleLogger()
    elif logger_type == 'wandb':
        return WandbLogger(**kwargs)
    elif logger_type == 'tensorboard':
        return TensorBoardLogger(**kwargs)
    else:
        raise ValueError(
            f"Unknown logger type: {logger_type}. "
            f"Available: 'console', 'wandb', 'tensorboard'"
        )


# Export
__all__ = [
    'TrainingLogger',
    'ConsoleLogger',
    'WandbLogger',
    'TensorBoardLogger',
    'MultiLogger',
    'create_logger',
]
