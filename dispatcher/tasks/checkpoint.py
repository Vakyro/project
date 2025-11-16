"""
Checkpoint-related tasks for dispatcher.

Provides tasks for downloading, validating, and loading CLIPZyme checkpoints.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

from ..core.task import Task, TaskConfig, TaskContext, TaskResult, TaskStatus
from ..monitoring.progress import ProgressTracker


logger = logging.getLogger(__name__)


class DownloadCheckpointTask(Task):
    """Download CLIPZyme checkpoint from Zenodo."""

    def __init__(self, config: TaskConfig, checkpoint_name: str, output_dir: Optional[Path] = None):
        """
        Initialize download checkpoint task.

        Args:
            config: Task configuration
            checkpoint_name: Name of checkpoint to download
            output_dir: Output directory for checkpoint
        """
        super().__init__(config)
        self.checkpoint_name = checkpoint_name
        self.output_dir = output_dir or Path("checkpoints")

    def execute(self, context: TaskContext) -> TaskResult:
        """Download checkpoint from Zenodo."""
        from datetime import datetime
        start_time = datetime.now()

        try:
            from checkpoints.downloader import CheckpointDownloader

            downloader = CheckpointDownloader(output_dir=str(self.output_dir))

            # Progress callback
            def progress_callback(current, total, message):
                if context.progress_callback:
                    percentage = (current / total * 100) if total > 0 else 0
                    context.progress_callback(percentage, message)

            # Download checkpoint
            checkpoint_path = downloader.download(
                self.checkpoint_name,
                progress_callback=progress_callback
            )

            # Store path in shared state for downstream tasks
            context.shared_state['checkpoint_path'] = checkpoint_path

            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=checkpoint_path,
                metadata={'checkpoint_name': self.checkpoint_name},
                start_time=start_time,
                end_time=datetime.now()
            )

        except Exception as e:
            logger.error(f"Checkpoint download failed: {str(e)}")
            import traceback
            return TaskResult(
                status=TaskStatus.FAILED,
                error=e,
                error_traceback=traceback.format_exc(),
                start_time=start_time,
                end_time=datetime.now()
            )


class ValidateCheckpointTask(Task):
    """Validate CLIPZyme checkpoint integrity."""

    def __init__(self, config: TaskConfig, checkpoint_path: Optional[Path] = None):
        """
        Initialize validate checkpoint task.

        Args:
            config: Task configuration
            checkpoint_path: Path to checkpoint (if None, uses shared state)
        """
        super().__init__(config)
        self.checkpoint_path = checkpoint_path

    def execute(self, context: TaskContext) -> TaskResult:
        """Validate checkpoint."""
        from datetime import datetime
        start_time = datetime.now()

        try:
            # Get checkpoint path
            checkpoint_path = self.checkpoint_path
            if checkpoint_path is None:
                checkpoint_path = context.shared_state.get('checkpoint_path')

            if checkpoint_path is None:
                raise ValueError("No checkpoint path provided")

            checkpoint_path = Path(checkpoint_path)

            # Validate existence
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # Validate it can be loaded
            import torch
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                logger.info(f"Checkpoint validated: {checkpoint_path}")
            except Exception as e:
                raise ValueError(f"Invalid checkpoint format: {str(e)}")

            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=True,
                metadata={'checkpoint_path': str(checkpoint_path)},
                start_time=start_time,
                end_time=datetime.now()
            )

        except Exception as e:
            logger.error(f"Checkpoint validation failed: {str(e)}")
            import traceback
            return TaskResult(
                status=TaskStatus.FAILED,
                error=e,
                error_traceback=traceback.format_exc(),
                start_time=start_time,
                end_time=datetime.now()
            )


class LoadCheckpointTask(Task):
    """Load CLIPZyme checkpoint into model."""

    def __init__(
        self,
        config: TaskConfig,
        checkpoint_path: Optional[Path] = None,
        device: str = "cuda"
    ):
        """
        Initialize load checkpoint task.

        Args:
            config: Task configuration
            checkpoint_path: Path to checkpoint (if None, uses shared state)
            device: Device to load model on
        """
        super().__init__(config)
        self.checkpoint_path = checkpoint_path
        self.device = device

    def execute(self, context: TaskContext) -> TaskResult:
        """Load checkpoint into model."""
        from datetime import datetime
        start_time = datetime.now()

        try:
            # Get checkpoint path
            checkpoint_path = self.checkpoint_path
            if checkpoint_path is None:
                checkpoint_path = context.shared_state.get('checkpoint_path')

            if checkpoint_path is None:
                raise ValueError("No checkpoint path provided")

            checkpoint_path = Path(checkpoint_path)

            # Load checkpoint
            from checkpoints.loader import CheckpointLoader

            loader = CheckpointLoader(device=self.device)
            model = loader.load(checkpoint_path)

            # Store model in shared state
            context.shared_state['model'] = model

            logger.info(f"Checkpoint loaded: {checkpoint_path}")

            return TaskResult(
                status=TaskStatus.COMPLETED,
                output=model,
                metadata={
                    'checkpoint_path': str(checkpoint_path),
                    'device': self.device
                },
                start_time=start_time,
                end_time=datetime.now()
            )

        except Exception as e:
            logger.error(f"Checkpoint loading failed: {str(e)}")
            import traceback
            return TaskResult(
                status=TaskStatus.FAILED,
                error=e,
                error_traceback=traceback.format_exc(),
                start_time=start_time,
                end_time=datetime.now()
            )


# Export
__all__ = [
    'DownloadCheckpointTask',
    'ValidateCheckpointTask',
    'LoadCheckpointTask',
]
