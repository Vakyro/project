"""
Configuration validator.

Validates dispatcher configurations against schemas.
"""

from typing import Dict, Any, List, Optional
import logging


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Configuration validation error."""
    pass


class ConfigValidator:
    """Validates dispatcher configurations."""

    def __init__(self):
        """Initialize validator."""
        self.errors: List[str] = []

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        self.errors.clear()

        # Validate scheduler config
        self._validate_scheduler(config.get('scheduler', {}))

        # Validate resources config
        self._validate_resources(config.get('resources', {}))

        # Validate tasks config
        self._validate_tasks(config.get('tasks', {}))

        # Validate logging config
        self._validate_logging(config.get('logging', {}))

        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in self.errors
            )
            raise ValidationError(error_msg)

        return True

    def _validate_scheduler(self, config: Dict[str, Any]):
        """Validate scheduler configuration."""
        if 'max_concurrent_jobs' in config:
            value = config['max_concurrent_jobs']
            if not isinstance(value, int) or value < 1:
                self.errors.append(
                    f"scheduler.max_concurrent_jobs must be integer >= 1, got {value}"
                )

        if 'max_workers_per_job' in config:
            value = config['max_workers_per_job']
            if not isinstance(value, int) or value < 1:
                self.errors.append(
                    f"scheduler.max_workers_per_job must be integer >= 1, got {value}"
                )

        if 'poll_interval' in config:
            value = config['poll_interval']
            if not isinstance(value, (int, float)) or value <= 0:
                self.errors.append(
                    f"scheduler.poll_interval must be number > 0, got {value}"
                )

    def _validate_resources(self, config: Dict[str, Any]):
        """Validate resources configuration."""
        # GPU config
        if 'gpu' in config:
            gpu_config = config['gpu']

            if 'auto_allocate' in gpu_config:
                if not isinstance(gpu_config['auto_allocate'], bool):
                    self.errors.append(
                        f"resources.gpu.auto_allocate must be boolean"
                    )

            if 'min_free_memory_mb' in gpu_config:
                value = gpu_config['min_free_memory_mb']
                if not isinstance(value, int) or value < 0:
                    self.errors.append(
                        f"resources.gpu.min_free_memory_mb must be integer >= 0"
                    )

        # Memory config
        if 'memory' in config:
            mem_config = config['memory']

            if 'reserve_ratio' in mem_config:
                value = mem_config['reserve_ratio']
                if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                    self.errors.append(
                        f"resources.memory.reserve_ratio must be number in [0, 1]"
                    )

        # Cache config
        if 'cache' in config:
            cache_config = config['cache']

            for size_key in ['l1_size_mb', 'l2_size_mb']:
                if size_key in cache_config:
                    value = cache_config[size_key]
                    if not isinstance(value, int) or value < 0:
                        self.errors.append(
                            f"resources.cache.{size_key} must be integer >= 0"
                        )

    def _validate_tasks(self, config: Dict[str, Any]):
        """Validate tasks configuration."""
        # Checkpoint config
        if 'checkpoint' in config:
            checkpoint_config = config['checkpoint']

            if 'device' in checkpoint_config:
                device = checkpoint_config['device']
                if device not in ['cpu', 'cuda', 'mps']:
                    self.errors.append(
                        f"tasks.checkpoint.device must be 'cpu', 'cuda', or 'mps'"
                    )

        # Screening config
        if 'screening' in config:
            screening_config = config['screening']

            if 'mode' in screening_config:
                mode = screening_config['mode']
                if mode not in ['interactive', 'batched']:
                    self.errors.append(
                        f"tasks.screening.mode must be 'interactive' or 'batched'"
                    )

            if 'batch_size' in screening_config:
                value = screening_config['batch_size']
                if not isinstance(value, int) or value < 1:
                    self.errors.append(
                        f"tasks.screening.batch_size must be integer >= 1"
                    )

            if 'top_k' in screening_config:
                value = screening_config['top_k']
                if not isinstance(value, int) or value < 1:
                    self.errors.append(
                        f"tasks.screening.top_k must be integer >= 1"
                    )

    def _validate_logging(self, config: Dict[str, Any]):
        """Validate logging configuration."""
        if 'level' in config:
            level = config['level']
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if level not in valid_levels:
                self.errors.append(
                    f"logging.level must be one of {valid_levels}"
                )

        for bool_key in ['console', 'file', 'structured']:
            if bool_key in config:
                if not isinstance(config[bool_key], bool):
                    self.errors.append(
                        f"logging.{bool_key} must be boolean"
                    )


# Export
__all__ = [
    'ConfigValidator',
    'ValidationError',
]
