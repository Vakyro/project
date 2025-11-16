"""
Structured logging for dispatcher.

Provides enhanced logging with structured data and multiple outputs.
"""

import logging
import json
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import sys


class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class DispatcherLogger:
    """
    Enhanced logger for dispatcher system.

    Provides structured logging with multiple outputs and filtering.
    """

    def __init__(
        self,
        name: str = "dispatcher",
        log_dir: Optional[Path] = None,
        level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True,
        structured: bool = False
    ):
        """
        Initialize dispatcher logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            console_output: Enable console output
            file_output: Enable file output
            structured: Use structured JSON logging
        """
        self.name = name
        self.log_dir = log_dir or (Path.home() / '.clipzyme' / 'dispatcher' / 'logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Remove existing handlers

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)

            if structured:
                console_formatter = StructuredFormatter()
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )

            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if file_output:
            log_file = self.log_dir / f"{name}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)

            if structured:
                file_formatter = StructuredFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )

            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, extra=kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg, extra=kwargs)

    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, extra=kwargs)

    def error(self, msg: str, **kwargs):
        """Log error message."""
        self.logger.error(msg, extra=kwargs)

    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, extra=kwargs)

    def log_task_start(self, task_name: str, task_id: str, **metadata):
        """Log task start."""
        self.info(
            f"Task started: {task_name}",
            task_id=task_id,
            task_name=task_name,
            event='task_start',
            **metadata
        )

    def log_task_complete(self, task_name: str, task_id: str, duration: float, **metadata):
        """Log task completion."""
        self.info(
            f"Task completed: {task_name} ({duration:.2f}s)",
            task_id=task_id,
            task_name=task_name,
            duration=duration,
            event='task_complete',
            **metadata
        )

    def log_task_failed(self, task_name: str, task_id: str, error: str, **metadata):
        """Log task failure."""
        self.error(
            f"Task failed: {task_name} - {error}",
            task_id=task_id,
            task_name=task_name,
            error=error,
            event='task_failed',
            **metadata
        )

    def log_workflow_start(self, workflow_name: str, workflow_id: str, **metadata):
        """Log workflow start."""
        self.info(
            f"Workflow started: {workflow_name}",
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            event='workflow_start',
            **metadata
        )

    def log_workflow_complete(
        self,
        workflow_name: str,
        workflow_id: str,
        duration: float,
        **metadata
    ):
        """Log workflow completion."""
        self.info(
            f"Workflow completed: {workflow_name} ({duration:.2f}s)",
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            duration=duration,
            event='workflow_complete',
            **metadata
        )

    def log_workflow_failed(
        self,
        workflow_name: str,
        workflow_id: str,
        error: str,
        **metadata
    ):
        """Log workflow failure."""
        self.error(
            f"Workflow failed: {workflow_name} - {error}",
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            error=error,
            event='workflow_failed',
            **metadata
        )

    def set_level(self, level: int):
        """Set logging level."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)


# Global logger instance
_global_logger: Optional[DispatcherLogger] = None


def get_logger(
    name: str = "dispatcher",
    **kwargs
) -> DispatcherLogger:
    """
    Get or create dispatcher logger.

    Args:
        name: Logger name
        **kwargs: Logger configuration

    Returns:
        Dispatcher logger
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = DispatcherLogger(name=name, **kwargs)

    return _global_logger


def configure_logging(
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
    structured: bool = False
):
    """
    Configure global logging.

    Args:
        level: Logging level
        console: Enable console output
        file: Enable file output
        structured: Use structured logging
    """
    global _global_logger

    _global_logger = DispatcherLogger(
        name="dispatcher",
        level=level,
        console_output=console,
        file_output=file,
        structured=structured
    )


# Export
__all__ = [
    'DispatcherLogger',
    'StructuredFormatter',
    'get_logger',
    'configure_logging',
]
